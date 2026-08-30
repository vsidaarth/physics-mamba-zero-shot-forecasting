#!/usr/bin/env python3
"""Create checksums, coverage audit, and geographic zero-shot splits for BSRN."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path


MEMBER_RE = re.compile(r"(?:^|/)([A-Z0-9]{3})_(\d{4})-(\d{2})_0100(?:\+0300)?_QC_combined\.txt$")
REGION_CODES = {
    "Africa": set("DAA GOB ILO RUN TAM".split()),
    "Antarctica": set("DOM GVN SPO SYO".split()),
    "Asia": set("ABS FUA GAN GUR HOW ISH LYU MNM SAP SBO SOV TAT TIR XIA YUS".split()),
    "Europe/Arctic": set("BUD CAB CAM CAP CAR CNR ENA INO IZA LER LIN NYA PAL PAY SON TIK TOR".split()),
    "North America": set("ALE BAR BER BIL BON BOS BOU CLH DRA E13 EUR FPE GCR GIM LRC NEW PSU REG SEL SXF".split()),
    "Oceania": set("ASP COC DAR DWN KWA LAU MAN NAU".split()),
    "South America": set("BRB FLO OHY PAR PTR RLM SMS".split()),
}
# Remove near-duplicate monitoring locations while retaining the longer record.
PREFERRED_NEAR_DUPLICATES = {"BIL": "E13", "BOS": "BOU", "DAR": "DWN"}


def region_for(code: str) -> str:
    matches = [region for region, codes in REGION_CODES.items() if code in codes]
    if len(matches) != 1:
        raise ValueError(f"Station {code} has {len(matches)} region assignments")
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def longest_month_run(months: set[tuple[int, int]]) -> int:
    values = sorted(year * 12 + month - 1 for year, month in months)
    longest = run = 0
    previous = None
    for value in values:
        run = run + 1 if previous is not None and value == previous + 1 else 1
        longest = max(longest, run)
        previous = value
    return longest


def haversine_km(a: dict, b: dict) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a["latitude"], a["longitude"], b["latitude"], b["longitude"]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--minimum-consecutive-months", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = json.loads((args.dataset_root / "inventory.json").read_text(encoding="utf-8"))
    by_code = {item["station_code"]: item for item in inventory}
    archive_dir = args.dataset_root / "archives"
    rows = []
    checksum_lines = []

    for index, code in enumerate(sorted(by_code), start=1):
        path = archive_dir / f"{code}.zip"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[{index:02d}/{len(by_code)}] audit {path.name}", flush=True)
        digest = sha256(path)
        checksum_lines.append(f"{digest}  archives/{path.name}")
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            months = set()
            variants = set()
            for name in names:
                match = MEMBER_RE.search(name)
                if match:
                    member_code, year, month = match.groups()
                    if member_code != code:
                        raise ValueError(f"{path.name}: member station {member_code}")
                    months.add((int(year), int(month)))
                    variants.add("0100+0300" if "0100+0300" in name else "0100")
            if not months:
                raise ValueError(f"{path.name}: no QC-combined LR0100 monthly files")
        item = by_code[code]
        counts_by_year = defaultdict(int)
        for year, _month in months:
            counts_by_year[year] += 1
        complete_years = sorted(year for year, count in counts_by_year.items() if count == 12)
        rows.append(
            {
                "station_code": code,
                "station_name": item["station_name"],
                "region": region_for(code),
                "latitude": item["latitude"],
                "longitude": item["longitude"],
                "elevation_m": item["elevation_m"],
                "archive_bytes": path.stat().st_size,
                "sha256": digest,
                "monthly_files": len(months),
                "first_month": f"{min(months)[0]:04d}-{min(months)[1]:02d}",
                "last_month": f"{max(months)[0]:04d}-{max(months)[1]:02d}",
                "longest_consecutive_months": longest_month_run(months),
                "complete_years": len(complete_years),
                "first_complete_year": complete_years[0] if complete_years else "",
                "last_complete_year": complete_years[-1] if complete_years else "",
                "record_variants": "+".join(sorted(variants)),
                "eligible": longest_month_run(months) >= args.minimum_consecutive_months,
                "exclusion_reason": "",
            }
        )

    row_by_code = {row["station_code"]: row for row in rows}
    for keep, drop in PREFERRED_NEAR_DUPLICATES.items():
        if row_by_code[keep]["eligible"] and row_by_code[drop]["eligible"]:
            distance = haversine_km(row_by_code[keep], row_by_code[drop])
            row_by_code[drop]["eligible"] = False
            row_by_code[drop]["exclusion_reason"] = f"near_duplicate_of_{keep}_{distance:.1f}km"

    selected = [row for row in rows if row["eligible"]]
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["region"]].append(row)
    for region, stations in grouped.items():
        stations.sort(key=lambda row: hashlib.sha256(row["station_code"].encode()).hexdigest())
        test_count = max(1, round(len(stations) * 0.20))
        val_count = max(1, round(len(stations) * 0.10))
        for index, row in enumerate(stations):
            row["split"] = "test" if index < test_count else "val" if index < test_count + val_count else "train"
    for row in rows:
        row.setdefault("split", "excluded")

    fieldnames = list(rows[0])
    with (args.dataset_root / "station_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (args.dataset_root / "final_station_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(selected, key=lambda row: (row["split"], row["region"], row["station_code"])))
    (args.dataset_root / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="ascii")

    split_counts = {split: sum(row["split"] == split for row in selected) for split in ("train", "val", "test")}
    region_counts = {region: sum(row["region"] == region for row in selected) for region in sorted(grouped)}
    summary = {
        "dataset_doi": "10.1594/PANGAEA.957398",
        "archives": len(rows),
        "archive_bytes": sum(row["archive_bytes"] for row in rows),
        "monthly_station_files": sum(row["monthly_files"] for row in rows),
        "minimum_consecutive_months": args.minimum_consecutive_months,
        "selected_stations": len(selected),
        "split_counts": split_counts,
        "region_counts": region_counts,
        "excluded_stations": [
            {"station_code": row["station_code"], "reason": row["exclusion_reason"] or "insufficient_consecutive_months"}
            for row in rows if not row["eligible"]
        ],
    }
    (args.dataset_root / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
