#!/usr/bin/env python3
"""Stream BSRN station ZIPs into leakage-controlled 15-minute Parquet files."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


MEMBER_RE = re.compile(
    r"(?:^|/)([A-Z0-9]{3})_(\d{4})-(\d{2})_0100(?:\+0300)?_QC_combined\.txt$"
)
CADENCE_CHOICES = np.asarray([1, 2, 3, 5, 10], dtype=float)

COLUMN_PREFIXES = {
    "DIF [": "dhi_wm2",
    "DIR [": "dni_wm2",
    "LWD [": "lwd_wm2",
    "SWD [": "ghi_wm2",
    "T2 [": "air_temperature_c",
    "RH [": "relative_humidity_pct",
    "PoPoPoPo [": "pressure_hpa",
    "SZA [": "solar_zenith_deg",
    "Mu0": "cos_zenith",
    "SWDTOA [": "toa_horizontal_wm2",
}
QC_COLUMNS = {
    "dhi_wm2": "DIFQc",
    "dni_wm2": "DIRQc",
    "lwd_wm2": "LWDQc",
    "ghi_wm2": "SWDQc",
    "air_temperature_c": "T2Qc",
}
MEAN_COLUMNS = list(COLUMN_PREFIXES.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/protocol_v1.json")
    )
    parser.add_argument(
        "--stations",
        help="Comma-separated station codes for a smoke test or targeted rebuild.",
    )
    parser.add_argument(
        "--max-months", type=int, help="Process at most this many months per station."
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Unlock test-station preprocessing after the development protocol is frozen.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path, include_test: bool, requested: set[str] | None) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    allowed_splits = {"train", "val"} | ({"test"} if include_test else set())
    selected = [
        row
        for row in rows
        if row["eligible"].lower() == "true"
        and row["split"] in allowed_splits
        and (requested is None or row["station_code"] in requested)
    ]
    if requested:
        missing = requested - {row["station_code"] for row in selected}
        if missing:
            raise ValueError(
                "Requested stations are unavailable under the current split lock: "
                + ", ".join(sorted(missing))
            )
    return selected


def canonical_columns(columns: list[str]) -> dict[str, str]:
    rename: dict[str, str] = {}
    for original in columns:
        if original == "Date/Time":
            rename[original] = "timestamp"
            continue
        if original in QC_COLUMNS.values():
            rename[original] = original
            continue
        for prefix, canonical in COLUMN_PREFIXES.items():
            if original == prefix or original.startswith(prefix):
                rename[original] = canonical
                break
    return rename


def infer_cadence_minutes(timestamp: pd.Series, maximum: int) -> int:
    deltas = timestamp.sort_values().diff().dt.total_seconds().div(60)
    deltas = deltas[(deltas > 0) & (deltas <= maximum)]
    if deltas.empty:
        return 1
    median = float(deltas.median())
    return int(CADENCE_CHOICES[np.abs(CADENCE_CHOICES - median).argmin()])


def valid_measurements(
    frame: pd.DataFrame, value: str, reject_mask: int
) -> tuple[pd.Series, pd.Series]:
    values = pd.to_numeric(frame.get(value), errors="coerce")
    qc_name = QC_COLUMNS.get(value)
    if qc_name is None:
        valid = values.notna()
    elif qc_name not in frame:
        valid = pd.Series(False, index=frame.index)
    else:
        qc = pd.to_numeric(frame[qc_name], errors="coerce").astype("Int64")
        valid = values.notna() & qc.notna() & ((qc.fillna(reject_mask).astype(int) & reject_mask) == 0)
    return values.where(valid), valid


def aggregate_member(
    raw: bytes,
    resolution_minutes: int,
    minimum_coverage: float,
    maximum_cadence: int,
    reject_mask: int,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    marker = b"Date/Time\t"
    header_offset = raw.find(marker)
    if header_offset < 0:
        raise ValueError("No tabular Date/Time header found")

    source = pd.read_csv(
        io.BytesIO(raw[header_offset:]),
        sep="\t",
        low_memory=False,
    )
    source = source.rename(columns=canonical_columns(list(source.columns)))
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True, errors="coerce")
    source = source.dropna(subset=["timestamp"]).sort_values("timestamp")
    source = source.drop_duplicates(subset=["timestamp"], keep="last")
    cadence = infer_cadence_minutes(source["timestamp"], maximum_cadence)
    expected_samples = max(1, resolution_minutes // cadence)
    minimum_samples = max(1, math.ceil(expected_samples * minimum_coverage))
    indexed = source.set_index("timestamp")
    rule = f"{resolution_minutes}min"

    output = pd.DataFrame(index=indexed.resample(rule).size().index)
    output["raw_samples"] = indexed.resample(rule).size().astype("int16")
    output["native_cadence_minutes"] = np.int16(cadence)
    output["expected_native_samples"] = np.int16(expected_samples)

    for column in MEAN_COLUMNS:
        if column not in indexed:
            output[column] = np.nan
            output[f"{column}_valid_samples"] = np.int16(0)
            continue
        values, valid = valid_measurements(indexed, column, reject_mask)
        counts = valid.resample(rule).sum().astype("int16")
        means = values.resample(rule).mean().where(counts >= minimum_samples)
        output[column] = means
        output[f"{column}_valid_samples"] = counts

    if "SWDQc" in indexed:
        swd_qc = pd.to_numeric(indexed["SWDQc"], errors="coerce").astype("Int64")
        swd_values = pd.to_numeric(indexed.get("ghi_wm2"), errors="coerce")
        strict = (swd_values.notna() & (swd_qc == 0)).resample(rule).sum().astype("int16")
    else:
        strict = pd.Series(0, index=output.index, dtype="int16")
    output["ghi_qc_zero_samples"] = strict
    output["ghi_primary_valid"] = output["ghi_wm2"].notna()
    output["ghi_strict_valid"] = strict >= minimum_samples
    output["ghi_coverage"] = (
        output["ghi_wm2_valid_samples"] / output["expected_native_samples"]
    ).clip(upper=1.0)
    output["ghi_wm2"] = output["ghi_wm2"].clip(lower=0)
    output = output[output["raw_samples"] > 0].reset_index()

    diagnostics = {
        "native_cadence_minutes": cadence,
        "input_rows": int(len(source)),
        "output_bins": int(len(output)),
        "valid_ghi_bins": int(output["ghi_primary_valid"].sum()),
        "strict_ghi_bins": int(output["ghi_strict_valid"].sum()),
    }
    return output, diagnostics


def add_context_features(frame: pd.DataFrame, station: dict[str, str]) -> pd.DataFrame:
    timestamp = frame["timestamp"]
    hour = timestamp.dt.hour + timestamp.dt.minute / 60
    day = timestamp.dt.dayofyear
    frame["utc_hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    frame["utc_hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    frame["day_of_year_sin"] = np.sin(2 * np.pi * (day - 1) / 365.2425).astype("float32")
    frame["day_of_year_cos"] = np.cos(2 * np.pi * (day - 1) / 365.2425).astype("float32")
    frame["station_code"] = station["station_code"]
    frame["split"] = station["split"]
    frame["latitude"] = np.float32(station["latitude"])
    frame["longitude"] = np.float32(station["longitude"])
    frame["elevation_m"] = np.float32(station["elevation_m"] or np.nan)
    frame["time_contiguous"] = frame["timestamp"].diff().eq(pd.Timedelta(minutes=15))
    return frame


def process_station(
    station: dict[str, str],
    archive: Path,
    destination: Path,
    config: dict,
    max_months: int | None,
) -> dict:
    aggregate = config["aggregation"]
    reject_mask = int(config["quality_control"]["primary_reject_bitmask"])
    monthly: list[pd.DataFrame] = []
    cadence_counts: Counter[int] = Counter()
    input_rows = valid_bins = strict_bins = 0

    with zipfile.ZipFile(archive) as handle:
        members = sorted(name for name in handle.namelist() if MEMBER_RE.search(name))
        if max_months is not None:
            members = members[:max_months]
        for member in members:
            frame, diagnostics = aggregate_member(
                handle.read(member),
                resolution_minutes=int(aggregate["resolution_minutes"]),
                minimum_coverage=float(aggregate["minimum_native_coverage"]),
                maximum_cadence=int(aggregate["maximum_inferred_native_cadence_minutes"]),
                reject_mask=reject_mask,
            )
            monthly.append(frame)
            cadence_counts[int(diagnostics["native_cadence_minutes"])] += 1
            input_rows += int(diagnostics["input_rows"])
            valid_bins += int(diagnostics["valid_ghi_bins"])
            strict_bins += int(diagnostics["strict_ghi_bins"])

    if not monthly:
        raise ValueError(f"No QC-combined monthly files in {archive}")
    combined = pd.concat(monthly, ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    combined = add_context_features(combined.reset_index(drop=True), station)

    float_columns = combined.select_dtypes(include=["float64"]).columns
    combined[float_columns] = combined[float_columns].astype("float32")
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(destination, index=False, compression="zstd")

    return {
        "station_code": station["station_code"],
        "split": station["split"],
        "archive": archive.name,
        "months": len(monthly),
        "input_rows": input_rows,
        "output_bins": int(len(combined)),
        "valid_ghi_bins": valid_bins,
        "strict_ghi_bins": strict_bins,
        "cadence_months": {str(key): cadence_counts[key] for key in sorted(cadence_counts)},
        "first_timestamp": combined["timestamp"].min().isoformat(),
        "last_timestamp": combined["timestamp"].max().isoformat(),
        "output": destination.as_posix(),
    }


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    requested = (
        {value.strip().upper() for value in args.stations.split(",") if value.strip()}
        if args.stations
        else None
    )
    manifest = load_manifest(
        args.dataset_root / "final_station_manifest.csv", args.include_test, requested
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for index, station in enumerate(manifest, start=1):
        code = station["station_code"]
        destination = args.output_dir / station["split"] / f"{code}.parquet"
        if destination.exists() and not args.overwrite:
            print(f"[{index:02d}/{len(manifest):02d}] skip {code}: output exists")
            continue
        print(f"[{index:02d}/{len(manifest):02d}] preprocess {code}", flush=True)
        reports.append(
            process_station(
                station,
                args.dataset_root / "archives" / f"{code}.zip",
                destination,
                config,
                args.max_months,
            )
        )

    summary = {
        "protocol_id": config["protocol_id"],
        "test_included": args.include_test,
        "station_reports": reports,
    }
    report_path = args.output_dir / "preprocessing_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
