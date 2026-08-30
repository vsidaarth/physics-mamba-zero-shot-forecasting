#!/usr/bin/env python3
"""Audit processed BSRN station files before any model training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REQUIRED_COLUMNS = {
    "timestamp", "ghi_wm2", "ghi_primary_valid", "ghi_strict_valid",
    "ghi_coverage", "toa_horizontal_wm2", "station_code", "split",
    "latitude", "longitude", "elevation_m", "native_cadence_minutes",
    "time_contiguous",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect-train", type=int, default=33)
    parser.add_argument("--expect-val", type=int, default=7)
    parser.add_argument("--allow-test", action="store_true")
    return parser.parse_args()


def audit_file(path: Path) -> dict[str, object]:
    schema_columns = set(pq.read_schema(path).names)
    missing_columns = sorted(REQUIRED_COLUMNS - schema_columns)
    frame = pd.read_parquet(path, columns=sorted(REQUIRED_COLUMNS & schema_columns))
    errors: list[str] = []
    if missing_columns:
        errors.append("missing columns: " + ", ".join(missing_columns))
        return {"path": str(path), "errors": errors}

    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    station = path.stem
    split = path.parent.name
    primary = frame["ghi_primary_valid"].to_numpy(dtype=bool)
    strict = frame["ghi_strict_valid"].to_numpy(dtype=bool)
    ghi = frame["ghi_wm2"].to_numpy(dtype=np.float64)

    if not timestamps.is_monotonic_increasing:
        errors.append("timestamps are not monotonic")
    if timestamps.has_duplicates:
        errors.append("timestamps contain duplicates")
    if not np.all((timestamps.minute % 15 == 0) & (timestamps.second == 0)):
        errors.append("timestamps are not aligned to 15-minute boundaries")
    if not frame["station_code"].eq(station).all():
        errors.append("station_code does not match filename")
    if not frame["split"].eq(split).all():
        errors.append("split column does not match directory")
    if np.any(strict & ~primary):
        errors.append("strict-valid bins are not a subset of primary-valid bins")
    if not np.array_equal(primary, np.isfinite(ghi)):
        errors.append("ghi_primary_valid disagrees with finite ghi_wm2")
    if np.any(ghi[np.isfinite(ghi)] < 0.0):
        errors.append("valid GHI contains negative values")
    if np.any((frame["ghi_coverage"] < 0.0) | (frame["ghi_coverage"] > 1.0)):
        errors.append("GHI coverage lies outside [0, 1]")

    expected_contiguous = np.r_[
        False,
        (timestamps[1:] - timestamps[:-1]) == pd.Timedelta(minutes=15),
    ]
    if not np.array_equal(
        frame["time_contiguous"].to_numpy(dtype=bool), expected_contiguous
    ):
        errors.append("time_contiguous is inconsistent with timestamps")

    return {
        "path": str(path),
        "station": station,
        "split": split,
        "rows": int(len(frame)),
        "primary_valid_bins": int(primary.sum()),
        "strict_valid_bins": int(strict.sum()),
        "start": timestamps.min().isoformat(),
        "end": timestamps.max().isoformat(),
        "native_cadence_minutes": sorted(
            int(value) for value in frame["native_cadence_minutes"].unique()
        ),
        "size_bytes": path.stat().st_size,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    files = sorted(args.processed_dir.glob("*/*.parquet"))
    reports = [audit_file(path) for path in files]
    split_counts = Counter(report.get("split") for report in reports)
    global_errors: list[str] = []
    if split_counts["train"] != args.expect_train:
        global_errors.append(
            f"expected {args.expect_train} train stations, found {split_counts['train']}"
        )
    if split_counts["val"] != args.expect_val:
        global_errors.append(
            f"expected {args.expect_val} val stations, found {split_counts['val']}"
        )
    if split_counts["test"] and not args.allow_test:
        global_errors.append("sealed test files are present")
    if any(report["errors"] for report in reports):
        global_errors.append("one or more station files failed integrity checks")

    summary = {
        "passed": not global_errors,
        "station_files": len(files),
        "split_counts": dict(split_counts),
        "total_rows": sum(int(report.get("rows", 0)) for report in reports),
        "total_primary_valid_bins": sum(
            int(report.get("primary_valid_bins", 0)) for report in reports
        ),
        "total_strict_valid_bins": sum(
            int(report.get("strict_valid_bins", 0)) for report in reports
        ),
        "total_size_bytes": sum(int(report.get("size_bytes", 0)) for report in reports),
        "errors": global_errors,
        "stations": reports,
    }
    output = args.output or args.processed_dir / "integrity_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "stations"}, indent=2))
    if global_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
