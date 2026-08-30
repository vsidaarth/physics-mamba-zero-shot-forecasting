#!/usr/bin/env python3
"""Fit a train-station-only global climatology and evaluate unseen locations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.evaluate_persistence import (
        CLEARSKY_FLOOR_WM2, CLEARSKY_INDEX_MAX, STEP_MINUTES,
        aggregate_metrics, forecast_origin_mask, ineichen_ghi, metric_row,
    )
except ModuleNotFoundError:  # Direct execution: python scripts/evaluate_climatology.py
    from evaluate_persistence import (
        CLEARSKY_FLOOR_WM2, CLEARSKY_INDEX_MAX, STEP_MINUTES,
        aggregate_metrics, forecast_origin_mask, ineichen_ghi, metric_row,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/protocol_v1.json"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument(
        "--climatology-model", type=Path,
        help="Reuse a previously fitted climatology.npz instead of refitting.",
    )
    return parser.parse_args()


def local_solar_slot(timestamps: pd.DatetimeIndex, longitude: float) -> np.ndarray:
    """Approximate 15-minute local-solar-time slot using longitude."""
    utc_minutes = timestamps.hour * 60 + timestamps.minute
    local_minutes = np.mod(utc_minutes + longitude * 4.0, 1440.0)
    return np.floor(local_minutes / STEP_MINUTES).astype(np.int16)


def station_inputs(path: Path) -> tuple[pd.DataFrame, pd.DatetimeIndex, np.ndarray]:
    frame = pd.read_parquet(
        path,
        columns=["timestamp", "ghi_wm2", "toa_horizontal_wm2", "latitude",
                 "longitude", "elevation_m", "station_code"],
    )
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    clear = ineichen_ghi(
        timestamps, float(frame["latitude"].iloc[0]),
        float(frame["longitude"].iloc[0]), float(frame["elevation_m"].iloc[0]),
    )
    return frame, timestamps, clear


def fit_climatology(files: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((12, 96), dtype=np.float64)
    counts = np.zeros((12, 96), dtype=np.int64)
    for index, path in enumerate(files, start=1):
        frame, timestamps, clear = station_inputs(path)
        print(f"Fit [{index:02d}/{len(files):02d}] {path.stem}", flush=True)
        ghi = frame["ghi_wm2"].to_numpy(dtype=np.float64)
        valid = np.isfinite(ghi) & np.isfinite(clear) & (clear >= CLEARSKY_FLOOR_WM2)
        clear_index = np.clip(
            ghi[valid] / clear[valid], 0.0, CLEARSKY_INDEX_MAX
        )
        months = timestamps.month.to_numpy()[valid] - 1
        slots = local_solar_slot(
            timestamps, float(frame["longitude"].iloc[0])
        )[valid]
        np.add.at(sums, (months, slots), clear_index)
        np.add.at(counts, (months, slots), 1)

    global_mean = float(sums.sum() / counts.sum())
    month_count = counts.sum(axis=1)
    month_mean = np.divide(
        sums.sum(axis=1), month_count,
        out=np.full(12, global_mean), where=month_count > 0,
    )
    table = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    for month in range(12):
        table[month, counts[month] == 0] = month_mean[month]
    return table, counts


def evaluate_station(
    frame: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    clear: np.ndarray,
    table: np.ndarray,
    horizons: list[int],
    context_steps: int = 1,
    minimum_context_fraction: float = 1.0,
) -> list[dict[str, object]]:
    station = str(frame["station_code"].iloc[0])
    longitude = float(frame["longitude"].iloc[0])
    month = timestamps.month.to_numpy() - 1
    slot = local_solar_slot(timestamps, longitude)
    prediction = np.maximum(table[month, slot] * clear, 0.0)
    ghi = frame["ghi_wm2"].to_numpy(dtype=np.float64)
    toa = frame["toa_horizontal_wm2"].to_numpy(dtype=np.float64)
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        valid = forecast_origin_mask(
            timestamps, ghi, horizon, context_steps, minimum_context_fraction
        )
        regimes = {
            "all": valid,
            "daylight": valid & (toa[horizon:] > 0.0),
            "daylight_to_daylight": valid & (toa[:-horizon] > 0.0) & (toa[horizon:] > 0.0),
        }
        for regime, mask in regimes.items():
            if np.any(mask):
                rows.append(metric_row(
                    station, horizon, regime, "global_climatology",
                    ghi[horizon:][mask], prediction[horizon:][mask],
                ))
    return rows


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.include_test:
        raise SystemExit("The test split is sealed; pass --include-test only after freeze.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    horizons = [int(value) for value in config["forecast_horizon_steps"]]
    eligibility = config["evaluation_eligibility"]
    train_files = sorted((args.processed_dir / "train").glob("*.parquet"))
    eval_files = sorted((args.processed_dir / args.split).glob("*.parquet"))
    if not train_files or not eval_files:
        raise SystemExit("Training or evaluation station files are missing.")
    if args.climatology_model:
        saved = np.load(args.climatology_model)
        table, counts = saved["clear_sky_index"], saved["counts"]
    else:
        table, counts = fit_climatology(train_files)

    rows: list[dict[str, object]] = []
    for path in eval_files:
        frame, timestamps, clear = station_inputs(path)
        print(f"Evaluate {path.stem}", flush=True)
        rows.extend(evaluate_station(
            frame, timestamps, clear, table, horizons,
            int(eligibility["context_steps"]),
            float(eligibility["minimum_context_ghi_fraction"]),
        ))
    station_metrics = pd.DataFrame(rows)
    aggregate = aggregate_metrics_without_smart(station_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    station_metrics.to_csv(args.output_dir / "station_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    np.savez_compressed(args.output_dir / "climatology.npz", clear_sky_index=table, counts=counts)
    metadata = {
        "protocol_id": config["protocol_id"], "split": args.split,
        "test_unlocked": bool(args.include_test), "train_stations": len(train_files),
        "evaluation_stations": len(eval_files),
        "evaluation_eligibility": eligibility,
        "reused_climatology_model": str(args.climatology_model) if args.climatology_model else None,
        "definition": "Mean clear-sky index by month and 15-minute local-solar-time slot, fit only on training stations.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(aggregate.to_string(index=False), flush=True)


def aggregate_metrics_without_smart(station_metrics: pd.DataFrame) -> pd.DataFrame:
    """Reuse the common aggregator by adding a temporary smart-reference copy."""
    reference = station_metrics.copy()
    reference["baseline"] = "smart_persistence"
    combined = pd.concat([station_metrics, reference], ignore_index=True)
    result = aggregate_metrics(combined)
    return result[result["baseline"] == "global_climatology"].drop(
        columns=["macro_smart_persistence_skill", "micro_smart_persistence_skill"]
    )


if __name__ == "__main__":
    main()
