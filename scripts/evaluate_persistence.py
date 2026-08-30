#!/usr/bin/env python3
"""Evaluate persistence and clear-sky-index smart persistence on BSRN data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib


STEP_MINUTES = 15
CLEARSKY_FLOOR_WM2 = 20.0
CLEARSKY_INDEX_MAX = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/protocol_v1.json"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--include-test", action="store_true",
        help="Required to evaluate the sealed test split.",
    )
    return parser.parse_args()


def ineichen_ghi(
    timestamps: pd.DatetimeIndex, latitude: float, longitude: float, elevation: float
) -> np.ndarray:
    """Return Ineichen clear-sky GHI in W/m2 for UTC timestamps."""
    location = pvlib.location.Location(
        latitude=latitude, longitude=longitude, tz="UTC", altitude=elevation
    )
    return location.get_clearsky(timestamps, model="ineichen")["ghi"].to_numpy(
        dtype=np.float64
    )


def metric_row(
    station: str,
    horizon_steps: int,
    regime: str,
    baseline: str,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, object]:
    error = predicted - observed
    absolute_error = np.abs(error)
    mae = float(np.mean(absolute_error))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mean_observed = float(np.mean(observed))
    return {
        "station": station,
        "horizon_steps": horizon_steps,
        "horizon_minutes": horizon_steps * STEP_MINUTES,
        "regime": regime,
        "baseline": baseline,
        "n": int(observed.size),
        "mae_wm2": mae,
        "rmse_wm2": rmse,
        "mean_observed_wm2": mean_observed,
        "nmae": mae / mean_observed if mean_observed > 0 else np.nan,
        "sum_absolute_error": float(np.sum(absolute_error)),
        "sum_squared_error": float(np.sum(np.square(error))),
        "sum_observed": float(np.sum(observed)),
    }


def forecast_origin_mask(
    timestamps: pd.DatetimeIndex,
    ghi: np.ndarray,
    horizon_steps: int,
    context_steps: int,
    minimum_context_fraction: float,
) -> np.ndarray:
    """Return the common eligibility mask aligned to pre-horizon origins."""
    origins = len(ghi) - horizon_steps
    if origins <= 0:
        return np.zeros(0, dtype=bool)
    finite = np.isfinite(ghi)
    context_count = (
        pd.Series(finite.astype(np.int16))
        .rolling(context_steps, min_periods=context_steps)
        .sum()
        .to_numpy()[:origins]
    )
    enough_context = context_count >= np.ceil(
        context_steps * minimum_context_fraction
    )
    origin_index = np.arange(origins)
    has_full_context = origin_index >= context_steps - 1
    context_start = np.maximum(origin_index - context_steps + 1, 0)
    context_contiguous = np.zeros(origins, dtype=bool)
    context_contiguous[has_full_context] = (
        timestamps[origin_index[has_full_context]]
        - timestamps[context_start[has_full_context]]
        == pd.Timedelta(minutes=(context_steps - 1) * STEP_MINUTES)
    )
    target_contiguous = (
        timestamps[horizon_steps:] - timestamps[:-horizon_steps]
        == pd.Timedelta(minutes=horizon_steps * STEP_MINUTES)
    )
    return (
        finite[:origins] & finite[horizon_steps:] & enough_context
        & context_contiguous & target_contiguous
    )


def evaluate_station(
    frame: pd.DataFrame,
    station: str,
    horizon_steps: list[int],
    context_steps: int = 1,
    minimum_context_fraction: float = 1.0,
) -> list[dict[str, object]]:
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    ghi = frame["ghi_wm2"].to_numpy(dtype=np.float64)
    toa = frame["toa_horizontal_wm2"].to_numpy(dtype=np.float64)
    clear = ineichen_ghi(
        timestamps,
        float(frame["latitude"].iloc[0]),
        float(frame["longitude"].iloc[0]),
        float(frame["elevation_m"].iloc[0]),
    )

    rows: list[dict[str, object]] = []
    for horizon in horizon_steps:
        if len(frame) <= horizon:
            continue
        current = slice(None, -horizon)
        future = slice(horizon, None)
        valid = forecast_origin_mask(
            timestamps, ghi, horizon, context_steps, minimum_context_fraction
        )
        persistence = ghi[current]
        clear_index = np.clip(
            ghi[current] / np.maximum(clear[current], CLEARSKY_FLOOR_WM2),
            0.0,
            CLEARSKY_INDEX_MAX,
        )
        smart = np.maximum(clear_index * clear[future], 0.0)

        regimes = {
            "all": valid,
            "daylight": valid & (toa[future] > 0.0),
            "daylight_to_daylight": valid
            & (toa[current] > 0.0)
            & (toa[future] > 0.0),
        }
        for regime, mask in regimes.items():
            if not np.any(mask):
                continue
            observed = ghi[future][mask]
            for name, prediction in (
                ("persistence", persistence[mask]),
                ("smart_persistence", smart[mask]),
            ):
                rows.append(
                    metric_row(station, horizon, regime, name, observed, prediction)
                )
    return rows


def aggregate_metrics(station_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["horizon_steps", "horizon_minutes", "regime", "baseline"]
    output: list[dict[str, object]] = []
    for key, group in station_metrics.groupby(keys, sort=True):
        horizon_steps, horizon_minutes, regime, baseline = key
        n = int(group["n"].sum())
        micro_mae = float(group["sum_absolute_error"].sum() / n)
        micro_rmse = float(np.sqrt(group["sum_squared_error"].sum() / n))
        micro_mean = float(group["sum_observed"].sum() / n)
        output.append(
            {
                "horizon_steps": horizon_steps,
                "horizon_minutes": horizon_minutes,
                "regime": regime,
                "baseline": baseline,
                "stations": int(group["station"].nunique()),
                "n": n,
                "macro_mae_wm2": float(group["mae_wm2"].mean()),
                "macro_rmse_wm2": float(group["rmse_wm2"].mean()),
                "macro_nmae": float(group["nmae"].mean()),
                "micro_mae_wm2": micro_mae,
                "micro_rmse_wm2": micro_rmse,
                "micro_nmae": micro_mae / micro_mean if micro_mean > 0 else np.nan,
            }
        )
    aggregated = pd.DataFrame(output)
    smart = aggregated.loc[
        aggregated["baseline"] == "smart_persistence",
        keys[:-1] + ["macro_mae_wm2", "micro_mae_wm2"],
    ].rename(
        columns={
            "macro_mae_wm2": "smart_macro_mae_wm2",
            "micro_mae_wm2": "smart_micro_mae_wm2",
        }
    )
    aggregated = aggregated.merge(smart, on=keys[:-1], how="left")
    aggregated["macro_smart_persistence_skill"] = 1.0 - (
        aggregated["macro_mae_wm2"] / aggregated["smart_macro_mae_wm2"]
    )
    aggregated["micro_smart_persistence_skill"] = 1.0 - (
        aggregated["micro_mae_wm2"] / aggregated["smart_micro_mae_wm2"]
    )
    return aggregated.drop(columns=["smart_macro_mae_wm2", "smart_micro_mae_wm2"])


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.include_test:
        raise SystemExit("The test split is sealed; pass --include-test only after freeze.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    horizons = [int(value) for value in config["forecast_horizon_steps"]]
    eligibility = config["evaluation_eligibility"]
    files = sorted((args.processed_dir / args.split).glob("*.parquet"))
    if not files:
        raise SystemExit(f"No Parquet stations found for split {args.split!r}.")

    rows: list[dict[str, object]] = []
    for path in files:
        frame = pd.read_parquet(
            path,
            columns=[
                "timestamp", "ghi_wm2", "toa_horizontal_wm2", "latitude",
                "longitude", "elevation_m", "station_code",
            ],
        )
        station = str(frame["station_code"].iloc[0])
        print(f"Evaluating {station} ({len(frame):,} bins)", flush=True)
        rows.extend(evaluate_station(
            frame, station, horizons,
            int(eligibility["context_steps"]),
            float(eligibility["minimum_context_ghi_fraction"]),
        ))

    station_metrics = pd.DataFrame(rows)
    aggregate = aggregate_metrics(station_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    station_metrics.to_csv(args.output_dir / "station_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    metadata = {
        "protocol_id": config["protocol_id"],
        "split": args.split,
        "test_unlocked": bool(args.include_test),
        "stations": len(files),
        "horizon_steps": horizons,
        "evaluation_eligibility": eligibility,
        "clear_sky_model": "pvlib Ineichen",
        "clear_sky_index_floor_wm2": CLEARSKY_FLOOR_WM2,
        "clear_sky_index_clip": [0.0, CLEARSKY_INDEX_MAX],
        "metrics": {
            "macro": "unweighted mean of station metrics",
            "micro": "pooled valid forecast pairs",
            "nmae_denominator": "mean observed GHI in the evaluated regime",
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
