#!/usr/bin/env python3
"""Train leakage-controlled LightGBM baselines for each forecast horizon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    from scripts.evaluate_persistence import (
        CLEARSKY_FLOOR_WM2, CLEARSKY_INDEX_MAX, aggregate_metrics,
        forecast_origin_mask, ineichen_ghi, metric_row,
    )
except ModuleNotFoundError:
    from evaluate_persistence import (
        CLEARSKY_FLOOR_WM2, CLEARSKY_INDEX_MAX, aggregate_metrics,
        forecast_origin_mask, ineichen_ghi, metric_row,
    )


LAGS = (0, 1, 2, 3, 4, 8, 12, 24, 48, 95)
ROLLING_WINDOWS = (4, 12, 24, 96)
CURRENT_MET = (
    "dni_wm2", "dhi_wm2", "lwd_wm2", "air_temperature_c",
    "relative_humidity_pct", "pressure_hpa",
)
FUTURE_FEATURES = (
    "solar_zenith_deg", "cos_zenith", "toa_horizontal_wm2",
    "utc_hour_sin", "utc_hour_cos", "day_of_year_sin", "day_of_year_cos",
)
READ_COLUMNS = (
    "timestamp", "ghi_wm2", *CURRENT_MET, *FUTURE_FEATURES,
    "latitude", "longitude", "elevation_m", "station_code",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/protocol_v1.json"))
    parser.add_argument("--horizons", help="Comma-separated step horizons; default is protocol list.")
    parser.add_argument("--max-train-per-station", type=int, default=30_000)
    parser.add_argument("--max-val-per-station", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=-1)
    return parser.parse_args()


def load_station(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(READ_COLUMNS))


def eligible_indices(
    frame: pd.DataFrame, horizon: int, context_steps: int, minimum_fraction: float
) -> np.ndarray:
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    ghi = frame["ghi_wm2"].to_numpy(dtype=np.float64)
    return np.flatnonzero(
        forecast_origin_mask(timestamps, ghi, horizon, context_steps, minimum_fraction)
    )


def sample_indices(indices: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    return np.sort(rng.choice(indices, size=maximum, replace=False))


def feature_frame(frame: pd.DataFrame, origins: np.ndarray, horizon: int) -> pd.DataFrame:
    ghi = frame["ghi_wm2"].to_numpy(dtype=np.float32)
    series = pd.Series(ghi)
    features: dict[str, np.ndarray] = {}
    for lag in LAGS:
        values = series.shift(lag).to_numpy(dtype=np.float32)[origins]
        features[f"ghi_lag_{lag}"] = values
        features[f"ghi_lag_{lag}_missing"] = (~np.isfinite(values)).astype(np.int8)
    for window in ROLLING_WINDOWS:
        rolling = series.rolling(window, min_periods=1)
        features[f"ghi_mean_{window}"] = rolling.mean().to_numpy(dtype=np.float32)[origins]
        features[f"ghi_std_{window}"] = rolling.std(ddof=0).to_numpy(dtype=np.float32)[origins]
        features[f"ghi_min_{window}"] = rolling.min().to_numpy(dtype=np.float32)[origins]
        features[f"ghi_max_{window}"] = rolling.max().to_numpy(dtype=np.float32)[origins]
    for name in CURRENT_MET:
        features[f"current_{name}"] = frame[name].to_numpy(dtype=np.float32)[origins]
    targets = origins + horizon
    for name in FUTURE_FEATURES:
        features[f"future_{name}"] = frame[name].to_numpy(dtype=np.float32)[targets]
    features["latitude"] = frame["latitude"].to_numpy(dtype=np.float32)[origins]
    features["longitude"] = frame["longitude"].to_numpy(dtype=np.float32)[origins]
    features["elevation_m"] = frame["elevation_m"].to_numpy(dtype=np.float32)[origins]
    toa_now = frame["toa_horizontal_wm2"].to_numpy(dtype=np.float32)[origins]
    features["current_toa_index"] = np.clip(
        ghi[origins] / np.maximum(toa_now, CLEARSKY_FLOOR_WM2), 0.0, CLEARSKY_INDEX_MAX
    )
    return pd.DataFrame(features)


def collect_sample(
    files: list[Path], horizon: int, context_steps: int, minimum_fraction: float,
    maximum_per_station: int, rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    feature_parts: list[pd.DataFrame] = []
    target_parts: list[np.ndarray] = []
    for path in files:
        frame = load_station(path)
        indices = sample_indices(
            eligible_indices(frame, horizon, context_steps, minimum_fraction),
            maximum_per_station, rng,
        )
        feature_parts.append(feature_frame(frame, indices, horizon))
        target_parts.append(frame["ghi_wm2"].to_numpy(dtype=np.float32)[indices + horizon])
    return pd.concat(feature_parts, ignore_index=True), np.concatenate(target_parts)


def evaluate_model(
    model: lgb.LGBMRegressor, files: list[Path], horizon: int,
    context_steps: int, minimum_fraction: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in files:
        frame = load_station(path)
        station = path.stem
        origins = eligible_indices(frame, horizon, context_steps, minimum_fraction)
        features = feature_frame(frame, origins, horizon)
        observed = frame["ghi_wm2"].to_numpy(dtype=np.float64)[origins + horizon]
        prediction = np.maximum(model.predict(features), 0.0)
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
        clear = ineichen_ghi(
            timestamps, float(frame["latitude"].iloc[0]),
            float(frame["longitude"].iloc[0]), float(frame["elevation_m"].iloc[0]),
        )
        current_ghi = frame["ghi_wm2"].to_numpy(dtype=np.float64)[origins]
        clear_index = np.clip(
            current_ghi / np.maximum(clear[origins], CLEARSKY_FLOOR_WM2),
            0.0, CLEARSKY_INDEX_MAX,
        )
        smart = np.maximum(clear_index * clear[origins + horizon], 0.0)
        toa = frame["toa_horizontal_wm2"].to_numpy(dtype=np.float64)
        regimes = {
            "all": np.ones(len(origins), dtype=bool),
            "daylight": toa[origins + horizon] > 0.0,
            "daylight_to_daylight": (toa[origins] > 0.0) & (toa[origins + horizon] > 0.0),
        }
        for regime, mask in regimes.items():
            rows.append(metric_row(station, horizon, regime, "lightgbm", observed[mask], prediction[mask]))
            rows.append(metric_row(station, horizon, regime, "smart_persistence", observed[mask], smart[mask]))
    return rows


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    horizons = (
        [int(value) for value in args.horizons.split(",")]
        if args.horizons else [int(value) for value in config["forecast_horizon_steps"]]
    )
    eligibility = config["evaluation_eligibility"]
    context_steps = int(eligibility["context_steps"])
    minimum_fraction = float(eligibility["minimum_context_ghi_fraction"])
    train_files = sorted((args.processed_dir / "train").glob("*.parquet"))
    val_files = sorted((args.processed_dir / "val").glob("*.parquet"))
    if len(train_files) != 33 or len(val_files) != 7:
        raise SystemExit("Expected exactly 33 train and 7 validation station files.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []

    for horizon in horizons:
        print(f"\n=== Horizon {horizon} steps ===", flush=True)
        train_rng = np.random.default_rng(args.seed + horizon)
        val_rng = np.random.default_rng(args.seed + 10_000 + horizon)
        x_train, y_train = collect_sample(
            train_files, horizon, context_steps, minimum_fraction,
            args.max_train_per_station, train_rng,
        )
        x_val, y_val = collect_sample(
            val_files, horizon, context_steps, minimum_fraction,
            args.max_val_per_station, val_rng,
        )
        print(f"Training rows: {len(y_train):,}; early-stop rows: {len(y_val):,}", flush=True)
        model = lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=2000, learning_rate=0.03,
            num_leaves=63, max_depth=-1, min_child_samples=100,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=args.seed, n_jobs=args.num_threads, verbosity=-1,
        )
        model.fit(
            x_train, y_train, eval_set=[(x_val, y_val)], eval_metric="l1",
            callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(100)],
        )
        model.booster_.save_model(str(args.output_dir / f"lightgbm_h{horizon}.txt"))
        all_rows.extend(evaluate_model(
            model, val_files, horizon, context_steps, minimum_fraction
        ))
        del x_train, y_train, x_val, y_val

    station_metrics = pd.DataFrame(all_rows)
    aggregate = aggregate_metrics(station_metrics)
    station_metrics.to_csv(args.output_dir / "station_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    metadata = {
        "protocol_id": config["protocol_id"], "model": "LightGBM",
        "seed": args.seed, "horizons": horizons,
        "max_train_samples_per_station_per_horizon": args.max_train_per_station,
        "max_validation_samples_per_station_for_early_stopping": args.max_val_per_station,
        "evaluation_eligibility": eligibility,
        "features": list(feature_frame(load_station(train_files[0]), np.array([95]), 1).columns),
        "test_evaluated": False,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(aggregate[aggregate["baseline"] == "lightgbm"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
