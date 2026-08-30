#!/usr/bin/env python3
"""Inference-only validation subgroup analysis for a frozen neural checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.train_lstm import (
    CLEARSKY_FLOOR_WM2,
    CLEARSKY_INDEX_MAX,
    CONTEXT_FEATURES,
    FUTURE_FEATURES,
    STATIC_FEATURES,
    ITransformerForecaster,
    PhysicsMambaForecaster,
    WindowDataset,
    ineichen_ghi,
    load_station,
    predict_dataset,
)


SOLAR_EDGES = np.array([0.0, 10.0, 20.0, 40.0, 60.0, np.inf])
SOLAR_LABELS = ("0-10", "10-20", "20-40", "40-60", ">60")
SEASON_BY_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}
MIN_STATION_BIN_N = 500
MIN_STATIONS_PER_AGGREGATE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_dir", type=Path)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/protocol_v1.json"))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def build_model(
    architecture: str, hyperparameters: dict[str, object], horizons: int,
    context_mean: np.ndarray, context_std: np.ndarray,
) -> torch.nn.Module:
    hidden = int(hyperparameters["hidden_size"])
    layers = int(hyperparameters["layers"])
    dropout = float(hyperparameters["dropout"])
    if architecture == "physics_mamba":
        toa_position = CONTEXT_FEATURES.index("toa_horizontal_wm2")
        return PhysicsMambaForecaster(
            len(CONTEXT_FEATURES) * 2, len(FUTURE_FEATURES), len(STATIC_FEATURES),
            horizons, hidden, layers, dropout,
            float(context_mean[toa_position]), float(context_std[toa_position]),
        )
    if architecture == "itransformer":
        return ITransformerForecaster(
            96, len(CONTEXT_FEATURES) * 2, len(FUTURE_FEATURES),
            len(STATIC_FEATURES), horizons, hidden, layers, dropout,
        )
    raise ValueError(f"Unsupported frozen architecture: {architecture}")


def local_seasons(months: np.ndarray, latitude: float) -> np.ndarray:
    adjusted = months.copy()
    if latitude < 0.0:
        adjusted = ((adjusted + 5) % 12) + 1
    return np.asarray([SEASON_BY_MONTH[int(month)] for month in adjusted], dtype=object)


def metric_record(
    station: str, horizon_steps: int, horizon_minutes: int,
    group_type: str, group: str, observed: np.ndarray, predicted: np.ndarray,
) -> dict[str, object]:
    error = predicted - observed
    absolute = np.abs(error)
    return {
        "station": station,
        "horizon_steps": horizon_steps,
        "horizon_minutes": horizon_minutes,
        "group_type": group_type,
        "group": group,
        "n": len(observed),
        "mae_wm2": float(absolute.mean()),
        "rmse_wm2": float(np.sqrt(np.square(error).mean())),
        "nmae": float(absolute.sum() / np.maximum(observed.sum(), 1e-9)),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    metadata = json.loads((args.result_dir / "metadata.json").read_text(encoding="utf-8"))
    architecture = str(metadata["hyperparameters"]["architecture"])
    checkpoint = torch.load(
        args.result_dir / f"{architecture}_checkpoint.pt",
        map_location="cpu", weights_only=False,
    )
    horizons = [int(value) for value in checkpoint["horizons"]]
    context_mean = np.asarray(checkpoint["context_mean"], dtype=np.float32)
    context_std = np.asarray(checkpoint["context_std"], dtype=np.float32)
    static_mean = np.asarray(checkpoint["static_mean"], dtype=np.float32)
    static_std = np.asarray(checkpoint["static_std"], dtype=np.float32)
    eligibility = config["evaluation_eligibility"]
    context_steps = int(eligibility["context_steps"])
    minimum_fraction = float(eligibility["minimum_context_ghi_fraction"])
    val_files = sorted((args.processed_dir / "val").glob("*.parquet"))
    if len(val_files) != 7:
        raise RuntimeError(f"Expected seven validation stations, found {len(val_files)}")
    stations = [
        load_station(
            path, horizons, context_steps, minimum_fraction,
            context_mean, context_std, static_mean, static_std,
        )
        for path in val_files
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        architecture, metadata["hyperparameters"], len(horizons),
        context_mean, context_std,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    inference_workers = 0 if "mamba" in architecture else args.workers

    records: list[dict[str, object]] = []
    for station_index, station in enumerate(stations):
        dataset = WindowDataset([station], horizons, context_steps, None, seed=0)
        predictions = np.maximum(
            predict_dataset(model, dataset, device, args.batch_size, inference_workers),
            0.0,
        )
        origins = dataset.pairs[:, 1]
        clear = ineichen_ghi(
            station.timestamps,
            float(station.static_raw[0]), float(station.static_raw[1]),
            float(station.static_raw[2]),
        )
        zenith_position = CONTEXT_FEATURES.index("solar_zenith_deg")
        for column, horizon in enumerate(horizons):
            valid = station.valid_targets[origins, column]
            selected_origins = origins[valid]
            targets = selected_origins + horizon
            observed = station.ghi[targets].astype(np.float64)
            predicted = predictions[valid, column].astype(np.float64)
            current = station.ghi[selected_origins].astype(np.float64)
            clear_index = np.clip(
                current / np.maximum(clear[selected_origins], CLEARSKY_FLOOR_WM2),
                0.0, CLEARSKY_INDEX_MAX,
            )
            smart = np.maximum(clear_index * clear[targets], 0.0)
            daylight = station.toa[targets] > 0.0
            observed = observed[daylight]
            predicted = predicted[daylight]
            smart = smart[daylight]
            targets = targets[daylight]

            normalized_zenith = station.context[targets, zenith_position]
            zenith = (
                normalized_zenith * context_std[zenith_position]
                + context_mean[zenith_position]
            )
            elevation = np.maximum(90.0 - zenith, 0.0)
            solar_groups = np.asarray(
                SOLAR_LABELS, dtype=object
            )[np.clip(np.digitize(elevation, SOLAR_EDGES[1:-1]), 0, 4)]
            months = station.timestamps[targets].month.to_numpy(dtype=np.int16)
            seasons = local_seasons(months, float(station.static_raw[0]))
            smart_absolute_error = np.abs(smart - observed)

            records.append({
                "station": station.code,
                "horizon_steps": horizon,
                "horizon_minutes": horizon * 15,
                "observed": observed,
                "predicted": predicted,
                "solar_group": solar_groups,
                "season": seasons,
                "smart_absolute_error": smart_absolute_error,
            })
        print(f"Predicted {station_index + 1}/{len(stations)}: {station.code}", flush=True)

    variability_edges: dict[int, np.ndarray] = {}
    for horizon in horizons:
        values = np.concatenate([
            record["smart_absolute_error"] for record in records
            if record["horizon_steps"] == horizon
        ])
        variability_edges[horizon] = np.quantile(values, [0.2, 0.4, 0.6, 0.8])

    rows: list[dict[str, object]] = []
    for record in records:
        observed = record["observed"]
        predicted = record["predicted"]
        group_sets = {
            "solar_elevation_deg": record["solar_group"],
            "local_season": record["season"],
            "observed_variability_quintile": np.asarray(
                [f"Q{value + 1}" for value in np.digitize(
                    record["smart_absolute_error"],
                    variability_edges[int(record["horizon_steps"])],
                )], dtype=object,
            ),
        }
        for group_type, assignments in group_sets.items():
            for group in np.unique(assignments):
                mask = assignments == group
                if int(mask.sum()) < MIN_STATION_BIN_N:
                    continue
                rows.append(metric_record(
                    str(record["station"]), int(record["horizon_steps"]),
                    int(record["horizon_minutes"]), group_type, str(group),
                    observed[mask], predicted[mask],
                ))

    station_metrics = pd.DataFrame(rows)
    station_metrics["model"] = architecture
    station_metrics["seed"] = int(metadata["seed"])
    aggregate = (
        station_metrics.groupby(
            ["model", "seed", "horizon_steps", "horizon_minutes", "group_type", "group"],
            as_index=False,
        )
        .agg(
            stations=("station", "nunique"),
            n=("n", "sum"),
            macro_mae_wm2=("mae_wm2", "mean"),
            macro_rmse_wm2=("rmse_wm2", "mean"),
            macro_nmae=("nmae", "mean"),
        )
    )
    aggregate = aggregate[aggregate["stations"] >= MIN_STATIONS_PER_AGGREGATE]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    station_metrics.to_csv(args.output_dir / "subgroup_station_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "subgroup_aggregate_metrics.csv", index=False)
    pd.DataFrame([
        {
            "horizon_steps": horizon,
            "horizon_minutes": horizon * 15,
            **{f"q{index + 1}_upper_wm2": value for index, value in enumerate(edges)},
        }
        for horizon, edges in variability_edges.items()
    ]).to_csv(args.output_dir / "variability_quintile_edges.csv", index=False)
    (args.output_dir / "subgroup_metadata.json").write_text(json.dumps({
        "model": architecture,
        "seed": int(metadata["seed"]),
        "split": "validation",
        "test_evaluated": False,
        "minimum_station_bin_n": MIN_STATION_BIN_N,
        "minimum_stations_per_aggregate": MIN_STATIONS_PER_AGGREGATE,
        "solar_elevation_bins_deg": list(SOLAR_LABELS),
        "variability_definition": "absolute smart-persistence error quintile by horizon",
        "season_definition": "meteorological season shifted six months south of equator",
    }, indent=2), encoding="utf-8")
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
