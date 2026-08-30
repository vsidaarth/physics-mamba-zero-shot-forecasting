#!/usr/bin/env python3
"""Consolidate the completed frozen sealed-test suite without model selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = (15, 30, 60, 180, 360)
MULTISEED = {
    "physics_mamba": "physics_mamba_seed{seed}",
    "itransformer": "itransformer_seed{seed}",
    "vanilla_mamba": "mamba_seed{seed}",
    "night_mask_mamba": "night_mask_mamba_seed{seed}",
}
SINGLE = {
    "lightgbm": ("lightgbm_seed42", "lightgbm"),
    "lstm": ("lstm_seed42", "lstm"),
    "tcn": ("tcn_seed42", "tcn"),
    "dlinear": ("dlinear_seed42", "dlinear"),
    "patchtst": ("patchtst_seed42", "patchtst"),
    "persistence": ("persistence", "persistence"),
    "smart_persistence": ("persistence", "smart_persistence"),
    "global_climatology": ("climatology", "global_climatology"),
}
SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    return parser.parse_args()


def load_run(root: Path, directory: str, baseline: str, seed: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    run = root / directory
    aggregate = pd.read_csv(run / "aggregate_metrics.csv")
    station = pd.read_csv(run / "station_metrics.csv")
    aggregate = aggregate[(aggregate["regime"] == "daylight") & (aggregate["baseline"] == baseline)].copy()
    station = station[(station["regime"] == "daylight") & (station["baseline"] == baseline)].copy()
    aggregate["seed"] = seed
    station["seed"] = seed
    return aggregate, station


def paired_bootstrap(
    station: pd.DataFrame, candidate: str, reference: str,
    replicates: int, rng: np.random.Generator,
) -> list[dict[str, object]]:
    per_station = (
        station[station["model"].isin([candidate, reference])]
        .groupby(["model", "station", "horizon_minutes"], as_index=False)["mae_wm2"]
        .mean()
    )
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        pivot = per_station[per_station["horizon_minutes"] == horizon].pivot(
            index="station", columns="model", values="mae_wm2"
        )
        difference = (pivot[candidate] - pivot[reference]).dropna().to_numpy()
        draws = rng.integers(0, len(difference), size=(replicates, len(difference)))
        bootstrap = difference[draws].mean(axis=1)
        rows.append({
            "candidate": candidate,
            "reference": reference,
            "horizon_minutes": horizon,
            "stations": len(difference),
            "paired_difference_wm2": difference.mean(),
            "ci_low_wm2": np.quantile(bootstrap, 0.025),
            "ci_high_wm2": np.quantile(bootstrap, 0.975),
            "candidate_station_wins": int((difference < 0).sum()),
        })
    return rows


def main() -> None:
    args = parse_args()
    root = args.project_root / "results/sealed_test"
    if not (root / "SUITE_COMPLETE.json").exists():
        raise SystemExit("Refusing analysis: SUITE_COMPLETE.json is absent.")
    aggregates: list[pd.DataFrame] = []
    stations: list[pd.DataFrame] = []
    for model, pattern in MULTISEED.items():
        baseline = model.replace("vanilla_", "")
        for seed in SEEDS:
            aggregate, station = load_run(root, pattern.format(seed=seed), baseline, seed)
            aggregate["model"] = model
            station["model"] = model
            aggregates.append(aggregate)
            stations.append(station)
    for model, (directory, baseline) in SINGLE.items():
        aggregate, station = load_run(root, directory, baseline, 42 if model not in {"persistence", "smart_persistence", "global_climatology"} else None)
        aggregate["model"] = model
        station["model"] = model
        aggregates.append(aggregate)
        stations.append(station)
    aggregate_all = pd.concat(aggregates, ignore_index=True)
    station_all = pd.concat(stations, ignore_index=True)

    summary = (
        aggregate_all.groupby(["model", "horizon_minutes"], as_index=False)
        .agg(
            seeds=("macro_mae_wm2", "size"),
            mean_macro_mae_wm2=("macro_mae_wm2", "mean"),
            sd_macro_mae_wm2=("macro_mae_wm2", "std"),
            mean_macro_rmse_wm2=("macro_rmse_wm2", "mean"),
            mean_macro_nmae=("macro_nmae", "mean"),
        )
        .sort_values(["model", "horizon_minutes"])
    )
    summary.loc[summary["seeds"] == 1, "sd_macro_mae_wm2"] = np.nan

    rng = np.random.default_rng(args.bootstrap_seed)
    comparisons = (
        ("physics_mamba", "itransformer"),
        ("night_mask_mamba", "vanilla_mamba"),
        ("physics_mamba", "night_mask_mamba"),
        ("physics_mamba", "vanilla_mamba"),
    )
    effects = pd.DataFrame([
        row
        for candidate, reference in comparisons
        for row in paired_bootstrap(
            station_all, candidate, reference, args.bootstrap_replicates, rng
        )
    ])
    primary = effects[
        (effects["candidate"] == "physics_mamba")
        & (effects["reference"] == "itransformer")
    ].copy()
    model_means = summary.pivot(index="horizon_minutes", columns="model", values="mean_macro_mae_wm2")
    primary["relative_mae_reduction_pct"] = [
        -100.0 * row.paired_difference_wm2 / model_means.loc[row.horizon_minutes, "itransformer"]
        for row in primary.itertuples()
    ]

    output = args.project_root / "results/sealed_test_analysis"
    output.mkdir(parents=True, exist_ok=True)
    aggregate_all.to_csv(output / "test_seed_metrics.csv", index=False)
    station_all.to_csv(output / "test_station_metrics.csv", index=False)
    summary.to_csv(output / "test_all_methods_summary.csv", index=False)
    effects.to_csv(output / "test_paired_station_bootstrap.csv", index=False)
    primary.to_csv(output / "test_primary_comparison.csv", index=False)
    print("All methods:")
    print(summary.to_string(index=False))
    print("\nPrimary paired comparison (negative favors Physics-Mamba):")
    print(primary.to_string(index=False))
    print("\nAblation paired comparisons:")
    print(effects[effects["reference"] != "itransformer"].to_string(index=False))


if __name__ == "__main__":
    main()
