#!/usr/bin/env python3
"""Consolidate the frozen three-seed Mamba ablation on validation stations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = (15, 30, 60, 180, 360)
MODELS = {
    "vanilla_mamba": ("mamba_seed{seed}", "mamba"),
    "night_mask_mamba": ("night_mask_mamba_seed{seed}", "night_mask_mamba"),
    "physics_mamba": ("physics_mamba_seed{seed}", "physics_mamba"),
}
SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    return parser.parse_args()


def load_metrics(project: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_parts: list[pd.DataFrame] = []
    station_parts: list[pd.DataFrame] = []
    for model, (directory_pattern, baseline) in MODELS.items():
        for seed in SEEDS:
            directory = directory_pattern.format(seed=seed)
            run_dir = project / "results" / directory
            if not run_dir.exists():
                run_dir = project / "results" / "models" / directory
            aggregate = pd.read_csv(run_dir / "aggregate_metrics.csv")
            aggregate = aggregate[
                (aggregate["regime"] == "daylight") & (aggregate["baseline"] == baseline)
            ].copy()
            aggregate["model"] = model
            aggregate["seed"] = seed
            aggregate_parts.append(aggregate)

            station = pd.read_csv(run_dir / "station_metrics.csv")
            station = station[
                (station["regime"] == "daylight") & (station["baseline"] == baseline)
            ].copy()
            station["model"] = model
            station["seed"] = seed
            station_parts.append(station)
    return pd.concat(aggregate_parts, ignore_index=True), pd.concat(station_parts, ignore_index=True)


def summarize(aggregate: pd.DataFrame) -> pd.DataFrame:
    summary = (
        aggregate.groupby(["model", "horizon_minutes"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_macro_mae_wm2=("macro_mae_wm2", "mean"),
            sd_macro_mae_wm2=("macro_mae_wm2", "std"),
            min_macro_mae_wm2=("macro_mae_wm2", "min"),
            max_macro_mae_wm2=("macro_mae_wm2", "max"),
        )
        .sort_values(["model", "horizon_minutes"])
    )
    return summary


def paired_effects(station: pd.DataFrame, replicates: int, bootstrap_seed: int) -> pd.DataFrame:
    per_station = (
        station.groupby(["model", "station", "horizon_minutes"], as_index=False)["mae_wm2"]
        .mean()
    )
    comparisons = (
        ("night_mask_minus_vanilla", "night_mask_mamba", "vanilla_mamba"),
        ("physics_minus_night_mask", "physics_mamba", "night_mask_mamba"),
        ("physics_minus_vanilla", "physics_mamba", "vanilla_mamba"),
    )
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(bootstrap_seed)
    for comparison, candidate, reference in comparisons:
        for horizon in HORIZONS:
            subset = per_station[per_station["horizon_minutes"] == horizon]
            pivot = subset.pivot(index="station", columns="model", values="mae_wm2")
            differences = (pivot[candidate] - pivot[reference]).dropna().to_numpy()
            draws = rng.integers(0, len(differences), size=(replicates, len(differences)))
            boot = differences[draws].mean(axis=1)
            rows.append({
                "comparison": comparison,
                "candidate": candidate,
                "reference": reference,
                "horizon_minutes": horizon,
                "stations": len(differences),
                "paired_difference_wm2": differences.mean(),
                "station_bootstrap_ci_low_wm2": np.quantile(boot, 0.025),
                "station_bootstrap_ci_high_wm2": np.quantile(boot, 0.975),
                "candidate_station_wins": int((differences < 0).sum()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    aggregate, station = load_metrics(args.project_root)
    output = args.project_root / "results" / "ablation_replications"
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(aggregate)
    effects = paired_effects(station, args.bootstrap_replicates, args.bootstrap_seed)
    aggregate.to_csv(output / "ablation_seed_metrics.csv", index=False)
    summary.to_csv(output / "ablation_3seed_summary.csv", index=False)
    effects.to_csv(output / "ablation_paired_station_bootstrap.csv", index=False)
    print(summary.to_string(index=False))
    print("\nPaired effects (negative favors candidate):")
    print(effects.to_string(index=False))


if __name__ == "__main__":
    main()
