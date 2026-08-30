#!/usr/bin/env python3
"""Combine frozen validation subgroup results across models and seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("physics_mamba", "itransformer")
SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subgroups_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def load_all(subgroups_dir: Path, filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model in MODELS:
        for seed in SEEDS:
            path = subgroups_dir / f"{model}_seed{seed}" / filename
            frame = pd.read_csv(path)
            frame["model"] = model
            frame["seed"] = seed
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def verify_variability_edges(subgroups_dir: Path) -> pd.DataFrame:
    reference: pd.DataFrame | None = None
    for model in MODELS:
        for seed in SEEDS:
            path = subgroups_dir / f"{model}_seed{seed}" / "variability_quintile_edges.csv"
            frame = pd.read_csv(path).sort_values("horizon_minutes").reset_index(drop=True)
            if reference is None:
                reference = frame
            elif not np.allclose(frame.to_numpy(dtype=float), reference.to_numpy(dtype=float)):
                raise RuntimeError(f"Observation-defined variability edges differ in {path}")
    assert reference is not None
    return reference


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges = verify_variability_edges(args.subgroups_dir)
    edges.to_csv(args.output_dir / "validated_variability_quintile_edges.csv", index=False)

    aggregate = load_all(args.subgroups_dir, "subgroup_aggregate_metrics.csv")
    aggregate.to_csv(args.output_dir / "subgroup_seed_aggregate_metrics.csv", index=False)
    seed_summary = (
        aggregate.groupby(
            ["model", "horizon_steps", "horizon_minutes", "group_type", "group"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            stations=("stations", "min"),
            mean_macro_mae_wm2=("macro_mae_wm2", "mean"),
            sd_macro_mae_wm2=("macro_mae_wm2", "std"),
            mean_macro_rmse_wm2=("macro_rmse_wm2", "mean"),
            mean_macro_nmae=("macro_nmae", "mean"),
        )
        .sort_values(["group_type", "horizon_minutes", "group", "model"])
    )
    seed_summary.to_csv(args.output_dir / "subgroup_3seed_model_summary.csv", index=False)

    station = load_all(args.subgroups_dir, "subgroup_station_metrics.csv")
    station_means = (
        station.groupby(
            ["model", "station", "horizon_steps", "horizon_minutes", "group_type", "group"],
            as_index=False,
        )
        .agg(seed_mean_mae_wm2=("mae_wm2", "mean"))
    )
    physics = station_means[station_means["model"] == "physics_mamba"]
    transformer = station_means[station_means["model"] == "itransformer"]
    paired = physics.merge(
        transformer,
        on=["station", "horizon_steps", "horizon_minutes", "group_type", "group"],
        suffixes=("_physics", "_itransformer"),
        validate="one_to_one",
    )
    paired["difference_wm2"] = (
        paired["seed_mean_mae_wm2_physics"]
        - paired["seed_mean_mae_wm2_itransformer"]
    )
    paired.to_csv(args.output_dir / "subgroup_paired_station_differences.csv", index=False)
    comparison = (
        paired.groupby(
            ["horizon_steps", "horizon_minutes", "group_type", "group"],
            as_index=False,
        )
        .agg(
            stations=("station", "nunique"),
            physics_mamba_macro_mae_wm2=("seed_mean_mae_wm2_physics", "mean"),
            itransformer_macro_mae_wm2=("seed_mean_mae_wm2_itransformer", "mean"),
            physics_minus_itransformer_wm2=("difference_wm2", "mean"),
            physics_station_wins=("difference_wm2", lambda values: int((values < 0).sum())),
        )
    )
    comparison["physics_relative_reduction_pct"] = (
        -100.0 * comparison["physics_minus_itransformer_wm2"]
        / comparison["itransformer_macro_mae_wm2"]
    )
    comparison = comparison.sort_values(["group_type", "horizon_minutes", "group"])
    comparison.to_csv(args.output_dir / "subgroup_paired_comparison.csv", index=False)

    for group_type in ("solar_elevation_deg", "observed_variability_quintile", "local_season"):
        print(f"\n{group_type}")
        print(comparison[comparison["group_type"] == group_type].to_string(index=False))


if __name__ == "__main__":
    main()
