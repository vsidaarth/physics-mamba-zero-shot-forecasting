#!/usr/bin/env python3
"""Create paper-ready station-level tables from completed validation metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (42, 43, 44)
SHORTLIST = ("physics_mamba", "itransformer")
ABLATION = ("mamba", "night_mask_mamba", "physics_mamba")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def read_station_metrics(models_dir: Path, model: str, seed: int) -> pd.DataFrame:
    path = models_dir / f"{model}_seed{seed}" / "station_metrics.csv"
    frame = pd.read_csv(path)
    frame = frame[
        (frame["baseline"] == model) & (frame["regime"] == "daylight")
    ].copy()
    frame["model"] = model
    frame["seed"] = seed
    return frame[[
        "model", "seed", "station", "horizon_steps", "horizon_minutes",
        "n", "mae_wm2", "rmse_wm2", "nmae",
    ]]


def shortlist_tables(models_dir: Path, output_dir: Path) -> None:
    raw = pd.concat([
        read_station_metrics(models_dir, model, seed)
        for model in SHORTLIST for seed in SEEDS
    ], ignore_index=True)
    expected = len(SHORTLIST) * len(SEEDS) * 7 * 5
    if len(raw) != expected:
        raise RuntimeError(f"Expected {expected} shortlist rows, found {len(raw)}")

    means = (
        raw.groupby(["model", "station", "horizon_steps", "horizon_minutes"], as_index=False)
        .agg(
            seed_mean_mae_wm2=("mae_wm2", "mean"),
            seed_sd_mae_wm2=("mae_wm2", "std"),
            seed_mean_rmse_wm2=("rmse_wm2", "mean"),
            seed_mean_nmae=("nmae", "mean"),
        )
    )
    wide = means.pivot(
        index=["station", "horizon_steps", "horizon_minutes"],
        columns="model",
        values=["seed_mean_mae_wm2", "seed_sd_mae_wm2"],
    )
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    wide["physics_minus_itransformer_wm2"] = (
        wide["seed_mean_mae_wm2_physics_mamba"]
        - wide["seed_mean_mae_wm2_itransformer"]
    )
    wide["physics_relative_reduction_pct"] = (
        -100.0 * wide["physics_minus_itransformer_wm2"]
        / wide["seed_mean_mae_wm2_itransformer"]
    )

    paired_seeds = raw.pivot(
        index=["seed", "station", "horizon_steps", "horizon_minutes"],
        columns="model", values="mae_wm2",
    ).reset_index()
    paired_seeds["physics_win"] = (
        paired_seeds["physics_mamba"] < paired_seeds["itransformer"]
    )
    seed_wins = (
        paired_seeds.groupby(["station", "horizon_steps", "horizon_minutes"], as_index=False)
        .agg(physics_seed_wins=("physics_win", "sum"))
    )
    wide = wide.merge(
        seed_wins, on=["station", "horizon_steps", "horizon_minutes"],
        validate="one_to_one",
    ).sort_values(["horizon_minutes", "station"])
    wide.to_csv(output_dir / "shortlist_station_horizon_comparison.csv", index=False)

    station_summary = (
        wide.groupby("station", as_index=False)
        .agg(
            mean_paired_difference_wm2=("physics_minus_itransformer_wm2", "mean"),
            mean_relative_reduction_pct=("physics_relative_reduction_pct", "mean"),
            physics_horizon_wins=(
                "physics_minus_itransformer_wm2", lambda values: int((values < 0).sum())
            ),
            physics_seed_horizon_wins=("physics_seed_wins", "sum"),
        )
        .sort_values("mean_paired_difference_wm2")
    )
    station_summary.to_csv(output_dir / "shortlist_station_summary.csv", index=False)


def ablation_table(models_dir: Path, output_dir: Path) -> None:
    frames = [read_station_metrics(models_dir, model, 42) for model in ABLATION]
    raw = pd.concat(frames, ignore_index=True)
    wide = raw.pivot(
        index=["station", "horizon_steps", "horizon_minutes"],
        columns="model", values="mae_wm2",
    ).reset_index()
    wide["night_mask_gain_vs_vanilla_wm2"] = wide["mamba"] - wide["night_mask_mamba"]
    wide["envelope_gain_vs_night_mask_wm2"] = (
        wide["night_mask_mamba"] - wide["physics_mamba"]
    )
    wide["total_physics_gain_vs_vanilla_wm2"] = wide["mamba"] - wide["physics_mamba"]
    wide.sort_values(["horizon_minutes", "station"]).to_csv(
        output_dir / "ablation_station_horizon_comparison.csv", index=False
    )

    summary = (
        wide.groupby("horizon_minutes", as_index=False)
        .agg(
            vanilla_mae_wm2=("mamba", "mean"),
            night_mask_mae_wm2=("night_mask_mamba", "mean"),
            physics_mamba_mae_wm2=("physics_mamba", "mean"),
            night_mask_gain_wm2=("night_mask_gain_vs_vanilla_wm2", "mean"),
            envelope_gain_wm2=("envelope_gain_vs_night_mask_wm2", "mean"),
            total_gain_wm2=("total_physics_gain_vs_vanilla_wm2", "mean"),
            stations_with_total_gain=(
                "total_physics_gain_vs_vanilla_wm2", lambda values: int((values > 0).sum())
            ),
        )
        .sort_values("horizon_minutes")
    )
    summary.to_csv(output_dir / "ablation_horizon_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shortlist_tables(args.models_dir, args.output_dir)
    ablation_table(args.models_dir, args.output_dir)
    station = pd.read_csv(args.output_dir / "shortlist_station_summary.csv")
    ablation = pd.read_csv(args.output_dir / "ablation_horizon_summary.csv")
    print("Station summary")
    print(station.to_string(index=False))
    print("\nAblation summary")
    print(ablation.to_string(index=False))


if __name__ == "__main__":
    main()
