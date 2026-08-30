#!/usr/bin/env python3
"""Paired station-level uncertainty analysis for the validation shortlist."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("physics_mamba", "itransformer")
SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--resamples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_metrics(models_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model in MODELS:
        for seed in SEEDS:
            path = models_dir / f"{model}_seed{seed}" / "station_metrics.csv"
            frame = pd.read_csv(path)
            frame = frame[
                (frame["baseline"] == model) & (frame["regime"] == "daylight")
            ].copy()
            frame["model"] = model
            frame["seed"] = seed
            frames.append(frame[[
                "model", "seed", "station", "horizon_steps",
                "horizon_minutes", "n", "mae_wm2", "rmse_wm2", "nmae",
            ]])
    metrics = pd.concat(frames, ignore_index=True)
    expected = len(MODELS) * len(SEEDS) * 7 * 5
    if len(metrics) != expected:
        raise RuntimeError(f"Expected {expected} metric rows, found {len(metrics)}")
    return metrics


def paired_bootstrap(
    paired: pd.DataFrame, resamples: int, rng: np.random.Generator
) -> tuple[float, float]:
    differences = paired["difference_wm2"].to_numpy(dtype=np.float64)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    draws = differences[indices].mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.models_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "shortlist_station_seed_metrics.csv", index=False)

    station_means = (
        metrics.groupby(["model", "station", "horizon_steps", "horizon_minutes"], as_index=False)
        .agg(mae_wm2=("mae_wm2", "mean"), seed_sd_wm2=("mae_wm2", "std"))
    )
    station_means.to_csv(args.output_dir / "shortlist_station_3seed_means.csv", index=False)

    physics = station_means[station_means["model"] == "physics_mamba"]
    transformer = station_means[station_means["model"] == "itransformer"]
    paired = physics.merge(
        transformer,
        on=["station", "horizon_steps", "horizon_minutes"],
        suffixes=("_physics", "_itransformer"),
        validate="one_to_one",
    )
    paired["difference_wm2"] = (
        paired["mae_wm2_physics"] - paired["mae_wm2_itransformer"]
    )
    paired.to_csv(args.output_dir / "paired_station_differences.csv", index=False)

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int]] = []
    for (steps, minutes), group in paired.groupby(["horizon_steps", "horizon_minutes"]):
        low, high = paired_bootstrap(group, args.resamples, rng)
        physics_mean = float(group["mae_wm2_physics"].mean())
        transformer_mean = float(group["mae_wm2_itransformer"].mean())
        difference = physics_mean - transformer_mean
        rows.append({
            "horizon_steps": int(steps),
            "horizon_minutes": int(minutes),
            "stations": len(group),
            "seeds": len(SEEDS),
            "physics_mamba_macro_mae_wm2": physics_mean,
            "itransformer_macro_mae_wm2": transformer_mean,
            "paired_difference_wm2": difference,
            "relative_mae_reduction_pct": -100.0 * difference / transformer_mean,
            "station_bootstrap_ci_low_wm2": low,
            "station_bootstrap_ci_high_wm2": high,
            "physics_station_wins": int((group["difference_wm2"] < 0).sum()),
        })
    summary = pd.DataFrame(rows).sort_values("horizon_minutes")
    summary.to_csv(args.output_dir / "paired_station_bootstrap.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
