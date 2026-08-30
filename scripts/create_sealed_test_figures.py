#!/usr/bin/env python3
"""Create the final sealed-test figures from consolidated frozen results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HORIZONS = (15, 30, 60, 180, 360)
LABELS = ("15 min", "30 min", "1 h", "3 h", "6 h")
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#4D4D4D"


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
        "axes.titlesize": 10, "legend.fontsize": 8.5, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": "#D9D9D9",
        "grid.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    analysis = args.project_root / "results/sealed_test_analysis"
    summary = pd.read_csv(analysis / "test_all_methods_summary.csv")
    effects = pd.read_csv(analysis / "test_primary_comparison.csv")
    paired = pd.read_csv(analysis / "test_paired_station_bootstrap.csv")
    x = np.arange(len(HORIZONS))

    fig, ax = plt.subplots(figsize=(7.1, 4.25), constrained_layout=True)
    for model, label, color, marker in (
        ("physics_mamba", "Physics-Mamba", BLUE, "o"),
        ("itransformer", "iTransformer", ORANGE, "s"),
    ):
        rows = summary[summary["model"] == model].set_index("horizon_minutes").loc[list(HORIZONS)]
        ax.plot(x, rows["mean_macro_mae_wm2"], color=color, marker=marker,
                linewidth=1.8, markersize=5.5, label=label)
    ax.set_xticks(x, LABELS)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Test daylight station-macro MAE (W/m²)")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.99, 0.02, "11 unseen stations; mean across 3 seeds",
            transform=ax.transAxes, ha="right", va="bottom", color=GREY, fontsize=8)
    save(fig, args.output_dir, "fig_test_horizon_mae")

    rows = effects.set_index("horizon_minutes").loc[list(HORIZONS)]
    value = rows["paired_difference_wm2"].to_numpy()
    low = rows["ci_low_wm2"].to_numpy()
    high = rows["ci_high_wm2"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.1, 4.25), constrained_layout=True)
    ax.axhline(0, color=GREY, linewidth=1)
    ax.errorbar(x, value, yerr=np.vstack([value - low, high - value]), fmt="o",
                color=BLUE, ecolor=BLUE, capsize=4, linewidth=1.6, markersize=6)
    for index, wins in enumerate(rows["candidate_station_wins"]):
        ax.text(index, low[index] - 0.07, f"{int(wins)}/11",
                ha="center", va="top", fontsize=8)
    ax.set_xticks(x, LABELS)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Paired test MAE difference (W/m²)\nPhysics-Mamba − iTransformer")
    ax.grid(axis="y")
    ax.set_ylim(float(low.min() - 0.35), float(high.max() + 0.15))
    ax.text(0.02, 0.95, "Negative values favor Physics-Mamba; labels show station wins",
            transform=ax.transAxes, ha="left", va="top", color=GREY, fontsize=8)
    save(fig, args.output_dir, "fig_test_paired_effects")

    fig, ax = plt.subplots(figsize=(7.1, 4.25), constrained_layout=True)
    for model, label, color, marker in (
        ("vanilla_mamba", "Vanilla Mamba", GREY, "^"),
        ("night_mask_mamba", "+ astronomical night mask", GREEN, "s"),
        ("physics_mamba", "+ complete solar envelope", BLUE, "o"),
    ):
        rows = summary[summary["model"] == model].set_index("horizon_minutes").loc[list(HORIZONS)]
        ax.plot(x, rows["mean_macro_mae_wm2"], color=color, marker=marker,
                linewidth=1.8, markersize=5.5, label=label)
    ax.set_xticks(x, LABELS)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Test daylight station-macro MAE (W/m²)")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.99, 0.02, "11 unseen stations; mean across 3 seeds",
            transform=ax.transAxes, ha="right", va="bottom", color=GREY, fontsize=8)
    save(fig, args.output_dir, "fig_test_physics_ablation")
    print(f"Created three PNG and three PDF sealed-test figures in {args.output_dir}")


if __name__ == "__main__":
    main()
