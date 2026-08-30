#!/usr/bin/env python3
"""Generate publication-ready validation figures from frozen result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


HORIZONS = (15, 30, 60, 180, 360)
HORIZON_LABELS = ("15 min", "30 min", "1 h", "3 h", "6 h")
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7B3294"
NEUTRAL = "#4D4D4D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def architecture_figure(output: Path) -> None:
    """Draw a compact two-row forecasting path that remains legible in print."""
    fig, ax = plt.subplots(figsize=(7.3, 4.2), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = (
        (0.04, 0.61, 0.25, 0.23,
         "Past 24 h context\n96 × 15-min samples\nGHI + meteorology\n+ availability masks", "#E8F2F8", 9.3),
        (0.39, 0.57, 0.27, 0.31,
         "Input projection\n3 × Mamba-2 residual blocks\nSequence summary", "#DDEBF7", 9.5),
        (0.04, 0.14, 0.25, 0.21,
         "Known-future covariates\nsolar/calendar\n+ station metadata", "#FFF4D6", 9.3),
        (0.40, 0.13, 0.25, 0.23,
         "Transmission head\n$\\widehat{\\kappa}=2\\,\\mathrm{sigmoid}(z)$\n$0\\leq\\widehat{\\kappa}\\leq2$", "#E5F5ED", 9.8),
        (0.73, 0.54, 0.23, 0.20,
         "Deterministic\nsolar geometry\nfuture TOA irradiance $I_0$", "#FFF4D6", 9.4),
        (0.73, 0.11, 0.23, 0.28,
         "GHI forecasts\n15 min to 6 h\n\n$\\widehat{y}=\\widehat{\\kappa}I_0$\n$0\\leq\\widehat{y}\\leq2I_0$", "#E5F5ED", 9.8),
    )
    for x, y, width, height, label, fill, font_size in nodes:
        patch = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.014,rounding_size=0.018",
            linewidth=1.2, edgecolor=NEUTRAL, facecolor=fill,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, label,
                ha="center", va="center", fontsize=font_size, linespacing=1.12)

    arrows = (
        ((0.304, 0.725), (0.376, 0.725), "arc3,rad=0"),
        ((0.525, 0.555), (0.525, 0.375), "arc3,rad=0"),
        ((0.304, 0.245), (0.386, 0.245), "arc3,rad=0"),
        ((0.664, 0.245), (0.716, 0.245), "arc3,rad=0"),
        ((0.845, 0.525), (0.845, 0.405), "arc3,rad=0"),
    )
    for start, end, connection in arrows:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=10,
            linewidth=1.15, color=NEUTRAL, connectionstyle=connection,
            shrinkA=0, shrinkB=0,
        ))

    ax.text(0.845, 0.045, r"Astronomical night: $I_0=0\Rightarrow\widehat{y}=0$",
            ha="center", va="center", color=GREEN, fontsize=9.8)
    save_figure(fig, output, "fig0_physics_mamba_architecture")


def main_horizon_figure(project: Path, output: Path) -> None:
    data = pd.read_csv(project / "results/replications/shortlist_3seed_summary.csv")
    fig, ax = plt.subplots(figsize=(7.1, 4.25), constrained_layout=True)
    x = np.arange(len(HORIZONS))
    for model, label, color, marker in (
        ("physics_mamba", "Physics-Mamba", BLUE, "o"),
        ("itransformer", "iTransformer", ORANGE, "s"),
    ):
        rows = data[data["model"] == model].set_index("horizon_minutes").loc[list(HORIZONS)]
        mean = rows["mean_macro_mae_wm2"].to_numpy()
        ax.plot(
            x, mean, color=color, marker=marker, markersize=5.5,
            linewidth=1.8, label=label,
        )
    ax.set_xticks(x, HORIZON_LABELS)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Daylight station-macro MAE (W/m²)")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.99, 0.02, "Mean across seeds 42, 43, 44",
        transform=ax.transAxes, ha="right", va="bottom", color=NEUTRAL, fontsize=8,
    )
    save_figure(fig, output, "fig1_validation_horizon_mae")


def paired_effect_figure(project: Path, output: Path) -> None:
    data = pd.read_csv(project / "results/uncertainty/paired_station_bootstrap.csv")
    data = data.set_index("horizon_minutes").loc[list(HORIZONS)]
    effect = data["paired_difference_wm2"].to_numpy()
    low = data["station_bootstrap_ci_low_wm2"].to_numpy()
    high = data["station_bootstrap_ci_high_wm2"].to_numpy()
    x = np.arange(len(HORIZONS))
    fig, ax = plt.subplots(figsize=(7.1, 4.25), constrained_layout=True)
    ax.axhline(0.0, color=NEUTRAL, linewidth=1.0)
    ax.errorbar(
        x, effect, yerr=np.vstack([effect - low, high - effect]),
        fmt="o", color=BLUE, ecolor=BLUE, markersize=6, capsize=4, linewidth=1.6,
    )
    for index, (value, wins) in enumerate(zip(effect, data["physics_station_wins"])):
        offset = -10 if value < 0 else 8
        ax.annotate(
            f"{int(wins)}/7 wins", (index, value), xytext=(0, offset),
            textcoords="offset points", ha="center",
            va="top" if value < 0 else "bottom", fontsize=8,
        )
    ax.set_xticks(x, HORIZON_LABELS)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Paired MAE difference (W/m²)\nPhysics-Mamba − iTransformer")
    ax.grid(axis="y")
    ax.text(
        0.02, 0.04, "Negative values favor Physics-Mamba",
        transform=ax.transAxes, color=NEUTRAL, fontsize=8,
    )
    save_figure(fig, output, "fig2_paired_station_effects")


def ablation_figure(project: Path, output: Path) -> None:
    data = pd.read_csv(project / "results/ablation_replications/ablation_3seed_summary.csv")
    x = np.arange(len(HORIZONS))
    fig, ax = plt.subplots(figsize=(7.1, 4.25), constrained_layout=True)
    series = (
        ("vanilla_mamba", "Vanilla Mamba", NEUTRAL, "^"),
        ("night_mask_mamba", "+ astronomical night mask", GREEN, "s"),
        ("physics_mamba", "+ complete solar envelope", BLUE, "o"),
    )
    for model, label, color, marker in series:
        rows = data[data["model"] == model].set_index("horizon_minutes").loc[list(HORIZONS)]
        ax.plot(
            x, rows["mean_macro_mae_wm2"], label=label, color=color,
            marker=marker, linewidth=1.8, markersize=5.5,
        )
    ax.set_xticks(x, HORIZON_LABELS)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Daylight station-macro MAE (W/m²)")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.99, 0.02, "Mean across seeds 42, 43, 44",
        transform=ax.transAxes, ha="right", va="bottom", color=NEUTRAL, fontsize=8,
    )
    save_figure(fig, output, "fig3_physics_ablation")


def heatmap(
    ax: plt.Axes, values: np.ndarray, row_labels: list[str], title: str,
    color_limit: float,
) -> None:
    image = ax.imshow(
        values, cmap="RdBu_r", vmin=-color_limit, vmax=color_limit,
        aspect="auto", interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(HORIZONS)), HORIZON_LABELS)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title(title, loc="left", fontweight="normal")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text_color = "white" if abs(value) > color_limit * 0.58 else "black"
            ax.text(column, row, f"{value:+.1f}", ha="center", va="center", color=text_color, fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("#B0B0B0")
    return image


def subgroup_figure(project: Path, output: Path) -> None:
    data = pd.read_csv(project / "results/subgroup_analysis/subgroup_paired_comparison.csv")
    panels = (
        ("solar_elevation_deg", ["0-10", "10-20", "20-40", "40-60", ">60"], "a  Target solar elevation (degrees)"),
        ("observed_variability_quintile", ["Q1", "Q2", "Q3", "Q4", "Q5"], "b  Observation-defined variability"),
        ("local_season", ["winter", "spring", "summer", "autumn"], "c  Local meteorological season"),
    )
    matrices: list[np.ndarray] = []
    for group_type, labels, _ in panels:
        subset = data[data["group_type"] == group_type]
        pivot = subset.pivot(index="group", columns="horizon_minutes", values="physics_minus_itransformer_wm2")
        matrices.append(pivot.loc[labels, list(HORIZONS)].to_numpy())
    color_limit = float(np.ceil(max(np.abs(matrix).max() for matrix in matrices)))
    fig, axes = plt.subplots(3, 1, figsize=(7.25, 8.1), constrained_layout=True)
    image = None
    for ax, matrix, (_, labels, title) in zip(axes, matrices, panels):
        image = heatmap(ax, matrix, labels, title, color_limit)
        ax.set_ylabel("Subgroup")
    axes[-1].set_xlabel("Forecast horizon")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, location="right", shrink=0.72, pad=0.03)
    colorbar.set_label("MAE difference (W/m²)\nPhysics-Mamba − iTransformer")
    save_figure(fig, output, "fig4_subgroup_effects")


def station_heatmap_figure(project: Path, output: Path) -> None:
    data = pd.read_csv(
        project / "results/station_analysis/shortlist_station_horizon_comparison.csv"
    )
    pivot = data.pivot(
        index="station", columns="horizon_minutes", values="physics_minus_itransformer_wm2"
    )
    order = data.groupby("station")["physics_minus_itransformer_wm2"].mean().sort_values().index
    matrix = pivot.loc[order, list(HORIZONS)].to_numpy()
    limit = float(np.ceil(np.abs(matrix).max()))
    fig, ax = plt.subplots(figsize=(7.1, 4.4), constrained_layout=True)
    image = heatmap(ax, matrix, list(order), "Paired effects by unseen validation station", limit)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Validation station")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.03)
    colorbar.set_label("MAE difference (W/m²)\nPhysics-Mamba − iTransformer")
    save_figure(fig, output, "fig5_station_horizon_effects")


def main() -> None:
    args = parse_args()
    configure_style()
    architecture_figure(args.output_dir)
    main_horizon_figure(args.project_root, args.output_dir)
    paired_effect_figure(args.project_root, args.output_dir)
    ablation_figure(args.project_root, args.output_dir)
    subgroup_figure(args.project_root, args.output_dir)
    station_heatmap_figure(args.project_root, args.output_dir)
    print(f"Created six PNG and six PDF figures in {args.output_dir}")


if __name__ == "__main__":
    main()
