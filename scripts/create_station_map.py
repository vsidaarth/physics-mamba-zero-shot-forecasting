#!/usr/bin/env python3
"""Create the station-disjoint BSRN split map used in the manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


COLORS = {"train": "#666666", "validation": "#0072B2", "test": "#D55E00"}
MARKERS = {"train": "o", "validation": "s", "test": "^"}
SIZES = {"train": 22, "validation": 48, "test": 48}
LABEL_OFFSETS = {
    "PSU": (4, 7), "CLH": (4, -8), "LRC": (4, 0),
    "LER": (4, 3), "CAB": (4, 7), "PAY": (4, 0), "CAR": (4, -8),
    "SMS": (-5, 2), "FLO": (5, 7), "KWA": (-5, 5), "NAU": (-5, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("station_manifest", type=Path)
    parser.add_argument("world_geojson", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def world_patches(path: Path) -> list[Polygon]:
    data = json.loads(path.read_text(encoding="utf-8"))
    patches: list[Polygon] = []
    for feature in data["features"]:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        if geometry["type"] == "Polygon":
            polygons = [geometry["coordinates"]]
        elif geometry["type"] == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            continue
        for polygon in polygons:
            exterior = polygon[0]
            patches.append(Polygon(exterior, closed=True))
    return patches


def main() -> None:
    args = parse_args()
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })
    stations = pd.read_csv(args.station_manifest)
    stations = stations[stations["eligible"].astype(str).str.lower().eq("true")].copy()
    stations["split"] = stations["split"].replace({"val": "validation"})
    counts = stations.groupby("split").size().to_dict()
    expected = {"train": 33, "validation": 7, "test": 11}
    if counts != expected:
        raise ValueError(f"Unexpected split counts: {counts}; expected {expected}")

    fig, ax = plt.subplots(figsize=(10.0, 5.35), constrained_layout=True)
    collection = PatchCollection(
        world_patches(args.world_geojson), facecolor="#F1F1EE", edgecolor="#B8B8B2",
        linewidth=0.35, zorder=0,
    )
    ax.add_collection(collection)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#EAF3F7")
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_xticklabels(["180°W", "120°W", "60°W", "0°", "60°E", "120°E", "180°E"])
    ax.set_yticks([-60, -30, 0, 30, 60, 90])
    ax.set_yticklabels(["60°S", "30°S", "0°", "30°N", "60°N", "90°N"])
    ax.grid(color="white", linewidth=0.6, alpha=0.9, zorder=-1)

    for split in ("train", "validation", "test"):
        subset = stations[stations["split"] == split]
        ax.scatter(
            subset["longitude"], subset["latitude"], s=SIZES[split],
            marker=MARKERS[split], c=COLORS[split], edgecolors="white",
            linewidths=0.55, zorder=3 if split != "train" else 2,
            label=f"{split.capitalize()} (n={len(subset)})",
        )

    for _, row in stations[stations["split"].isin(["validation", "test"])].iterrows():
        default_dx = 3.0 if row["longitude"] < 150 else -3.0
        default_dy = 2.2 if row["latitude"] < 70 else -3.2
        dx, dy = LABEL_OFFSETS.get(row["station_code"], (default_dx, default_dy))
        ha = "left" if dx > 0 else "right"
        ax.annotate(
            row["station_code"], (row["longitude"], row["latitude"]),
            xytext=(dx, dy), textcoords="offset points", ha=ha,
            va="bottom" if dy > 0 else ("top" if dy < 0 else "center"), fontsize=6.8,
            color="#333333", zorder=4,
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Station-disjoint BSRN zero-shot forecasting protocol", loc="left")
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#B8B8B2", ncol=3)
    for spine in ax.spines.values():
        spine.set_color("#9A9A94")
        spine.set_linewidth(0.7)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "fig_station_split_map.png", dpi=300)
    fig.savefig(args.output_dir / "fig_station_split_map.pdf")
    plt.close(fig)
    print(f"Created station map with split counts {counts}")


if __name__ == "__main__":
    main()
