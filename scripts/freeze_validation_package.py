#!/usr/bin/env python3
"""Create a checksum manifest for the frozen pre-test validation package."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


FREEZE_ID = "bsrn-location-zero-shot-v1-final-pretest-freeze-2026-08-10"
REQUIRED_RUNS = {
    "itransformer": (42, 43, 44),
    "physics_mamba": (42, 43, 44),
    "mamba": (42, 43, 44),
    "night_mask_mamba": (42, 43, 44),
}
CORE_FILES = (
    "configs/protocol_v1.json",
    "data/raw/bsrn_snapshot_957398/final_station_manifest.csv",
    "scripts/train_lstm.py",
    "scripts/train_lightgbm.py",
    "scripts/evaluate_persistence.py",
    "scripts/evaluate_climatology.py",
    "scripts/evaluate_validation_subgroups.py",
    "scripts/analyze_shortlist_uncertainty.py",
    "scripts/analyze_station_results.py",
    "scripts/analyze_subgroup_results.py",
    "scripts/analyze_ablation_replications.py",
    "scripts/create_paper_figures.py",
    "scripts/create_station_map.py",
    "scripts/freeze_validation_package.py",
    "scripts/evaluate_frozen_test_neural.py",
    "scripts/evaluate_frozen_test_lightgbm.py",
    "scripts/run_frozen_test_suite.py",
    "TEST_EVALUATION_PLAN.md",
    "ANALYSIS_PLAN.md",
    "VALIDATION_ANALYSIS.md",
    "RESULTS_STATUS.md",
    "PAPER_PREVIEW.md",
    "MANUSCRIPT_TABLES.md",
    "FIGURE_CAPTIONS.md",
)
RESULT_FILES = (
    "results/replications/shortlist_3seed_summary.csv",
    "results/uncertainty/paired_station_bootstrap.csv",
    "results/subgroup_analysis/subgroup_paired_comparison.csv",
    "results/subgroup_analysis/subgroup_3seed_model_summary.csv",
    "results/ablation_replications/ablation_seed_metrics.csv",
    "results/ablation_replications/ablation_3seed_summary.csv",
    "results/ablation_replications/ablation_paired_station_bootstrap.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_run(project: Path, model: str, seed: int) -> Path:
    name = f"{model}_seed{seed}"
    candidates = (project / "results" / name, project / "results" / "models" / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing required run: {name}")


def collect_files(project: Path, run_dirs: list[Path]) -> list[Path]:
    files = [project / relative for relative in CORE_FILES + RESULT_FILES]
    files.extend(path for path in (project / "figures").glob("*") if path.is_file())
    files.extend(path for path in (project / "results" / "baselines").rglob("*") if path.is_file())
    files.extend(path for path in (project / "results" / "models").rglob("*") if path.is_file())
    for run_dir in run_dirs:
        files.extend(path for path in run_dir.glob("*") if path.is_file())
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen files:\n" + "\n".join(map(str, missing)))
    return sorted(set(files), key=lambda path: path.relative_to(project).as_posix())


def main() -> None:
    args = parse_args()
    project = args.project_root.resolve()
    output = args.output_dir.resolve()

    stations = pd.read_csv(project / "data/raw/bsrn_snapshot_957398/final_station_manifest.csv")
    stations = stations[stations["eligible"].astype(str).str.lower().eq("true")].copy()
    stations["split"] = stations["split"].replace({"val": "validation"})
    split_counts = stations.groupby("split").size().to_dict()
    expected_counts = {"train": 33, "validation": 7, "test": 11}
    if split_counts != expected_counts:
        raise ValueError(f"Station split mismatch: {split_counts}")

    run_dirs: list[Path] = []
    run_records: list[dict[str, object]] = []
    for model, seeds in REQUIRED_RUNS.items():
        for seed in seeds:
            run_dir = resolve_run(project, model, seed)
            metadata_path = run_dir / "metadata.json"
            checkpoint_path = run_dir / "best_state.pt"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("test_evaluated") is not False:
                raise ValueError(f"Test status is not sealed in {metadata_path}")
            if int(metadata["seed"]) != seed:
                raise ValueError(f"Seed mismatch in {metadata_path}")
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            run_dirs.append(run_dir)
            run_records.append({
                "model": model,
                "seed": seed,
                "run_directory": run_dir.relative_to(project).as_posix(),
                "epochs_run": int(metadata["epochs_run"]),
                "test_evaluated": False,
                "checkpoint_sha256": sha256(checkpoint_path),
                "metadata_sha256": sha256(metadata_path),
            })

    for metadata_path in (project / "results").rglob("metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("test_evaluated") is True:
            raise ValueError(f"Found an evaluated test result: {metadata_path}")

    files = collect_files(project, run_dirs)
    file_records = [
        {
            "path": path.relative_to(project).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    manifest = {
        "freeze_id": FREEZE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": "bsrn-location-zero-shot-v1",
        "station_split": split_counts,
        "test_evaluated": False,
        "test_opening_count": 0,
        "context_steps": 96,
        "resolution_minutes": 15,
        "forecast_horizons_minutes": [15, 30, 60, 180, 360],
        "primary_metric": "daylight station-macro MAE",
        "primary_test_comparison": "physics_mamba versus itransformer, seeds averaged within station",
        "uncertainty_unit": "paired station bootstrap",
        "frozen_runs": run_records,
        "files": file_records,
    }

    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "validation_freeze_manifest.json"
    json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_digest = sha256(json_path)
    (output / "validation_freeze_manifest.sha256").write_text(
        f"{manifest_digest}  validation_freeze_manifest.json\n", encoding="utf-8"
    )
    markdown = f"""# Validation freeze

- Freeze ID: `{FREEZE_ID}`
- Protocol: `bsrn-location-zero-shot-v1`
- Created (UTC): `{manifest['created_utc']}`
- Station split: 33 train / 7 validation / 11 sealed test
- Required frozen neural runs: {len(run_records)}
- Hashed files: {len(file_records)}
- Test evaluations performed: **0**
- Manifest SHA-256: `{manifest_digest}`

The next permitted model operation is inference-only evaluation of these frozen
checkpoints on the eleven test stations. No hyperparameter, checkpoint,
subgroup, metric, or claim may be changed in response to the test results.
"""
    (output / "VALIDATION_FREEZE.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
