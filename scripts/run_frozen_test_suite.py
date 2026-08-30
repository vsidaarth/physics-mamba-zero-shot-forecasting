#!/usr/bin/env python3
"""Run the complete predeclared sealed-test suite without inspecting partial results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


NEURAL_RUNS = (
    "physics_mamba_seed42", "physics_mamba_seed43", "physics_mamba_seed44",
    "itransformer_seed42", "itransformer_seed43", "itransformer_seed44",
    "mamba_seed42", "mamba_seed43", "mamba_seed44",
    "night_mask_mamba_seed42", "night_mask_mamba_seed43", "night_mask_mamba_seed44",
    "lstm_seed42", "tcn_seed42", "dlinear_seed42", "patchtst_seed42",
)


def resolve_run(project: Path, name: str) -> Path:
    for candidate in (
        project / "checkpoints" / name,
        project / "results" / name,
        project / "results" / "models" / name,
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(name)


def execute(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False,
        )
    if completed.returncode:
        raise SystemExit(f"Test suite failed; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("freeze_manifest", type=Path)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/sealed_test_reproduction"),
        help="Fresh output directory. Existing published results are never overwritten.",
    )
    args = parser.parse_args()
    project = args.project_root.resolve()
    processed = project / "data/processed/bsrn_15min"
    output = args.output_root.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to reuse sealed-test root: {output}")
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    python = sys.executable

    for name in NEURAL_RUNS:
        result_dir = resolve_run(project, name)
        command = [
            python, str(project / "scripts/evaluate_frozen_test_neural.py"),
            str(project), str(processed), str(result_dir), str(output / name),
            str(args.freeze_manifest),
            "--expected-freeze-sha256", args.expected_freeze_sha256,
            "--config", str(project / "configs/protocol_v1.json"),
            "--include-test",
        ]
        execute(command, logs / f"{name}.log")

    execute([
        python, str(project / "scripts/evaluate_frozen_test_lightgbm.py"),
        str(project), str(processed), str(project / "checkpoints"),
        str(output / "lightgbm_seed42"), str(args.freeze_manifest),
        "--expected-freeze-sha256", args.expected_freeze_sha256,
        "--config", str(project / "configs/protocol_v1.json"), "--include-test",
    ], logs / "lightgbm_seed42.log")

    execute([
        python, str(project / "scripts/evaluate_persistence.py"), str(processed),
        str(output / "persistence"), "--config", str(project / "configs/protocol_v1.json"),
        "--split", "test", "--include-test",
    ], logs / "persistence.log")
    execute([
        python, str(project / "scripts/evaluate_climatology.py"), str(processed),
        str(output / "climatology"), "--config", str(project / "configs/protocol_v1.json"),
        "--split", "test", "--include-test",
    ], logs / "climatology.log")

    completion = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_manifest": str(args.freeze_manifest),
        "freeze_manifest_sha256": args.expected_freeze_sha256,
        "neural_runs": list(NEURAL_RUNS),
        "other_runs": ["lightgbm_seed42", "persistence", "climatology"],
        "partial_results_inspected_during_run": False,
    }
    (output / "SUITE_COMPLETE.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
