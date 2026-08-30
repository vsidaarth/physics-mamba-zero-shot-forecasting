#!/usr/bin/env python3
"""Resume only supporting baselines after the audited legacy-LSTM metadata failure."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


COMPLETED_PRIMARY = (
    "physics_mamba_seed42", "physics_mamba_seed43", "physics_mamba_seed44",
    "itransformer_seed42", "itransformer_seed43", "itransformer_seed44",
    "mamba_seed42", "mamba_seed43", "mamba_seed44",
    "night_mask_mamba_seed42", "night_mask_mamba_seed43", "night_mask_mamba_seed44",
)
REMAINING_NEURAL = ("lstm_seed42", "tcn_seed42", "dlinear_seed42", "patchtst_seed42")


def resolve_run(project: Path, name: str) -> Path:
    for candidate in (project / "results" / name, project / "results/models" / name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(name)


def execute(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise SystemExit(f"Supporting-suite resume failed; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("freeze_manifest", type=Path)
    parser.add_argument("--expected-freeze-sha256", required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = project / "results/sealed_test"
    logs = output / "logs"
    if (output / "SUITE_COMPLETE.json").exists():
        raise SystemExit("The sealed-test suite is already complete.")
    for name in COMPLETED_PRIMARY:
        metadata_path = output / name / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("split") != "test" or metadata.get("test_evaluated") is not True:
            raise SystemExit(f"Incomplete primary run: {name}")
    for name in REMAINING_NEURAL + ("lightgbm_seed42", "persistence", "climatology"):
        if (output / name).exists():
            raise SystemExit(f"Refusing to overwrite supporting output: {name}")

    python = sys.executable
    processed = project / "data/processed/bsrn_15min"
    for name in REMAINING_NEURAL:
        execute([
            python, str(project / "scripts/evaluate_frozen_test_neural.py"),
            str(project), str(processed), str(resolve_run(project, name)), str(output / name),
            str(args.freeze_manifest), "--expected-freeze-sha256", args.expected_freeze_sha256,
            "--config", str(project / "configs/protocol_v1.json"), "--include-test",
        ], logs / f"{name}_retry.log")
    execute([
        python, str(project / "scripts/evaluate_frozen_test_lightgbm.py"),
        str(project), str(processed), str(project / "results"),
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
        "freeze_manifest_sha256": args.expected_freeze_sha256,
        "primary_runs_preserved": list(COMPLETED_PRIMARY),
        "supporting_runs_completed_after_audited_resume": list(REMAINING_NEURAL)
            + ["lightgbm_seed42", "persistence", "climatology"],
        "resume_reason": "legacy LSTM metadata lacked hyperparameters.architecture",
        "partial_metrics_inspected_before_resume": False,
    }
    (output / "SUITE_COMPLETE.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
