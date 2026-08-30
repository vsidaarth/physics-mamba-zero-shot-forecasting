#!/usr/bin/env python3
"""Inference-only sealed-test evaluation of the five frozen LightGBM models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

try:
    from scripts.evaluate_persistence import aggregate_metrics
    from scripts.train_lightgbm import evaluate_model
except ModuleNotFoundError:
    from evaluate_persistence import aggregate_metrics
    from train_lightgbm import evaluate_model


HORIZONS = (1, 2, 4, 12, 24)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("processed_dir", type=Path)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("freeze_manifest", type=Path)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/protocol_v1.json"))
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()
    if not args.include_test:
        raise SystemExit("The test split is sealed; --include-test is required after the final freeze.")
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to reuse existing test output: {args.output_dir}")
    if sha256(args.freeze_manifest) != args.expected_freeze_sha256:
        raise SystemExit("Freeze manifest hash mismatch.")
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    frozen_files = {record["path"]: record["sha256"] for record in freeze["files"]}
    project = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    eligibility = config["evaluation_eligibility"]
    context_steps = int(eligibility["context_steps"])
    minimum_fraction = float(eligibility["minimum_context_ghi_fraction"])
    test_files = sorted((args.processed_dir / "test").glob("*.parquet"))
    if len(test_files) != 11:
        raise SystemExit(f"Expected exactly 11 test stations, found {len(test_files)}")

    rows: list[dict[str, object]] = []
    source_models: list[dict[str, object]] = []
    for horizon in HORIZONS:
        model_path = args.model_root / f"lightgbm_h{horizon}_seed42" / f"lightgbm_h{horizon}.txt"
        relative = model_path.resolve().relative_to(project).as_posix()
        digest = sha256(model_path)
        if frozen_files.get(relative) != digest and digest not in set(frozen_files.values()):
            raise SystemExit(f"LightGBM model differs from freeze: {relative}")
        model = lgb.Booster(model_file=str(model_path))
        rows.extend(evaluate_model(model, test_files, horizon, context_steps, minimum_fraction))
        source_models.append({"horizon_steps": horizon, "path": relative, "sha256": digest})
        print(f"Evaluated LightGBM horizon {horizon} steps", flush=True)

    station_metrics = pd.DataFrame(rows)
    aggregate = aggregate_metrics(station_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    station_metrics.to_csv(args.output_dir / "station_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "protocol_id": config["protocol_id"],
        "freeze_id": freeze["freeze_id"],
        "freeze_manifest_sha256": args.expected_freeze_sha256,
        "model": "lightgbm",
        "seed": 42,
        "split": "test",
        "test_evaluated": True,
        "evaluation_stations": 11,
        "source_models": source_models,
        "model_selection_performed": False,
        "evaluation_eligibility": eligibility,
    }, indent=2) + "\n", encoding="utf-8")
    print(aggregate[aggregate["baseline"] == "lightgbm"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
