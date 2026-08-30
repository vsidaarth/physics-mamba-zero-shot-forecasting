#!/usr/bin/env python3
"""Inference-only evaluation of one frozen neural checkpoint on sealed test stations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from scripts.train_lstm import (
        CONTEXT_FEATURES, FUTURE_FEATURES, STATIC_FEATURES,
        DLinearForecaster, ITransformerForecaster, LSTMForecaster,
        MambaForecaster, NightMaskedMambaForecaster, PatchTSTForecaster,
        PhysicsMambaForecaster, TCNForecaster, aggregate_metrics, evaluate,
        load_station,
    )
except ModuleNotFoundError:
    from train_lstm import (
        CONTEXT_FEATURES, FUTURE_FEATURES, STATIC_FEATURES,
        DLinearForecaster, ITransformerForecaster, LSTMForecaster,
        MambaForecaster, NightMaskedMambaForecaster, PatchTSTForecaster,
        PhysicsMambaForecaster, TCNForecaster, aggregate_metrics, evaluate,
        load_station,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("processed_dir", type=Path)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("freeze_manifest", type=Path)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/protocol_v1.json"))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--include-test", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(
    architecture: str, hp: dict[str, object], context_steps: int, horizons: int,
    context_mean: np.ndarray, context_std: np.ndarray,
) -> torch.nn.Module:
    hidden = int(hp["hidden_size"])
    layers = int(hp["layers"])
    dropout = float(hp["dropout"])
    dimensions = (len(CONTEXT_FEATURES) * 2, len(FUTURE_FEATURES), len(STATIC_FEATURES))
    if architecture == "lstm":
        return LSTMForecaster(*dimensions, horizons, hidden, layers, dropout)
    if architecture == "tcn":
        return TCNForecaster(*dimensions, horizons, hidden, dropout)
    if architecture == "dlinear":
        return DLinearForecaster(context_steps, len(FUTURE_FEATURES), len(STATIC_FEATURES), horizons, dropout)
    if architecture == "patchtst":
        return PatchTSTForecaster(context_steps, len(FUTURE_FEATURES), len(STATIC_FEATURES), horizons, hidden, layers, dropout)
    if architecture == "itransformer":
        return ITransformerForecaster(context_steps, *dimensions, horizons, hidden, layers, dropout)
    if architecture == "mamba":
        return MambaForecaster(*dimensions, horizons, hidden, layers, dropout)
    toa_position = CONTEXT_FEATURES.index("toa_horizontal_wm2")
    toa_mean = float(context_mean[toa_position])
    toa_std = float(context_std[toa_position])
    if architecture == "night_mask_mamba":
        return NightMaskedMambaForecaster(*dimensions, horizons, hidden, layers, dropout, toa_mean, toa_std)
    if architecture == "physics_mamba":
        return PhysicsMambaForecaster(*dimensions, horizons, hidden, layers, dropout, toa_mean, toa_std)
    raise ValueError(f"Unsupported architecture: {architecture}")


def main() -> None:
    args = parse_args()
    if not args.include_test:
        raise SystemExit("The test split is sealed; --include-test is required after the final freeze.")
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to reuse existing test output: {args.output_dir}")
    if sha256(args.freeze_manifest) != args.expected_freeze_sha256:
        raise SystemExit("Freeze manifest hash mismatch.")
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if freeze.get("test_evaluated") is not False or freeze.get("test_opening_count") != 0:
        raise SystemExit("Freeze manifest does not certify an unopened test split.")

    project = args.project_root.resolve()
    metadata_path = args.result_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    architecture = str(
        metadata["hyperparameters"].get("architecture", metadata["model"].lower())
    )
    checkpoint_path = args.result_dir / f"{architecture}_checkpoint.pt"
    relative_checkpoint = checkpoint_path.resolve().relative_to(project).as_posix()
    frozen_files = {record["path"]: record["sha256"] for record in freeze["files"]}
    checkpoint_digest = sha256(checkpoint_path)
    if (
        frozen_files.get(relative_checkpoint) != checkpoint_digest
        and checkpoint_digest not in set(frozen_files.values())
    ):
        raise SystemExit(f"Checkpoint is absent from or differs from freeze: {relative_checkpoint}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    eligibility = config["evaluation_eligibility"]
    context_steps = int(eligibility["context_steps"])
    minimum_fraction = float(eligibility["minimum_context_ghi_fraction"])
    test_files = sorted((args.processed_dir / "test").glob("*.parquet"))
    if len(test_files) != 11:
        raise SystemExit(f"Expected exactly 11 sealed test stations, found {len(test_files)}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    horizons = [int(value) for value in checkpoint["horizons"]]
    context_mean = np.asarray(checkpoint["context_mean"], dtype=np.float32)
    context_std = np.asarray(checkpoint["context_std"], dtype=np.float32)
    static_mean = np.asarray(checkpoint["static_mean"], dtype=np.float32)
    static_std = np.asarray(checkpoint["static_std"], dtype=np.float32)
    stations = [
        load_station(path, horizons, context_steps, minimum_fraction,
                     context_mean, context_std, static_mean, static_std)
        for path in test_files
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        architecture, metadata["hyperparameters"], context_steps, len(horizons),
        context_mean, context_std,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    workers = 0 if "mamba" in architecture else args.workers
    rows = evaluate(
        model, stations, horizons, context_steps, device,
        args.batch_size, workers, architecture,
    )
    station_metrics = pd.DataFrame(rows)
    aggregate = aggregate_metrics(station_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    station_metrics.to_csv(args.output_dir / "station_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    test_metadata = {
        "protocol_id": config["protocol_id"],
        "freeze_id": freeze["freeze_id"],
        "freeze_manifest_sha256": args.expected_freeze_sha256,
        "model": architecture,
        "seed": int(metadata["seed"]),
        "split": "test",
        "test_evaluated": True,
        "evaluation_stations": len(stations),
        "source_checkpoint": relative_checkpoint,
        "source_checkpoint_sha256": checkpoint_digest,
        "model_selection_performed": False,
        "evaluation_eligibility": eligibility,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(test_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(aggregate[aggregate["baseline"] == architecture].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
