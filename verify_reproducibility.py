#!/usr/bin/env python3
"""Run the fast, data-free verification checks for the public release."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EXPECTED_FREEZE_SHA256 = "35435f3639f0ee6583504376cf4450d7a29800f6661624cd3c3421d1b73ddfbf"
EXPECTED_SPLIT = {"train": 33, "val": 7, "test": 11}
EXPECTED_HORIZONS = [1, 2, 4, 12, 24]
REFERENCE_TABLES = (
    "test_seed_metrics.csv",
    "test_station_metrics.csv",
    "test_all_methods_summary.csv",
    "test_paired_station_bootstrap.csv",
    "test_primary_comparison.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoints() -> None:
    checked = 0
    for line in (ROOT / "CHECKPOINTS.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Checkpoint file is missing: {relative}")
        if sha256(path) != expected:
            raise RuntimeError(f"Checkpoint checksum mismatch: {relative}")
        checked += 1
    print(f"PASS: checkpoint checksums ({checked} files)")


def verify_protocol() -> None:
    protocol = json.loads((ROOT / "configs/protocol_v1.json").read_text(encoding="utf-8"))
    if protocol["context_steps"]["primary"] != 96:
        raise RuntimeError("Unexpected context length")
    if protocol["forecast_horizon_steps"] != EXPECTED_HORIZONS:
        raise RuntimeError("Unexpected forecast horizons")

    stations = pd.read_csv(ROOT / "manifests/final_station_manifest.csv")
    eligible = stations[stations["eligible"].astype(str).str.lower().eq("true")]
    counts = eligible.groupby("split").size().to_dict()
    if counts != EXPECTED_SPLIT:
        raise RuntimeError(f"Unexpected station split: {counts}")

    freeze = ROOT / "freezes/validation_freeze_manifest.json"
    if sha256(freeze) != EXPECTED_FREEZE_SHA256:
        raise RuntimeError("Frozen validation manifest checksum mismatch")
    completion = json.loads(
        (ROOT / "results/sealed_test/SUITE_COMPLETE.json").read_text(encoding="utf-8")
    )
    if completion["freeze_manifest_sha256"] != EXPECTED_FREEZE_SHA256:
        raise RuntimeError("Sealed-test completion record does not match the freeze")
    print("PASS: frozen protocol, station split, and sealed-test marker")


def compare_csv(expected_path: Path, actual_path: Path) -> None:
    expected = pd.read_csv(expected_path)
    actual = pd.read_csv(actual_path)
    if list(expected.columns) != list(actual.columns) or len(expected) != len(actual):
        raise RuntimeError(f"Shape mismatch in {expected_path.name}")
    for column in expected.columns:
        left = expected[column]
        right = actual[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            if not np.allclose(
                left.to_numpy(dtype=float), right.to_numpy(dtype=float),
                rtol=1e-10, atol=1e-10, equal_nan=True,
            ):
                raise RuntimeError(f"Numeric mismatch in {expected_path.name}: {column}")
        elif not left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str)):
            raise RuntimeError(f"Text mismatch in {expected_path.name}: {column}")


def reproduce_analysis() -> None:
    with tempfile.TemporaryDirectory(prefix="verify_", dir=ROOT) as temporary:
        work = Path(temporary)
        (work / "scripts").mkdir()
        (work / "results").mkdir()
        shutil.copy2(ROOT / "scripts/analyze_sealed_test.py", work / "scripts")
        shutil.copytree(ROOT / "results/sealed_test", work / "results/sealed_test")
        subprocess.run(
            [
                sys.executable,
                str(work / "scripts/analyze_sealed_test.py"),
                str(work),
                "--bootstrap-replicates", "100000",
                "--bootstrap-seed", "20260810",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        for name in REFERENCE_TABLES:
            compare_csv(
                ROOT / "results/sealed_test_analysis" / name,
                work / "results/sealed_test_analysis" / name,
            )
    print("PASS: sealed-test aggregation and paired bootstrap")


def run_unit_tests() -> None:
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        check=True,
    )
    print("PASS: unit tests")


def main() -> None:
    verify_checkpoints()
    verify_protocol()
    run_unit_tests()
    reproduce_analysis()
    print("PASS: quick reproducibility verification")


if __name__ == "__main__":
    main()
