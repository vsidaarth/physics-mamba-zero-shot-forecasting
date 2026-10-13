#!/usr/bin/env python3
"""Compare a fresh sealed-test run with the released station and aggregate tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare_table(
    reference_path: Path,
    reproduced_path: Path,
    *,
    rtol: float,
    atol: float,
) -> tuple[float, float, str]:
    reference = pd.read_csv(reference_path)
    reproduced = pd.read_csv(reproduced_path)
    if list(reference.columns) != list(reproduced.columns):
        raise RuntimeError(f"Column mismatch: {reference_path}")
    if len(reference) != len(reproduced):
        raise RuntimeError(f"Row-count mismatch: {reference_path}")

    maximum_absolute = 0.0
    maximum_relative = 0.0
    maximum_location = ""
    for column in reference.columns:
        left = reference[column]
        right = reproduced[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            left_values = left.to_numpy(dtype=float)
            right_values = right.to_numpy(dtype=float)
            if pd.api.types.is_integer_dtype(left) and pd.api.types.is_integer_dtype(right):
                if not np.array_equal(left_values, right_values):
                    raise RuntimeError(f"Integer mismatch in {reference_path}, column {column}")
                continue
            if not np.allclose(
                left_values,
                right_values,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ):
                difference = np.abs(left_values - right_values)
                maximum = float(np.nanmax(difference))
                raise RuntimeError(
                    f"Numeric mismatch in {reference_path}, column {column}: {maximum}"
                )
            absolute = np.abs(left_values - right_values)
            denominator = np.maximum(np.abs(left_values), atol)
            relative = absolute / denominator
            if np.isfinite(absolute).any():
                column_absolute = float(np.nanmax(absolute))
                column_relative = float(np.nanmax(relative))
                if column_absolute > maximum_absolute:
                    maximum_absolute = column_absolute
                    maximum_location = f"{reference_path.parent.name}/{reference_path.name}:{column}"
                maximum_relative = max(maximum_relative, column_relative)
        elif not left.fillna("<NA>").astype(str).equals(
            right.fillna("<NA>").astype(str)
        ):
            raise RuntimeError(f"Text mismatch in {reference_path}, column {column}")
    return maximum_absolute, maximum_relative, maximum_location


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("reproduced_root", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    project = args.project_root.resolve()
    reference_root = project / "results/sealed_test"
    reproduced_root = args.reproduced_root.resolve()
    completion_path = reproduced_root / "SUITE_COMPLETE.json"
    if not completion_path.is_file():
        raise SystemExit("The reproduced suite has no completion marker.")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("partial_results_inspected_during_run") is not False:
        raise SystemExit("The reproduced suite was not completed under the sealed-run rule.")

    reference_tables = sorted(reference_root.glob("*/*.csv"))
    if not reference_tables:
        raise SystemExit("No released sealed-test tables were found.")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    maximum_location = ""
    for reference_path in reference_tables:
        relative = reference_path.relative_to(reference_root)
        reproduced_path = reproduced_root / relative
        if not reproduced_path.is_file():
            raise SystemExit(f"Missing reproduced table: {relative}")
        absolute, relative_error, location = compare_table(
            reference_path,
            reproduced_path,
            rtol=args.rtol,
            atol=args.atol,
        )
        if absolute > maximum_absolute:
            maximum_absolute = absolute
            maximum_location = location
        maximum_relative = max(maximum_relative, relative_error)

    reproduced_tables = sorted(reproduced_root.glob("*/*.csv"))
    if len(reference_tables) != len(reproduced_tables):
        raise SystemExit(
            f"Table-count mismatch: released={len(reference_tables)}, "
            f"reproduced={len(reproduced_tables)}"
        )
    print(
        f"PASS: {len(reference_tables)} sealed-test tables reproduced; "
        f"maximum absolute difference={maximum_absolute:.6g} "
        f"({maximum_location}); maximum relative difference={maximum_relative:.6g}; "
        f"tolerances rtol={args.rtol:g}, atol={args.atol:g}"
    )


if __name__ == "__main__":
    main()
