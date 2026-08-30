from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.preprocess_bsrn import aggregate_member, load_manifest


def synthetic_month(
    cadence_minutes: int,
    rows: int,
    qc_by_row: dict[int, int] | None = None,
    missing_rows: set[int] | None = None,
) -> bytes:
    qc_by_row = qc_by_row or {}
    missing_rows = missing_rows or set()
    timestamp = pd.Timestamp("2020-01-01T00:00:00")
    lines = [
        "/* metadata */",
        "Date/Time\tSWD [W/m**2]\tSWDQc\tSZA [deg]\tMu0\tSWDTOA [W/m**2]",
    ]
    for index in range(rows):
        qc = qc_by_row.get(index, 0)
        value = "" if index in missing_rows else str(index + 1)
        lines.append(f"{timestamp.isoformat()}\t{value}\t{qc}\t80\t0.1736\t237")
        timestamp += pd.Timedelta(minutes=cadence_minutes)
    return ("\n".join(lines) + "\n").encode()


class AggregateMemberTests(unittest.TestCase):
    def test_primary_qc_rejects_physical_flag_but_keeps_rare_flag(self) -> None:
        frame, diagnostics = aggregate_member(
            synthetic_month(1, 15, {0: 1, 1: 4}),
            resolution_minutes=15,
            minimum_coverage=0.8,
            maximum_cadence=10,
            reject_mask=3,
        )
        self.assertEqual(diagnostics["native_cadence_minutes"], 1)
        self.assertEqual(int(frame.loc[0, "ghi_wm2_valid_samples"]), 14)
        self.assertEqual(int(frame.loc[0, "ghi_qc_zero_samples"]), 13)
        self.assertTrue(bool(frame.loc[0, "ghi_primary_valid"]))
        self.assertTrue(bool(frame.loc[0, "ghi_strict_valid"]))
        self.assertAlmostEqual(float(frame.loc[0, "ghi_wm2"]), sum(range(2, 16)) / 14)

    def test_cadence_aware_coverage_accepts_four_of_five_samples(self) -> None:
        frame, diagnostics = aggregate_member(
            synthetic_month(3, 5, {0: 2}),
            resolution_minutes=15,
            minimum_coverage=0.8,
            maximum_cadence=10,
            reject_mask=3,
        )
        self.assertEqual(diagnostics["native_cadence_minutes"], 3)
        self.assertEqual(int(frame.loc[0, "expected_native_samples"]), 5)
        self.assertEqual(int(frame.loc[0, "ghi_wm2_valid_samples"]), 4)
        self.assertTrue(bool(frame.loc[0, "ghi_primary_valid"]))

    def test_strict_count_does_not_include_missing_measurements(self) -> None:
        frame, _ = aggregate_member(
            synthetic_month(1, 15, missing_rows={0, 1, 2}),
            resolution_minutes=15,
            minimum_coverage=0.8,
            maximum_cadence=10,
            reject_mask=3,
        )
        self.assertEqual(int(frame.loc[0, "ghi_qc_zero_samples"]), 12)
        self.assertEqual(int(frame.loc[0, "ghi_wm2_valid_samples"]), 12)
        self.assertTrue(bool(frame.loc[0, "ghi_strict_valid"]))


class ManifestLockTests(unittest.TestCase):
    def test_test_stations_are_locked_by_default(self) -> None:
        columns = ["station_code", "eligible", "split"]
        rows = [
            {"station_code": "AAA", "eligible": "True", "split": "train"},
            {"station_code": "BBB", "eligible": "True", "split": "val"},
            {"station_code": "CCC", "eligible": "True", "split": "test"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
            unlocked = load_manifest(path, include_test=True, requested=None)
            locked = load_manifest(path, include_test=False, requested=None)
        self.assertEqual({row["station_code"] for row in unlocked}, {"AAA", "BBB", "CCC"})
        self.assertEqual({row["station_code"] for row in locked}, {"AAA", "BBB"})


if __name__ == "__main__":
    unittest.main()
