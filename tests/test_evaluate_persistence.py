import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.evaluate_persistence import aggregate_metrics, evaluate_station


class PersistenceEvaluationTests(unittest.TestCase):
    @patch("scripts.evaluate_persistence.ineichen_ghi")
    def test_exact_continuity_and_missing_values_are_required(self, clear_mock):
        clear_mock.return_value = np.full(5, 100.0)
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2020-01-01 00:00Z", "2020-01-01 00:15Z", "2020-01-01 00:45Z",
                     "2020-01-01 01:00Z", "2020-01-01 01:15Z"], utc=True
                ),
                "ghi_wm2": [10.0, 20.0, 30.0, np.nan, 50.0],
                "toa_horizontal_wm2": [100.0] * 5,
                "latitude": [0.0] * 5, "longitude": [0.0] * 5, "elevation_m": [0.0] * 5,
            }
        )
        rows = evaluate_station(frame, "AAA", [1])
        result = next(row for row in rows if row["regime"] == "all" and row["baseline"] == "persistence")
        self.assertEqual(result["n"], 1)
        self.assertAlmostEqual(result["mae_wm2"], 10.0)

    @patch("scripts.evaluate_persistence.ineichen_ghi")
    def test_smart_persistence_uses_clear_sky_index(self, clear_mock):
        clear_mock.return_value = np.asarray([100.0, 200.0])
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=2, freq="15min", tz="UTC"),
                "ghi_wm2": [50.0, 120.0], "toa_horizontal_wm2": [100.0, 200.0],
                "latitude": [0.0, 0.0], "longitude": [0.0, 0.0], "elevation_m": [0.0, 0.0],
            }
        )
        rows = evaluate_station(frame, "AAA", [1])
        smart = next(row for row in rows if row["regime"] == "all" and row["baseline"] == "smart_persistence")
        self.assertAlmostEqual(smart["mae_wm2"], 20.0)

    def test_smart_persistence_has_zero_skill_against_itself(self):
        station = pd.DataFrame([
            {"station": "AAA", "horizon_steps": 1, "horizon_minutes": 15,
             "regime": "daylight", "baseline": name, "n": 2,
             "mae_wm2": mae, "rmse_wm2": mae, "mean_observed_wm2": 100.0,
             "nmae": mae / 100.0, "sum_absolute_error": 2 * mae,
             "sum_squared_error": 2 * mae**2, "sum_observed": 200.0}
            for name, mae in (("persistence", 20.0), ("smart_persistence", 10.0))
        ])
        result = aggregate_metrics(station)
        smart = result[result["baseline"] == "smart_persistence"].iloc[0]
        persistence = result[result["baseline"] == "persistence"].iloc[0]
        self.assertAlmostEqual(smart["macro_smart_persistence_skill"], 0.0)
        self.assertAlmostEqual(persistence["macro_smart_persistence_skill"], -1.0)


if __name__ == "__main__":
    unittest.main()
