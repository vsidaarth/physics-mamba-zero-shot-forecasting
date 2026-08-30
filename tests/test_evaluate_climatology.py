import unittest

import numpy as np
import pandas as pd

from scripts.evaluate_climatology import local_solar_slot


class ClimatologyTests(unittest.TestCase):
    def test_local_solar_slot_respects_longitude_and_wraps(self):
        timestamps = pd.DatetimeIndex(pd.to_datetime(["2020-01-01 23:00Z"], utc=True))
        self.assertEqual(int(local_solar_slot(timestamps, 30.0)[0]), 4)
        self.assertEqual(int(local_solar_slot(timestamps, -30.0)[0]), 84)

    def test_slots_are_in_range(self):
        timestamps = pd.date_range("2020-01-01", periods=96, freq="15min", tz="UTC")
        slots = local_solar_slot(timestamps, 179.9)
        self.assertTrue(np.all((slots >= 0) & (slots < 96)))


if __name__ == "__main__":
    unittest.main()
