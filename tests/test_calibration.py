import unittest

import numpy as np

from modules.calibration import Calibration, load_calibration
from tests.helpers import make_temp_dir, remove_temp_dir


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.quantiles = np.linspace(-1.0, 1.0, 101)
        self.calibration = Calibration(
            threshold=-0.9,
            target_fpr=0.1,
            achieved_fpr={"holdout": 0.1},
            quantiles=self.quantiles,
        )

    def test_low_score_is_max_anomaly(self):
        self.assertEqual(self.calibration.percentile(-2.0), 0.0)
        self.assertEqual(self.calibration.anomaly_percentile(-2.0), 100.0)

    def test_high_score_is_normal(self):
        self.assertEqual(self.calibration.percentile(2.0), 100.0)
        self.assertEqual(self.calibration.anomaly_percentile(2.0), 0.0)

    def test_percentiles_are_monotonic(self):
        values = self.calibration.anomaly_percentiles(np.array([-2.0, 0.0, 2.0]))
        self.assertEqual(values.tolist(), [100.0, 50.0, 0.0])

    def test_missing_file_returns_none(self):
        path = make_temp_dir()
        try:
            self.assertIsNone(load_calibration(path))
        finally:
            remove_temp_dir(path)


if __name__ == "__main__":
    unittest.main()
