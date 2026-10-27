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

    def test_score_on_quantile_node_is_exact(self):
        # quantiles[2] — 2-й перцентиль нормы: аномальнее ровно 98%, а не 97/99
        values = self.calibration.anomaly_percentiles(self.quantiles[[1, 2, 50]])
        np.testing.assert_allclose(values, [99.0, 98.0, 50.0])

    def test_interpolates_between_nodes(self):
        middle = (self.quantiles[2] + self.quantiles[3]) / 2
        self.assertAlmostEqual(self.calibration.percentile(middle), 2.5)

    def test_non_finite_score_is_max_anomaly(self):
        values = self.calibration.anomaly_percentiles(np.array([np.nan, np.inf]))
        self.assertEqual(values.tolist(), [100.0, 100.0])
        self.assertEqual(self.calibration.anomaly_percentile(float("nan")), 100.0)

    def test_missing_file_returns_none(self):
        path = make_temp_dir()
        try:
            self.assertIsNone(load_calibration(path))
        finally:
            remove_temp_dir(path)


if __name__ == "__main__":
    unittest.main()
