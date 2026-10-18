import unittest

import numpy as np
import pandas as pd

from modules.explainer import explain_anomaly


class ExplainerTests(unittest.TestCase):
    def setUp(self):
        self.means = pd.Series({"a": 5.0, "b": 1.0, "c": 0.0})
        self.stds = pd.Series({"a": 1.0, "b": 0.0, "c": 2.0})

    def test_normal_output(self):
        record = pd.Series({"a": 10.0, "b": 1.0, "c": 1.0})
        text = explain_anomaly(record, self.means, self.stds)
        self.assertIn("a:", text)
        self.assertIn("выше", text)

    def test_zero_std_is_skipped(self):
        record = pd.Series({"a": 5.0, "b": 1.0, "c": 0.0})
        text = explain_anomaly(record, self.means, self.stds)
        self.assertNotIn("b:", text)

    def test_all_nan_returns_message(self):
        record = pd.Series({"a": np.nan, "b": np.nan})
        text = explain_anomaly(record, self.means, self.stds)
        self.assertIn("не рассчитаны", text)

    def test_non_numeric_value_does_not_crash(self):
        record = pd.Series({"a": "text", "b": 1.0})
        text = explain_anomaly(record, self.means, self.stds)
        self.assertIsInstance(text, str)

    def test_empty_top_n(self):
        record = pd.Series({"a": 10.0})
        text = explain_anomaly(record, self.means, self.stds, top_n=0)
        self.assertIn("a:", text)


if __name__ == "__main__":
    unittest.main()
