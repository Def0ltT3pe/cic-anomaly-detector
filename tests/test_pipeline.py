import unittest

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from modules.pipeline import FeatureValidationError, detect_anomalies, prepare_features


class PrepareFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.cols = ["a", "b", "c"]
        self.means = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})

    def test_valid_frame(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0], "c": [3.0, 4.0]})
        X, info = prepare_features(df, self.cols)
        self.assertEqual(X.shape, (2, 3))
        self.assertEqual(info["rows"], 2)
        self.assertEqual(info["inf_replaced"], 0)

    def test_missing_columns(self):
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        with self.assertRaises(FeatureValidationError) as ctx:
            prepare_features(df, self.cols)
        self.assertIn("нужных колонок", str(ctx.exception))

    def test_empty_frame(self):
        df = pd.DataFrame(columns=self.cols)
        with self.assertRaises(FeatureValidationError):
            prepare_features(df, self.cols)

    def test_text_values(self):
        df = pd.DataFrame({"a": ["tcp"], "b": [2.0], "c": [3.0]})
        with self.assertRaises(FeatureValidationError) as ctx:
            prepare_features(df, self.cols)
        self.assertIn("текстовые", str(ctx.exception))

    def test_inf_and_nan_are_filled(self):
        df = pd.DataFrame(
            {"a": [np.inf, 1.0], "b": [np.nan, 2.0], "c": [3.0, 3.0]}
        )
        X, info = prepare_features(df, self.cols, fill_values=self.means)
        self.assertTrue(np.isfinite(X.to_numpy()).all())
        self.assertEqual(info["inf_replaced"], 1)
        self.assertEqual(info["nan_filled"], 2)

    def test_nan_without_fill_values(self):
        df = pd.DataFrame({"a": [np.nan], "b": [2.0], "c": [3.0]})
        with self.assertRaises(FeatureValidationError):
            prepare_features(df, self.cols)

    def test_max_rows(self):
        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0], "c": [3.0, 4.0, 5.0]}
        )
        with self.assertRaises(FeatureValidationError):
            prepare_features(df, self.cols, max_rows=2)

    def test_duplicate_columns(self):
        df = pd.DataFrame(
            [[1.0, 2.0, 3.0]], columns=["a", "a", "c"]
        )
        with self.assertRaises(FeatureValidationError):
            prepare_features(df, self.cols)


class DetectAnomaliesTests(unittest.TestCase):
    def test_isolation_forest_single_pass_matches_predict(self):
        rng = np.random.default_rng(0)
        cols = ["a", "b", "c"]
        df = pd.DataFrame(rng.normal(size=(300, 3)), columns=cols)
        df.iloc[:10] *= 8  # явные выбросы
        scaler = StandardScaler().fit(df)
        model = IsolationForest(random_state=0, contamination=0.05).fit(scaler.transform(df))
        result = detect_anomalies(df, model, scaler, None, cols)
        X = scaler.transform(df)
        np.testing.assert_array_equal(result.predictions, model.predict(X))
        np.testing.assert_array_equal(result.scores, model.decision_function(X))
        self.assertGreater(result.n_anomalies, 0)


if __name__ == "__main__":
    unittest.main()
