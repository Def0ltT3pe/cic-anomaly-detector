import argparse
import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from adapt_model import AdaptationError, _display_path, adapt
from modules.bruteforce import load_bruteforce
from modules.calibration import load_calibration
from modules.model_loader import load_model_objects, load_normal_stats
from modules.pipeline import detect_anomalies
from tests.helpers import make_temp_dir, remove_temp_dir

COLS = ["Flow Duration", "Total Fwd Packets", "Fwd IAT Mean", "URG Flag Count"]


def _benign(rng, n, scale=1.0):
    return pd.DataFrame(
        {
            "Flow Duration": rng.lognormal(3, 0.5, n) * scale,
            "Total Fwd Packets": rng.lognormal(1, 0.3, n),
            "Fwd IAT Mean": rng.lognormal(2, 0.5, n) * scale,
            "URG Flag Count": np.zeros(n),  # константа в целевой сети
        }
    )


def _brute(rng, n, scale=1.0):
    frame = _benign(rng, n, scale)
    frame["Total Fwd Packets"] = rng.lognormal(3, 0.2, n)
    return frame


def _write_base(directory: str) -> None:
    """Базовая модель «другой сети» в формате models/."""
    rng = np.random.default_rng(0)
    benign = _benign(rng, 3000)
    benign["URG Flag Count"] = rng.integers(0, 2, len(benign))
    scaler = StandardScaler().fit(benign)
    model = IsolationForest(n_estimators=50, random_state=0).fit(scaler.transform(benign))
    scores = model.decision_function(scaler.transform(benign))
    joblib.dump(model, os.path.join(directory, "isolation_forest.pkl"))
    joblib.dump(scaler, os.path.join(directory, "scaler.pkl"))
    joblib.dump(COLS, os.path.join(directory, "feature_cols_filtered.pkl"))
    with open(os.path.join(directory, "normal_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {"means": benign.mean().to_dict(), "stds": benign.std().to_dict()}, handle
        )
    with open(os.path.join(directory, "calibration.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "threshold": float(np.quantile(scores, 0.02)),
                "target_fpr": 0.02,
                "quantiles": np.quantile(scores, np.linspace(0, 1, 101)).tolist(),
            },
            handle,
        )
    attacks = _brute(rng, 300)
    X = pd.concat([benign, attacks], ignore_index=True)
    y = np.r_[np.zeros(len(benign)), np.ones(len(attacks))]
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    joblib.dump(rf, os.path.join(directory, "bruteforce.pkl"))
    with open(os.path.join(directory, "bruteforce.json"), "w", encoding="utf-8") as handle:
        json.dump({"threshold": 0.5, "target_fpr": 0.001, "feature_cols": COLS}, handle)


def _args(**overrides):
    defaults = dict(
        data=[],
        test_data=[],
        output_dir="",
        base_dir="",
        force=False,
        label_column="Label",
        classifier="all",
        target_fpr=0.02,
        brute_target_fpr=0.01,
        holdout=0.3,
        split="time",
        limit_per_file=0,
        trim=0.0,
        log1p=False,
        n_estimators=50,
        rf_trees=20,
        rf_depth=20,
        rf_leaf=5,
        rf_max_samples=0.5,
        seed=0,
        n_jobs=1,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class AdaptModelTests(unittest.TestCase):
    def setUp(self):
        self.dir = make_temp_dir()
        self.base = os.path.join(self.dir, "base")
        self.out = os.path.join(self.dir, "site")
        os.makedirs(self.base)
        _write_base(self.base)
        self.env = patch.dict(
            os.environ, {"VERIFY_MODEL_CHECKSUMS": "0", "AUDIT_ENABLED": "0"}, clear=False
        )
        self.env.start()
        # Целевая сеть: другая шкала времени (как IAT в 2017 и 2018)
        rng = np.random.default_rng(1)
        self.target = _benign(rng, 4000, scale=1000.0)

    def tearDown(self):
        self.env.stop()
        remove_temp_dir(self.dir)

    def _run(self, frames, **overrides):
        paths = []
        for number, frame in enumerate(frames):
            path = os.path.join(self.dir, f"target{number}.csv")
            frame.to_csv(path, index=False)
            paths.append(path)
        args = _args(data=paths, output_dir=self.out, base_dir=self.base, **overrides)
        with redirect_stdout(io.StringIO()):
            return adapt(args)

    def test_unlabeled_baseline_fixes_fpr(self):
        result = self._run([self.target])
        iforest = result["evaluation"]["isolation_forest"]
        self.assertFalse(result["labeled"])
        self.assertGreater(iforest["base"]["fpr"], 0.5)  # чужая шкала — всё «аномально»
        self.assertLess(abs(iforest["adapted"]["fpr"] - 0.02), 0.02)
        self.assertEqual(result["evaluation"]["supervised"]["mode"], "recalibrated")
        self.assertEqual(result["evaluation"]["supervised"]["display_name"], "BruteForce")

    def test_unlabeled_recalibration_never_lowers_threshold(self):
        result = self._run([self.target], brute_target_fpr=0.5)  # квантиль заведомо ниже
        brute = load_bruteforce(self.out)
        self.assertGreaterEqual(brute.threshold, 0.5)  # базовый порог в _write_base
        self.assertEqual(result["evaluation"]["supervised"]["mode"], "recalibrated")

    def test_output_dir_is_a_working_model(self):
        self._run([self.target], log1p=True)
        model, scaler, selector, cols = load_model_objects(self.out)
        means, stds = load_normal_stats(cols, model_dir=self.out)
        calibration = load_calibration(self.out)
        brute = load_bruteforce(self.out)
        self.assertIsNone(selector)
        self.assertEqual(cols, COLS)
        self.assertGreater(stds["URG Flag Count"], 0)  # константа -> std из базы
        self.assertEqual(brute.metadata["adaptation"]["mode"], "recalibrated")

        result = detect_anomalies(
            self.target, model, scaler, selector, cols, fill_values=means, calibration=calibration
        )
        self.assertLess(result.share, 0.1)
        with open(os.path.join(self.out, "metadata.json"), encoding="utf-8") as handle:
            metadata = json.load(handle)
        with open(os.path.join(self.out, "normal_stats.json"), encoding="utf-8") as handle:
            stats = json.load(handle)
        self.assertTrue(metadata["log1p"])
        self.assertIn("URG Flag Count", stats["std_from_base"])

    def _labeled(self):
        rng = np.random.default_rng(2)
        benign = self.target.assign(Label="Benign")
        brute = _brute(rng, 400, scale=1000.0).assign(Label="SSH-Patator")
        dos = _benign(rng, 400, scale=1000.0).assign(Label="DoS Hulk")
        dos["Flow Duration"] *= 50
        # Атаки равномерно по времени, чтобы попали и в train, и в holdout
        return pd.concat([benign, brute, dos]).sample(frac=1.0, random_state=0)

    def test_labeled_trains_classifier_of_all_attacks(self):
        frame = self._labeled()
        result = self._run([frame])
        block = result["evaluation"]["supervised"]
        self.assertTrue(result["labeled"])
        self.assertEqual((block["mode"], block["classifier"]), ("retrained", "all"))
        self.assertGreater(block["adapted"]["recall"], 0.9)
        self.assertLessEqual(block["adapted"]["fpr"], 0.05)
        self.assertEqual(set(block["adapted"]["families"]), {"BruteForce", "DoS"})
        brute = load_bruteforce(self.out)
        self.assertEqual(brute.display_name, "Атаки")
        self.assertEqual(brute.metadata["params"]["n_estimators"], 20)
        self.assertEqual(brute.model.max_depth, 20)
        self.assertEqual(brute.model.max_samples, 0.5)
        self.assertEqual(result["coverage"]["DoS"]["train"] + result["coverage"]["DoS"]["holdout"], 400)
        self.assertEqual(result["unchecked_families"], [])
        # IsolationForest учится только на benign
        self.assertLess(result["train_benign_rows"], len(self.target))

    def test_labeled_bruteforce_only(self):
        result = self._run([self._labeled()], classifier="bruteforce")
        block = result["evaluation"]["supervised"]
        self.assertEqual((block["mode"], block["classifier"]), ("retrained", "bruteforce"))
        self.assertGreater(block["adapted"]["families"]["BruteForce"]["recall"], 0.9)
        self.assertEqual(load_bruteforce(self.out).display_name, "BruteForce")

    def test_family_missing_from_holdout_is_reported(self):
        frame = self._labeled().sort_values("Label", key=lambda s: s != "SSH-Patator")
        result = self._run([frame])  # брутфорс в начале файла -> не в time-holdout
        self.assertIn("BruteForce", result["unchecked_families"])

    def test_rf_depth_zero_means_unlimited(self):
        self._run([self._labeled()], rf_depth=0)
        self.assertIsNone(load_bruteforce(self.out).model.max_depth)

    def test_separate_test_files_are_evaluated(self):
        train = self._labeled()
        test_path = os.path.join(self.dir, "day2.csv")
        rng = np.random.default_rng(5)
        day2 = pd.concat([
            _benign(rng, 2000, scale=1000.0).assign(Label="Benign"),
            _brute(rng, 200, scale=1000.0).assign(Label="SSH-Bruteforce"),
        ])
        day2.to_csv(test_path, index=False)
        result = self._run([train], test_data=[test_path])
        test = result["evaluation_test"]
        self.assertEqual(test["files"], ["day2.csv"])
        self.assertEqual(test["supervised"]["adapted"]["attacks"], 200)
        self.assertGreater(test["supervised"]["adapted"]["recall"], 0.9)
        self.assertLess(test["isolation_forest"]["adapted"]["fpr"], 0.1)
        with open(os.path.join(self.out, "metadata.json"), encoding="utf-8") as handle:
            self.assertIn("evaluation_test", json.load(handle))

    def test_test_data_must_not_overlap_training(self):
        path = os.path.join(self.dir, "target0.csv")
        with self.assertRaises(AdaptationError):
            self._run([self.target], test_data=[path])

    def test_trim_drops_rows(self):
        result = self._run([self.target], trim=0.05)
        self.assertGreater(result["trimmed_rows"], 0)

    def test_refuses_to_overwrite(self):
        self._run([self.target])
        with self.assertRaises(AdaptationError):
            self._run([self.target])
        self._run([self.target], force=True)

    def test_display_path_survives_other_drive(self):
        with patch("adapt_model.os.path.relpath", side_effect=ValueError("other mount")):
            self.assertEqual(_display_path(self.out), os.path.abspath(self.out))

    def test_refuses_base_as_output(self):
        path = os.path.join(self.dir, "t.csv")
        self.target.to_csv(path, index=False)
        with self.assertRaises(AdaptationError):
            adapt(_args(data=[path], output_dir=self.base, base_dir=self.base))

    def test_missing_features_are_reported(self):
        with self.assertRaises(AdaptationError):
            self._run([self.target.drop(columns=["Fwd IAT Mean"])])

    def test_too_little_benign(self):
        with self.assertRaises(AdaptationError):
            self._run([self.target.head(500)])


if __name__ == "__main__":
    unittest.main()
