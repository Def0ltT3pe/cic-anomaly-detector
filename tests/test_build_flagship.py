import os
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from build_flagship import FlagshipError, _cap, _weighted_quantile, load_site
from tests.helpers import make_temp_dir, remove_temp_dir


class CapTests(unittest.TestCase):
    def test_benign_and_families_are_capped(self):
        labels = pd.Series(["Benign"] * 100 + ["SSH-Bruteforce"] * 50 + ["SSH-Probe"] * 5)
        rows = _cap(labels, max_benign=30, max_attack=20, rng=np.random.default_rng(0))
        picked = labels.iloc[rows].value_counts()
        self.assertEqual(picked["Benign"], 30)
        # малое семейство берётся целиком, остаток бюджета — крупному
        self.assertEqual(picked["SSH-Probe"], 5)
        self.assertEqual(picked["SSH-Bruteforce"], 15)
        self.assertTrue(np.all(np.diff(rows) > 0))

    def test_weighted_quantile_gives_networks_equal_weight(self):
        big = np.zeros(900)
        small = np.ones(100)
        values = np.concatenate([big, small])
        weights = np.concatenate([np.full(900, 1 / 900), np.full(100, 1 / 100)])
        # без весов 60% квантиль был бы 0; с равным весом сетей — 1
        self.assertEqual(_weighted_quantile(values, weights, 0.6), 1.0)
        self.assertEqual(_weighted_quantile(values, weights, 0.4), 0.0)


class LoadSiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = make_temp_dir("flagship-")
        self.addCleanup(remove_temp_dir, self.tmp)
        self.args = SimpleNamespace(max_benign=1000, max_attack=1000)
        for n in range(4):
            pd.DataFrame({
                "a": np.arange(10, dtype=float) + n,
                "b": np.inf,
                "Label": ["Benign"] * 8 + ["SSH-Bruteforce"] * 2,
            }).to_parquet(os.path.join(self.tmp, f"capture_{n:05d}_labeled.parquet"))

    def test_last_files_are_test(self):
        network = load_site(f"siteA={self.tmp}:1", ["a", "b"], self.args, np.random.default_rng(0))
        self.assertEqual(network.name, "siteA")
        self.assertEqual(set(network.data.files.iloc[network.test]), {"capture_00003_labeled.parquet"})
        self.assertFalse(set(network.data.files.iloc[network.train]) & {"capture_00003_labeled.parquet"})
        self.assertFalse(network.data.y[network.calib].any())
        self.assertTrue(network.data.X["b"].isna().all())  # inf -> NaN, заполнит pipeline

    def test_bad_spec(self):
        with self.assertRaises(FlagshipError):
            load_site(f"siteA={self.tmp}:4", ["a", "b"], self.args, np.random.default_rng(0))
        with self.assertRaises(FlagshipError):
            load_site("без-знака-равно", ["a"], self.args, np.random.default_rng(0))
