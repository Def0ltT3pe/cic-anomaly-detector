import unittest

import numpy as np
import pandas as pd

from evaluate import (
    NETWORK_FEATURES,
    Dataset,
    attack_family,
    augment_dataset,
    evaluate_scores,
    run,
    transform,
)


def _dataset(name: str, shift: float, seed: int) -> Dataset:
    """Benign ~ N(0, 1), атаки сдвинуты по feature a; шум в сетевом признаке."""
    rng = np.random.default_rng(seed)
    n_benign, n_attack = 600, 60
    X = pd.DataFrame(
        {
            "a": np.r_[rng.normal(0, 1, n_benign), rng.normal(6, 1, n_attack)],
            "b": rng.normal(0, 1, n_benign + n_attack),
            "Bwd Header Length": rng.normal(shift, 1, n_benign + n_attack),
        }
    )
    labels = pd.Series(["Benign"] * n_benign + ["DoS Hulk"] * n_attack)
    return Dataset(name=name, X=X, labels=labels, files=pd.Series(["f"] * len(X)))


class AttackFamilyTests(unittest.TestCase):
    def test_labels_from_both_datasets(self):
        cases = {
            "Benign": "Benign",
            "DoS Hulk": "DoS",
            "DoS attacks-Slowloris": "DoS",
            "DDoS": "DDoS",
            "DDOS attack-HOIC": "DDoS",
            "SSH-Patator": "BruteForce",
            "FTP-BruteForce": "BruteForce",
            "Web Attack � XSS": "Web",
            "Brute Force -Web": "Web",
            "SQL Injection": "Web",
            "Bot": "Bot",
            "Infilteration": "Infiltration",
            "PortScan": "PortScan",
            "Heartbleed": "DoS",
            "Something new": "Other",
            # CIC-DDoS2019: всё — DDoS, MSSQL не Web, DrDoS не DoS
            "DrDoS_DNS": "DDoS",
            "DrDoS_MSSQL": "DDoS",
            "MSSQL": "DDoS",
            "Syn": "DDoS",
            "UDP-lag": "DDoS",
            "UDPLag": "DDoS",
            "WebDDoS": "DDoS",
            "Portmap": "DDoS",
        }
        for label, family in cases.items():
            self.assertEqual(attack_family(label), family, label)


class TransformTests(unittest.TestCase):
    def test_drop_network_features(self):
        X = pd.DataFrame({name: [1.0] for name in NETWORK_FEATURES + ["Flow Duration"]})
        result = transform(X, drop_net=True, log=False)
        self.assertEqual(list(result.columns), ["Flow Duration"])

    def test_drop_time_features(self):
        names = ["Flow Duration", "Fwd IAT Mean", "Flow Bytes/s", "Active Mean",
                 "Idle Max", "Total Fwd Packets", "Packet Length Mean"]
        X = pd.DataFrame({name: [1.0] for name in names})
        result = transform(X, drop_time=True)
        self.assertEqual(list(result.columns), ["Total Fwd Packets", "Packet Length Mean"])

    def test_signed_log1p(self):
        X = pd.DataFrame({"a": [-1.0, 0.0, np.e - 1]})
        result = transform(X, drop_net=False, log=True)
        np.testing.assert_allclose(result["a"], [-np.log(2), 0.0, 1.0])


class EvaluateScoresTests(unittest.TestCase):
    def test_tied_scores_do_not_flag_everything(self):
        data = _dataset("x", 0.0, 0)
        scores = np.zeros(len(data.X))  # как predict_proba RF на «чистом» benign
        metrics = evaluate_scores(data, scores, threshold=0.0, target_fpr=0.01)
        self.assertEqual(metrics["fpr"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)

    def test_oracle_threshold_uses_target_benign(self):
        data = _dataset("x", 0.0, 0)
        scores = data.X["a"].to_numpy() + 100  # порог источника заведомо мимо
        metrics = evaluate_scores(data, scores, threshold=0.0, target_fpr=0.05)
        self.assertEqual(metrics["fpr"], 1.0)
        self.assertGreater(metrics["recall_oracle"], 0.9)
        self.assertIn("DoS", metrics["families"])


class RunTests(unittest.TestCase):
    def test_both_directions_and_variants(self):
        datasets = {"src": _dataset("src", 0.0, 1), "dst": _dataset("dst", 50.0, 2)}
        results = run(
            datasets,
            detectors=["if", "rf"],
            variants=["baseline", "drop_net", "log1p"],
            target_fpr={"if": 0.05, "rf": 0.05},
            holdout=0.3,
            max_train=0,
            seed=0,
            n_jobs=1,
            log=lambda *_: None,
        )
        routes = {(r["source"], r["target"], r["detector"], r["variant"]) for r in results}
        self.assertIn(("src", "dst", "if", "log1p"), routes)
        self.assertIn(("dst", "src", "rf", "drop_net"), routes)
        self.assertNotIn(("src", "dst", "rf", "log1p"), routes)  # log1p не меняет деревья

        by_route = {(r["source"], r["target"], r["detector"], r["variant"]): r for r in results}
        # Сдвиг сетевого признака превращает весь трафик цели в «аномалии»,
        # а без этого признака FPR возвращается к целевому
        baseline = by_route[("src", "dst", "if", "baseline")]
        dropped = by_route[("src", "dst", "if", "drop_net")]
        self.assertGreater(baseline["fpr"], 0.5)
        self.assertLess(dropped["fpr"], 0.2)
        self.assertGreater(dropped["recall"], 0.9)
        self.assertEqual(by_route[("src", "dst", "if", "drop_net")]["features"], 2)

    def test_mix_is_never_tested_on_its_components(self):
        datasets = {
            "a": _dataset("a", 0.0, 1),
            "b": _dataset("b", 0.0, 2),
            "c": _dataset("c", 0.0, 3),
        }
        results = run(
            datasets,
            detectors=["if"],
            variants=["baseline"],
            target_fpr={"if": 0.05},
            holdout=0.3,
            max_train=0,
            seed=0,
            n_jobs=1,
            mixes=[["a", "b"]],
            sources=["a+b", "a"],
            targets=["c", "b"],
            log=lambda *_: None,
        )
        routes = {(r["source"], r["target"], r["in_domain"]) for r in results}
        self.assertEqual(
            routes,
            {
                ("a+b", "a+b", True),
                ("a+b", "c", False),
                ("a", "a", True),
                ("a", "c", False),
                ("a", "b", False),
            },
        )
        mixed = next(r for r in results if r["source"] == "a+b" and r["in_domain"])
        total = len(datasets["a"].X) + len(datasets["b"].X)
        self.assertEqual(mixed["rows"], total - int(total * (1.0 - 0.3)))

    def test_augment_adds_only_donor_attacks_and_excludes_them_from_eval(self):
        base = _dataset("a", 0.0, 1)
        donor = _dataset("b", 0.0, 2)
        donor.labels = pd.Series(
            ["Benign"] * 600 + ["DDOS attack-HOIC"] * 30 + ["SSH-Bruteforce"] * 30
        )
        augmented = augment_dataset(base, donor, ["DDoS"])
        self.assertEqual(augmented.name, "a+b[DDoS]")
        self.assertEqual(len(augmented.X), len(base.X) + 30)  # без benign донора
        self.assertEqual((augmented.labels == "DDOS attack-HOIC").sum(), 30)

        results = run(
            {"a": base, "b": donor},
            detectors=["rf"],
            variants=["baseline"],
            target_fpr={"rf": 0.05},
            holdout=0.3,
            max_train=0,
            seed=0,
            n_jobs=1,
            sources=["a+b[DDoS]"],
            augments=[("a", "b", ["DDoS"])],
            log=lambda *_: None,
        )
        on_donor = next(r for r in results if r["target"] == "b")
        self.assertEqual(on_donor["excluded_families"], ["DDoS"])
        self.assertEqual(set(on_donor["families"]), {"BruteForce"})
        self.assertNotIn("a", {r["target"] for r in results if not r["in_domain"]})

    def test_domain_normalization_fixes_scale_shift(self):
        # Цель — та же сеть, но все признаки в 1000 раз крупнее (как IAT
        # в мкс против мс): без нормировки весь трафик выглядит аномальным.
        rng = np.random.default_rng(3)
        n_benign, n_attack = 800, 80

        def make(name, factor):
            X = pd.DataFrame(
                {
                    "a": np.r_[rng.lognormal(0, 0.3, n_benign), rng.lognormal(2, 0.3, n_attack)],
                    "b": rng.lognormal(0, 0.3, n_benign + n_attack),
                }
            ) * factor
            labels = pd.Series(["Benign"] * n_benign + ["DoS Hulk"] * n_attack)
            return Dataset(name, X, labels, pd.Series(["f"] * len(X)))

        datasets = {"src": make("src", 1.0), "dst": make("dst", 1000.0)}
        results = run(
            datasets,
            detectors=["if", "hgb"],
            variants=["baseline", "domain"],
            target_fpr={"if": 0.05, "hgb": 0.05},
            holdout=0.3,
            max_train=0,
            seed=0,
            n_jobs=1,
            log=lambda *_: None,
        )
        by_route = {(r["source"], r["target"], r["detector"], r["variant"]): r for r in results}
        baseline = by_route[("src", "dst", "if", "baseline")]
        domain = by_route[("src", "dst", "if", "domain")]
        self.assertGreater(baseline["fpr"], 0.5)
        self.assertLess(domain["fpr"], 0.15)
        self.assertGreater(domain["recall"], 0.9)
        self.assertEqual(domain["threshold_from"], "target benign")
        self.assertGreater(by_route[("src", "dst", "hgb", "domain")]["recall"], 0.9)


if __name__ == "__main__":
    unittest.main()
