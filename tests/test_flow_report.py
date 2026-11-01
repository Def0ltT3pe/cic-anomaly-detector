import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from modules.alerts import aggregate_alerts
from modules.flow_report import build_flow_report, first_message, flagged_candidates, mask_ip
from modules.pipeline import DetectionResult

SERVER = "198.51.100.7"


def make_result():
    rows = []
    for n in range(6):
        attacker = n < 3
        rows.append({
            "Src IP": "185.12.34.56" if attacker else f"203.0.113.{n}",
            "Src Port": 50000 + n,
            "Dst IP": SERVER,
            "Dst Port": 22 if attacker else 443,
            "Timestamp": f"2026-10-30 10:0{n}:00",
            "Protocol": 6,
            "Flow Duration": 2_000_000,
            "Total Fwd Packets": 12,
            "Total Backward Packets": 10,
            "Fwd Packets Length Total": 900,
            "Bwd Packets Length Total": 1500,
            "Label": "SSH-Bruteforce" if attacker else "Benign",
        })
    df = pd.DataFrame(rows)
    X = pd.DataFrame({"a": [10.0, 9, 8, 1, 1, 1], "b": [0.0] * 6})
    mask = np.array([False, True, False, False, False, True])
    scores = np.array([0.1, -0.2, 0.05, 0.2, 0.2, -0.3])
    result = DetectionResult(df=df, X=X, predictions=np.where(mask, -1, 1), mask=mask,
                             indices=np.flatnonzero(mask), scores=scores, threshold=-0.1,
                             percentiles=np.full(6, 99.0))
    probs = np.array([0.99, 0.95, 0.7, 0.01, 0.02, 0.03])
    brute = SimpleNamespace(display_name="BruteForce", threshold=0.5)
    return result, brute, probs, probs >= brute.threshold


class MaskIpTests(unittest.TestCase):
    def test_roles(self):
        self.assertEqual(mask_ip(SERVER, SERVER), "сервер")
        self.assertEqual(mask_ip("185.12.34.56", SERVER), "внешний 185.12.x.x")
        self.assertEqual(mask_ip("10.1.2.3"), "внутренний 10.1.x.x")
        self.assertEqual(mask_ip("2001:db8::1"), "внутренний 2001:0db8::x")
        self.assertEqual(mask_ip("не адрес"), "адрес скрыт")


class FlowReportTests(unittest.TestCase):
    def setUp(self):
        self.result, self.brute, self.probs, self.flagged = make_result()
        self.means = pd.Series({"a": 1.0, "b": 0.0})
        self.stds = pd.Series({"a": 1.0, "b": 1.0})
        self.incidents = aggregate_alerts(self.result.df, self.flagged, self.probs)

    def report(self, index, mask_ips):
        return build_flow_report(self.result, index, self.means, self.stds, brute=self.brute,
                                 brute_probs=self.probs, brute_mask=self.flagged,
                                 incidents=self.incidents, mask_ips=mask_ips)

    def test_masked_report_has_no_real_addresses(self):
        text = self.report(0, mask_ips=True)
        self.assertNotIn("185.12.34.56", text)
        self.assertNotIn(SERVER, text)
        self.assertIn("внешний 185.12.x.x:50000 → сервер:22, TCP (входящий на сервер)", text)
        self.assertIn("BruteForce (supervised): атака (вероятность 0.990", text)
        self.assertIn("IsolationForest: норма", text)
        self.assertIn("потоков с ним в файле: 3", text)
        self.assertIn("Входит в инцидент", text)
        self.assertIn("a: 10.00", text)

    def test_unmasked_report_shows_addresses(self):
        text = self.report(5, mask_ips=False)
        self.assertIn("203.0.113.5:50005", text)
        self.assertIn("IsolationForest: аномалия", text)
        self.assertIn("В инцидент не входит", text)

    def test_report_without_meta(self):
        self.result.df = self.result.df.drop(columns=["Src IP", "Dst IP", "Timestamp"])
        text = build_flow_report(self.result, 1, self.means, self.stds)
        self.assertNotIn("Контекст адреса", text)
        self.assertIn("Длительность: 2.000 с", text)

    def test_candidates_order(self):
        # сначала supervised по вероятности, затем IF по score, без повторов
        self.assertEqual(flagged_candidates(self.result, self.probs, self.flagged), [0, 1, 2, 5])
        self.assertEqual(flagged_candidates(self.result), [5, 1])

    def test_first_message(self):
        message = first_message("ФАКТЫ", "Почему?")
        self.assertEqual(message["role"], "user")
        self.assertIn("ФАКТЫ", message["content"])
        self.assertTrue(message["content"].endswith("Вопрос: Почему?"))
