import unittest

import numpy as np
import pandas as pd

from modules.explainer import feature_deviations
from modules.overview import (
    ANOMALY,
    ATTACK,
    NORMAL,
    SUSPICIOUS,
    alert_timeline,
    bucket_for,
    flow_table,
    incident_peers,
    llm_verdict,
    model_verdict,
    top_peers,
    top_ports,
)
from tests.test_flow_report import SERVER, make_result


class OverviewTests(unittest.TestCase):
    def setUp(self):
        self.result, self.brute, self.probs, self.flagged = make_result()

    def test_timeline_fills_empty_buckets(self):
        timeline = alert_timeline(self.result.df, {"IF": self.result.mask, "RF": self.flagged})
        # 6 потоков за 5 минут -> шаг 1 минута, 6 точек на каждую модель
        self.assertEqual(len(timeline), 12)
        rf = timeline[timeline["Модель"] == "RF"]["Тревог"].tolist()
        self.assertEqual(rf, [1, 1, 1, 0, 0, 0])
        self.assertEqual(timeline["Тревог"].sum(), 5)

    def test_timeline_without_time(self):
        self.assertIsNone(alert_timeline(self.result.df.drop(columns=["Timestamp"]), {"IF": self.result.mask}))

    def test_bucket_for(self):
        self.assertEqual(bucket_for(pd.Timedelta(hours=1)), "1min")
        self.assertEqual(bucket_for(pd.Timedelta(hours=10)), "10min")
        self.assertEqual(bucket_for(pd.Timedelta(days=2)), "1h")
        self.assertEqual(bucket_for(pd.Timedelta(days=60)), "1D")

    def test_top_peers_and_ports(self):
        peers = top_peers(self.result.df, self.flagged)
        self.assertEqual(peers.iloc[0].tolist(), ["185.12.34.56", 3])
        ports = top_ports(self.result.df, self.flagged)
        self.assertEqual(ports.iloc[0].tolist(), ["22", 3])
        self.assertNotIn(SERVER, set(peers["Адрес"]))
        self.assertIsNone(top_peers(self.result.df.drop(columns=["Src IP"]), self.flagged))

    def test_incident_peers(self):
        self.assertEqual(incident_peers(None), 0)
        self.assertEqual(incident_peers(pd.DataFrame({"peer": ["a", "a", "b"]})), 2)

    def test_flow_table(self):
        table = flow_table(self.result, np.array([1, 5]), self.probs)
        self.assertEqual(table["#"].tolist(), [1, 5])
        self.assertEqual(table["Источник"].iloc[0], "185.12.34.56:50001")
        self.assertEqual(table["Назначение"].iloc[0], f"{SERVER}:22")
        self.assertEqual(table["Протокол"].iloc[0], "TCP")
        self.assertEqual(table["IF"].tolist(), ["аномалия", "аномалия"])
        self.assertAlmostEqual(table["Вероятность атаки"].iloc[0], 0.95)
        self.assertEqual(table["Время"].iloc[0], "30.10 10:01:00")


class VerdictTests(unittest.TestCase):
    def test_model_verdict(self):
        self.assertEqual(model_verdict(False, True, True)[0], ATTACK)
        self.assertEqual(model_verdict(True, True, False)[0], SUSPICIOUS)
        self.assertEqual(model_verdict(True, False, False)[0], ANOMALY)
        self.assertEqual(model_verdict(True, None, False)[0], ANOMALY)
        self.assertEqual(model_verdict(False, False, False)[0], NORMAL)

    def test_llm_verdict(self):
        self.assertEqual(llm_verdict("1. **Вердикт** — атака, уверенность высокая"), ATTACK)
        self.assertEqual(llm_verdict("**Вердикт:** подозрительно"), SUSPICIOUS)
        self.assertEqual(llm_verdict("Вердикт — вероятно ложное срабатывание атаки"), NORMAL)
        self.assertIsNone(llm_verdict("Ответ без раздела"))


class DeviationTests(unittest.TestCase):
    def test_feature_deviations(self):
        record = pd.Series({"a": 10.0, "b": 0.0, "c": 5.0})
        means = pd.Series({"a": 2.0, "b": 0.0, "c": 5.0})
        stds = pd.Series({"a": 2.0, "b": 0.0, "c": 1.0})
        deviations = feature_deviations(record, means, stds, top_n=5)
        self.assertEqual(deviations["feature"].tolist(), ["a", "c"])  # b: std=0 пропущен
        self.assertAlmostEqual(deviations["z"].iloc[0], 4.0)
        self.assertAlmostEqual(deviations["ratio"].iloc[0], 5.0)

    def test_ratio_is_nan_for_zero_mean(self):
        deviations = feature_deviations(pd.Series({"a": 3.0}), pd.Series({"a": 0.0}),
                                        pd.Series({"a": 1.0}))
        self.assertTrue(np.isnan(deviations["ratio"].iloc[0]))
