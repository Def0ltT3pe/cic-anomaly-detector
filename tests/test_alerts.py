import unittest

import numpy as np
import pandas as pd

from modules.alerts import (
    aggregate_alerts,
    detect_local_ip,
    parse_timestamps,
    parse_trusted,
    summarize,
    trusted_mask,
)

SERVER = "10.0.0.1"


def flows(rows):
    """rows: (src, sport, dst, dport, time)"""
    frame = pd.DataFrame(rows, columns=["Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"])
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"])
    return frame


class AggregateAlertsTests(unittest.TestCase):
    def setUp(self):
        rows = [
            # бот: одна попытка на SSH — инцидент сразу
            ("192.0.2.10", 40000, SERVER, 22, "2026-10-29 10:05"),
            # другой адрес: три подозрительных HTTPS за час — инцидент по счёту
            ("192.0.2.20", 40001, SERVER, 443, "2026-10-29 10:10"),
            ("192.0.2.20", 40002, SERVER, 443, "2026-10-29 10:20"),
            ("192.0.2.20", 40003, SERVER, 443, "2026-10-29 10:50"),
            # единичное ложное на HTTPS — не инцидент
            ("192.0.2.30", 40004, SERVER, 443, "2026-10-29 10:15"),
            # исходящее соединение сервера: внешний адрес — Dst IP, сервис — Src Port
            (SERVER, 22, "192.0.2.40", 50000, "2026-10-29 11:00"),
            # фон, чтобы сервер был самым частым адресом
            (SERVER, 51000, "198.51.100.1", 443, "2026-10-29 10:00"),
            (SERVER, 51001, "198.51.100.2", 443, "2026-10-29 10:01"),
        ]
        self.df = flows(rows)
        self.flagged = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
        self.proba = np.array([0.9, 0.6, 0.7, 0.8, 0.95, 0.5, 0, 0])

    def test_rules(self):
        incidents = aggregate_alerts(self.df, self.flagged, self.proba)
        by_peer = incidents.set_index("peer")
        self.assertEqual(set(by_peer.index), {"192.0.2.10", "192.0.2.20", "192.0.2.40"})
        self.assertEqual(by_peer.loc["192.0.2.10", "reason"], "порт 22")
        self.assertEqual(by_peer.loc["192.0.2.20", "reason"], "≥3 потоков")
        self.assertEqual(by_peer.loc["192.0.2.20", "flows"], 3)
        self.assertAlmostEqual(by_peer.loc["192.0.2.20", "max_proba"], 0.8)
        self.assertEqual(by_peer.loc["192.0.2.20", "rows"], [1, 2, 3])
        self.assertNotIn("192.0.2.30", by_peer.index)

    def test_window_splits_groups(self):
        df = flows([
            ("192.0.2.20", 1, SERVER, 443, "2026-10-29 10:50"),
            ("192.0.2.20", 2, SERVER, 443, "2026-10-29 10:55"),
            ("192.0.2.20", 3, SERVER, 443, "2026-10-29 11:05"),
            (SERVER, 4, "198.51.100.1", 443, "2026-10-29 10:00"),
            (SERVER, 5, "198.51.100.2", 443, "2026-10-29 10:00"),
            (SERVER, 6, "198.51.100.3", 443, "2026-10-29 10:00"),
        ])
        flagged = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
        self.assertEqual(len(aggregate_alerts(df, flagged, window="1h")), 0)  # 2 + 1 в разных часах
        self.assertEqual(len(aggregate_alerts(df, flagged, window="1D")), 1)

    def test_options(self):
        none_immediate = aggregate_alerts(self.df, self.flagged, immediate_ports=())
        self.assertEqual(set(none_immediate["peer"]), {"192.0.2.20"})
        single = aggregate_alerts(self.df, self.flagged, min_flows=1, immediate_ports=())
        self.assertEqual(len(single), 4)

    def test_local_ip_detection_and_override(self):
        self.assertEqual(detect_local_ip(self.df), SERVER)
        incidents = aggregate_alerts(self.df, self.flagged, local_ip="192.0.2.20")
        # если «сервер» — 192.0.2.20, его потоки группируются по собеседнику SERVER
        self.assertIn(SERVER, set(incidents["peer"]))

    def test_trusted_addresses_and_networks(self):
        trusted = aggregate_alerts(self.df, self.flagged, trusted=["192.0.2.10", "192.0.2.40/30"])
        self.assertEqual(set(trusted["peer"]), {"192.0.2.20"})  # .10 и .40 — доверенные
        self.assertEqual(len(parse_trusted("192.0.2.10, 198.51.100.0/24;2001:db8::/32")), 3)
        with self.assertRaises(ValueError):
            parse_trusted("not-an-ip")
        mask = trusted_mask(["192.0.2.10", "2001:db8::1", "garbage"], parse_trusted("2001:db8::/32"))
        self.assertEqual(mask.tolist(), [False, True, False])

    def test_without_meta_or_flags(self):
        self.assertIsNone(aggregate_alerts(self.df.drop(columns=["Src IP"]), self.flagged))
        empty = aggregate_alerts(self.df, np.zeros(len(self.df), dtype=bool))
        self.assertEqual(len(empty), 0)
        self.assertIn("rows", empty.columns)

    def test_summary(self):
        incidents = aggregate_alerts(self.df, self.flagged)
        text = summarize(incidents, self.flagged)
        self.assertIn("подозрительных потоков 6 -> инцидентов 3", text)
        self.assertIn("нет адресов", summarize(None, self.flagged))


class ParseTimestampsTests(unittest.TestCase):
    def test_cic_dates_are_day_first(self):
        # CIC-IDS2017: «03/07/2017» — 3 июля (среда), а не 7 марта
        stamps = parse_timestamps(["03/07/2017 08:55:58", "7/7/2017 3:30"])
        self.assertEqual(list(stamps), [pd.Timestamp("2017-07-03 08:55:58"), pd.Timestamp("2017-07-07 03:30")])

    def test_iso_and_datetime_values_are_not_swapped(self):
        stamps = parse_timestamps(pd.Series(
            ["2026-01-11 10:05:00", "2026-11-01T10:05:00Z", pd.Timestamp("2026-01-02 03:04")], dtype=object))
        self.assertEqual(list(stamps), [pd.Timestamp("2026-01-11 10:05"), pd.Timestamp("2026-11-01 10:05"),
                                        pd.Timestamp("2026-01-02 03:04")])
        datetimes = pd.Series(pd.to_datetime(["2026-01-11"]))
        self.assertEqual(parse_timestamps(datetimes).iloc[0], pd.Timestamp("2026-01-11"))

    def test_garbage_becomes_nat(self):
        self.assertTrue(parse_timestamps(["not a date", None, "![x](https://example.com/t.png)"]).isna().all())

    def test_incidents_from_cic_csv_dates(self):
        frame = pd.DataFrame({
            "Src IP": ["203.0.113.5"] * 3, "Src Port": [40001, 40002, 40003],
            "Dst IP": [SERVER] * 3, "Dst Port": [443] * 3,
            "Timestamp": ["03/07/2017 08:05:00", "03/07/2017 08:25:00", "03/07/2017 08:45:00"],
        })
        incidents = aggregate_alerts(frame, np.ones(3, dtype=bool), local_ip=SERVER)
        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents.iloc[0]["window_start"], pd.Timestamp("2017-07-03 08:00"))


if __name__ == "__main__":
    unittest.main()
