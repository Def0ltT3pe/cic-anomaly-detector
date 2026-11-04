import io
import json
import os
import time
import unittest
import urllib.error

import pandas as pd

from modules.alerts import parse_trusted
from modules.monitor import (
    MAX_SEPARATE_MESSAGES,
    Server,
    daily_counts,
    load_incidents,
    load_servers,
    load_state,
    merge_flagged,
    run,
)
from modules.notify import NotifyError, send_telegram
from tests.helpers import make_temp_dir, remove_temp_dir

SERVER_IP = "10.0.0.1"


def flagged(*flows):
    """(внешний адрес, порт сервера, время, вероятность) -> тревожные потоки."""
    rows = [
        {"Src IP": peer, "Src Port": 40000 + n, "Dst IP": SERVER_IP, "Dst Port": port,
         "Timestamp": pd.Timestamp(stamp), "proba": proba}
        for n, (peer, port, stamp, proba) in enumerate(flows)
    ]
    return pd.DataFrame(rows, columns=["Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp", "proba"])


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = make_temp_dir()
        self.local = os.path.join(self.tmp, "serverA")
        self.state = os.path.join(self.tmp, "state")
        os.makedirs(os.path.join(self.local, "session1"))
        self.server = Server("serverA", self.local)
        self.results = {}
        self.sent = []

    def tearDown(self):
        remove_temp_dir(self.tmp)

    def capture(self, name, rows, age_days=0.0):
        path = os.path.join(self.local, "session1", name)
        with open(path, "wb") as handle:
            handle.write(b"pcap")
        stamp = time.time() - age_days * 86400
        os.utime(path, (stamp, stamp))
        self.results[name] = rows
        return path

    def analyze(self, path, server):
        rows = self.results[os.path.basename(path)]
        if rows is None:
            raise ValueError("битый pcap")
        return rows, SERVER_IP

    def run_once(self, notify=True, **kwargs):
        return run([self.server], self.analyze, self.sent.append if notify else None,
                   directory=self.state, log=lambda message: None, **kwargs)

    def test_incident_notified_once(self):
        self.capture("capture_1.pcapng", flagged(("203.0.113.5", 22, "2026-11-01 10:05", 0.9)))
        summary = self.run_once()
        self.assertEqual((summary.files, summary.new_incidents, summary.notified), (1, 1, 1))
        self.assertIn("203.0.113.5", self.sent[0])
        self.assertIn("serverA", self.sent[0])

        summary = self.run_once()  # ничего нового
        self.assertEqual((summary.files, summary.new_incidents), (0, 0))
        self.assertEqual(len(self.sent), 1)
        self.assertTrue(load_incidents(self.state)[0]["notified"].all())

    def test_incident_builds_up_across_files(self):
        # «≥3 потоков за час» складывается из двух файлов одного часа
        self.capture("capture_1.pcapng", flagged(("198.51.100.7", 443, "2026-11-01 10:05", 0.8),
                                                 ("198.51.100.7", 443, "2026-11-01 10:10", 0.8)))
        self.assertEqual(self.run_once().incidents, 0)
        self.capture("capture_2.pcapng", flagged(("198.51.100.7", 443, "2026-11-01 10:40", 0.7)))
        summary = self.run_once()
        self.assertEqual((summary.incidents, summary.new_incidents), (1, 1))
        incident = load_incidents(self.state)[0].iloc[0]
        self.assertEqual(incident["flows"], 3)

    def test_redownloaded_file_does_not_duplicate_flows(self):
        rows = flagged(("198.51.100.7", 443, "2026-11-01 10:05", 0.8))
        path = self.capture("capture_1.pcapng", rows)
        self.run_once()
        with open(path, "ab") as handle:  # pull_captures докачал файл: размер другой
            handle.write(b"more")
        self.results["capture_1.pcapng"] = pd.concat(
            [rows, flagged(("198.51.100.9", 443, "2026-11-01 10:06", 0.6))], ignore_index=True
        )
        self.assertEqual(self.run_once().files, 1)
        stored = pd.read_parquet(os.path.join(self.state, "serverA", "flagged.parquet"))
        self.assertEqual(len(stored), 2)

    def test_failed_notification_is_retried(self):
        self.capture("capture_1.pcapng", flagged(("203.0.113.5", 22, "2026-11-01 10:05", 0.9)))

        def broken(text):
            raise NotifyError("нет сети")

        summary = run([self.server], self.analyze, broken, directory=self.state, log=lambda m: None)
        self.assertEqual(summary.notified, 0)
        self.assertTrue(summary.errors)
        self.assertEqual(self.run_once().notified, 1)
        self.assertEqual(len(self.sent), 1)

    def test_without_notifier_incidents_wait(self):
        self.capture("capture_1.pcapng", flagged(("203.0.113.5", 22, "2026-11-01 10:05", 0.9)))
        self.run_once(notify=False)
        self.assertFalse(load_incidents(self.state)[0]["notified"].any())
        self.assertEqual(self.run_once().notified, 1)

    def test_many_incidents_become_one_summary(self):
        peers = [(f"203.0.113.{n}", 22, "2026-11-01 10:05", 0.9) for n in range(MAX_SEPARATE_MESSAGES + 1)]
        self.capture("capture_1.pcapng", flagged(*peers))
        summary = self.run_once()
        self.assertEqual((summary.new_incidents, summary.notified), (len(peers), len(peers)))
        self.assertEqual(len(self.sent), 1)
        self.assertIn(f"Новых инцидентов: {len(peers)}", self.sent[0])

    def test_first_run_skips_old_archive(self):
        self.capture("capture_old.pcapng", flagged(("203.0.113.5", 22, "2026-10-01 10:05", 0.9)), age_days=5)
        self.capture("capture_new.pcapng", flagged(("203.0.113.6", 22, "2026-11-01 10:05", 0.9)))
        summary = self.run_once(backfill_days=1)
        self.assertEqual(summary.files, 1)
        self.assertEqual(len(load_state(self.state)["processed"]), 2)
        self.assertNotIn("203.0.113.5", " ".join(self.sent))

    def test_trusted_and_broken_files(self):
        self.server = Server("serverA", self.local, trusted=tuple(parse_trusted("203.0.113.0/24")))
        self.capture("capture_1.pcapng", flagged(("203.0.113.5", 22, "2026-11-01 10:05", 0.9)))
        self.capture("capture_2.pcapng", None)  # analyze упадёт на этом файле
        summary = self.run_once()
        self.assertEqual(summary.new_incidents, 0)
        self.assertEqual(summary.files, 1)
        self.assertEqual(len(summary.errors), 1)
        # Битый файл не помечен проверенным — повторится в следующий прогон
        self.assertEqual(len(load_state(self.state)["processed"]), 1)

    def test_daily_counts_fill_empty_days(self):
        incidents = pd.DataFrame({"window_start": pd.to_datetime(
            ["2026-11-03 10:00", "2026-11-03 18:00", "2026-11-01 02:00", "2026-10-01 02:00"])})
        counts = daily_counts(incidents, days=3, today="2026-11-03 23:00")
        self.assertEqual(list(counts["Инцидентов"]), [1, 0, 2])

    def test_merge_flagged_drops_old_rows(self):
        history = flagged(("203.0.113.5", 22, "2026-10-01 10:05", 0.9))
        new = flagged(("203.0.113.6", 22, "2026-11-01 10:05", 0.9))
        self.assertEqual(list(merge_flagged(history, new, retain_days=14)["Src IP"]), ["203.0.113.6"])

    def test_load_servers_reads_pull_config(self):
        path = os.path.join(self.tmp, "pull_captures.json")
        with open(path, "w", encoding="utf-8-sig") as handle:
            json.dump([{"name": "serverA", "target": "root@x", "remote": "/r", "local": self.local,
                        "trusted_ips": "198.51.100.0/24", "model": "models"}], handle)
        server = load_servers(path, trusted_env="203.0.113.5")[0]
        self.assertEqual((server.name, server.model_dir, len(server.trusted)), ("serverA", "models", 2))

    def test_load_servers_rejects_path_in_name(self):
        path = os.path.join(self.tmp, "pull_captures.json")
        for name in (r"..\..\x", "../x", "a/b"):
            with self.subTest(name=name):
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump([{"name": name, "local": self.local}], handle)
                with self.assertRaises(ValueError):
                    load_servers(path)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TelegramTests(unittest.TestCase):
    def test_sends_plain_text(self):
        seen = {}

        def opener(request, timeout):
            seen["url"] = request.full_url
            seen["body"] = json.loads(request.data)
            return FakeResponse(b'{"ok": true}')

        send_telegram("x" * 5000, "123:SECRET", "42", opener=opener)
        self.assertTrue(seen["url"].endswith("/bot123:SECRET/sendMessage"))
        self.assertEqual(seen["body"]["chat_id"], "42")
        self.assertNotIn("parse_mode", seen["body"])
        self.assertLessEqual(len(seen["body"]["text"]), 4096)

    def test_error_hides_token(self):
        def opener(request, timeout):
            raise urllib.error.URLError(f"cannot reach {request.full_url}")

        with self.assertRaises(NotifyError) as caught:
            send_telegram("hi", "123:SECRET", "42", opener=opener)
        self.assertNotIn("SECRET", str(caught.exception))

    def test_api_refusal_is_error(self):
        def opener(request, timeout):
            return FakeResponse(b'{"ok": false, "description": "chat not found"}')

        with self.assertRaises(NotifyError) as caught:
            send_telegram("hi", "123:SECRET", "42", opener=opener)
        self.assertIn("chat not found", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
