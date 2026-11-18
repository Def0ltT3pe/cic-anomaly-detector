import io
import json
import os
import time
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

from modules.alerts import parse_trusted
from modules.monitor import (
    MAX_SEPARATE_MESSAGES,
    RETAIN_DAYS,
    Server,
    attacked_ports,
    daily_counts,
    format_summary,
    hourly_counts,
    last_download,
    load_incidents,
    load_servers,
    load_state,
    mark_backlog,
    merge_flagged,
    merge_hourly,
    run,
    server_incidents,
    server_traffic,
    state_dir,
    top_attackers,
    traffic_summary,
)
from modules.notify import NotifyError, send_telegram, telegram_settings
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


class MonitorCase(unittest.TestCase):
    """Помощники: папка сервера, захваты-заглушки, прогон run()."""

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


class MonitorTests(MonitorCase):
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

    def test_hourly_totals_and_share(self):
        rows = flagged(("203.0.113.5", 22, "2026-11-01 10:05", 0.9))
        all_flows = pd.DataFrame({"Timestamp": pd.to_datetime(
            ["2026-11-01 10:01", "2026-11-01 10:30", "2026-11-01 10:50", "2026-11-01 11:10"])})
        self.results["capture_1.pcapng"] = rows
        path = self.capture("capture_1.pcapng", rows)
        analyze = lambda p, server: (rows, SERVER_IP, hourly_counts(all_flows))
        run([self.server], analyze, None, directory=self.state, log=lambda m: None)
        # Файл скачан заново: его часы заменяются, а не удваиваются
        with open(path, "ab") as handle:
            handle.write(b"x")
        run([self.server], analyze, None, directory=self.state, log=lambda m: None)
        traffic = server_traffic(self.state, "serverA")
        self.assertEqual(list(traffic["Потоков"]), [3, 1])
        self.assertEqual(list(traffic["Тревог"]), [1, 0])
        summary = traffic_summary(traffic, "2026-11-01 00:00")
        self.assertEqual((summary["flows"], summary["alerts"]), (4, 1))
        self.assertAlmostEqual(summary["share"], 0.25)

    def test_old_analyzer_without_hourly_still_works(self):
        self.capture("capture_1.pcapng", flagged(("203.0.113.5", 22, "2026-11-01 10:05", 0.9)))
        self.assertEqual(self.run_once().files, 1)
        self.assertEqual(list(server_traffic(self.state, "serverA")["Потоков"]), [0])

    def test_top_attackers(self):
        incidents = pd.DataFrame({
            "peer": ["203.0.113.5", "203.0.113.5", "198.51.100.7"],
            "window_start": pd.to_datetime(["2026-11-01 10:00", "2026-11-01 11:00", "2026-11-01 10:00"]),
            "flows": [3, 4, 9], "services": ["22", "22, 443", "443"],
            "last_seen": pd.to_datetime(["2026-11-01 10:50", "2026-11-01 11:20", "2026-11-01 10:40"]),
        })
        top = top_attackers(incidents)
        self.assertEqual(list(top["Адрес"]), ["203.0.113.5", "198.51.100.7"])
        self.assertEqual(top.iloc[0]["Порты"], "22, 443")
        self.assertEqual(top.iloc[0]["Потоков"], 7)

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


class MonitorEdgeCaseTests(MonitorCase):
    """Ветки, которые нашли выжившие мутанты (cosmic-ray)."""

    def test_missing_folder_does_not_stop_other_servers(self):
        self.capture("capture_1.pcapng", flagged(("203.0.113.5", 22, "2026-11-01 10:05", 0.9)))
        missing = Server("serverZ", os.path.join(self.tmp, "absent"))
        summary = run([missing, self.server], self.analyze, self.sent.append, directory=self.state,
                      log=lambda message: None)
        self.assertEqual((summary.files, summary.flagged, summary.new_incidents), (1, 1, 1))
        self.assertEqual(len(summary.errors), 1)

    def test_broken_file_does_not_stop_later_files(self):
        self.capture("capture_0.pcapng", None, age_days=0.01)  # старше — разбирается первым
        self.capture("capture_1.pcapng", flagged(("203.0.113.5", 22, "2026-11-01 10:05", 0.9)))
        summary = self.run_once()
        self.assertEqual((summary.files, summary.incidents), (1, 1))

    def test_exactly_max_incidents_are_separate_messages(self):
        peers = [(f"203.0.113.{n}", 22, "2026-11-01 10:05", 0.9) for n in range(MAX_SEPARATE_MESSAGES)]
        self.capture("capture_1.pcapng", flagged(*peers))
        self.run_once()
        self.assertEqual(len(self.sent), MAX_SEPARATE_MESSAGES)

    def test_summary_lists_servers_and_top_addresses(self):
        rows = pd.DataFrame({
            "server": ["serverB", "serverA", "serverA", "serverA"],
            "peer": ["198.51.100.1", "203.0.113.1", "203.0.113.2", "203.0.113.1"],
            "flows": [3, 4, 9, 2],
        })
        text = format_summary(rows)
        self.assertTrue(text.startswith("Новых инцидентов: 4"))
        self.assertLess(text.index("serverA:"), text.index("serverB:"))
        self.assertIn("serverA: 3 инцидентов, 2 адресов", text)
        # Сначала адрес с большим числом потоков: 203.0.113.2 (9) раньше 203.0.113.1 (6)
        self.assertLess(text.index("203.0.113.2: 9"), text.index("203.0.113.1: 6"))

    def test_backfill_days_zero_and_disabled(self):
        self.capture("capture_old.pcapng", flagged(("203.0.113.5", 22, "2026-10-01 10:05", 0.9)), age_days=0.5)
        self.assertEqual(self.run_once(backfill_days=0).files, 0)  # 0 — проверять только новое
        other = os.path.join(self.tmp, "state2")
        summary = run([self.server], self.analyze, None, directory=other, backfill_days=-1, log=lambda m: None)
        self.assertEqual(summary.files, 1)  # отрицательное — разобрать весь архив

    def test_mark_backlog_counts_files(self):
        for n in range(3):
            self.capture(f"capture_{n}.pcapng", flagged(), age_days=3)
        state = load_state(self.state)
        self.assertEqual(mark_backlog(self.server, state, 1, time.time()), 3)
        self.assertEqual(len(state["processed"]), 3)

    def test_old_notified_keys_are_pruned(self):
        os.makedirs(self.state)
        old_key = "serverA|203.0.113.9|2026-01-01T10:00"
        recent = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=RETAIN_DAYS)
        recent_key = f"serverA|203.0.113.8|{recent:%Y-%m-%dT%H:%M}"
        with open(os.path.join(self.state, "state.json"), "w", encoding="utf-8") as handle:
            json.dump({"processed": {"serverA/x": 1}, "notified": [old_key, recent_key]}, handle)
        self.run_once()
        self.assertEqual(load_state(self.state)["notified"], [recent_key])

    def test_counters_start_from_zero(self):
        summary = self.run_once()
        self.assertEqual((summary.files, summary.flagged, summary.incidents, summary.new_incidents,
                          summary.notified), (0, 0, 0, 0, 0))

    def test_incidents_file_starts_with_server_column(self):
        self.capture("capture_1.pcapng", flagged(("203.0.113.5", 22, "2026-11-01 10:05", 0.9)))
        self.run_once()
        self.assertEqual(list(load_incidents(self.state)[0].columns[:2]), ["server", "peer"])

    def test_no_local_ip_means_no_incidents(self):
        rows = flagged(("203.0.113.5", 22, "2026-11-01 10:05", 0.9))
        self.assertEqual(len(server_incidents(self.server, rows, None)), 0)
        self.assertEqual(len(server_incidents(self.server, rows, SERVER_IP)), 1)

    def test_merge_flagged_keeps_highest_probability_and_cutoff(self):
        low = flagged(("203.0.113.5", 22, "2026-11-01 10:05", 0.4))
        high = flagged(("203.0.113.5", 22, "2026-11-01 10:05", 0.9))
        merged = merge_flagged(low, high)
        self.assertEqual(list(merged["proba"]), [0.9])
        self.assertEqual(list(merged.index), [0])
        edge = flagged(("203.0.113.6", 22, "2026-10-18 10:05", 0.5))  # ровно 14 дней до самого свежего
        self.assertEqual(len(merge_flagged(edge, high, retain_days=14)), 2)

    def test_merge_hourly_replaces_only_that_file(self):
        hours = lambda *pairs: pd.DataFrame({"hour": pd.to_datetime([h for h, _ in pairs]),  # noqa: E731
                                            "flows": [f for _, f in pairs]})
        history = merge_hourly(pd.DataFrame(columns=["file", "hour", "flows"]),
                               {"a": hours(("2026-11-01 10:00", 5)), "b": hours(("2026-11-01 10:00", 7))})
        updated = merge_hourly(history, {"a": hours(("2026-11-01 10:00", 6))})
        self.assertEqual(sorted(zip(updated["file"], updated["flows"])), [("a", 6), ("b", 7)])
        self.assertEqual(list(updated.index), [0, 1])
        edge = merge_hourly(updated, {"c": hours(("2026-10-18 10:00", 1))}, retain_days=14)
        self.assertIn("c", set(edge["file"]))

    def test_server_without_history(self):
        # Сервер уже в списке, но monitor.py его ещё не проверял: вкладка не должна падать
        traffic = server_traffic(self.state, "serverNew")
        self.assertEqual(len(traffic), 0)
        summary = traffic_summary(traffic, pd.Timestamp("2026-11-01"))
        self.assertEqual((summary["flows"], summary["alerts"], summary["share"]), (0, 0, None))

    def test_traffic_summary_includes_start_hour(self):
        traffic = pd.DataFrame({"Час": pd.to_datetime(["2026-11-01 09:00", "2026-11-01 10:00"]),
                                "Потоков": [5, 3], "Тревог": [1, 1]})
        summary = traffic_summary(traffic, "2026-11-01 10:00")
        self.assertEqual((summary["flows"], summary["alerts"]), (3, 1))
        self.assertIsNone(traffic_summary(traffic, "2026-11-02")["share"])

    def test_attacked_ports_by_direction(self):
        rows = flagged(("203.0.113.5", 22, "2026-11-01 10:05", 0.9), ("203.0.113.6", 22, "2026-11-01 10:06", 0.9))
        outbound = pd.DataFrame([{"Src IP": SERVER_IP, "Src Port": 443, "Dst IP": "203.0.113.7",
                                  "Dst Port": 51000, "Timestamp": pd.Timestamp("2026-11-01 10:07"), "proba": 0.8}])
        os.makedirs(os.path.join(self.state, "serverA"))
        pd.concat([rows, outbound], ignore_index=True).to_parquet(
            os.path.join(self.state, "serverA", "flagged.parquet"), index=False)
        ports = attacked_ports(self.state, "serverA", SERVER_IP)
        self.assertEqual(list(zip(ports["Порт"], ports["Тревог"])), [("22", 2), ("443", 1)])
        self.assertEqual(len(attacked_ports(self.state, "serverA", None)), 0)
        self.assertEqual(len(attacked_ports(self.state, "absent", SERVER_IP)), 0)

    def test_last_download(self):
        self.assertIsNone(last_download(self.local))
        path = self.capture("capture_1.pcapng", flagged(), age_days=1)
        self.assertAlmostEqual(last_download(self.local).timestamp(), os.path.getmtime(path), places=3)

    def test_state_dir_and_server_config(self):
        with patch.dict(os.environ, {"MONITOR_DIR": "relative/mon"}):
            self.assertTrue(state_dir().endswith(os.path.join("relative", "mon")))
            self.assertTrue(os.path.isabs(state_dir()))
        with patch.dict(os.environ, {"MONITOR_DIR": self.state}):
            self.assertEqual(state_dir(), os.path.abspath(self.state))
        path = os.path.join(self.tmp, "pull.json")
        for entry in ({"name": "serverA"}, {"local": self.local}):
            with self.subTest(entry=entry):
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump([entry], handle)
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

    def sent_body(self, text):
        seen = {}

        def opener(request, timeout):
            seen.update(json.loads(request.data))
            return FakeResponse(b'{"ok": true}')

        send_telegram(text, "123:SECRET", "42", opener=opener)
        return seen

    def test_length_limit_boundary(self):
        self.assertEqual(self.sent_body("x" * 4000)["text"], "x" * 4000)  # ровно лимит — как есть
        cut = self.sent_body("x" * 4001)["text"]
        self.assertEqual(len(cut), 4000)
        self.assertTrue(cut.endswith("x…"))

    def test_links_are_not_previewed(self):
        # Адрес из захвата не должен превращаться в превью чужой страницы
        self.assertIs(self.sent_body("hi")["disable_web_page_preview"], True)

    def test_http_error_details(self):
        def failing(body):
            def opener(request, timeout):
                raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(body))
            return opener

        with self.assertRaises(NotifyError) as caught:
            send_telegram("hi", "123:SECRET", "42", opener=failing(b'{"ok": false, "description": "Unauthorized"}'))
        self.assertEqual(str(caught.exception), "Telegram: HTTP 401: Unauthorized")
        with self.assertRaises(NotifyError) as caught:
            send_telegram("hi", "123:SECRET", "42", opener=failing(b"<html>bad gateway</html>"))
        self.assertEqual(str(caught.exception), "Telegram: HTTP 401: <html>bad gateway</html>")
        with self.assertRaises(NotifyError) as caught:
            send_telegram("hi", "123:SECRET", "42", opener=failing(b'{"description": ""}'))
        self.assertEqual(str(caught.exception), 'Telegram: HTTP 401: {"description": ""}')

    def test_settings_from_environment(self):
        cases = [
            ({"TELEGRAM_BOT_TOKEN": " 123:SECRET ", "TELEGRAM_CHAT_ID": " 42 "}, ("123:SECRET", "42")),
            ({"TELEGRAM_BOT_TOKEN": "123:SECRET", "TELEGRAM_CHAT_ID": ""}, None),
            ({"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": "42"}, None),
        ]
        for env, expected in cases:
            with self.subTest(env=env), patch.dict(os.environ, env):
                self.assertEqual(telegram_settings(), expected)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(telegram_settings())


if __name__ == "__main__":
    unittest.main()
