import unittest

import pandas as pd

from label_from_authlog import AuthLogError, label_flows, parse_authlog

# Адреса из диапазонов для документации (RFC 5737 / 3849)
LOG = """\
Oct 28 19:01:56 host sshd[100]: Invalid user update from 192.0.2.10 port 46438
Oct 28 19:01:56 host sshd[100]: pam_unix(sshd:auth): check pass; user unknown
Oct 28 19:01:58 host sshd[100]: Failed password for invalid user update from 192.0.2.10 port 46438 ssh2
Oct 28 19:02:00 host sshd[100]: Connection closed by invalid user update 192.0.2.10 port 46438 [preauth]
Oct 28 19:06:40 host sshd[200]: Accepted password for root from 198.51.100.7 port 1803 ssh2
Oct 28 19:06:40 host sshd[200]: pam_unix(sshd:session): session opened for user root(uid=0) by root(uid=0)
Oct 28 19:11:33 host sshd[200]: Disconnected from user root 198.51.100.7 port 1803
Oct 28 19:17:38 host sshd[300]: error: kex_exchange_identification: read: Connection reset by peer
Oct 28 19:17:38 host sshd[300]: Connection reset by 203.0.113.5 port 21987
2026-10-28T19:20:03+0000 host sshd[400]: Invalid user admin from 2001:db8::1 port 40158
2026-10-28T19:20:05+0000 host sshd[400]: Failed password for invalid user admin from 2001:db8::1 port 40158 ssh2
Oct 28 19:21:00 host CRON[1]: unrelated line
"""


def flows(rows):
    columns = ["Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"]
    frame = pd.DataFrame(rows, columns=columns)
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"])
    frame["Flow Duration"] = 1
    return frame


class ParseAuthLogTests(unittest.TestCase):
    def test_sessions_and_kinds(self):
        parsed = parse_authlog(LOG.splitlines(), year=2026)
        kinds = {s.pid: (s.ip, s.port, s.kind) for s in parsed.sessions}
        self.assertEqual(kinds["100"], ("192.0.2.10", 46438, "failed"))
        self.assertEqual(kinds["200"], ("198.51.100.7", 1803, "accepted"))
        self.assertEqual(kinds["300"], ("203.0.113.5", 21987, "probe"))
        self.assertEqual(kinds["400"], ("2001:db8::1", 40158, "failed"))
        self.assertEqual(parsed.skipped, 1)  # строка CRON

    def test_session_spans_until_last_line(self):
        session = next(s for s in parse_authlog(LOG.splitlines(), 2026).sessions if s.pid == "200")
        self.assertEqual(session.start, pd.Timestamp("2026-10-28 19:06:40"))
        self.assertEqual(session.end, pd.Timestamp("2026-10-28 19:11:33"))

    def test_timezone_is_converted_to_utc(self):
        line = ["Oct 28 22:01:56 host sshd[1]: Invalid user x from 192.0.2.1 port 5"]
        session = parse_authlog(line, 2026, tz="Europe/Moscow").sessions[0]
        self.assertEqual(session.start, pd.Timestamp("2026-10-28 19:01:56"))


class LabelFlowsTests(unittest.TestCase):
    def setUp(self):
        self.sessions = parse_authlog(LOG.splitlines(), year=2026).sessions
        self.frame = flows([
            ("192.0.2.10", 46438, "10.0.0.1", 22, "2026-10-28 19:01:55"),   # брутфорс
            ("198.51.100.7", 1803, "10.0.0.1", 22, "2026-10-28 19:06:39"),  # свой вход
            ("198.51.100.7", 1803, "10.0.0.1", 22, "2026-10-28 19:09:00"),  # тот же, после таймаута
            ("203.0.113.5", 21987, "10.0.0.1", 22, "2026-10-28 19:17:37"),  # сканер
            ("192.0.2.99", 5555, "10.0.0.1", 22, "2026-10-28 19:30:00"),    # нет в журнале
            ("10.0.0.1", 22, "192.0.2.10", 46438, "2026-10-28 19:01:55"),   # исходящий с 22 — не входящий
            ("10.0.0.1", 50000, "93.184.216.34", 443, "2026-10-28 19:00:00"),
        ])

    def test_labels_and_unlogged_drop(self):
        labeled, summary = label_flows(self.frame, self.sessions)
        self.assertEqual(
            labeled["Label"].tolist(),
            ["SSH-Bruteforce", "Benign", "Benign", "SSH-Probe", "Benign", "Benign"],
        )
        self.assertEqual(summary["unlogged_flows"], 1)
        self.assertEqual(summary["sessions_matched"], 3)

    def test_unlogged_modes(self):
        kept, _ = label_flows(self.frame, self.sessions, unlogged="keep")
        self.assertIn("SSH-Unlogged", kept["Label"].tolist())
        benign, _ = label_flows(self.frame, self.sessions, unlogged="benign")
        self.assertEqual(len(benign), len(self.frame))
        self.assertNotIn("SSH-Unlogged", benign["Label"].tolist())

    def test_time_window_prevents_port_reuse_mismatch(self):
        late = flows([("192.0.2.10", 46438, "10.0.0.1", 22, "2026-10-28 23:00:00")])
        labeled, _ = label_flows(late, self.sessions, unlogged="keep")
        self.assertEqual(labeled["Label"].tolist(), ["SSH-Unlogged"])

    def test_requires_meta_columns(self):
        with self.assertRaises(AuthLogError):
            label_flows(self.frame.drop(columns=["Src IP"]), self.sessions)


if __name__ == "__main__":
    unittest.main()
