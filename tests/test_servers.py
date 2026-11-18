import json
import os
import shlex
import subprocess
import unittest
from unittest.mock import patch

from modules import servers
from modules.servers import (
    REMOTE_SCRIPT,
    ServerEntry,
    ServerError,
    add_to_config,
    authorized_keys_command,
    check_all,
    check_key_login,
    collection_command,
    collector,
    collector_command,
    collector_state,
    parse_collection,
    entry_from_config,
    install_collector,
    install_key_with_password,
    load_config,
    remove_from_config,
    start_options,
    validate,
)
from tests.helpers import make_temp_dir, remove_temp_dir

LOCAL = os.path.abspath(os.path.join(os.sep, "captures", "serverC"))


class FakeRunner:
    """Вместо subprocess.run: запоминает команды, отвечает заданным выводом."""

    def __init__(self, stdout=b"", returncode=0):
        self.calls = []
        self.stdout = stdout
        self.returncode = returncode

    def __call__(self, args, input=None, capture_output=True, timeout=None):
        self.calls.append((args, input))
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, b"")


class ValidateTests(unittest.TestCase):
    def test_valid_entry(self):
        entry = validate("serverC", "root", "203.0.113.10", "/var/lib/cic-capture", LOCAL)
        self.assertEqual(entry.target, "root@203.0.113.10")
        self.assertEqual(entry.as_config()["local"], LOCAL)

    def test_rejects_injection(self):
        bad = [
            ("-oProxyCommand=x", "root", "203.0.113.10", "/r", LOCAL),   # имя
            ("s", "-oProxyCommand=calc", "203.0.113.10", "/r", LOCAL),   # пользователь как опция
            ("s", "root", "-oProxyCommand=calc", "/r", LOCAL),           # хост как опция
            ("s", "root", "host;reboot", "/r", LOCAL),
            ("s", "root", "203.0.113.10:2222", "/r", LOCAL),             # порт не поддерживается
            ("s", "root", "203.0.113.10", "/var/$(reboot)", LOCAL),
            ("s", "root", "203.0.113.10", "/var/../etc", LOCAL),
            ("s", "root", "203.0.113.10", "relative", LOCAL),
            ("s", "root", "203.0.113.10", "/r", "relative\\dir"),
        ]
        for values in bad:
            with self.subTest(values=values), self.assertRaises(ServerError):
                validate(*values)

    def test_start_options(self):
        self.assertEqual(start_options("7d", 5, 100),
                         ["--duration", "7d", "--reserve-gb", "5", "--file-mb", "100"])
        for duration in ("7 d", "1w", "; reboot", "-5h"):
            with self.subTest(duration=duration), self.assertRaises(ServerError):
                start_options(duration, 5, 100)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = make_temp_dir()
        self.path = os.path.join(self.tmp, "pull_captures.json")
        with open(self.path, "w", encoding="utf-8-sig") as handle:  # как пишет Блокнот
            json.dump([{"name": "serverA", "target": "root@198.51.100.1",
                        "remote": "/var/lib/cic-capture", "local": "D:\\caps\\serverA"}], handle)

    def tearDown(self):
        remove_temp_dir(self.tmp)

    def test_add_and_remove(self):
        add_to_config(validate("serverC", "root", "203.0.113.10", "/var/lib/cic-capture", LOCAL), self.path)
        names = [e["name"] for e in load_config(self.path)]
        self.assertEqual(names, ["serverA", "serverC"])
        with open(self.path, "rb") as handle:
            self.assertFalse(handle.read().startswith(b"\xef\xbb\xbf"))
        remove_from_config("serverA", self.path)
        self.assertEqual([e["name"] for e in load_config(self.path)], ["serverC"])

    def test_duplicates_rejected(self):
        for values in (("serverA", "root", "203.0.113.10"), ("serverC", "admin", "198.51.100.1")):
            with self.subTest(values=values), self.assertRaises(ServerError):
                add_to_config(validate(*values, "/var/lib/cic-capture", LOCAL), self.path)

    def test_entry_from_config(self):
        entry = entry_from_config(load_config(self.path)[0])
        self.assertEqual((entry.user, entry.host, entry.name), ("root", "198.51.100.1", "serverA"))
        bare = entry_from_config({"name": "x", "target": "root@203.0.113.9", "local": LOCAL})
        self.assertEqual(bare.remote, servers.DEFAULT_REMOTE_DIR)

    def test_duplicate_check_is_exact_not_ordered(self):
        # Имя и адрес «меньше» существующих по алфавиту — не дубль
        add_to_config(validate("alpha", "root", "198.51.100.0", "/var/lib/cic-capture", LOCAL), self.path)
        add_to_config(validate("zulu", "root", "203.0.113.200", "/var/lib/cic-capture", LOCAL), self.path)
        self.assertEqual([e["name"] for e in load_config(self.path)], ["serverA", "alpha", "zulu"])

    def test_remove_only_exact_name(self):
        add_to_config(validate("serverB", "root", "203.0.113.10", "/var/lib/cic-capture", LOCAL), self.path)
        with self.assertRaises(ServerError):
            remove_from_config("serverZ", self.path)
        remove_from_config("serverB", self.path)
        self.assertEqual([e["name"] for e in load_config(self.path)], ["serverA"])

    def test_config_path_from_environment(self):
        with patch.dict(os.environ, {"PULL_CONFIG": self.path}):
            self.assertEqual(servers.config_path(), self.path)
        with patch.dict(os.environ, {"PULL_CONFIG": ""}):
            self.assertEqual(servers.config_path(), servers.DEFAULT_CONFIG)


class SshCommandTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(servers, "_ssh_binary", return_value="ssh")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.root = ServerEntry("serverC", "root", "203.0.113.10", "/var/lib/cic-capture", LOCAL)
        self.admin = ServerEntry("serverC", "deploy", "203.0.113.10", "/var/lib/cic-capture", LOCAL)

    def test_ssh_args_are_safe(self):
        runner = FakeRunner(b"ok")
        collector(self.root, "status", runner=runner)
        args, _ = runner.calls[0]
        self.assertIn("BatchMode=yes", args)
        self.assertIn("StrictHostKeyChecking=accept-new", args)
        # Цель стоит после «--»: она не может стать опцией ssh
        self.assertEqual(args[args.index("--") + 1], "root@203.0.113.10")
        self.assertEqual(shlex.split(args[-1]),
                         [REMOTE_SCRIPT, "status", "--dir", "/var/lib/cic-capture"])

    def test_non_root_uses_sudo(self):
        runner = FakeRunner()
        collector(self.admin, "start", *start_options("12h", 1, 100), runner=runner)
        self.assertTrue(runner.calls[0][0][-1].startswith("sudo -n "))

    def test_unknown_action(self):
        with self.assertRaises(ServerError):
            collector(self.root, "rm", runner=FakeRunner())

    def test_key_login_failure_is_explained(self):
        with self.assertRaises(ServerError) as caught:
            collector(self.root, "status", runner=FakeRunner(b"Permission denied", returncode=255))
        self.assertIn("по SSH", str(caught.exception))

    def test_privileges(self):
        self.assertEqual(check_key_login(self.root, runner=FakeRunner(b"0\n")), "root")
        self.assertEqual(check_key_login(self.admin, runner=FakeRunner(b"1000\nsudo-ok\n")), "sudo")
        with self.assertRaises(ServerError):
            check_key_login(self.admin, runner=FakeRunner(b"1000\n"))

    def test_install_collector_sends_script_with_lf(self):
        runner = FakeRunner()
        install_collector(self.root, runner=runner)
        args, script = runner.calls[0]
        self.assertTrue(script.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r\n", script)
        self.assertIn(REMOTE_SCRIPT, args[-1])

    def test_authorized_keys_command_quotes_key(self):
        key = "ssh-ed25519 AAAAC3Nza user@pc'; rm -rf ~; echo '"
        command = authorized_keys_command(key)
        self.assertIn(shlex.quote(key), command)
        self.assertEqual(command.count(shlex.quote(key)), 2)


class PasswordBootstrapTests(unittest.TestCase):
    """Первый вход по паролю: paramiko подменён, сеть не нужна."""

    def setUp(self):
        import paramiko

        self.paramiko = paramiko
        self.tmp = make_temp_dir()
        patcher = patch.object(servers, "ssh_dir", return_value=self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(remove_temp_dir, self.tmp)
        self.entry = ServerEntry("serverC", "root", "203.0.113.10", "/var/lib/cic-capture", LOCAL)
        self.commands = []
        # Что подменённый клиент увидел
        self.connect_kwargs = {}
        self.loaded_known_hosts = None
        self.password_seen = None

    def fake_client(self, fail=None, exit_status=0, known=False):
        test = self
        key = self.paramiko.RSAKey.generate(1024)

        class Channel:
            def recv_exit_status(self):
                return exit_status

        class Stream:
            channel = Channel()

            def read(self):
                return b"permission denied" if exit_status else b""

        class Client:
            def load_host_keys(self, path):
                test.loaded_known_hosts = path

            def set_missing_host_key_policy(self, policy):
                self.policy = policy

            def connect(self, host, username, password, **kwargs):
                test.connect_kwargs = kwargs
                if fail:
                    raise fail
                test.password_seen = password
                if not known:  # известный хост paramiko сверяет сам, политику не зовёт
                    self.policy.missing_host_key(self, host, key)

            def exec_command(self, command, timeout=None):
                test.commands.append(command)
                return None, Stream(), Stream()

            def get_transport(self):
                class Transport:
                    def get_remote_server_key(self_inner):
                        return key
                return Transport()

            def close(self):
                pass

        return Client, key

    def test_key_installed_and_host_remembered(self):
        client, key = self.fake_client()
        with patch.object(self.paramiko, "SSHClient", client):
            fingerprint = install_key_with_password(self.entry, "s3cret-pass", "ssh-ed25519 AAAA me@pc")
        self.assertTrue(fingerprint.startswith("ssh-rsa SHA256:"))
        self.assertIn(shlex.quote("ssh-ed25519 AAAA me@pc"), self.commands[0])
        with open(os.path.join(self.tmp, "known_hosts"), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), f"203.0.113.10 ssh-rsa {key.get_base64()}\n")
        # Пароль не попадает ни в команды, ни в файлы
        self.assertNotIn("s3cret-pass", " ".join(self.commands))

    def test_wrong_password(self):
        client, _ = self.fake_client(fail=self.paramiko.AuthenticationException())
        with patch.object(self.paramiko, "SSHClient", client), self.assertRaises(ServerError) as caught:
            install_key_with_password(self.entry, "bad", "ssh-ed25519 AAAA me@pc")
        self.assertIn("пароль", str(caught.exception))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "known_hosts")))

    def test_password_login_does_not_use_local_keys(self):
        # Вход по паролю — ровно по паролю: без агента и чужих ключей с диска
        client, _ = self.fake_client()
        with patch.object(self.paramiko, "SSHClient", client):
            install_key_with_password(self.entry, "pass", "ssh-ed25519 AAAA me@pc")
        self.assertIs(self.connect_kwargs["allow_agent"], False)
        self.assertIs(self.connect_kwargs["look_for_keys"], False)

    def test_known_hosts_loaded_and_not_duplicated(self):
        known = os.path.join(self.tmp, "known_hosts")
        with open(known, "w", encoding="utf-8") as handle:
            handle.write("203.0.113.10 ssh-rsa AAAAold\n")
        client, _ = self.fake_client(known=True)
        with patch.object(self.paramiko, "SSHClient", client):
            install_key_with_password(self.entry, "pass", "ssh-ed25519 AAAA me@pc")
        self.assertEqual(self.loaded_known_hosts, known)
        with open(known, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "203.0.113.10 ssh-rsa AAAAold\n")

    def test_changed_host_key_is_refused(self):
        key = self.paramiko.RSAKey.generate(1024)
        client, _ = self.fake_client(fail=self.paramiko.BadHostKeyException("203.0.113.10", key, key))
        with patch.object(self.paramiko, "SSHClient", client), self.assertRaises(ServerError) as caught:
            install_key_with_password(self.entry, "pass", "ssh-ed25519 AAAA me@pc")
        self.assertIn("подмена", str(caught.exception))
        self.assertIn("ssh-keygen -R 203.0.113.10", str(caught.exception))

    def test_unreachable_host(self):
        client, _ = self.fake_client(fail=OSError("timed out"))
        with patch.object(self.paramiko, "SSHClient", client), self.assertRaises(ServerError) as caught:
            install_key_with_password(self.entry, "pass", "ssh-ed25519 AAAA me@pc")
        self.assertIn("Не удалось подключиться", str(caught.exception))

    def test_banner_reset_is_retried(self):
        # sshd (MaxStartups) сбросил первые два соединения до приветствия
        resets = [self.paramiko.SSHException("Error reading SSH protocol banner[WinError 10054]"),
                  ConnectionResetError(10054, "reset")]
        client, key = self.fake_client()

        class Flaky(client):
            def connect(self_inner, host, username, password, **kwargs):
                if resets:
                    raise resets.pop(0)
                super().connect(host, username, password, **kwargs)

        sleeps = []
        with patch.object(self.paramiko, "SSHClient", Flaky), patch.object(servers, "_sleep", sleeps.append):
            install_key_with_password(self.entry, "pass", "ssh-ed25519 AAAA me@pc")
        self.assertEqual(len(sleeps), 2)
        self.assertEqual(len(self.commands), 1)  # ключ записан один раз
        with open(os.path.join(self.tmp, "known_hosts"), encoding="utf-8") as handle:
            self.assertEqual(handle.read().count(key.get_base64()), 1)

    def test_banner_reset_gives_up_with_hint(self):
        client, _ = self.fake_client(fail=self.paramiko.SSHException("Error reading SSH protocol banner"))
        sleeps = []
        with patch.object(self.paramiko, "SSHClient", client), patch.object(servers, "_sleep", sleeps.append), \
                self.assertRaises(ServerError) as caught:
            install_key_with_password(self.entry, "pass", "ssh-ed25519 AAAA me@pc")
        self.assertEqual(len(sleeps), servers.CONNECT_ATTEMPTS - 1)
        self.assertIn("MaxStartups", str(caught.exception))

    def test_key_write_failure(self):
        client, _ = self.fake_client(exit_status=1)
        with patch.object(self.paramiko, "SSHClient", client), self.assertRaises(ServerError) as caught:
            install_key_with_password(self.entry, "pass", "ssh-ed25519 AAAA me@pc")
        self.assertIn("permission denied", str(caught.exception))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "known_hosts")))


class StateTests(unittest.TestCase):
    def test_collector_state(self):
        self.assertEqual(collector_state("Состояние:        идёт запись (dumpcap PID 42)"), "running")
        self.assertEqual(collector_state("Состояние:        запись не идёт"), "stopped")
        self.assertEqual(collector_state("Сессий в /var/lib/cic-capture нет."), "empty")

    def test_parse_collection_full_ring(self):
        output = "\n".join([
            "Состояние:        идёт запись (dumpcap PID 7)",
            "Файлов захвата:   3, журналов SSH: 2",
            "Занято:           1.5G, свободно на диске: 9000 МБ",
            "--- session.json ---",
            '{"started_utc": "2026-11-01T10:00:00Z", "ring_files": 3, "file_mb": 500}',
            "--- capture mtimes ---",
            "1793520000",
            "1793530800",
        ])
        stats = parse_collection(output)
        self.assertEqual((stats["state"], stats["files"], stats["free_mb"]), ("running", 3, 9000))
        self.assertEqual(stats["used_mb"], 1536.0)
        self.assertEqual(stats["ring_hours"], 3.0)  # от старейшего файла до новейшего
        self.assertAlmostEqual(stats["rate_mb_h"], 512.0)

    def test_parse_collection_filling_ring(self):
        from datetime import datetime
        output = "\n".join([
            "Состояние:        идёт запись (dumpcap PID 7)",
            "Файлов захвата:   2, журналов SSH: 1",
            "Занято:           200M, свободно на диске: 9000 МБ",
            "--- session.json ---",
            '{"started_utc": "2026-11-01T10:00:00Z", "ring_files": 10, "file_mb": 100}',
        ])
        stats = parse_collection(output, now=datetime(2026, 11, 1, 12, 0))
        self.assertAlmostEqual(stats["rate_mb_h"], 100.0)
        self.assertAlmostEqual(stats["ring_hours"], 10.0)  # 1000 МБ кольца / 100 МБ в час

    def test_parse_collection_without_session(self):
        stats = parse_collection("Сессий в /var/lib/cic-capture нет.")
        self.assertEqual(stats["state"], "empty")
        self.assertIsNone(stats["ring_hours"])

    def test_check_all_marks_unreachable(self):
        entries = [ServerEntry("ok", "root", "203.0.113.10"), ServerEntry("down", "root", "203.0.113.11")]

        def runner(args, input=None, capture_output=True, timeout=None):
            if "root@203.0.113.11" in args:
                return subprocess.CompletedProcess(args, 255, b"", b"Connection timed out")
            return subprocess.CompletedProcess(args, 0, "Состояние: идёт запись".encode(), b"")

        with patch.object(servers, "_ssh_binary", return_value="ssh"):
            states = check_all(entries, runner=runner)
        self.assertEqual(states["ok"][0], "running")
        self.assertEqual(states["down"][0], "unreachable")
        self.assertIn("timed out", states["down"][1])


class EdgeCaseTests(unittest.TestCase):
    """Границы и ветки, которые нашли выжившие мутанты (cosmic-ray)."""

    def setUp(self):
        patcher = patch.object(servers, "_ssh_binary", return_value="ssh")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.root = ServerEntry("serverC", "root", "203.0.113.10", "/var/lib/cic-capture", LOCAL)

    def test_start_option_bounds(self):
        for reserve, file_mb in ((0, 10), (1000, 2000)):
            with self.subTest(reserve=reserve, file_mb=file_mb):
                start_options("7d", reserve, file_mb)
        for reserve, file_mb in ((-1, 100), (1001, 100), (5, 9), (5, 2001)):
            with self.subTest(reserve=reserve, file_mb=file_mb), self.assertRaises(ServerError):
                start_options("7d", reserve, file_mb)

    def test_sudo_only_for_non_root(self):
        # «ubuntu» по алфавиту после «root»: сравнение должно быть точным
        for user, sudo in (("root", False), ("ubuntu", True), ("admin", True)):
            entry = ServerEntry("s", user, "203.0.113.10")
            with self.subTest(user=user):
                self.assertEqual(collector_command(entry, "status").startswith("sudo -n "), sudo)

    def test_root_wins_over_sudo(self):
        self.assertEqual(check_key_login(self.root, runner=FakeRunner(b"0\nsudo-ok\n")), "root")

    def test_remote_failure_is_not_a_login_failure(self):
        with self.assertRaises(ServerError) as caught:
            collector(self.root, "status", runner=FakeRunner(b"dumpcap: not found", returncode=1))
        self.assertEqual(str(caught.exception), "dumpcap: not found")
        with self.assertRaises(ServerError) as caught:
            collector(self.root, "status", runner=FakeRunner(b"", returncode=3))
        self.assertIn("3", str(caught.exception))

    def test_handshake_reset_is_retried_for_ssh(self):
        outputs = [(255, b"kex_exchange_identification: read: Connection reset"),
                   (255, b"kex_exchange_identification: Connection closed by remote host"),
                   (0, "Состояние: идёт запись".encode())]
        calls = []

        def runner(args, input=None, capture_output=True, timeout=None):
            calls.append(args)
            code, out = outputs.pop(0)
            return subprocess.CompletedProcess(args, code, out, b"")

        with patch.object(servers, "_sleep", lambda seconds: None):
            self.assertIn("идёт запись", collector(self.root, "status", runner=runner))
        self.assertEqual(len(calls), 3)

    def test_other_ssh_failures_are_not_retried(self):
        runner = FakeRunner(b"Permission denied (publickey)", returncode=255)
        with patch.object(servers, "_sleep", lambda seconds: None), self.assertRaises(ServerError) as caught:
            collector(self.root, "status", runner=runner)
        self.assertEqual(len(runner.calls), 1)
        self.assertNotIn("MaxStartups", str(caught.exception))
        runner = FakeRunner(b"kex_exchange_identification: read: Connection reset", returncode=255)
        with patch.object(servers, "_sleep", lambda seconds: None), self.assertRaises(ServerError) as caught:
            collector(self.root, "status", runner=runner)
        self.assertEqual(len(runner.calls), servers.CONNECT_ATTEMPTS)
        self.assertIn("MaxStartups", str(caught.exception))

    def test_collection_command_reads_status_and_session(self):
        for entry, prefix in ((self.root, ""), (ServerEntry("s", "deploy", "203.0.113.10"), "sudo -n ")):
            with self.subTest(user=entry.user):
                command = collection_command(entry)
                status, session = command.split("; ", 1)
                self.assertEqual(status, collector_command(entry, "status"))
                self.assertTrue(session.startswith(prefix + "sh -c "))
                script = shlex.split(session)[-1]
                self.assertIn("ls -1d /var/lib/cic-capture/*/", script)
                self.assertIn(servers.SESSION_JSON_MARK, script)
                self.assertIn(servers.SPAN_MARK, script)

    def test_size_units(self):
        for text, mb in (("640K", 0.625), ("200M", 200.0), ("1,5G", 1536.0), ("2T", 2097152.0), ("12", 12.0)):
            with self.subTest(text=text):
                self.assertAlmostEqual(servers._size_mb(text), mb)
        self.assertIsNone(servers._size_mb("—"))

    def test_collector_state_needs_both_words(self):
        # «нет» в выводе ещё не значит «сессий нет»
        self.assertEqual(collector_state("Состояние: запись не идёт\nСтатус: нет места"), "stopped")

    def test_parse_collection_edges(self):
        from datetime import datetime

        base = ("Состояние:        идёт запись (dumpcap PID 7)\n"
                "Файлов захвата:   4, журналов SSH: 9\n"
                "Занято:           400M, свободно на диске: 9000 МБ\n"
                "--- session.json ---\n")
        # Кольцо заполнено ровно до ring_files: время жизни — по файлам
        full = parse_collection(base + '{"started_utc": "2026-11-01T10:00:00Z", "ring_files": 4, "file_mb": 100}\n'
                                "--- capture mtimes ---\n1793520000\n1793527200\n")
        self.assertEqual((full["files"], full["ssh_logs"]), (4, 9))
        self.assertEqual(full["ring_hours"], 2.0)
        # Нет file_mb — объём кольца неизвестен
        partial = parse_collection(base + '{"started_utc": "2026-11-01T10:00:00Z", "ring_files": 10}\n',
                                   now=datetime(2026, 11, 1, 12, 0))
        self.assertIsNone(partial["ring_mb"])
        self.assertIsNone(partial["ring_hours"])
        self.assertAlmostEqual(partial["rate_mb_h"], 200.0)
        # Только что запущено (меньше 6 минут) — темп ещё не считается
        fresh = parse_collection(base + '{"started_utc": "2026-11-01T10:00:00Z", "ring_files": 10, "file_mb": 100}\n',
                                 now=datetime(2026, 11, 1, 10, 3))
        self.assertIsNone(fresh["rate_mb_h"])
        # Битый session.json и время — без исключений
        broken = parse_collection(base + "{not json\n")
        self.assertIsNone(broken["started"])
        odd_time = parse_collection(base + '{"started_utc": "вчера"}\n')
        self.assertIsNone(odd_time["started"])


class PublicKeyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = make_temp_dir()
        self.addCleanup(remove_temp_dir, self.tmp)
        patcher = patch.object(servers, "ssh_dir", return_value=self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, name, text):
        with open(os.path.join(self.tmp, name), "w", encoding="utf-8") as handle:
            handle.write(text + "\n")

    def test_prefers_ed25519(self):
        self.write("id_rsa.pub", "ssh-rsa AAAArsa me@pc")
        self.write("id_ed25519.pub", "ssh-ed25519 AAAAed me@pc")
        self.assertEqual(servers.public_key(), "ssh-ed25519 AAAAed me@pc")

    def test_missing_key_without_create(self):
        self.assertIsNone(servers.public_key())

    def test_creates_key_without_passphrase(self):
        calls = []

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            self.write("id_ed25519.pub", "ssh-ed25519 AAAAnew me@pc")
            return subprocess.CompletedProcess(args, 0, b"", b"")

        with patch("shutil.which", return_value="ssh-keygen"), patch("subprocess.run", fake_run):
            self.assertEqual(servers.public_key(create=True), "ssh-ed25519 AAAAnew me@pc")
        args, kwargs = calls[0]
        self.assertEqual(args[args.index("-N") + 1], "")  # без пароля: задача ходит сама
        self.assertEqual(args[args.index("-t") + 1], "ed25519")
        self.assertTrue(kwargs["check"])

    def test_no_keygen(self):
        with patch("shutil.which", return_value=None), self.assertRaises(ServerError):
            servers.public_key(create=True)


if __name__ == "__main__":
    unittest.main()
