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
    collector,
    collector_state,
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

    def fake_client(self, fail=None):
        test = self
        key = self.paramiko.RSAKey.generate(1024)

        class Channel:
            def recv_exit_status(self):
                return 0

        class Stream:
            channel = Channel()

            def read(self):
                return b""

        class Client:
            def load_host_keys(self, path):
                pass

            def set_missing_host_key_policy(self, policy):
                self.policy = policy

            def connect(self, host, username, password, **kwargs):
                if fail:
                    raise fail
                test.password_seen = password
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


class StateTests(unittest.TestCase):
    def test_collector_state(self):
        self.assertEqual(collector_state("Состояние:        идёт запись (dumpcap PID 42)"), "running")
        self.assertEqual(collector_state("Состояние:        запись не идёт"), "stopped")
        self.assertEqual(collector_state("Сессий в /var/lib/cic-capture нет."), "empty")

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


if __name__ == "__main__":
    unittest.main()
