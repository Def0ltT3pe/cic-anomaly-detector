import json
import os
import unittest
from unittest.mock import patch

from modules.auth import (
    AuthError,
    LoginThrottle,
    add_user,
    authenticate,
    can_manage_servers,
    can_use_llm,
    can_view_audit,
    hash_password,
    load_users,
    remove_user,
    set_password,
    verify_password,
)
from tests.helpers import make_temp_dir, remove_temp_dir


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = make_temp_dir()
        self.path = os.path.join(self.tmp, "users.json")

    def tearDown(self):
        remove_temp_dir(self.tmp)

    def test_hash_roundtrip(self):
        encoded = hash_password("password123")
        self.assertTrue(encoded.startswith("pbkdf2_sha256$600000$"))
        self.assertTrue(verify_password("password123", encoded))
        self.assertFalse(verify_password("wrong-password", encoded))

    def test_add_and_authenticate(self):
        add_user("alice", "password123", "analyst", self.path)
        user = authenticate("alice", "password123", self.path)
        self.assertIsNotNone(user)
        self.assertEqual(user.username, "alice")
        self.assertEqual(user.role, "analyst")
        self.assertIsNone(authenticate("alice", "bad", self.path))
        self.assertIsNone(authenticate("nobody", "password123", self.path))

    def test_duplicate_and_remove(self):
        add_user("bob", "password123", "viewer", self.path)
        with self.assertRaises(AuthError):
            add_user("bob", "password123", "viewer", self.path)
        remove_user("bob", self.path)
        self.assertEqual(load_users(self.path), {})

    def test_password_change(self):
        add_user("carol", "password123", "viewer", self.path)
        set_password("carol", "newpassword", self.path)
        self.assertIsNotNone(authenticate("carol", "newpassword", self.path))
        self.assertIsNone(authenticate("carol", "password123", self.path))

    def test_permissions(self):
        add_user("dave", "password123", "viewer", self.path)
        self.assertTrue(can_use_llm("admin"))
        self.assertTrue(can_use_llm("analyst"))
        self.assertFalse(can_use_llm("viewer"))
        self.assertTrue(can_view_audit("admin"))
        self.assertFalse(can_view_audit("analyst"))

    def test_short_password_rejected(self):
        with self.assertRaises(AuthError):
            hash_password("short")


class LoginThrottleTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        self.throttle = LoginThrottle(max_attempts=3, lock_seconds=60, clock=lambda: self.now)

    def test_locks_after_max_attempts_and_expires(self):
        for _ in range(2):
            self.throttle.failed("admin")
        self.assertEqual(self.throttle.locked_for("admin"), 0)
        self.assertEqual(self.throttle.failed("admin"), 3)
        self.assertEqual(self.throttle.locked_for("admin"), 60)
        self.now += 61
        self.assertEqual(self.throttle.locked_for("admin"), 0)

    def test_login_key_ignores_case_and_spaces(self):
        # Иначе блокировку обходят вариантами написания логина
        for name in ("admin", " Admin", "ADMIN "):
            self.throttle.failed(name)
        self.assertGreater(self.throttle.locked_for("admin"), 0)

    def test_success_resets_series_and_logins_are_independent(self):
        self.throttle.failed("admin")
        self.throttle.failed("admin")
        self.throttle.succeeded("admin")
        self.assertEqual(self.throttle.failed("admin"), 1)
        for _ in range(3):
            self.throttle.failed("other")
        self.assertEqual(self.throttle.locked_for("admin"), 0)

    def test_many_logins_do_not_grow_unbounded(self):
        self.throttle.MAX_TRACKED = 10
        for number in range(50):
            self.throttle.failed(f"user{number}")
        self.assertLessEqual(len(self.throttle._state), 10)


class AuthEdgeCaseTests(unittest.TestCase):
    """Ветки, которые нашли выжившие мутанты (cosmic-ray). Пароли — дешёвые (10 000 итераций)."""

    def setUp(self):
        self.tmp = make_temp_dir()
        self.addCleanup(remove_temp_dir, self.tmp)
        self.path = os.path.join(self.tmp, "users.json")
        self.encoded = hash_password("correct-horse", iterations=10_000)

    def test_rejects_other_algorithms_and_garbage(self):
        _, iterations, salt, digest = self.encoded.split("$")
        bad = [
            f"pbkdf2_sha1${iterations}${salt}${digest}",       # «меньше» по строке
            f"pbkdf2_sha512${iterations}${salt}${digest}",     # «больше» по строке
            f"md5${iterations}${salt}${digest}",
            f"pbkdf2_sha256$9999${salt}${digest}",             # слишком мало итераций
            f"pbkdf2_sha256$5000001${salt}${digest}",          # слишком много
            f"pbkdf2_sha256$abc${salt}${digest}",              # не число
            f"pbkdf2_sha256${iterations}$не-base64!${digest}",
            "pbkdf2_sha256$10000$only-three",
            "",
        ]
        for encoded in bad:
            with self.subTest(encoded=encoded[:40]):
                # Правильный пароль к испорченной записи — всё равно отказ, а не пропуск
                self.assertIs(verify_password("correct-horse", encoded), False)
        self.assertIs(verify_password("correct-horse", self.encoded), True)
        self.assertIs(verify_password("wrong", self.encoded), False)

    def test_lowest_allowed_iterations_still_verify(self):
        # Граница включительна: 10 000 итераций — ещё допустимо
        self.assertIs(verify_password("correct-horse", hash_password("correct-horse", 10_000)), True)

    def test_minimum_password_length(self):
        hash_password("x" * 8, iterations=10_000)
        with self.assertRaises(AuthError):
            hash_password("x" * 7, iterations=10_000)

    def test_roles(self):
        cases = {"admin": (True, True, True), "analyst": (True, False, False), "viewer": (False, False, False),
                 "Admin": (False, False, False), "": (False, False, False), None: (False, False, False)}
        for role, (llm, audit, servers) in cases.items():
            with self.subTest(role=role):
                self.assertIs(can_use_llm(role), llm)
                self.assertIs(can_view_audit(role), audit)
                # Вкладка «Серверы» запускает на серверах команды от root — только admin
                self.assertIs(can_manage_servers(role), servers)

    def write_users(self, users):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "users": users}, handle)

    def test_unknown_role_becomes_default(self):
        self.write_users({"bob": {"password": self.encoded, "role": "root"}})
        self.assertEqual(load_users(self.path)["bob"]["role"], "analyst")
        user = authenticate("bob", "correct-horse", path=self.path)
        self.assertEqual((user.username, user.role), ("bob", "analyst"))

    def test_malformed_users_file(self):
        for content in ("{not json", json.dumps({"users": []}), json.dumps({"other": {}}),
                        json.dumps({"users": {"bob": "plain"}}), json.dumps({"users": {"bob": {"role": "admin"}}})):
            with self.subTest(content=content):
                with open(self.path, "w", encoding="utf-8") as handle:
                    handle.write(content)
                with self.assertRaises(AuthError):
                    load_users(self.path)
        self.assertEqual(load_users(os.path.join(self.tmp, "absent.json")), {})

    def test_users_path_from_environment(self):
        from modules import auth

        previous = os.getcwd()
        os.chdir(self.tmp)  # относительный путь — от проекта, а не от рабочего каталога
        self.addCleanup(os.chdir, previous)
        with patch.dict(os.environ, {"USERS_FILE": "config/other.json"}):
            self.assertEqual(auth.users_path(), os.path.join(auth.BASE_DIR, "config", "other.json"))
        with patch.dict(os.environ, {"USERS_FILE": self.path}):
            self.assertEqual(auth.users_path(), os.path.abspath(self.path))
        with patch.dict(os.environ, {"USERS_FILE": ""}):
            self.assertEqual(auth.users_path(), os.path.join(auth.BASE_DIR, "config", "users.json"))


class LoginThrottleEdgeCaseTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0

    def test_defaults_five_attempts_sixty_seconds(self):
        throttle = LoginThrottle(clock=lambda: self.now)
        for _ in range(4):
            throttle.failed("admin")
        self.assertEqual(throttle.locked_for("admin"), 0)
        self.assertEqual(throttle.failed("admin"), 5)
        self.assertEqual(throttle.locked_for("admin"), 60)
        # После блокировки серия начинается заново
        self.now += 61
        self.assertEqual(throttle.failed("admin"), 1)

    def test_unknown_login_is_not_locked(self):
        throttle = LoginThrottle(clock=lambda: self.now)
        self.assertEqual(throttle.locked_for("nobody"), 0)
        self.assertEqual(throttle.locked_for(None), 0)
        throttle.failed(None)  # пустой логин — тоже ключ, без исключений
        self.assertEqual(throttle.failed(""), 2)

    def test_pruning_keeps_active_locks(self):
        throttle = LoginThrottle(max_attempts=2, lock_seconds=60, clock=lambda: self.now)
        throttle.MAX_TRACKED = 4
        throttle.failed("victim")
        throttle.failed("victim")  # заблокирован
        for n in range(10):
            throttle.failed(f"spray{n}")  # перебор логинов чистит память
        self.assertLessEqual(len(throttle._state), 4)
        self.assertGreater(throttle.locked_for("victim"), 0)


if __name__ == "__main__":
    unittest.main()
