import os
import unittest

from modules.auth import (
    AuthError,
    LoginThrottle,
    add_user,
    authenticate,
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


if __name__ == "__main__":
    unittest.main()
