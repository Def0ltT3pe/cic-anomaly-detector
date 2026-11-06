import os
import unittest
from unittest.mock import patch

from modules.fsutil import replace_file
from tests.helpers import make_temp_dir, remove_temp_dir


class ReplaceFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = make_temp_dir()
        self.addCleanup(remove_temp_dir, self.tmp)
        self.source = os.path.join(self.tmp, "a.tmp")
        self.target = os.path.join(self.tmp, "a.json")
        with open(self.source, "w", encoding="utf-8") as handle:
            handle.write("new")

    def test_retries_while_file_is_busy(self):
        # Антивирус держит файл: первые две замены — «Отказано в доступе»
        real = os.replace
        calls = []

        def flaky(source, target):
            calls.append(source)
            if len(calls) < 3:
                raise PermissionError(5, "Отказано в доступе")
            real(source, target)

        delays = []
        with patch("os.replace", flaky):
            replace_file(self.source, self.target, sleep=delays.append)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(delays), 2)
        with open(self.target, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "new")

    def test_gives_up_after_attempts(self):
        def busy(source, target):
            raise PermissionError(5, "Отказано в доступе")

        with patch("os.replace", busy), self.assertRaises(PermissionError):
            replace_file(self.source, self.target, attempts=3, sleep=lambda seconds: None)

    def test_other_errors_are_not_retried(self):
        calls = []

        def missing(source, target):
            calls.append(source)
            raise FileNotFoundError(source)

        with patch("os.replace", missing), self.assertRaises(FileNotFoundError):
            replace_file(self.source, self.target, sleep=lambda seconds: None)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
