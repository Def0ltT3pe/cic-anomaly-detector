import json
import os
import unittest
from unittest.mock import patch

from modules.audit import _tail_lines, log_event, read_recent
from tests.helpers import make_temp_dir, remove_temp_dir


class ReadRecentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = make_temp_dir()
        self.addCleanup(remove_temp_dir, self.tmp)
        self.path = os.path.join(self.tmp, "audit.log")
        env = patch.dict(os.environ, {"AUDIT_LOG": self.path, "AUDIT_ENABLED": "1"})
        env.start()
        self.addCleanup(env.stop)

    def test_last_entries_in_order(self):
        for number in range(30):
            log_event("admin", "admin", "upload", n=number)
        entries = read_recent(5)
        self.assertEqual([entry["n"] for entry in entries], [25, 26, 27, 28, 29])
        self.assertEqual(read_recent(0), [])

    def test_tail_matches_whole_file_across_block_borders(self):
        # Строки разной длины (и не ASCII), куски меньше строки — ответ как у чтения целиком
        lines = [json.dumps({"n": n, "text": "абв" * (n % 7)}, ensure_ascii=False) + "\n"
                 for n in range(200)]
        with open(self.path, "w", encoding="utf-8", newline="\n") as handle:
            handle.writelines(lines)
        for limit in (1, 3, 20, 199, 200, 500):
            for block in (7, 64, 4096):
                with self.subTest(limit=limit, block=block):
                    self.assertEqual(_tail_lines(self.path, limit, block), lines[-limit:])

    def test_missing_or_empty_file(self):
        self.assertEqual(read_recent(5), [])
        open(self.path, "w").close()
        self.assertEqual(read_recent(5), [])


if __name__ == "__main__":
    unittest.main()
