import unittest
from pathlib import Path

from modules import __version__


class VersionTests(unittest.TestCase):
    def test_version_matches_file(self):
        version_file = Path(__file__).resolve().parents[1] / "VERSION"
        self.assertEqual(__version__, version_file.read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    unittest.main()
