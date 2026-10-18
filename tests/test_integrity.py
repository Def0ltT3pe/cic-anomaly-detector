import os
import unittest
from unittest.mock import patch

from modules.integrity import (
    IntegrityError,
    generate_keypair,
    verify_checksums,
    write_checksums,
)
from tests.helpers import make_temp_dir, remove_temp_dir


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.dir = make_temp_dir()
        self.private = os.path.join(self.dir, "private.pem")
        self.public = os.path.join(self.dir, "public.pem")
        self.artifact = os.path.join(self.dir, "dummy.pkl")
        with open(self.artifact, "wb") as handle:
            handle.write(b"model-bytes")
        generate_keypair(self.private, self.public, force=True)
        self.env = {
            "MODEL_PUBLIC_KEY": self.public,
            "VERIFY_MODEL_CHECKSUMS": "1",
        }

    def tearDown(self):
        remove_temp_dir(self.dir)

    def test_signed_checksums_verify(self):
        with patch.dict(os.environ, self.env, clear=False):
            write_checksums(self.dir, ["dummy.pkl"], key_path=self.private)
            payload = verify_checksums(self.dir, ["dummy.pkl"])
        self.assertIn("signature", payload)
        self.assertEqual(payload["signature"]["algorithm"], "ed25519")

    def test_tampered_file_is_detected(self):
        with patch.dict(os.environ, self.env, clear=False):
            write_checksums(self.dir, ["dummy.pkl"], key_path=self.private)
            with open(self.artifact, "ab") as handle:
                handle.write(b"tampered")
            with self.assertRaises(IntegrityError):
                verify_checksums(self.dir, ["dummy.pkl"])

    def test_unsigned_checksums_are_rejected(self):
        with patch.dict(os.environ, self.env, clear=False):
            write_checksums(self.dir, ["dummy.pkl"], sign=False)
            with self.assertRaises(IntegrityError):
                verify_checksums(self.dir, ["dummy.pkl"])

    def test_tampered_signature_is_detected(self):
        with patch.dict(os.environ, self.env, clear=False):
            write_checksums(self.dir, ["dummy.pkl"], key_path=self.private)
            checksums = os.path.join(self.dir, "checksums.json")
            with open(checksums, encoding="utf-8") as handle:
                text = handle.read()
            with open(checksums, "w", encoding="utf-8") as handle:
                handle.write(text.replace('"value": "', '"value": "AAAA', 1))
            with self.assertRaises(IntegrityError):
                verify_checksums(self.dir, ["dummy.pkl"])

    def test_unsigned_override(self):
        with patch.dict(
            os.environ,
            {**self.env, "ALLOW_UNSIGNED_MODELS": "1"},
            clear=False,
        ):
            write_checksums(self.dir, ["dummy.pkl"], sign=False)
            payload = verify_checksums(self.dir, ["dummy.pkl"])
        self.assertNotIn("signature", payload)


if __name__ == "__main__":
    unittest.main()
