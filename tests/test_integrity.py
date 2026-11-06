import json
import os
import unittest
from unittest.mock import patch

from modules.calibration import CalibrationError, load_calibration
from modules.integrity import (
    IntegrityError,
    generate_keypair,
    verify_checksums,
    verify_if_listed,
    write_checksums,
)
from tests.helpers import make_temp_dir, remove_temp_dir


class IntegrityCase(unittest.TestCase):
    """Временная папка, пара ключей и один файл модели."""

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


class IntegrityTests(IntegrityCase):
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

    def test_missing_checksums_are_rejected(self):
        # Удалённый checksums.json не должен отключать проверку подписи
        with patch.dict(os.environ, self.env, clear=False):
            with self.assertRaises(IntegrityError):
                verify_checksums(self.dir, ["dummy.pkl"])
            with self.assertRaises(IntegrityError):
                verify_if_listed(self.dir, ["dummy.pkl"])

    def test_missing_checksums_allowed_without_key_or_with_override(self):
        missing_key = {**self.env, "MODEL_PUBLIC_KEY": os.path.join(self.dir, "absent.pem")}
        for env in (missing_key, {**self.env, "ALLOW_UNSIGNED_MODELS": "1"}):
            with patch.dict(os.environ, env, clear=False):
                self.assertIsNone(verify_checksums(self.dir, ["dummy.pkl"]))

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

    def _write_config(self, name: str, text: str = '{"threshold": 0.1}') -> str:
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def test_unlisted_config_is_reported_not_blocked(self):
        with patch.dict(os.environ, self.env, clear=False):
            write_checksums(self.dir, ["dummy.pkl"], key_path=self.private)
            self._write_config("calibration.json")
            unprotected = verify_if_listed(self.dir, ["calibration.json", "absent.json"])
        self.assertEqual(unprotected, ["calibration.json"])

    def test_listed_config_tampering_is_detected(self):
        path = self._write_config("calibration.json")
        with patch.dict(os.environ, self.env, clear=False):
            write_checksums(
                self.dir, ["dummy.pkl", "calibration.json"], key_path=self.private
            )
            self.assertEqual(verify_if_listed(self.dir, ["calibration.json"]), [])
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('{"threshold": -99}')
            with self.assertRaises(IntegrityError):
                verify_if_listed(self.dir, ["calibration.json"])

    def test_tampered_calibration_is_not_loaded(self):
        quantiles = ", ".join(str(value / 100) for value in range(101))
        path = self._write_config(
            "calibration.json",
            f'{{"threshold": 0.02, "target_fpr": 0.02, "quantiles": [{quantiles}]}}',
        )
        with patch.dict(os.environ, self.env, clear=False):
            write_checksums(self.dir, ["calibration.json"], key_path=self.private)
            self.assertTrue(load_calibration(self.dir).protected)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(" ")
            with self.assertRaises(CalibrationError):
                load_calibration(self.dir)

    def test_unsigned_override(self):
        with patch.dict(
            os.environ,
            {**self.env, "ALLOW_UNSIGNED_MODELS": "1"},
            clear=False,
        ):
            write_checksums(self.dir, ["dummy.pkl"], sign=False)
            payload = verify_checksums(self.dir, ["dummy.pkl"])
        self.assertNotIn("signature", payload)


class IntegrityEdgeCaseTests(IntegrityCase):
    """Ветки, которые нашли выжившие мутанты (cosmic-ray)."""

    def signed(self):
        with patch.dict(os.environ, self.env, clear=False):
            write_checksums(self.dir, ["dummy.pkl"], key_path=self.private)
        return os.path.join(self.dir, "checksums.json")

    def rewrite(self, path, change):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        change(payload)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def test_any_hash_mismatch_is_detected(self):
        # Несовпадение — в любую сторону, а не только когда хэш «больше» ожидаемого
        for fake in ("0" * 64, "f" * 64):
            with self.subTest(fake=fake):
                with patch.dict(os.environ, {**self.env, "MODEL_PUBLIC_KEY": os.path.join(self.dir, "absent.pem")}):
                    write_checksums(self.dir, ["dummy.pkl"], sign=False)
                    self.rewrite(os.path.join(self.dir, "checksums.json"),
                                 lambda p: p["files"].update({"dummy.pkl": fake}))
                    with self.assertRaises(IntegrityError):
                        verify_checksums(self.dir, ["dummy.pkl"])

    def test_every_problem_is_reported(self):
        with open(os.path.join(self.dir, "second.pkl"), "wb") as handle:
            handle.write(b"x")
        self.signed()
        with patch.dict(os.environ, self.env, clear=False), self.assertRaises(IntegrityError) as caught:
            verify_checksums(self.dir, ["absent.pkl", "second.pkl", "dummy.pkl"])
        message = str(caught.exception)
        self.assertIn("absent.pkl: файл отсутствует", message)
        self.assertIn("second.pkl: нет записи", message)

    def test_key_order_in_file_does_not_break_signature(self):
        path = self.signed()
        # Тот же JSON с ключами в обратном порядке — подпись по каноническому виду
        self.rewrite(path, lambda p: p.update(dict(reversed(list(p.items())))))
        with patch.dict(os.environ, self.env, clear=False):
            self.assertIn("signature", verify_checksums(self.dir, ["dummy.pkl"]))

    def test_signature_from_other_key_is_rejected(self):
        path = self.signed()
        other_private = os.path.join(self.dir, "other.pem")
        other_public = os.path.join(self.dir, "other.pub.pem")
        generate_keypair(other_private, other_public, force=True)
        with patch.dict(os.environ, {**self.env, "MODEL_PUBLIC_KEY": other_public}):
            with self.assertRaises(IntegrityError) as caught:
                verify_checksums(self.dir, ["dummy.pkl"])
        self.assertIn("Отпечаток", str(caught.exception))
        # Без отпечатка в файле подпись всё равно не сходится с чужим ключом
        self.rewrite(path, lambda p: p["signature"].pop("fingerprint"))
        with patch.dict(os.environ, {**self.env, "MODEL_PUBLIC_KEY": other_public}):
            with self.assertRaises(IntegrityError):
                verify_checksums(self.dir, ["dummy.pkl"])

    def test_malformed_signature_and_checksums(self):
        broken = [
            lambda p: p["signature"].update({"value": "не base64!"}),
            lambda p: p["signature"].pop("value"),
            lambda p: p.pop("files"),
        ]
        for change in broken:
            with self.subTest(change=change):
                path = self.signed()
                self.rewrite(path, change)
                with patch.dict(os.environ, self.env, clear=False), self.assertRaises(IntegrityError):
                    verify_checksums(self.dir, ["dummy.pkl"])
        with open(os.path.join(self.dir, "checksums.json"), "w", encoding="utf-8") as handle:
            handle.write("{not json")
        with patch.dict(os.environ, self.env, clear=False), self.assertRaises(IntegrityError):
            verify_checksums(self.dir, ["dummy.pkl"])

    def test_only_exact_flag_values(self):
        with patch.dict(os.environ, self.env, clear=False):
            write_checksums(self.dir, ["dummy.pkl"], sign=False)
        for value in ("2", "true", "yes", " 0 "):
            with self.subTest(allow=value), patch.dict(os.environ, {**self.env, "ALLOW_UNSIGNED_MODELS": value}):
                with self.assertRaises(IntegrityError):
                    verify_checksums(self.dir, ["dummy.pkl"])
        with patch.dict(os.environ, {**self.env, "ALLOW_UNSIGNED_MODELS": " 1 "}):
            self.assertIsNotNone(verify_checksums(self.dir, ["dummy.pkl"]))
        for value, disabled in (("0", True), (" 0 ", True), ("1", False), ("", False), ("-1", False)):
            with self.subTest(verify=value), patch.dict(os.environ, {**self.env, "VERIFY_MODEL_CHECKSUMS": value}):
                if disabled:
                    self.assertIsNone(verify_checksums(self.dir, ["dummy.pkl"]))
                else:
                    with self.assertRaises(IntegrityError):
                        verify_checksums(self.dir, ["dummy.pkl"])

    def test_explicit_key_path_signs_without_default_key(self):
        env = {**self.env, "MODEL_PRIVATE_KEY": os.path.join(self.dir, "absent", "key.pem")}
        with patch.dict(os.environ, env):
            self.assertIn("signature", write_checksums(self.dir, ["dummy.pkl"], key_path=self.private))
            self.assertNotIn("signature", write_checksums(self.dir, ["dummy.pkl"]))
            with self.assertRaises(IntegrityError):
                write_checksums(self.dir, ["dummy.pkl"], sign=True)  # ключа нет — понятная ошибка

    def test_generate_keypair_refuses_to_overwrite(self):
        with self.assertRaises(IntegrityError):
            generate_keypair(self.private, self.public)
        with self.assertRaises(IntegrityError):
            generate_keypair(self.private, os.path.join(self.dir, "new.pub"))
        nested = os.path.join(self.dir, "nested", "deeper")
        generate_keypair(os.path.join(nested, "k.pem"), os.path.join(nested, "k.pub"))
        self.assertTrue(os.path.isfile(os.path.join(nested, "k.pem")))
        if os.name != "nt":  # права файлов Windows не хранит
            self.assertEqual(os.stat(os.path.join(nested, "k.pem")).st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(os.path.join(nested, "k.pub")).st_mode & 0o777, 0o644)

    def test_file_order_in_checksums_does_not_break_signature(self):
        with open(os.path.join(self.dir, "second.pkl"), "wb") as handle:
            handle.write(b"x")
        with patch.dict(os.environ, self.env, clear=False):
            write_checksums(self.dir, ["dummy.pkl", "second.pkl"], key_path=self.private)
        # Другой инструмент мог переписать список файлов в ином порядке
        self.rewrite(os.path.join(self.dir, "checksums.json"),
                     lambda p: p.update(files=dict(reversed(list(p["files"].items())))))
        with patch.dict(os.environ, self.env, clear=False):
            self.assertIn("signature", verify_checksums(self.dir, ["dummy.pkl", "second.pkl"]))

    def test_relative_key_paths_resolve_from_project(self):
        from modules import integrity

        # Из другого рабочего каталога: путь считается от проекта, а не от cwd
        previous = os.getcwd()
        os.chdir(self.dir)
        self.addCleanup(os.chdir, previous)
        with patch.dict(os.environ, {"MODEL_PUBLIC_KEY": "trust/x.pem", "MODEL_PRIVATE_KEY": "keys/y.pem"}):
            self.assertEqual(integrity.public_key_path(), os.path.join(integrity.BASE_DIR, "trust", "x.pem"))
            self.assertEqual(integrity.private_key_path(), os.path.join(integrity.BASE_DIR, "keys", "y.pem"))
        with patch.dict(os.environ, {"MODEL_PUBLIC_KEY": self.public}):
            self.assertEqual(integrity.public_key_path(), os.path.abspath(self.public))


if __name__ == "__main__":
    unittest.main()
