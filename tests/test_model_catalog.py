import json
import os
import unittest
from unittest.mock import patch

from modules.model_catalog import list_models
from tests.helpers import make_temp_dir, remove_temp_dir


def make_model(path, signed=True, metadata=None):
    os.makedirs(path, exist_ok=True)
    open(os.path.join(path, "isolation_forest.pkl"), "wb").close()
    if signed:
        open(os.path.join(path, "checksums.json"), "w").close()
    if metadata is not None:
        with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle)


class ModelCatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = make_temp_dir("catalog-")
        self.addCleanup(remove_temp_dir, self.tmp)
        self.root = os.path.join(self.tmp, "site")

    def names(self, default=None):
        env = {"MODEL_DIR": default or os.path.join(self.tmp, "нет")}
        with patch.dict(os.environ, env):
            return [entry.name for entry in list_models([(self.tmp, [self.root])])]

    def test_nested_signed_models(self):
        make_model(os.path.join(self.root, "serverA", "model2"), metadata={"version": "site-adapted"})
        make_model(os.path.join(self.root, "serverB"))
        self.assertEqual(self.names(), ["site/serverA/model2", "site/serverB"])

    def test_unsigned_and_too_deep_are_skipped(self):
        make_model(os.path.join(self.root, "unsigned"), signed=False)
        make_model(os.path.join(self.root, "a", "b", "c"))
        self.assertEqual(self.names(), [])

    def test_default_model_first(self):
        make_model(os.path.join(self.root, "a"))
        make_model(os.path.join(self.root, "b"))
        self.assertEqual(self.names(default=os.path.join(self.root, "b")), ["site/b", "site/a"])

    def test_label_uses_metadata_version(self):
        make_model(os.path.join(self.root, "a"), metadata={"version": "v9"})
        with patch.dict(os.environ, {"MODEL_DIR": os.path.join(self.tmp, "нет")}):
            entry = list_models([(self.tmp, [self.root])])[0]
        self.assertEqual(entry.label, "site/a (v9)")

    def test_project_models_are_found(self):
        names = [entry.name for entry in list_models()]
        self.assertIn("models", names)
