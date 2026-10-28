# -*- coding: utf-8 -*-
"""splits.json 统一读取口径的测试。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hok_decision.data.splits import load_splits, split_of  # noqa: E402


class SplitsTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def _write(self, payload: dict) -> Path:
        path = self.dir / "splits.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_loads_base_namespace_and_strips_view_suffix(self):
        path = self._write({
            "tencent_rt": {"a#2": "train"},
            "tencent_rt_base": {"a": "val", "b": "test"},
        })
        splits = load_splits(path)
        self.assertEqual(splits, {"a": "val", "b": "test"})
        self.assertEqual(split_of(splits, "a#2"), "val")
        self.assertIsNone(split_of(splits, "missing"))

    def test_missing_file_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            load_splits(self.dir / "missing.json")

    def test_missing_namespace_is_an_error(self):
        path = self._write({"tencent_rt": {"a": "train"}})
        with self.assertRaisesRegex(ValueError, "tencent_rt_base"):
            load_splits(path)

    def test_suffixed_base_key_is_rejected(self):
        path = self._write({"tencent_rt_base": {"a#2": "train"}})
        with self.assertRaisesRegex(ValueError, "#"):
            load_splits(path)

    def test_invalid_split_value_is_rejected(self):
        path = self._write({"tencent_rt_base": {"a": "holdout"}})
        with self.assertRaisesRegex(ValueError, "非法"):
            load_splits(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
