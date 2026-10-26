# -*- coding: utf-8 -*-
"""数据契约审计单测（DATA-03）。"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hok_decision.data_contract import audit_dataset, load_splits  # noqa: E402


class DataContractTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)
        self.splits = self.dir / "splits.json"
        self.splits.write_text(json.dumps({"wzry": {"s1": "train", "s2": "val"}}),
                               encoding="utf-8")
        self.data = self.dir / "d.jsonl"
        rows = [
            {"frame_id": "f1", "session": "s1", "game_time_s": 2.0,
             "label": "push", "action": {"type": "map_move"},
             "ui": {"hp": 0.8, "mp": 0.5}},
            {"frame_id": "f2", "session": "s1", "game_time_s": 2.0,
             "label": "push", "action": {"type": "skill"},
             "ui": {"hp": 0.7, "mp": 0.4}},
            {"frame_id": "f2", "session": "s2", "game_time_s": 2.0,
             "label": "farm", "action": {"type": "attack"},
             "ui": {"hp": 0.6, "mp": 0.3}},
        ]
        self.data.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                             encoding="utf-8")

    def tearDown(self):
        self.td.cleanup()

    def test_split_namespace(self):
        got = load_splits(self.splits, "wzry")
        self.assertEqual(got["s1"], "train")
        with self.assertRaises(ValueError):
            load_splits(self.splits)

    def test_audit_finds_duplicate_and_zero_variance(self):
        rep = audit_dataset(self.data, load_splits(self.splits, "wzry"))
        self.assertEqual(rep["rows"], 3)
        self.assertEqual(rep["duplicate_frame_ids"], 1)
        self.assertTrue(any("frame_id 重复" in e for e in rep["errors"]))
        self.assertTrue(rep["features"]["game_time_s"]["zero_variance"])
        self.assertFalse(rep["features"]["ui.hp"]["zero_variance"])
        self.assertEqual(rep["labels"]["label"].get("push"), 2)
        self.assertEqual(rep["labels"]["action.type"].get("map_move"), 1)

    def test_audit_flags_empty_features(self):
        rep = audit_dataset(self.data, load_splits(self.splits, "wzry"))
        self.assertIn("ui.gold", rep["features"])
        self.assertEqual(rep["features"]["ui.gold"]["n"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
