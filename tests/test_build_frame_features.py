# -*- coding: utf-8 -*-
"""特征/标签构建单测（DATA-03）。"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_frame_features import FEATURES, build, extract, load_splits  # noqa: E402


class FeatureBuildTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def test_extract_mask(self):
        x, mask = extract({
            "ui": {"hp": 0.5}, "self_state": {"hp_frac": 0.4, "dead": False},
            "towers": {"visible_blue": 3, "visible_red": 2,
                       "enemy_towers_remaining_est": 8, "kills_self": 1},
            "reward": {"total": 10, "delta": 2}, "phase": "in_match",
            "action": {"age_s": 0.3}, "ally_hp": [], "enemy_hp": [[0.5]],
            "frame_quality": "ok", "perception_ok": True,
        })
        self.assertEqual(len(x), len(FEATURES))
        self.assertEqual(mask[FEATURES.index("hp_frac")], 1)
        self.assertEqual(mask[FEATURES.index("mp_frac")], 0)
        self.assertEqual(mask[FEATURES.index("has_enemy_hp")], 1)
        self.assertEqual(x[FEATURES.index("has_ally_hp")], 0.0)
        self.assertEqual(mask[FEATURES.index("has_ally_hp")], 1)

    def test_build_uses_session_split(self):
        splits = {"s1": "train", "s2": "val"}
        rows = [
            {"frame_id": "f1", "session": "s1", "phase": "in_match",
             "action": {"type": "map_move", "reason": "push", "age_s": 1.0},
             "label": "idle", "self_state": {"hp_frac": 0.5}},
            {"frame_id": "f2", "session": "s2", "phase": "in_match",
             "action": {"type": "skill", "reason": "hook", "age_s": 2.0},
             "label": "attacking_enemy", "self_state": {"hp_frac": 0.5}},
            {"frame_id": "f3", "session": "unknown", "action": {"type": "attack"}},
        ]
        data = self.dir / "in.jsonl"
        data.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                        encoding="utf-8")
        out = self.dir / "out.jsonl"
        summary = build(data, splits, out)
        got = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["skipped_unsplit"], 1)
        self.assertEqual({r["split"] for r in got}, {"train", "val"})
        self.assertEqual(got[0]["y_action"], "map_move")
        self.assertEqual(got[1]["y_reason"], "hook")
        self.assertIn("hp_frac", summary["zero_variance"])

    def test_load_splits_namespace(self):
        p = self.dir / "splits.json"
        p.write_text(json.dumps({"wzry": {"s1": "train"}}), encoding="utf-8")
        self.assertEqual(load_splits(p, "wzry"), {"s1": "train"})
        with self.assertRaises(ValueError):
            load_splits(p, "nope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
