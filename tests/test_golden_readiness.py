# -*- coding: utf-8 -*-
"""黄金集就绪度单测（EVAL-01）。"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from golden_readiness import readiness  # noqa: E402


class GoldenReadinessTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)
        self.img = self.dir / "a.png"
        self.img.write_bytes(b"x")
        self.data = self.dir / "c.jsonl"

    def tearDown(self):
        self.td.cleanup()

    def _write(self, rows):
        self.data.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                             encoding="utf-8")

    def test_ready(self):
        self._write([
            {"image": str(self.img), "split": s, "scenario": "teamfight",
             "verified": True, "ground_truth": {"action": "hook"}}
            for s in ("train", "val", "test")
        ])
        rep = readiness(self.data)
        self.assertTrue(rep["ready"])
        self.assertEqual(rep["verified"], 3)
        self.assertEqual(rep["actions"].get("hook"), 3)

    def test_unlabelled_not_ready(self):
        self._write([
            {"image": str(self.img), "split": "test", "scenario": "other",
             "verified": False, "ground_truth": None},
        ])
        rep = readiness(self.data)
        self.assertFalse(rep["ready"])
        self.assertTrue(any("ground_truth" in e for e in rep["errors"]))
        self.assertTrue(any("verified" in e for e in rep["errors"]))
        self.assertTrue(any("缺少 split" in e for e in rep["errors"]))

    def test_missing_image(self):
        self._write([
            {"image": str(self.dir / "nope.png"), "split": "test",
             "verified": True, "ground_truth": {"action": "x"}},
        ])
        rep = readiness(self.data)
        self.assertIn("1 条图片不存在", rep["errors"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
