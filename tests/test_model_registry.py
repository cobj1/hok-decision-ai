# -*- coding: utf-8 -*-
"""模型注册表单测（MODEL-01）。"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from model_registry import _metrics_of, scan  # noqa: E402


class ModelRegistryTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.dir = Path(self.td.name)

    def tearDown(self):
        self.td.cleanup()

    def _pt(self, name):
        (self.dir / f"{name}.pt").write_bytes(b"fake-weights-" + name.encode())

    def test_validated_and_missing(self):
        self._pt("good")
        (self.dir / "good.json").write_text(json.dumps({
            "metrics": {"samples": 100, "val_accuracy": 0.7,
                        "majority_baseline": 0.6, "seeds": [0.69, 0.71]},
        }), encoding="utf-8")
        self._pt("bare")
        reg = scan(self.dir)
        status = {m["name"]: m["status"] for m in reg["models"]}
        self.assertEqual(status["good"], "validated")
        self.assertEqual(status["bare"], "no_report")
        self.assertTrue(reg["errors"])

    def test_sequence_report(self):
        self._pt("prior")
        (self.dir / "prior.json").write_text(json.dumps({
            "rows": 724106, "test": {"acc": 0.48, "macro_f1": 0.43},
            "baselines": {"majority_acc": 0.32},
            "matches": {"train": 38, "val": 8, "test": 9},
        }), encoding="utf-8")
        reg = scan(self.dir)
        m = reg["models"][0]
        self.assertEqual(m["status"], "validated")
        self.assertEqual(m["metrics"]["samples"], 724106)
        self.assertAlmostEqual(m["metrics"]["baseline_accuracy"], 0.32)

    def test_bad_metrics_rejected(self):
        self.assertIsNone(_metrics_of({"metrics": {"samples": 0, "val_accuracy": 0.9}}))
        self.assertIsNone(_metrics_of({"metrics": {"samples": 10, "val_accuracy": None}}))
        self.assertIsNone(_metrics_of({"foo": 1}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
