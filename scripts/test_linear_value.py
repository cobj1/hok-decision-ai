# -*- coding: utf-8 -*-
"""线性价值模型 JSON 推理回归。"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hok_decision.search.linear_value import LinearValue  # noqa: E402

model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "hokdata" / "value_linear_model.json"
model = LinearValue.load(model_path)
payload = json.loads(model_path.read_text(encoding="utf-8"))

checks = {
    "dims": len(model.features) == 23 and len(model.weights) == 23,
    "zero_logit_matches": abs(
        model.logit([0.0] * 13, [0.0] * 10)
        - (model.bias + sum(w * ((0.0 - m) / (s if s else 1.0))
                            for w, m, s in zip(model.weights, model.mean, model.sigma)))
    ) < 1e-9,
    "score_in_0_1": 0.0 <= model.score([0.0] * 13, [0.0] * 10) <= 1.0,
    "roundtrip": LinearValue(model.to_dict()).to_dict() == model.to_dict(),
    "metrics_present": "test_acc" in payload and "test_majority" in payload,
}
for name, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
if not all(checks.values()):
    raise SystemExit(1)
print("LINEAR_VALUE_OK")
