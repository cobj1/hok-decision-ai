# -*- coding: utf-8 -*-
"""线性价值模型 JSON 推理回归。"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hok_decision.search.linear_value import (  # noqa: E402
    LinearValue,
    LinearValueEvaluator,
    context_from_state,
)
from hok_decision.search.planner import SearchState  # noqa: E402
from hok_decision.simulator.units import Hero  # noqa: E402

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

state = SearchState(
    heroes={
        "blue_0": Hero("blue_0", "self", side="blue", position=(100.0, 300.0), hp=0.8, max_hp=1.0),
        "red_0": Hero("red_0", "enemy", side="red", position=(300.0, 100.0), hp=0.4, max_hp=1.0),
    },
    game_time=300.0,
    gold={"blue_0": 5000.0, "red_0": 3000.0},
    tower_hp={"tower_red_1": 0.5, "tower_blue_1": 1.0},
)
context = context_from_state(state)
state_score = LinearValueEvaluator(model).evaluate(state)
checks["context_dim_13"] = len(context) == 13
checks["state_score_in_0_1"] = 0.0 <= state_score <= 1.0

for name, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
if not all(checks.values()):
    raise SystemExit(1)
print("LINEAR_VALUE_OK")
