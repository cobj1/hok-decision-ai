"""线性价值模型推理（S2-1 基线）。

模型 JSON 由 `scripts/win_train_linear_baseline.py` 生成：
  features = CONTEXT_FEATURES + VALUE_FEATURES
  value = sigmoid(Σ w_i * (x_i - mean_i) / sigma_i + bias)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence


class LinearValue:
    def __init__(self, payload: dict) -> None:
        self.features: list[str] = list(payload["features"])
        self.mean: list[float] = [float(v) for v in payload["mean"]]
        self.sigma: list[float] = [float(v) for v in payload["sigma"]]
        self.weights: list[float] = [float(v) for v in payload["weights"]]
        self.bias: float = float(payload["bias"])
        if not (len(self.features) == len(self.mean) == len(self.sigma) == len(self.weights)):
            raise ValueError("LinearValue 维度不一致")

    @classmethod
    def load(cls, path: str | Path) -> "LinearValue":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def logit(self, context: Sequence[float], features: Sequence[float]) -> float:
        values = list(context) + list(features)
        if len(values) != len(self.weights):
            raise ValueError(f"输入维度 {len(values)} != 模型维度 {len(self.weights)}")
        total = self.bias
        for value, mean, sigma, weight in zip(values, self.mean, self.sigma, self.weights):
            total += weight * ((float(value) - mean) / (sigma if sigma else 1.0))
        return total

    def score(self, context: Sequence[float], features: Sequence[float]) -> float:
        z = self.logit(context, features)
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def to_dict(self) -> dict:
        return {
            "features": self.features,
            "mean": self.mean,
            "sigma": self.sigma,
            "weights": self.weights,
            "bias": self.bias,
        }


def context_from_state(state) -> list[float]:
    """SearchState -> 13 维 context（与训练口径一致；缺失字段用保守默认值）。"""
    blues = [hero for hero in state.heroes.values() if hero.side == "blue"]
    reds = [hero for hero in state.heroes.values() if hero.side == "red"]
    own_hp = (
        sum(hero.hp / hero.max_hp for hero in blues if hero.max_hp > 0) / len(blues)
        if blues else 0.7
    )
    red_loss = sum(
        1.0 - hp for tower_id, hp in state.tower_hp.items() if tower_id.startswith("tower_red")
    )
    blue_loss = sum(
        1.0 - hp for tower_id, hp in state.tower_hp.items() if tower_id.startswith("tower_blue")
    )
    time_norm = min(1.0, state.game_time / 1500.0)
    return [
        time_norm,
        time_norm,                    # econ_sat（与训练数据同口径）
        0.5,                          # enemy_threat
        own_hp,
        (red_loss - blue_loss) / 9.0,  # tower_adv
        1.0 if state.game_time >= 120.0 else 0.0,
        0.8,                          # team_aa_power
        0.8,                          # team_caster_power
        0.8,                          # build_attack_power
        0.8,                          # build_defense_power
        0.8,                          # build_utility_power
        0.8,                          # skill_power
        3.0 / 7.0,                    # match_level
    ]


class LinearValueEvaluator:
    """供 beam_search 使用的 value_context 适配器。"""

    def __init__(self, model: LinearValue) -> None:
        self.model = model

    @classmethod
    def load(cls, path: str | Path) -> "LinearValueEvaluator":
        return cls(LinearValue.load(path))

    def evaluate(self, state, action=None) -> float:
        from hok_decision.search.attention_weights import value_features_from_state

        return self.model.score(context_from_state(state), value_features_from_state(state))
