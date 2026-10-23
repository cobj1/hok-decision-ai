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
