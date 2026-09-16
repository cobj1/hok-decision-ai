"""末端价值模型：可参数化、可校准。

特征（当前）：
  hp_diff   = 蓝方存活英雄血量比例之和 - 红方存活英雄血量比例之和
  resource  = 蓝方已捕获资源点数量
  kill_diff = 红方阵亡数 - 蓝方阵亡数

价值函数形式（线性）：
  score = w_hp * hp_diff + w_resource * resource + w_kill * kill_diff + bias
校准后可用 sigmoid(score) 作为蓝方胜率估计；搜索排序只用 score。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from hok_decision.config import GAME_VERSION, NORMALIZED_ROOT

DEFAULT_WEIGHTS = {
    "w_hp": 100.0,
    "w_resource": 50.0,
    "w_kill": 200.0,
    "w_economy": 100.0,
    "w_tower": 100.0,
    "bias": 0.0,
}

WEIGHTS_FILE = NORMALIZED_ROOT / "value_weights.json"


@dataclass(frozen=True)
class ValueWeights:
    """价值函数权重（hp/resource/kill 由 logistic 校准，economy/tower 为规则初值）。"""

    w_hp: float = DEFAULT_WEIGHTS["w_hp"]
    w_resource: float = DEFAULT_WEIGHTS["w_resource"]
    w_kill: float = DEFAULT_WEIGHTS["w_kill"]
    w_economy: float = DEFAULT_WEIGHTS["w_economy"]
    w_tower: float = DEFAULT_WEIGHTS["w_tower"]
    bias: float = DEFAULT_WEIGHTS["bias"]

    def score(self, hp_diff: float, resource: float, kill_diff: float) -> float:
        return (
            self.w_hp * hp_diff
            + self.w_resource * resource
            + self.w_kill * kill_diff
            + self.bias
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "w_hp": self.w_hp,
            "w_resource": self.w_resource,
            "w_kill": self.w_kill,
            "w_economy": self.w_economy,
            "w_tower": self.w_tower,
            "bias": self.bias,
        }


def load_weights() -> ValueWeights:
    """加载校准权重；缺失字段回退默认值。"""
    if not WEIGHTS_FILE.exists():
        return ValueWeights()
    try:
        payload = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
        weights = payload["weights"]
        merged = {**DEFAULT_WEIGHTS, **weights}
        return ValueWeights(
            w_hp=float(merged["w_hp"]),
            w_resource=float(merged["w_resource"]),
            w_kill=float(merged["w_kill"]),
            w_economy=float(merged["w_economy"]),
            w_tower=float(merged["w_tower"]),
            bias=float(merged["bias"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, OSError):
        return ValueWeights()


def save_weights(
    weights: ValueWeights,
    *,
    source: dict[str, object],
    metrics: dict[str, object],
) -> Path:
    """把校准结果写入正式数据文件。"""
    payload = {
        "version": GAME_VERSION,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "metrics": metrics,
        "weights": weights.to_dict(),
    }
    WEIGHTS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return WEIGHTS_FILE
