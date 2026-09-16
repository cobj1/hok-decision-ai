"""搜索用的注意力权重价值评估器。

架构分工（用户确认）：
  - 动作来自 search（候选生成 + rollout + beam search）；
  - 模型只负责给出一阶价值特征的二阶权重 w(s)=attention(局势;θ)；
  - V(s,a)=Σ w_j(s)·F_j(s,a)，search 用它排序候选动作；
  - DecisionOutput 由搜索最优解编码，不是模型输出的动作。

特征 F 有两套：
  - VALUE_DIMS(10)：一阶事件收益（血/经济/野/buff/龙/塔/击杀），TD 模型使用；
  - ACTION_DIMS(20)：上者 + 几何推进特征(4) + 动作语义指示(6)，
    attention_value_asr.pt 用它把解说动作偏好学进权重 w(s)。
模型文件只在 Win。无 torch/无模型时回退 base_weights，search 仍可运行。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from hok_decision.config import NORMALIZED_ROOT
from hok_decision.search.planner import SearchState

DEFAULT_MODEL_PATH = NORMALIZED_ROOT.parent / "models" / "attention_value_td.pt"
DEFAULT_BASE_WEIGHTS = [4.8, 100.0, 50.0, 20.0, 20.0, 15.0, 25.0, 50.0, 100.0, 2.7]
VALUE_DIMS = [
    "hp_diff",
    "economy_diff",
    "small_camp_diff",
    "red_buff_diff",
    "blue_buff_diff",
    "tyrant_diff",
    "overlord_diff",
    "storm_dragon_diff",
    "tower_diff",
    "kill_diff",
]
GEOMETRY_DIMS = [
    "advance_abs",
    "centroid_dist_norm",
    "self_spread_norm",
    "nearest_pair_norm",
]
ACTION_KIND_DIMS = [
    "is_attack_enemy",
    "is_retreat",
    "is_capture_camp",
    "is_dragon",
    "is_push_tower",
    "is_defend_tower",
]
ACTION_DIMS = VALUE_DIMS + GEOMETRY_DIMS + ACTION_KIND_DIMS

MAP_MAX = math.sqrt(416.0**2 + 424.0**2)
BLUE_BASE = (25.0, 395.0)
RED_BASE = (397.0, 28.0)


def value_features_from_state(state: SearchState) -> list[float]:
    """10 维一阶价值特征 F(s)（TD 模型口径，差值均为蓝方相对红方）。"""
    blues = [h for h in state.heroes.values() if h.side == "blue"]
    reds = [h for h in state.heroes.values() if h.side == "red"]

    hp_diff = sum(h.hp / h.max_hp for h in blues if h.max_hp > 0) - sum(
        h.hp / h.max_hp for h in reds if h.max_hp > 0
    )
    gold_blue = sum(state.gold.get(h.unit_id, 0.0) for h in blues)
    gold_red = sum(state.gold.get(h.unit_id, 0.0) for h in reds)

    counts = {dim: 0 for dim in VALUE_DIMS[2:8]}
    for rid in state.captured_resources:
        counts[_resource_dimension(rid)] += 1

    red_loss = sum(1.0 - hp for tower_id, hp in state.tower_hp.items() if tower_id.startswith("tower_red"))
    blue_loss = sum(1.0 - hp for tower_id, hp in state.tower_hp.items() if tower_id.startswith("tower_blue"))

    return [
        hp_diff,
        (gold_blue - gold_red) / 10000.0,
        float(counts["small_camp_diff"]),
        float(counts["red_buff_diff"]),
        float(counts["blue_buff_diff"]),
        float(counts["tyrant_diff"]),
        float(counts["overlord_diff"]),
        float(counts["storm_dragon_diff"]),
        red_loss - blue_loss,
        float(sum(1 for h in reds if not h.alive) - sum(1 for h in blues if not h.alive)),
    ]


def action_features_from_state(
    state: SearchState,
    action: Any | None = None,
    *,
    feature_dim: int = len(ACTION_DIMS),
) -> list[float]:
    """20 维 F(s,a)：10 维事件 + 4 维几何 + 6 维动作语义。"""
    features = value_features_from_state(state)
    if feature_dim == len(VALUE_DIMS):
        return features
    features += geometry_features_from_state(state)
    if feature_dim == len(VALUE_DIMS) + len(GEOMETRY_DIMS):
        return features
    features += action_kind_features(action)
    return features[:feature_dim]


def geometry_features_from_state(state: SearchState) -> list[float]:
    blues = [h for h in state.heroes.values() if h.side == "blue" and h.alive]
    reds = [h for h in state.heroes.values() if h.side == "red" and h.alive]
    if not blues or not reds:
        return [0.0] * len(GEOMETRY_DIMS)
    bc = _centroid(blues)
    rc = _centroid(reds)

    def dist(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    self_progress = 1.0 - dist(bc, RED_BASE) / MAP_MAX
    enemy_progress = 1.0 - dist(rc, BLUE_BASE) / MAP_MAX
    return [
        self_progress - enemy_progress,
        dist(bc, rc) / MAP_MAX,
        sum(dist(h.position, bc) for h in blues) / len(blues) / MAP_MAX,
        min(dist(h1.position, h2.position) for h1 in blues for h2 in reds) / MAP_MAX,
    ]


def action_kind_features(action: Any | None) -> list[float]:
    kind = getattr(action, "kind", None)
    target = getattr(action, "target_id", "") or ""
    dragon = any(d in target for d in ("tyrant", "overlord", "storm_dragon"))
    return [
        1.0 if kind == "attack_enemy" else 0.0,
        1.0 if kind == "retreat" else 0.0,
        1.0 if kind == "capture_resource" and not dragon else 0.0,
        1.0 if kind == "capture_resource" and dragon else 0.0,
        1.0 if kind == "push_tower" else 0.0,
        1.0 if kind == "defend_tower" else 0.0,
    ]


def _centroid(heroes) -> tuple[float, float]:
    return (
        sum(h.position[0] for h in heroes) / len(heroes),
        sum(h.position[1] for h in heroes) / len(heroes),
    )


def _resource_dimension(resource_id: str) -> str:
    if resource_id in {"tyrant", "overlord", "storm_dragon"}:
        return f"{resource_id}_diff"
    if "blue_buff" in resource_id:
        return "blue_buff_diff"
    if "red_buff" in resource_id:
        return "red_buff_diff"
    return "small_camp_diff"


class AttentionWeightedValue:
    """w(s)=AttentionValueModel(context;θ) 的动态权重 + 一阶价值特征线性打分。

    feature_dim 自动按模型 state_dict 推断：TD=10，ASR 拟合=20。
    evaluate(state, action=None)：action 提供时用 F(s,a)（20 维），否则 F(s)。
    """

    def __init__(
        self,
        *,
        context: Sequence[float] | None = None,
        model_path: Path | None = DEFAULT_MODEL_PATH,
        base_weights: Sequence[float] | None = None,
        feature_dim: int | None = None,
        device: str | None = None,
    ) -> None:
        self.context = [float(v) for v in (context or [])]
        self.base_weights = [float(v) for v in (base_weights or DEFAULT_BASE_WEIGHTS)]
        self.model: Any = None
        self._device = device or "cpu"
        self.model_path = model_path
        self.feature_dim = feature_dim or len(VALUE_DIMS)

        if model_path is None or not Path(model_path).exists():
            self._ensure_base_width()
            return
        try:
            import torch

            from hok_decision.model.attention_value import AttentionValueModel
        except Exception:
            self._ensure_base_width()
            return
        try:
            state_dict = torch.load(Path(model_path), map_location=self._device)
            model_feature_dim = int(state_dict["keys"].shape[0])
            context_dim = max(len(self.context), 13)
            model = AttentionValueModel(
                context_dim=context_dim,
                feature_dim=model_feature_dim,
                base_weights=[1.0] * model_feature_dim,
            )
            model.load_state_dict(state_dict)
            model.to(self._device)
            model.eval()
            self.model = model
            self.feature_dim = model_feature_dim
            self.base_weights = [1.0] * model_feature_dim
            self._device = str(device) if device else str(next(model.parameters()).device)
        except Exception:
            self.model = None
            self._ensure_base_width()

    def _ensure_base_width(self) -> None:
        if len(self.base_weights) < self.feature_dim:
            self.base_weights = self.base_weights + [1.0] * (self.feature_dim - len(self.base_weights))
        self.base_weights = self.base_weights[: self.feature_dim]

    def weights(self) -> list[float]:
        if self.model is None:
            return self.base_weights[: self.feature_dim]
        import torch

        context = torch.tensor([self.context], dtype=torch.float32, device=self._device)
        with torch.no_grad():
            weights, _ = self.model.dynamic_weights(context)
        return [float(v) for v in weights[0].cpu().tolist()]

    def evaluate(self, state: SearchState, action: Any | None = None) -> float:
        if action is not None and self.feature_dim > len(VALUE_DIMS):
            features = action_features_from_state(state, action, feature_dim=self.feature_dim)
        else:
            features = value_features_from_state(state)
            if self.feature_dim > len(VALUE_DIMS):
                features += [0.0] * (self.feature_dim - len(features))
        weights = self.weights()
        return float(sum(w * f for w, f in zip(weights, features)))


__all__ = [
    "ACTION_DIMS",
    "ACTION_KIND_DIMS",
    "AttentionWeightedValue",
    "DEFAULT_BASE_WEIGHTS",
    "DEFAULT_MODEL_PATH",
    "GEOMETRY_DIMS",
    "VALUE_DIMS",
    "action_features_from_state",
    "action_kind_features",
    "geometry_features_from_state",
    "value_features_from_state",
]
