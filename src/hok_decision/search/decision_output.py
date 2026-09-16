"""决策输出层：把搜索最优解编码成"做什么"。

抛弃固定动作集。输出：
  goal_vector:   目标收益向量（击杀/经济/推塔/龙/生存/推进）
  difficulty:    达成该动作的难度 0~1（距离/敌方威胁/己方状态）
  target_position: 需要位置的动作附带坐标（ref_416x424 地图系）
  relative_position: 语义相对位置（哪条线/己方或敌方半区/距离）
  description:    给真人的一句话说明
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hok_decision.search.planner import SearchState, extract_features
from hok_decision.simulator.units import Hero

BENEFIT_DIMS = ["kill", "economy", "tower", "dragon", "survival", "advance"]

BLUE_BASE = (25.0, 395.0)
RED_BASE = (397.0, 28.0)
MAP_MAX = (416.0**2 + 424.0**2) ** 0.5


@dataclass(frozen=True)
class DecisionOutput:
    goal_vector: list[float]
    difficulty: float
    target_position: tuple[float, float] | None
    coordinate_system: str = "ref_416x424"
    relative_position: dict[str, Any] = field(default_factory=dict)
    description: str = ""


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _lane_name(y: float) -> str:
    return "上路" if y < 140 else ("中路" if y < 280 else "下路")


def _side_name(x: float, self_side: str) -> str:
    if x < 180:
        return "己方半区" if self_side == "blue" else "敌方半区"
    if x > 240:
        return "敌方半区" if self_side == "blue" else "己方半区"
    return "河道/中路"


def relative_position(
    target: tuple[float, float], self_pos: tuple[float, float], self_side: str = "blue"
) -> dict[str, Any]:
    """把目标坐标转成语义相对位置。"""
    return {
        "lane": _lane_name(target[1]),
        "side": _side_name(target[0], self_side),
        "distance_norm": round(_distance(target, self_pos) / MAP_MAX, 4),
    }


def estimate_difficulty(
    *,
    target: tuple[float, float],
    state: SearchState,
    threat_radius: float = 220.0,
) -> float:
    """达成难度：距离 + 敌方威胁 + 己方血量。

    difficulty ∈ [0,1]，越大越难。不参与训练，只用于输出提示。
    """
    blues = [h for h in state.heroes.values() if h.side == "blue" and h.alive]
    reds = [h for h in state.heroes.values() if h.side == "red" and h.alive]
    if not blues:
        return 1.0
    self_pos = min(
        (h.position for h in blues),
        key=lambda p: _distance(p, target),
    )
    distance_score = min(1.0, _distance(self_pos, target) / MAP_MAX)
    threats = [h for h in reds if _distance(h.position, target) <= threat_radius]
    threat_score = min(1.0, len(threats) / 3.0)
    hp_score = 1.0 - min(1.0, sum(h.hp / h.max_hp for h in blues if h.max_hp > 0) / len(blues))
    difficulty = 0.45 * distance_score + 0.35 * threat_score + 0.20 * hp_score
    return round(min(1.0, difficulty), 4)


def build_decision_output(
    *,
    initial_state: SearchState,
    final_state: SearchState,
    target_position: tuple[float, float] | None,
    self_side: str = "blue",
    description: str = "",
) -> DecisionOutput:
    """从搜索前后状态差计算目标收益向量，并给出难度与位置。"""
    hp0, res0, kill0 = extract_features(initial_state)
    hp1, res1, kill1 = extract_features(final_state)

    gold0 = sum(initial_state.gold.get(h, 0.0) for h in initial_state.heroes if initial_state.heroes[h].side == "blue")
    gold1 = sum(final_state.gold.get(h, 0.0) for h in final_state.heroes if final_state.heroes[h].side == "blue")

    def tower_damage(state: SearchState, side: str) -> float:
        return sum(1.0 - hp for tower_id, hp in state.tower_hp.items() if tower_id.startswith(f"tower_{side}"))

    def dragons(state: SearchState) -> int:
        return sum(1 for rid in state.captured_resources if rid in {"tyrant", "overlord", "storm_dragon"})

    goal_vector = [
        round(kill1 - kill0, 4),                    # 击杀收益
        round((gold1 - gold0) / 10000.0, 4),        # 经济收益
        round(tower_damage(final_state, "red") - tower_damage(initial_state, "red"), 4),  # 推塔
        round(float(dragons(final_state) - dragons(initial_state)), 4),                    # 龙
        round(hp1 - hp0, 4),                        # 生存
        round(_advance(final_state) - _advance(initial_state), 4),                        # 推进
    ]

    difficulty = estimate_difficulty(target=target_position or (0.0, 0.0), state=initial_state)
    blues0 = [h for h in initial_state.heroes.values() if h.side == "blue" and h.alive]
    self_pos = (
        sum(h.position[0] for h in blues0) / len(blues0),
        sum(h.position[1] for h in blues0) / len(blues0),
    ) if blues0 else (0.0, 0.0)
    rel = relative_position(target_position, self_pos, self_side) if target_position else {}
    return DecisionOutput(
        goal_vector=goal_vector,
        difficulty=difficulty,
        target_position=target_position,
        relative_position=rel,
        description=description,
    )


def _advance(state: SearchState) -> float:
    blues = [h for h in state.heroes.values() if h.side == "blue" and h.alive]
    reds = [h for h in state.heroes.values() if h.side == "red" and h.alive]
    if not blues or not reds:
        return 0.0
    bc = (sum(h.position[0] for h in blues) / len(blues), sum(h.position[1] for h in blues) / len(blues))
    rc = (sum(h.position[0] for h in reds) / len(reds), sum(h.position[1] for h in reds) / len(reds))
    self_progress = 1.0 - _distance(bc, RED_BASE) / MAP_MAX
    enemy_progress = 1.0 - _distance(rc, BLUE_BASE) / MAP_MAX
    return self_progress - enemy_progress
