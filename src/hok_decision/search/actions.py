"""P2 搜索骨架：宏观动作定义与候选生成。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

ActionKind = Literal[
    "attack_enemy", "retreat", "capture_resource", "defend_tower", "push_tower"
]

if TYPE_CHECKING:  # pragma: no cover
    from hok_decision.schema.models import MapData
    from hok_decision.search.planner import SearchState


@dataclass(frozen=True)
class MacroAction:
    """一个决策级宏观动作：语义 + 目标点/目标资源，持续时间固定。"""

    kind: ActionKind
    target_id: str | None = None
    target_position: tuple[float, float] | None = None
    duration_s: float = 4.0

    @property
    def label(self) -> str:
        prefix = {
            "attack_enemy": "进攻",
            "retreat": "后撤",
            "capture_resource": "打资源",
            "defend_tower": "守塔",
            "push_tower": "推塔",
        }[self.kind]
        if self.target_id:
            return f"{prefix}:{self.target_id}"
        return prefix


def generate_candidates(
    state: "SearchState",
    map_data: "MapData",
    anchors: dict[str, dict],
    *,
    max_candidates: int = 8,
) -> list[MacroAction]:
    """从当前状态生成 4~8 个候选宏观动作。"""
    blue = next((h for h in state.heroes.values() if h.side == "blue" and h.alive), None)
    if blue is None:
        return []
    pos = blue.position

    def dist(p: tuple[float, float] | None) -> float:
        if p is None:
            return float("inf")
        return ((pos[0] - p[0]) ** 2 + (pos[1] - p[1]) ** 2) ** 0.5

    candidates: list[MacroAction] = []

    def add(action: MacroAction) -> None:
        if action.target_position is not None and action.target_position not in seen:
            seen.add(action.target_position)
            candidates.append(action)

    seen: set[tuple[float, float]] = set()

    # 1. 进攻最近敌方英雄
    enemies = [h for h in state.heroes.values() if h.side == "red" and h.alive]
    if enemies:
        target = min(enemies, key=lambda h: dist(h.position))
        add(
            MacroAction(
                kind="attack_enemy",
                target_id=target.unit_id,
                target_position=target.position,
            )
        )

    # 2. 后撤回泉水/蓝方水晶
    fountain = anchors.get("fountain_blue", {}).get("position")
    if fountain is not None:
        add(
            MacroAction(kind="retreat", target_id="fountain_blue", target_position=tuple(fountain))
        )

    # 3. 龙/大野目标：按出现时间放进候选（不占用小野位）
    dragons = [
        rp
        for rp in map_data.resource_points
        if rp.position is not None
        and rp.id in {"tyrant", "overlord"}
        and (rp.initial_spawn_s is None or state.game_time >= rp.initial_spawn_s)
        and (rp.active_until_s is None or state.game_time <= rp.active_until_s)
    ]
    if dragons:
        dragon = min(dragons, key=lambda rp: dist(rp.position))
        add(
            MacroAction(
                kind="capture_resource",
                target_id=dragon.id,
                target_position=dragon.position,
            )
        )

    # 4. 守最近蓝方塔 / 推最近红方塔（目标动作，优先于填充用小野）
    blue_towers = [
        rp for rp in map_data.resource_points if rp.kind == "tower" and rp.side == "blue"
    ]
    if blue_towers:
        tower = min(blue_towers, key=lambda rp: dist(rp.position))
        add(
            MacroAction(kind="defend_tower", target_id=tower.id, target_position=tower.position)
        )
    red_towers = [
        rp for rp in map_data.resource_points if rp.kind == "tower" and rp.side == "red"
    ]
    if red_towers:
        tower = min(red_towers, key=lambda rp: dist(rp.position))
        add(
            MacroAction(kind="push_tower", target_id=tower.id, target_position=tower.position)
        )

    # 5. 小野填充：按距离补满候选
    resources = [
        rp
        for rp in map_data.resource_points
        if rp.position is not None and rp.kind in {"buff", "camp", "neutral"}
        and rp.id not in {"tyrant", "overlord", "storm_dragon"}
    ]
    resources.sort(key=lambda rp: dist(rp.position))
    for rp in resources:
        if len(candidates) >= max_candidates:
            break
        add(
            MacroAction(
                kind="capture_resource",
                target_id=rp.id,
                target_position=rp.position,
            )
        )

    return candidates[:max_candidates]
