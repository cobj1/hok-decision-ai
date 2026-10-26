"""P2 搜索骨架：确定性 rollout、手写末端价值与束搜索。"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from hok_decision.schema.models import DamageConfig, MapData
from hok_decision.search.actions import MacroAction, generate_candidates
from hok_decision.search.value import ValueWeights, load_weights
from hok_decision.simulator.combat import (
    attack_damage_enemy,
    attack_damage_friendly,
)
from hok_decision.simulator.units import Hero

_CAPTURE_RANGE = 40.0
_CONTEST_RANGE = 120.0


@dataclass
class SearchState:
    """搜索期状态快照。map_data 与 anchors 由 planner 持有，不复制。"""

    heroes: dict[str, Hero] = field(default_factory=dict)
    game_time: float = 0.0
    captured_resources: set[str] = field(default_factory=set)
    gold: dict[str, float] = field(default_factory=dict)  # hero_id -> 经济
    tower_hp: dict[str, float] = field(default_factory=dict)  # tower_id -> hp 比例(0~1)


@dataclass
class BeamNode:
    actions: tuple[MacroAction, ...]
    state: SearchState
    value: float
    decision_output: Any = None


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    """平方距离。仅用于**比较**（与半径比较、取最近）时等价且省掉 sqrt。

    排序/取 min 用平方距离与原距离严格同序（单调变换），阈值比较两边都平方即可，
    因此不改变任何挑选结果；唯一差异是浮点舍入在"恰好等于半径"的边界上，
    实测不可触发（见 scripts/refactor_gate.py 的决策 digest 校验）。
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


# map_data -> [(point_id, (x, y))]，只含可占领且带坐标的点。
# `_capture_resources` 每次 rollout 都会调用（实测占 beam_search 时间 ~50%），
# 而 map_data 在一次搜索内是不变的，过滤结果必须缓存，否则每 tick 重扫 30+ 个点。
_POI_CACHE: dict[int, list[tuple[str, tuple[float, float]]]] = {}


def _capture_points(map_data: MapData) -> list[tuple[str, tuple[float, float]]]:
    key = id(map_data)
    cached = _POI_CACHE.get(key)
    if cached is not None:
        return cached
    points = [(rp.id, rp.position) for rp in map_data.resource_points
              if rp.position is not None and rp.kind in {"buff", "camp", "neutral"}]
    _POI_CACHE[key] = points
    return points


def _nearest_enemy(state: SearchState, hero: Hero) -> Hero | None:
    enemies = [h for h in state.heroes.values() if h.side != hero.side and h.alive]
    if not enemies:
        return None
    return min(enemies, key=lambda h: _dist2(hero.position, h.position))


def _nearest_blue(state: SearchState, red: Hero) -> Hero | None:
    blues = [h for h in state.heroes.values() if h.side == "blue" and h.alive]
    if not blues:
        return None
    return min(blues, key=lambda h: _dist2(red.position, h.position))


def _move_toward(hero: Hero, target: tuple[float, float], dt: float) -> None:
    dx = target[0] - hero.position[0]
    dy = target[1] - hero.position[1]
    dist = (dx * dx + dy * dy) ** 0.5
    if dist <= 1e-6:
        return
    step = min(dist, hero.move_speed * dt)
    hero.position = (
        hero.position[0] + dx / dist * step,
        hero.position[1] + dy / dist * step,
    )


def _hero_vs_hero_damage(attacker: Hero, target: Hero, cfg: DamageConfig) -> float:
    """普攻伤害：蓝方为下界（无暴击），红方按期望/必暴。"""
    if attacker.side == "blue":
        return attack_damage_friendly(
            attacker.physical_attack,
            damage_type="physical",
            target_defense=target.physical_defense,
            config=cfg,
        )
    return attack_damage_enemy(
        attacker.physical_attack,
        damage_type="physical",
        target_defense=target.physical_defense,
        crit_rate=attacker.crit_rate,
        crit_multiplier=attacker.crit_multiplier,
        config=cfg,
    )


def _capture_resources(state: SearchState, map_data: MapData) -> None:
    blues = [h for h in state.heroes.values() if h.side == "blue" and h.alive]
    reds = [h for h in state.heroes.values() if h.side == "red" and h.alive]
    if not blues:
        return
    cap2 = _CAPTURE_RANGE * _CAPTURE_RANGE
    con2 = _CONTEST_RANGE * _CONTEST_RANGE
    captured = state.captured_resources
    for point_id, position in _capture_points(map_data):
        if point_id in captured:
            continue
        blue_near = any(_dist2(h.position, position) <= cap2 for h in blues)
        if not blue_near:
            continue                     # 先判蓝方: 大部分点蓝方都够不到, 省掉红方那一轮
        red_near = any(_dist2(h.position, position) <= con2 for h in reds)
        if not red_near:
            captured.add(point_id)


def rollout_decision(
    state: SearchState,
    blue_action: MacroAction,
    *,
    map_data: MapData,
    duration_s: float,
    damage_config: DamageConfig | None = None,
    dt: float = 0.25,
) -> None:
    """按宏观动作推演 duration_s 秒。蓝方执行动作，红方追击最近蓝方。"""
    cfg = damage_config or DamageConfig()
    ticks = max(1, int(duration_s / dt))
    for _ in range(ticks):
        for hero in list(state.heroes.values()):
            if not hero.alive:
                continue
            if hero.side == "blue":
                target_pos = blue_action.target_position
            else:
                nearest_blue = _nearest_blue(state, hero)
                target_pos = nearest_blue.position if nearest_blue else None

            enemy = _nearest_enemy(state, hero)
            if (
                enemy is not None
                and hero.attack_cooldown <= 0.0
                and _distance(hero.position, enemy.position) <= hero.attack_range
            ):
                damage = _hero_vs_hero_damage(hero, enemy, cfg)
                enemy.take_damage(damage)
                hero.attack_cooldown = hero.attack_interval
            elif (
                blue_action.kind == "push_tower"
                and blue_action.target_id is not None
                and blue_action.target_id in state.tower_hp
                and hero.attack_cooldown <= 0.0
                and _distance(hero.position, blue_action.target_position or target_pos) <= hero.attack_range
            ):
                # 推塔动作：普攻对塔造成伤害（塔按约 8000 血折算 0~1）
                state.tower_hp[blue_action.target_id] = max(
                    0.0,
                    state.tower_hp[blue_action.target_id] - hero.physical_attack / 8000.0,
                )
                hero.attack_cooldown = hero.attack_interval
            elif target_pos is not None:
                _move_toward(hero, target_pos, dt)

            hero.attack_cooldown = max(0.0, hero.attack_cooldown - dt)
            for skill_id in hero.skills:
                hero.skills[skill_id] = max(0.0, hero.skills[skill_id] - dt)

        _capture_resources(state, map_data)
        state.game_time += dt


def extract_features(state: SearchState) -> tuple[float, float, float]:
    """从状态提取价值特征：hp_diff, resource, kill_diff。"""
    blues = [h for h in state.heroes.values() if h.side == "blue"]
    reds = [h for h in state.heroes.values() if h.side == "red"]

    hp_diff = sum(h.hp / h.max_hp for h in blues if h.max_hp > 0) - sum(
        h.hp / h.max_hp for h in reds if h.max_hp > 0
    )
    resource = float(len(state.captured_resources))
    kill_diff = float(
        sum(1 for h in reds if not h.alive) - sum(1 for h in blues if not h.alive)
    )
    return hp_diff, resource, kill_diff


def terminal_value(
    state: SearchState, *, weights: ValueWeights | None = None
) -> float:
    """末端价值：可校准的线性价值函数。"""
    w = weights or load_weights()
    hp_diff, resource, kill_diff = extract_features(state)
    return w.score(hp_diff, resource, kill_diff)


def beam_search(
    initial_state: SearchState,
    *,
    map_data: MapData,
    anchors: dict[str, dict],
    horizon_s: float = 16.0,
    beam_width: int = 5,
    action_duration_s: float = 4.0,
    max_candidates: int = 8,
    damage_config: DamageConfig | None = None,
    value_context: Any | None = None,
    with_decision_output: bool = False,
) -> BeamNode | None:
    """束搜索：在 10~20s 视界内选近似最优宏观动作序列。

    value_context 可选：传入 ContextualValue 时按阶段/英雄/塔级/经济
    调节价值；不传则使用纯 logistic 校准的线性价值。
    """
    cfg = damage_config or DamageConfig()
    weights = load_weights()
    depth = max(1, int(horizon_s / action_duration_s))
    beam: list[BeamNode] = [BeamNode((), copy.deepcopy(initial_state), 0.0)]

    for _ in range(depth):
        expanded: list[BeamNode] = []
        for node in beam:
            candidates = generate_candidates(
                node.state, map_data, anchors, max_candidates=max_candidates
            )
            for action in candidates:
                state = copy.deepcopy(node.state)
                rollout_decision(
                    state,
                    action,
                    map_data=map_data,
                    duration_s=action_duration_s,
                    damage_config=cfg,
                )
                if value_context is not None:
                    try:
                        value = value_context.evaluate(state, action=action)
                    except TypeError:
                        value = value_context.evaluate(state)
                else:
                    value = terminal_value(state, weights=weights)
                expanded.append(BeamNode(node.actions + (action,), state, value))
        if not expanded:
            break
        expanded.sort(key=lambda n: n.value, reverse=True)
        beam = expanded[:beam_width]

    best = beam[0] if beam else None
    if best is not None and with_decision_output:
        # 局部导入避免与 decision_output -> planner 的循环依赖
        from hok_decision.search.decision_output import build_decision_output

        target_position = best.actions[0].target_position if best.actions else None
        description = "；".join(a.label for a in best.actions)
        best.decision_output = build_decision_output(
            initial_state=initial_state,
            final_state=best.state,
            target_position=target_position,
            self_side="blue",
            description=description,
        )
    return best
