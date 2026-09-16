"""上下文价值函数测试：阶段/英雄/塔级/经济 权重。"""

from __future__ import annotations

import json

import pytest

from hok_decision.config import NORMALIZED_ROOT
from hok_decision.search.planner import SearchState
from hok_decision.search.value_context import ContextualValue
from hok_decision.simulator.map import load_map, load_terrain_anchors
from hok_decision.simulator.units import Hero


def _heroes() -> dict[str, Hero]:
    blue = Hero(
        unit_id="blue_105",
        name="廉颇",
        side="blue",
        position=(100.0, 100.0),
        hp=1000.0,
        max_hp=1000.0,
        physical_attack=100.0,
        physical_defense=100.0,
    )
    red = Hero(
        unit_id="red_105",
        name="廉颇",
        side="red",
        position=(300.0, 300.0),
        hp=1000.0,
        max_hp=1000.0,
        physical_attack=100.0,
        physical_defense=100.0,
    )
    return {"blue_105": blue, "red_105": red}


def _state(game_time: float = 0.0) -> SearchState:
    return SearchState(heroes=_heroes(), game_time=game_time)


def _context() -> ContextualValue:
    return ContextualValue.load_defaults(
        map_data=load_map(), anchors=load_terrain_anchors()
    )


def test_dynamic_weights_follow_time_axis() -> None:
    ctx = _context()
    assert ctx.dynamic_weights(0)["economy"] == 1.3
    assert ctx.dynamic_weights(300)["tower"] == pytest.approx(1.0667, abs=1e-3)
    assert ctx.dynamic_weights(700)["kill"] == pytest.approx(1.4667, abs=1e-3)
    assert ctx.dynamic_weights(1000)["tower"] == 1.8
    assert ctx.dynamic_weights(950)["economy"] == 0.0  # 15 分钟后经济饱和


def test_economy_value_zero_after_saturation() -> None:
    ctx = _context()
    state = _state(game_time=950.0)
    state.gold = {"blue_105": 5000.0, "red_105": 0.0}
    with_gold = ctx.evaluate(state)
    state.gold = {"blue_105": 0.0, "red_105": 0.0}
    without_gold = ctx.evaluate(state)
    assert with_gold == without_gold  # 经济权重已归零，经济差不影响价值


def _hero_by_type(hero_type: int, side: str, *, hp: float = 1000.0) -> Hero:
    heroes = json.loads(
        (NORMALIZED_ROOT / "heroes.json").read_text(encoding="utf-8")
    )["data"]
    item = next(h for h in heroes if h["hero_type"] == hero_type)
    ename = item["ename"]
    return Hero(
        unit_id=f"{side}_{ename}",
        name=item["cname"],
        side=side,
        position=(100.0, 100.0),
        hp=hp,
        max_hp=hp,
        physical_attack=100.0,
        physical_defense=100.0,
    )


def test_buff_value_depends_on_team_composition() -> None:
    ctx = _context()
    # 射手阵容：红buff > 蓝buff
    state_marksman = _state(game_time=240.0)
    state_marksman.heroes = {
        "blue_carry": _hero_by_type(5, "blue"),
        "red_105": _heroes()["red_105"],
    }
    state_marksman.captured_resources = {"red_buff"}
    value_red = ctx.evaluate(state_marksman)
    state_marksman.captured_resources = {"blue_buff"}
    value_blue = ctx.evaluate(state_marksman)
    assert value_red > value_blue

    # 法师阵容：蓝buff > 红buff
    state_mage = _state(game_time=240.0)
    state_mage.heroes = {
        "blue_mage": _hero_by_type(2, "blue"),
        "red_105": _heroes()["red_105"],
    }
    state_mage.captured_resources = {"red_buff"}
    value_red = ctx.evaluate(state_mage)
    state_mage.captured_resources = {"blue_buff"}
    value_blue = ctx.evaluate(state_mage)
    assert value_blue > value_red


def test_dragon_types_have_different_values() -> None:
    ctx = _context()
    # 700s（10分钟后）：主宰（兵线/推塔）> 暴君（伤害）
    state = _state(game_time=700.0)
    state.captured_resources = {"tyrant"}
    value_tyrant = ctx.evaluate(state)
    state.captured_resources = {"overlord"}
    value_overlord = ctx.evaluate(state)
    assert value_overlord > value_tyrant
    # 风暴龙王（20分钟后）价值最高
    state.game_time = 1250.0
    state.captured_resources = {"storm_dragon"}
    value_storm = ctx.evaluate(state)
    assert value_storm > value_overlord


def test_tower_tiers_assigned_all_18() -> None:
    ctx = _context()
    assert len(ctx.tower_tiers) == 18
    assert set(ctx.tower_tiers.values()) == {"t1", "t2", "t3"}


def test_tower_damage_increases_value_and_high_tier_worth_more() -> None:
    ctx = _context()
    baseline = ctx.evaluate(_state())
    state = _state()
    state.tower_hp = {tower_id: 1.0 for tower_id in ctx.tower_tiers}
    # 摧毁一座红方高地塔(t3) > 摧毁一座红方一塔(t1)
    t1 = next(tid for tid, tier in ctx.tower_tiers.items() if tier == "t1" and tid.startswith("tower_red"))
    t3 = next(tid for tid, tier in ctx.tower_tiers.items() if tier == "t3" and tid.startswith("tower_red"))
    state.tower_hp[t1] = 0.0
    value_t1 = ctx.evaluate(state)
    state.tower_hp[t1] = 1.0
    state.tower_hp[t3] = 0.0
    value_t3 = ctx.evaluate(state)
    assert value_t1 > baseline
    assert value_t3 > value_t1


def test_economy_and_kill_value_hero_conditioned() -> None:
    ctx = _context()
    state = _state(game_time=300.0)  # 中期，击杀/推塔权重提升
    state.gold = {"blue_105": 600.0, "red_105": 100.0}
    base = ctx.evaluate(state)

    # 击杀敌方高价值目标（射手）比击杀坦克更值钱
    heroes = json.loads(
        (NORMALIZED_ROOT / "heroes.json").read_text(encoding="utf-8")
    )["data"]
    carry = next(h for h in heroes if h["hero_type"] == 5)["ename"]
    tank = next(h for h in heroes if h["hero_type"] == 3)["ename"]

    state_carry = _state(game_time=300.0)
    state_carry.gold = dict(state.gold)
    state_carry.heroes = {
        "blue_105": _heroes()["blue_105"],
        f"red_{carry}": Hero(
            unit_id=f"red_{carry}",
            name="敌方",
            side="red",
            position=(300.0, 300.0),
            hp=1000.0,
            max_hp=1000.0,
            physical_attack=100.0,
            physical_defense=100.0,
            alive=False,
        ),
    }
    value_carry = ctx.evaluate(state_carry)

    state_tank = _state(game_time=300.0)
    state_tank.gold = dict(state.gold)
    state_tank.heroes = {
        "blue_105": _heroes()["blue_105"],
        f"red_{tank}": Hero(
            unit_id=f"red_{tank}",
            name="敌方",
            side="red",
            position=(300.0, 300.0),
            hp=1000.0,
            max_hp=1000.0,
            physical_attack=100.0,
            physical_defense=100.0,
            alive=False,
        ),
    }
    value_tank = ctx.evaluate(state_tank)
    assert value_carry > value_tank
