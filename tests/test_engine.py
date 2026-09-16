"""模拟引擎测试：确定性、伤害结算与移动。"""

from __future__ import annotations

import pytest

from hok_decision.schema.models import DamageConfig
from hok_decision.simulator.engine import SimEngine
from hok_decision.simulator.units import Hero


def _duel_engine(defense: float = 602.0) -> SimEngine:
    engine = SimEngine(damage_config=DamageConfig(enemy_damage_mode="none"))
    blue = Hero(
        unit_id="blue_1",
        name="蓝方",
        side="blue",
        position=(0.0, 0.0),
        hp=1000.0,
        max_hp=1000.0,
        physical_attack=100.0,
        physical_defense=defense,
        attack_range=300.0,
        attack_interval=1.0,
        move_speed=360.0,
    )
    red = Hero(
        unit_id="red_1",
        name="红方",
        side="red",
        position=(0.0, 0.0),
        hp=1000.0,
        max_hp=1000.0,
        physical_attack=100.0,
        physical_defense=defense,
        crit_rate=0.5,
        crit_multiplier=2.0,
        attack_range=300.0,
        attack_interval=1.0,
        move_speed=360.0,
    )
    engine.add_unit(blue)
    engine.add_unit(red)
    return engine


def test_duel_deterministic_and_hand_checked() -> None:
    """602 防御 → 每次普攻 50 血。

    同一帧内蓝方先结算，第 20 次普攻击杀红方后，红方当帧不再出手，
    因此蓝方 20 次、红方 19 次，蓝方存活且剩 50 血。
    """
    engine = _duel_engine()
    events = engine.run_until(max_time=30.0, dt=0.1)
    attack_events = [e for e in events if e.kind == "auto_attack"]
    assert len(attack_events) == 39
    assert all(e.payload["damage"] == pytest.approx(50.0) for e in attack_events)
    assert sum(1 for e in attack_events if e.actor == "blue_1") == 20
    assert sum(1 for e in attack_events if e.actor == "red_1") == 19
    assert engine.units["blue_1"].alive is True
    assert engine.units["blue_1"].hp == pytest.approx(50.0)
    assert engine.units["red_1"].alive is False
    assert engine.units["red_1"].hp == 0.0


def test_duel_reproducible() -> None:
    first = _duel_engine()
    second = _duel_engine()
    first.run_until(max_time=15.0)
    second.run_until(max_time=15.0)
    assert [(e.t, e.kind, e.actor, e.target, e.payload) for e in first.events] == [
        (e.t, e.kind, e.actor, e.target, e.payload) for e in second.events
    ]


def test_enemy_expected_crit_mode_changes_outcome() -> None:
    engine = _duel_engine()
    engine.damage_config = DamageConfig(enemy_damage_mode="expected")
    events = engine.run_until(max_time=12.0)
    red_attacks = [
        e for e in events if e.kind == "auto_attack" and e.actor == "red_1"
    ]
    # 红方期望伤害 = 50 × (1 + 0.5 × 1) = 75
    assert red_attacks and all(e.payload["damage"] == pytest.approx(75.0) for e in red_attacks)


def test_movement_toward_enemy() -> None:
    engine = SimEngine()
    blue = Hero(
        unit_id="blue_1",
        name="蓝方",
        side="blue",
        position=(0.0, 0.0),
        physical_attack=10.0,
        attack_range=100.0,
        attack_interval=10.0,
        move_speed=360.0,
    )
    red = Hero(
        unit_id="red_1",
        name="红方",
        side="red",
        position=(1000.0, 0.0),
        physical_attack=10.0,
        attack_range=100.0,
        attack_interval=10.0,
    )
    engine.add_unit(blue)
    engine.add_unit(red)
    engine.step(1.0)
    assert blue.position[0] == pytest.approx(360.0)
    assert blue.position[1] == pytest.approx(0.0)


def test_skill_cooldowns_tick_when_no_enemy() -> None:
    engine = SimEngine()
    blue = Hero(
        unit_id="blue_1",
        name="蓝方",
        side="blue",
        skills={"s1": 5.0, "s2": 0.5},
    )
    engine.add_unit(blue)
    engine.step(1.0)
    assert blue.skills["s1"] == pytest.approx(4.0)
    assert blue.skills["s2"] == pytest.approx(0.0)
