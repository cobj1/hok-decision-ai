"""伤害结算公式测试（与手算对照）。"""

from __future__ import annotations

import pytest

from hok_decision.schema.models import DamageConfig
from hok_decision.simulator.combat import (
    CombatError,
    attack_damage_enemy,
    attack_damage_friendly,
    compute_damage,
    effective_defense,
    mitigation,
)


def test_mitigation_baseline() -> None:
    cfg = DamageConfig()
    assert mitigation(0.0, config=cfg) == pytest.approx(0.0)
    assert mitigation(602.0, config=cfg) == pytest.approx(0.5)
    assert mitigation(100.0, config=cfg) == pytest.approx(100 / 702)


def test_effective_defense_penetration() -> None:
    assert effective_defense(100.0) == pytest.approx(100.0)
    assert effective_defense(100.0, fixed_pen=30.0) == pytest.approx(70.0)
    assert effective_defense(100.0, pct_pen=0.4) == pytest.approx(60.0)
    assert effective_defense(50.0, fixed_pen=100.0) == pytest.approx(0.0)


def test_friendly_damage_no_crit_lower_bound() -> None:
    # 100 攻击 vs 0 防御：100
    assert attack_damage_friendly(100.0, damage_type="physical", target_defense=0.0) == pytest.approx(
        100.0
    )
    # 100 攻击 vs 602 防御：50
    assert attack_damage_friendly(100.0, damage_type="physical", target_defense=602.0) == pytest.approx(
        50.0
    )


def test_enemy_expected_crit_damage() -> None:
    cfg = DamageConfig(enemy_damage_mode="expected")
    # 期望 = 100 × (1 + 0.5 × (2-1)) = 150
    assert attack_damage_enemy(
        100.0, damage_type="physical", target_defense=0.0, crit_rate=0.5, config=cfg
    ) == pytest.approx(150.0)
    cfg_max = DamageConfig(enemy_damage_mode="max")
    assert attack_damage_enemy(
        100.0, damage_type="physical", target_defense=0.0, crit_rate=0.5, config=cfg_max
    ) == pytest.approx(200.0)


def test_true_damage_ignores_defense() -> None:
    assert compute_damage(
        100.0,
        damage_type="true",
        target_defense=9999.0,
        crit_rate=0.0,
        crit_multiplier=2.0,
        crit_mode="none",
    ) == pytest.approx(100.0)


def test_invalid_crit_rate_raises() -> None:
    with pytest.raises(CombatError):
        compute_damage(
            10.0,
            damage_type="physical",
            target_defense=0.0,
            crit_rate=1.5,
            crit_multiplier=2.0,
            crit_mode="none",
        )
