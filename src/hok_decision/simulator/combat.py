"""伤害结算：保守/期望口径的确定性公式。

公式（公开资料，常数进配置）：
    实际防御 = (防御 - 固定穿透) × (1 - 百分比穿透)，下限 0
    免伤比 = 实际防御 / (实际防御 + 602)
    物理/法术伤害 = 原始伤害 × (1 - 免伤比)
    真实伤害 = 原始伤害
    我方暴击：默认关闭（伤害取下界）
    敌方暴击：none=不暴击 / expected=期望 / max=必暴击
"""

from __future__ import annotations

from typing import Literal

from hok_decision.schema.models import DamageConfig

DamageType = Literal["physical", "magic", "true"]


class CombatError(ValueError):
    """伤害计算参数非法。"""


def effective_defense(
    defense: float,
    *,
    fixed_pen: float = 0.0,
    pct_pen: float = 0.0,
) -> float:
    """计算穿透后的有效防御（下限 0）。"""
    if defense < 0 or fixed_pen < 0 or not 0.0 <= pct_pen <= 1.0:
        raise CombatError(
            f"防御/穿透参数非法: defense={defense}, fixed_pen={fixed_pen}, pct_pen={pct_pen}"
        )
    return max(0.0, (defense - fixed_pen) * (1.0 - pct_pen))


def mitigation(defense: float, *, config: DamageConfig) -> float:
    """免伤比 = 防御 / (防御 + 602)。"""
    return defense / (defense + config.defense_constant)


def crit_factor(
    crit_rate: float,
    crit_multiplier: float,
    *,
    mode: Literal["none", "expected", "max"],
) -> float:
    """暴击系数：none=0 加成 / expected=期望 / max=满暴击。"""
    if not 0.0 <= crit_rate <= 1.0:
        raise CombatError(f"暴击率非法: {crit_rate}")
    if crit_multiplier < 1.0:
        raise CombatError(f"暴击倍率非法: {crit_multiplier}")
    if mode == "none":
        return 1.0
    if mode == "expected":
        return 1.0 + crit_rate * (crit_multiplier - 1.0)
    if mode == "max":
        return crit_multiplier
    raise CombatError(f"未知暴击模式: {mode}")


def compute_damage(
    raw_damage: float,
    *,
    damage_type: DamageType,
    target_defense: float,
    crit_rate: float,
    crit_multiplier: float,
    crit_mode: Literal["none", "expected", "max"],
    fixed_pen: float = 0.0,
    pct_pen: float = 0.0,
    config: DamageConfig | None = None,
) -> float:
    """一次攻击的确定性伤害。

    我方调用时 crit_mode 固定为 "none"（下界）；
    敌方按 DamageConfig.enemy_damage_mode（none/expected/max）。
    """
    cfg = config or DamageConfig()
    if raw_damage < 0:
        raise CombatError(f"原始伤害非法: {raw_damage}")
    factor = crit_factor(crit_rate, crit_multiplier, mode=crit_mode)
    if damage_type == "true":
        return raw_damage * factor
    defense = effective_defense(target_defense, fixed_pen=fixed_pen, pct_pen=pct_pen)
    return raw_damage * factor * (1.0 - mitigation(defense, config=cfg))


def attack_damage_friendly(
    attack: float,
    *,
    damage_type: DamageType,
    target_defense: float,
    config: DamageConfig | None = None,
) -> float:
    """我方普攻伤害：暴击=0 的下界口径。"""
    cfg = config or DamageConfig()
    return compute_damage(
        attack,
        damage_type=damage_type,
        target_defense=target_defense,
        crit_rate=0.0,
        crit_multiplier=cfg.crit_default_multiplier,
        crit_mode="none",
        config=cfg,
    )


def attack_damage_enemy(
    attack: float,
    *,
    damage_type: DamageType,
    target_defense: float,
    crit_rate: float,
    crit_multiplier: float | None = None,
    config: DamageConfig | None = None,
) -> float:
    """敌方普攻伤害：按配置的 none/expected/max 口径。"""
    cfg = config or DamageConfig()
    return compute_damage(
        attack,
        damage_type=damage_type,
        target_defense=target_defense,
        crit_rate=crit_rate,
        crit_multiplier=crit_multiplier or cfg.crit_default_multiplier,
        crit_mode=cfg.enemy_damage_mode,
        config=cfg,
    )
