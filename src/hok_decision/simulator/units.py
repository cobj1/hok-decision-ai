"""模拟单位：英雄/野怪/塔。战斗数值来自数据层，代码不含游戏数值。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Unit:
    """带位置与生命值的通用单位。"""

    unit_id: str
    name: str
    side: str = "neutral"
    position: tuple[float, float] = (0.0, 0.0)
    hp: float = 1000.0
    max_hp: float = 1000.0
    alive: bool = True

    def take_damage(self, damage: float) -> float:
        """结算伤害，返回实际扣血。"""
        if not self.alive:
            return 0.0
        actual = min(self.hp, max(0.0, damage))
        self.hp -= actual
        if self.hp <= 0.0:
            self.hp = 0.0
            self.alive = False
        return actual


@dataclass
class Hero(Unit):
    """英雄单位：攻击/防御/暴击等结算参数全部由调用方注入。"""

    physical_attack: float = 100.0
    magic_attack: float = 0.0
    physical_defense: float = 80.0
    magic_defense: float = 60.0
    crit_rate: float = 0.0
    crit_multiplier: float = 2.0
    attack_range: float = 300.0
    attack_interval: float = 1.0
    move_speed: float = 360.0
    attack_cooldown: float = 0.0
    skills: dict[str, float] = field(default_factory=dict)  # skill_id -> cd 剩余


@dataclass
class Monster(Unit):
    """野怪单位。"""

    camp_id: str | None = None


@dataclass
class Tower(Unit):
    """防御塔单位。"""

    physical_attack: float = 470.0
    attack_range: float = 900.0
    attack_interval: float = 1.0
    attack_cooldown: float = 0.0
