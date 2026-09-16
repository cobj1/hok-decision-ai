"""确定性模拟引擎（P0 骨架）。

当前支持：单位移动、普攻循环、伤害结算（保守/期望口径）、
技能冷却倒计时与死亡判定。所有随机项已按配置消除，结果可复现。
后续接入：资源点刷新调度、兵线、塔仇恨、技能模板。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from hok_decision.schema.models import DamageConfig
from hok_decision.simulator.combat import (
    DamageType,
    attack_damage_enemy,
    attack_damage_friendly,
)
from hok_decision.simulator.resources import ResourceScheduler
from hok_decision.simulator.units import Hero, Unit


@dataclass
class SimEvent:
    """一帧内发生的事件，用于断言与调试。"""

    t: float
    kind: str
    actor: str
    target: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimEngine:
    """确定性步进模拟器。"""

    damage_config: DamageConfig = field(default_factory=DamageConfig)
    units: dict[str, Unit] = field(default_factory=dict)
    resource_scheduler: ResourceScheduler | None = None
    game_time: float = 0.0
    events: list[SimEvent] = field(default_factory=list)
    step_count: int = 0

    def attach_resources(self, scheduler: ResourceScheduler) -> None:
        self.resource_scheduler = scheduler

    def add_unit(self, unit: Unit) -> None:
        if unit.unit_id in self.units:
            raise ValueError(f"单位 ID 重复: {unit.unit_id}")
        self.units[unit.unit_id] = unit

    def alive_units(self) -> list[Unit]:
        return [u for u in self.units.values() if u.alive]

    def enemies_of(self, unit: Unit) -> list[Unit]:
        return [u for u in self.alive_units() if u.side != unit.side]

    @staticmethod
    def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def nearest_enemy(self, hero: Hero) -> Hero | None:
        candidates = [u for u in self.enemies_of(hero) if isinstance(u, Hero)]
        if not candidates:
            return None
        return min(candidates, key=lambda u: self.distance(hero.position, u.position))

    def move_toward(self, unit: Unit, target: tuple[float, float], dt: float) -> None:
        """向目标点移动（速度取英雄移速，非英雄保持静止）。"""
        if not isinstance(unit, Hero):
            return
        dx = target[0] - unit.position[0]
        dy = target[1] - unit.position[1]
        dist = math.hypot(dx, dy)
        if dist <= 1e-6:
            return
        step = min(dist, unit.move_speed * dt)
        unit.position = (
            unit.position[0] + dx / dist * step,
            unit.position[1] + dy / dist * step,
        )

    def _try_auto_attack(self, hero: Hero) -> None:
        """普攻：范围内有敌人且冷却结束时结算一次伤害。"""
        hero.attack_cooldown = max(0.0, hero.attack_cooldown)
        if hero.attack_cooldown > 0.0 or hero.physical_attack <= 0.0:
            return
        target = self.nearest_enemy(hero)
        if target is None:
            return
        if self.distance(hero.position, target.position) > hero.attack_range:
            return
        damage = self._hero_vs_hero_damage(hero, target)
        dealt = target.take_damage(damage)
        hero.attack_cooldown = hero.attack_interval
        self.events.append(
            SimEvent(
                t=self.game_time,
                kind="auto_attack",
                actor=hero.unit_id,
                target=target.unit_id,
                payload={"damage": round(dealt, 4), "target_hp": round(target.hp, 4)},
            )
        )

    def _hero_vs_hero_damage(self, attacker: Hero, target: Hero) -> float:
        """双方英雄普攻结算：攻方是我方→下界；是敌方→配置口径。"""
        damage_type: DamageType = (
            "magic" if attacker.magic_attack > attacker.physical_attack else "physical"
        )
        attack = max(attacker.magic_attack, attacker.physical_attack)
        target_defense = (
            target.magic_defense if damage_type == "magic" else target.physical_defense
        )
        if attacker.side == "blue":
            return attack_damage_friendly(
                attack,
                damage_type=damage_type,
                target_defense=target_defense,
                config=self.damage_config,
            )
        return attack_damage_enemy(
            attack,
            damage_type=damage_type,
            target_defense=target_defense,
            crit_rate=attacker.crit_rate,
            crit_multiplier=attacker.crit_multiplier,
            config=self.damage_config,
        )

    def step(self, dt: float) -> list[SimEvent]:
        """推进 dt 秒。返回本步新增事件。"""
        if dt <= 0:
            raise ValueError(f"dt 必须为正: {dt}")
        start = len(self.events)
        for unit in list(self.units.values()):
            if not unit.alive:
                continue
            if isinstance(unit, Hero):
                target = self.nearest_enemy(unit)
                if target is not None and self.distance(
                    unit.position, target.position
                ) > unit.attack_range:
                    self.move_toward(unit, target.position, dt)
                self._try_auto_attack(unit)
                unit.attack_cooldown = max(0.0, unit.attack_cooldown - dt)
                for skill_id in unit.skills:
                    unit.skills[skill_id] = max(0.0, unit.skills[skill_id] - dt)
        self.game_time += dt
        if self.resource_scheduler is not None:
            self.resource_scheduler.tick(self.game_time)
        self.step_count += 1
        return self.events[start:]

    def run_until(self, *, max_time: float, dt: float = 0.1) -> list[SimEvent]:
        """推进至游戏结束或 max_time，返回全部事件。

        若存在对战单位，则任一方全灭时提前结束；纯资源模拟（无单位）时
        会一直推进到 max_time。
        """
        while self.game_time < max_time:
            if self.units:
                alive_blue = any(
                    u.side == "blue" and u.alive for u in self.units.values()
                )
                alive_red = any(
                    u.side == "red" and u.alive for u in self.units.values()
                )
                if not (alive_blue and alive_red):
                    break
            self.step(min(dt, max_time - self.game_time))
        return self.events
