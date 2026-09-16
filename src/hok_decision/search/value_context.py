"""上下文价值函数：阶段/英雄/塔级/经济 权重调节。

规则来源：
  data/S45/normalized/value_rules.json（由用户思路文档手工数值化）
  data/S45/normalized/value_weights.json（logistic 校准基础权重）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hok_decision.config import NORMALIZED_ROOT
from hok_decision.schema.models import MapData
from hok_decision.search.planner import SearchState, extract_features
from hok_decision.search.value import ValueWeights, load_weights

RULES_FILE = NORMALIZED_ROOT / "value_rules.json"
_ENAME_RE = re.compile(r"(\d+)$")

_DEFAULT_RULES: dict[str, Any] = {
    "phases": [
        {
            "name": "前期",
            "start_s": 0,
            "end_s": 240,
            "multipliers": {"hp": 1.0, "economy": 1.3, "resource": 1.2, "tower": 0.6, "kill": 0.4},
        },
        {
            "name": "中期",
            "start_s": 240,
            "end_s": 600,
            "multipliers": {"hp": 1.0, "economy": 1.0, "resource": 1.0, "tower": 1.3, "kill": 0.8},
        },
        {
            "name": "后期",
            "start_s": 600,
            "end_s": 900,
            "multipliers": {"hp": 1.2, "economy": 0.7, "resource": 0.7, "tower": 1.5, "kill": 1.4},
        },
        {
            "name": "大后期",
            "start_s": 900,
            "end_s": None,
            "multipliers": {"hp": 1.3, "economy": 0.5, "resource": 0.5, "tower": 1.8, "kill": 1.6},
        },
    ],
    "tower_tier_values": {"t1": 0.6, "t2": 0.9, "t3": 1.4, "crystal": 3.0},
    "role_economy_multiplier": {"1": 1.0, "2": 1.1, "3": 0.8, "4": 1.0, "5": 1.2, "6": 0.8},
    "role_kill_value": {"1": 1.0, "2": 1.3, "3": 0.6, "4": 1.3, "5": 1.5, "6": 0.6},
    "gold_scale": 500,
}


def _load_rules() -> dict[str, Any]:
    if not RULES_FILE.exists():
        return _DEFAULT_RULES
    try:
        return json.loads(RULES_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _DEFAULT_RULES


@dataclass(frozen=True)
class ContextualValue:
    """按阶段/英雄/塔级/经济调节的价值评估器。"""

    weights: ValueWeights
    rules: dict[str, Any]
    hero_types: dict[int, int]  # ename -> hero_type(职业)
    hero_attacks: dict[int, float]  # ename -> 物攻+法攻
    hero_mana: dict[int, float]  # ename -> 基础蓝量
    tower_tier_values: dict[str, float]
    tower_tiers: dict[str, str]  # tower_id -> t1/t2/t3

    @classmethod
    def load_defaults(
        cls,
        *,
        map_data: MapData | None = None,
        anchors: dict[str, dict] | None = None,
    ) -> "ContextualValue":
        """从正式数据文件构建（英雄类型/攻击、塔级按离己方水晶距离推断）。"""
        weights = load_weights()
        rules = _load_rules()
        hero_types, hero_attacks, hero_mana = _load_hero_tables()
        tiers = _assign_tower_tiers(map_data, anchors) if map_data else {}
        return cls(
            weights=weights,
            rules=rules,
            hero_types=hero_types,
            hero_attacks=hero_attacks,
            hero_mana=hero_mana,
            tower_tier_values=_numeric_table(rules.get("tower_tier_values", {})),
            tower_tiers=tiers,
        )

    def _curve_value(self, name: str, game_time: float) -> float:
        """从 weight_curves 分段线性插值，得到时间 t 的动态权重。"""
        curve = self.rules.get("weight_curves", {}).get(name)
        if not curve:
            return 1.0
        points = curve.get("points", [])
        if not points:
            return 1.0
        post = float(curve.get("post", points[-1][1]))
        if game_time <= float(points[0][0]):
            return float(points[0][1])
        for (t0, v0), (t1, v1) in zip(points, points[1:]):
            t0f, t1f = float(t0), float(t1)
            if t0f <= game_time <= t1f:
                return float(v0) + (float(v1) - float(v0)) * (game_time - t0f) / (t1f - t0f)
        return post

    def dynamic_weights(self, game_time: float) -> dict[str, float]:
        return {
            key: self._curve_value(key, game_time)
            for key in (
                "hp",
                "economy",
                "small_camp",
                "red_buff",
                "blue_buff",
                "tyrant",
                "overlord",
                "storm_dragon",
                "tower",
                "kill",
            )
        }

    def phase_multipliers(self, game_time: float) -> dict[str, float]:
        """兼容接口：返回当前时刻动态权重。"""
        return self.dynamic_weights(game_time)

    def _ename(self, unit_id: str) -> int | None:
        match = _ENAME_RE.search(unit_id)
        return int(match.group(1)) if match else None

    def _role(self, unit_id: str) -> int:
        ename = self._ename(unit_id)
        if ename is None:
            return 1
        role = self.hero_types.get(ename, 1)
        return int(role)

    def evaluate(self, state: SearchState) -> float:
        m = self.dynamic_weights(state.game_time)
        hp_diff, resource, _kill_diff = extract_features(state)

        # 英雄条件：血量差权重随敌方输出能力调整
        reds = [h for h in state.heroes.values() if h.side == "red" and h.alive]
        attacks = [
            self.hero_attacks.get(self._ename(h.unit_id) or 0, 200.0) for h in reds
        ]
        enemy_avg = sum(attacks) / len(attacks) if attacks else 200.0
        burst = max(-0.5, min(1.0, enemy_avg / 300.0 - 1.0))
        hp_scale = 1.0 + 0.5 * burst

        # 经济：我方职业需求 × 经济差 × (1 - 经济饱和度)
        blues = [h for h in state.heroes.values() if h.side == "blue"]
        economy_mult = self.rules.get("role_economy_multiplier", {}).get(
            str(self._role(blues[0].unit_id)) if blues else "1", 1.0
        )
        gold_blue = sum(state.gold.get(h.unit_id, 0.0) for h in blues)
        gold_red = sum(state.gold.get(h.unit_id, 0.0) for h in reds)
        economy_diff = (gold_blue - gold_red) / float(
            self.rules.get("gold_scale", 500)
        )
        time_norm = min(1.0, state.game_time / 900.0)
        avg_gold = (gold_blue + gold_red) / max(1.0, len(blues) + len(reds))
        econ_sat = min(1.0, max(time_norm, avg_gold / 12000.0))
        economy_active = max(0.0, 1.0 - econ_sat)

        # 资源拆分：小野 / 红buff / 蓝buff / 暴君 / 主宰 / 风暴龙王
        counts: dict[str, int] = {}
        for rid in state.captured_resources:
            dimension = self._resource_dimension(rid)
            counts[dimension] = counts.get(dimension, 0) + 1

        # buff 价值随己方阵容：红buff看普攻英雄，蓝buff看技能/蓝耗英雄
        resource_value = 0.0
        for dimension, count in counts.items():
            curve = m.get(dimension, 1.0)
            multiplier = 1.0
            if dimension == "small_camp":
                multiplier = economy_active
            elif dimension in {"red_buff", "blue_buff"}:
                multiplier = self._buff_team_multiplier(state, dimension)
            resource_value += self.weights.w_resource * curve * multiplier * count

        # 推塔价值：敌方塔损 - 我方塔损，按塔级加权
        tower_diff = self._tower_diff(state)

        # 击杀价值：按被杀英雄职业区分（切 C 更值钱）
        kill_values = self.rules.get("role_kill_value", {})
        red_death_value = sum(
            float(kill_values.get(str(self._role(h.unit_id)), 1.0))
            for h in state.heroes.values()
            if h.side == "red" and not h.alive
        )
        blue_death_value = sum(
            float(kill_values.get(str(self._role(h.unit_id)), 1.0))
            for h in state.heroes.values()
            if h.side == "blue" and not h.alive
        )

        score = (
            m.get("hp", 1.0) * hp_scale * self.weights.w_hp * hp_diff
            + m.get("economy", 1.0) * economy_active * economy_mult * self.weights.w_economy * economy_diff
            + resource_value
            + m.get("tower", 1.0) * self.weights.w_tower * tower_diff
            + m.get("kill", 1.0) * self.weights.w_kill * (red_death_value - blue_death_value)
            + self.weights.bias
        )
        return score

    @staticmethod
    def _resource_dimension(resource_id: str) -> str:
        """把捕获资源 id 映射到价值维度。"""
        if resource_id in {"tyrant", "overlord", "storm_dragon"}:
            return resource_id
        if "blue_buff" in resource_id:
            return "blue_buff"
        if "red_buff" in resource_id:
            return "red_buff"
        return "small_camp"

    def _buff_team_multiplier(self, state: SearchState, dimension: str) -> float:
        """红/蓝 Buff 对己方阵容的价值乘数。

        红buff：按职业表 + 特定英雄覆盖；蓝buff：职业表 × 蓝耗因子 + 覆盖。
        取全队最大值（buff 会交给收益最高的英雄）。
        """
        table = self.rules.get("buff_role_multipliers", {}).get(dimension, {})
        overrides = self.rules.get("hero_buff_overrides", {}).get(dimension, {})
        blues = [h for h in state.heroes.values() if h.side == "blue" and h.alive]
        values: list[float] = []
        for hero in blues:
            ename = self._ename(hero.unit_id)
            if ename is not None and str(ename) in overrides:
                values.append(float(overrides[str(ename)]))
                continue
            role_mult = float(table.get(str(self._role(hero.unit_id)), 1.0))
            if dimension == "blue_buff":
                mana = self._hero_mana(hero.unit_id)
                mana_factor = min(1.5, max(0.6, mana / 400.0))
                role_mult *= mana_factor
            values.append(role_mult)
        return max(values) if values else 1.0

    def _hero_mana(self, unit_id: str) -> float:
        """从营地数据取英雄基础蓝量（用于蓝buff收益判断）。"""
        ename = self._ename(unit_id)
        if ename is None:
            return 400.0
        mana = self.hero_mana.get(ename)
        return 400.0 if mana is None else mana

    def _tower_diff(self, state: SearchState) -> float:
        red_loss = 0.0
        blue_loss = 0.0
        for tower_id, hp_frac in state.tower_hp.items():
            tier = self.tower_tiers.get(tower_id, "t2")
            tier_value = self.tower_tier_values.get(tier, 0.9)
            damage = (1.0 - hp_frac) * tier_value
            if tower_id.startswith("tower_red"):
                red_loss += damage
            elif tower_id.startswith("tower_blue"):
                blue_loss += damage
        return red_loss - blue_loss


def _numeric_table(raw: dict[str, Any]) -> dict[str, float]:
    """只保留数值项，忽略 note 等说明字段。"""
    result: dict[str, float] = {}
    for key, value in raw.items():
        try:
            result[key] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _load_hero_tables() -> tuple[dict[int, int], dict[int, float], dict[int, float]]:
    hero_types: dict[int, int] = {}
    hero_attacks: dict[int, float] = {}
    hero_mana: dict[int, float] = {}
    try:
        heroes = json.loads(
            (NORMALIZED_ROOT / "heroes.json").read_text(encoding="utf-8")
        )["data"]
        for hero in heroes:
            ename = int(hero["ename"])
            hero_types[ename] = int(hero.get("hero_type", 1))
    except (json.JSONDecodeError, KeyError, OSError, FileNotFoundError):
        pass
    try:
        stats = json.loads(
            (NORMALIZED_ROOT / "hero_stats.json").read_text(encoding="utf-8")
        )["data"]
        for stat in stats:
            ename = int(stat["ename"])
            attack = float(stat.get("base_physical_attack") or 0.0) + float(
                stat.get("base_magic_attack") or 0.0
            )
            hero_attacks[ename] = attack
            mana = stat.get("base_mp")
            if mana is not None:
                hero_mana[ename] = float(mana)
    except (json.JSONDecodeError, KeyError, OSError, FileNotFoundError):
        pass
    return hero_types, hero_attacks, hero_mana


def _assign_tower_tiers(
    map_data: MapData, anchors: dict[str, dict]
) -> dict[str, str]:
    """塔级推断：按离己方水晶距离，远→t1(一塔)、中→t2(二塔)、近→t3(高地塔)。"""
    tiers: dict[str, str] = {}
    fountain = {
        "red": (anchors.get("fountain_red") or {}).get("position"),
        "blue": (anchors.get("fountain_blue") or {}).get("position"),
    }
    for side, tier_names in (("red", ["t1", "t2", "t3"]), ("blue", ["t1", "t2", "t3"])):
        towers = [
            rp
            for rp in map_data.resource_points
            if rp.kind == "tower" and rp.side == side and rp.position is not None
        ]
        base = fountain[side]
        if base is None:
            base = (0.0, 0.0)
        towers.sort(key=lambda rp: -((rp.position[0] - base[0]) ** 2 + (rp.position[1] - base[1]) ** 2))
        for index, rp in enumerate(towers):
            tier = tier_names[min(index // 3, 2)]
            tiers[rp.id] = tier
    return tiers
