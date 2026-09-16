"""装备/技能向量 → 英雄战力与模型权重特征。

items_vector / skills_vector 在这里作为"权重"进入模型：
  - 装备属性向量按出装求和，修正英雄面板（effective stats）；
  - 生成 4 维 loadout 特征（build_attack / build_defense / build_utility / skill_power），
    作为注意力二阶权重模型的局势上下文。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hok_decision.config import NORMALIZED_ROOT

LOADOUT_DIMS: list[str] = [
    "build_attack_power",
    "build_defense_power",
    "build_utility_power",
    "skill_power",
]

_ATTR_TO_FIELD: dict[str, str] = {
    "物理攻击": "physical_attack",
    "法术攻击": "magic_attack",
    "物理防御": "physical_defense",
    "法术防御": "magic_defense",
    "最大生命": "max_hp",
    "最大法力": "max_mp",
    "攻击速度": "attack_speed",
    "移动速度": "move_speed",
    "冷却缩减": "cooldown_reduction",
    "物理吸血": "physical_lifesteal",
    "法术吸血": "magic_lifesteal",
    "暴击率": "crit_rate",
    "暴击效果": "crit_effect",
}


def load_items_vector() -> dict[int, dict[str, Any]]:
    payload = json.loads(
        (NORMALIZED_ROOT / "items_vector.json").read_text(encoding="utf-8")
    )
    return {int(row["equipment_id"]): row for row in payload["data"]}


def load_hero_base_stats() -> dict[int, dict[str, float]]:
    payload = json.loads(
        (NORMALIZED_ROOT / "hero_stats.json").read_text(encoding="utf-8")
    )
    result: dict[int, dict[str, float]] = {}
    for row in payload["data"]:
        ename = int(row["ename"])
        result[ename] = {
            "hp": float(row.get("base_hp") or 0.0),
            "mp": float(row.get("base_mp") or 0.0),
            "physical_attack": float(row.get("base_physical_attack") or 0.0),
            "magic_attack": float(row.get("base_magic_attack") or 0.0),
            "physical_defense": float(row.get("base_physical_defense") or 0.0),
            "magic_defense": float(row.get("base_magic_defense") or 0.0),
            "attack_speed": float(row.get("base_attack_speed") or 0.0),
            "move_speed": float(row.get("move_speed") or 0.0),
            "crit_rate": float(row.get("crit_rate") or 0.0),
            "crit_effect": float(row.get("crit_effect") or 2.0),
        }
    return result


def effective_hero_stats(
    hero_ename: int, item_ids: list[int]
) -> dict[str, float]:
    """英雄基础属性 + 装备属性（按字段求和）。"""
    bases = load_hero_base_stats().get(hero_ename, {})
    stats: dict[str, float] = {
        "hp": bases.get("hp", 0.0),
        "mp": bases.get("mp", 0.0),
        "physical_attack": bases.get("physical_attack", 0.0),
        "magic_attack": bases.get("magic_attack", 0.0),
        "physical_defense": bases.get("physical_defense", 0.0),
        "magic_defense": bases.get("magic_defense", 0.0),
        "attack_speed": bases.get("attack_speed", 0.0),
        "move_speed": bases.get("move_speed", 0.0),
        "crit_rate": bases.get("crit_rate", 0.0),
        "crit_effect": bases.get("crit_effect", 2.0),
        "cooldown_reduction": 0.0,
        "physical_lifesteal": 0.0,
        "magic_lifesteal": 0.0,
    }
    items = load_items_vector()
    for item_id in item_ids:
        row = items.get(item_id)
        if row is None:
            continue
        for attr_name, value in row.get("attributes", {}).items():
            field = _ATTR_TO_FIELD.get(attr_name)
            if field:
                if attr_name in {"暴击率"}:
                    stats[field] = min(1.0, stats[field] + value / 100.0)
                else:
                    stats[field] += value
    return stats


def loadout_context_vector(hero_ename: int, item_ids: list[int]) -> list[float]:
    """4 维 loadout 特征（进入注意力二阶权重的 Q）。"""
    stats = effective_hero_stats(hero_ename, item_ids)
    items = load_items_vector()

    attack = (
        stats["physical_attack"] + stats["magic_attack"]
    ) / 400.0 + stats["crit_rate"]
    defense = (
        stats["physical_defense"] + stats["magic_defense"] + stats["hp"] / 10.0
    ) / 1200.0
    utility = (
        stats["cooldown_reduction"] / 40.0
        + stats["move_speed"] / 500.0
        + (stats["physical_lifesteal"] + stats["magic_lifesteal"]) / 50.0
        + min(1.0, sum(1 for i in item_ids if items.get(i) and items[i]["effects_text"]) / 6.0)
    )
    skill_power = _skill_power(hero_ename)
    return [
        round(min(2.0, attack), 4),
        round(min(2.0, defense), 4),
        round(min(2.0, utility), 4),
        round(min(2.0, skill_power), 4),
    ]


def _skill_power(hero_ename: int) -> float:
    payload = json.loads(
        (NORMALIZED_ROOT / "skills_vector.json").read_text(encoding="utf-8")
    )
    skills = [row for row in payload["data"] if int(row["ename"]) == hero_ename]
    if not skills:
        return 1.0
    max_damage = max(
        (row["damage"]["max"] or 0.0) for row in skills if row["damage"]["max"]
    )
    effect_count = max(sum(row["effect_vector"]) for row in skills)
    return 0.5 + max_damage / 1200.0 + min(1.0, effect_count / 10.0)
