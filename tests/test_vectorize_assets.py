"""装备/技能结构化向量测试。"""

from __future__ import annotations

import json

from hok_decision.config import NORMALIZED_ROOT


def _load(name: str) -> dict:
    return json.loads((NORMALIZED_ROOT / name).read_text(encoding="utf-8"))


def test_items_vector_coverage() -> None:
    data = _load("items_vector.json")
    assert data["count"] == 126
    assert len(data["attr_dims"]) == 20
    assert len(data["passive_dims"]) == 20
    assert all(len(row["attr_vector"]) == 20 for row in data["data"])
    assert all(len(row["passive_vector"]) == 20 for row in data["data"])
    assert data["coverage"]["with_effects_text"] >= 90
    assert data["coverage"]["effects_missing"] <= 40


def test_known_item_attributes_parsed() -> None:
    data = _load("items_vector.json")
    by_name = {row["item_name"]: row for row in data["data"]}
    assert by_name["铁剑"]["attributes"]["物理攻击"] == 20.0
    assert by_name["末世"]["attributes"]["物理攻击"] == 40.0
    assert by_name["末世"]["attributes"]["攻击速度"] == 20.0
    assert by_name["无尽战刃"]["attributes"]["暴击率"] == 20.0
    assert by_name["神速之靴"]["attributes"]["移动速度"] == 30.0
    assert "月之守护" in by_name["难知·月神"]["effects_text"]
    assert by_name["末世"]["effects_text"].startswith("唯一被动")


def test_skills_vector_coverage() -> None:
    data = _load("skills_vector.json")
    assert data["count"] == 538
    assert len(data["effect_dims"]) == 20
    assert data["coverage"]["with_cooldown"] == 400
    assert data["coverage"]["with_damage"] >= 530


def test_known_skill_parsed() -> None:
    data = _load("skills_vector.json")
    skill = next(
        row
        for row in data["data"]
        if row["ename"] == 105 and row["name"] == "爆裂冲撞"
    )
    assert skill["cooldown_min"] == 8.0
    assert skill["cooldown_max"] == 10.0
    assert skill["damage"]["base"] == 55.0
    assert skill["damage"]["growth"] == 55.0
    assert skill["effect_vector"][data["effect_dims"].index("伤害")] == 1.0
    assert skill["effect_vector"][data["effect_dims"].index("位移")] == 1.0
