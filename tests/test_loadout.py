"""装备/技能向量作为模型权重特征的测试。"""

from __future__ import annotations

from hok_decision.search.loadout import (
    LOADOUT_DIMS,
    effective_hero_stats,
    loadout_context_vector,
)


def _item_id(name: str) -> int:
    import json

    from hok_decision.config import NORMALIZED_ROOT

    payload = json.loads(
        (NORMALIZED_ROOT / "items_vector.json").read_text(encoding="utf-8")
    )
    return next(r["equipment_id"] for r in payload["data"] if r["item_name"] == name)


def test_equipment_attributes_apply_to_hero() -> None:
    base = effective_hero_stats(105, [])
    with_iron = effective_hero_stats(105, [_item_id("铁剑")])
    assert with_iron["physical_attack"] == base["physical_attack"] + 20.0


def test_loadout_context_vector_shape_and_effect() -> None:
    no_item = loadout_context_vector(105, [])
    with_iron = loadout_context_vector(105, [_item_id("铁剑")])
    assert len(LOADOUT_DIMS) == 4
    assert len(no_item) == 4 and len(with_iron) == 4
    assert with_iron[0] > no_item[0]  # build_attack_power 随装备提升
