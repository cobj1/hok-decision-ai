"""出装硬规则测试（辅助判定 = 局内手动角色输入）。"""

from __future__ import annotations

import json

from hok_decision.config import NORMALIZED_ROOT
from hok_decision.rules.hard_rules import PurchaseRuleEngine, ROLE_OPTIONS


def _engine() -> PurchaseRuleEngine:
    return PurchaseRuleEngine.load_defaults()


def _hero_of_type(hero_type: int) -> int:
    payload = json.loads(
        (NORMALIZED_ROOT / "heroes.json").read_text(encoding="utf-8")
    )
    return next(h["ename"] for h in payload["data"] if h["hero_type"] == hero_type)


def _item_id(name: str) -> int:
    payload = json.loads(
        (NORMALIZED_ROOT / "items_vector.json").read_text(encoding="utf-8")
    )
    return next(r["equipment_id"] for r in payload["data"] if r["item_name"] == name)


def test_jungle_item_requires_smite() -> None:
    engine = _engine()
    jungle_id = _item_id("狩猎宽刃")
    assert engine.category_of(jungle_id) == "jungle"
    ok, reason = engine.can_buy(
        item_id=jungle_id, summoner_spell_names=["闪现"], role_input="打野"
    )
    assert ok is False and "惩击" in reason
    ok, _ = engine.can_buy(
        item_id=jungle_id, summoner_spell_names=["闪现", "惩击"], role_input="打野"
    )
    assert ok is True


def test_support_item_requires_manual_support_role() -> None:
    engine = _engine()
    support_id = _item_id("学识宝石")
    assert engine.category_of(support_id) == "support"
    for role in ("辅助", "游走"):
        ok, _ = engine.can_buy(
            item_id=support_id, summoner_spell_names=["闪现"], role_input=role
        )
        assert ok is True, role
    for role in ("打野", "中路", "对抗路", "发育路"):
        ok, reason = engine.can_buy(
            item_id=support_id, summoner_spell_names=["闪现"], role_input=role
        )
        assert ok is False and "辅助" in reason, role


def test_normal_item_always_allowed() -> None:
    engine = _engine()
    iron_id = _item_id("铁剑")
    assert engine.category_of(iron_id) == "normal"
    ok, _ = engine.can_buy(
        item_id=iron_id, summoner_spell_names=[], role_input="中路"
    )
    assert ok is True


def test_filter_item_candidates_masks_before_model() -> None:
    engine = _engine()
    allowed, rejected = engine.filter_item_candidates(
        item_ids=[_item_id("铁剑"), _item_id("狩猎宽刃"), _item_id("学识宝石")],
        summoner_spell_names=["闪现"],
        role_input="中路",
    )
    assert allowed == [_item_id("铁剑")]
    assert len(rejected) == 2
    # 换成带惩击的辅助角色，则全部合法
    allowed2, rejected2 = engine.filter_item_candidates(
        item_ids=[_item_id("铁剑"), _item_id("狩猎宽刃"), _item_id("学识宝石")],
        summoner_spell_names=["闪现", "惩击"],
        role_input="辅助",
    )
    assert allowed2 == [_item_id("铁剑"), _item_id("狩猎宽刃"), _item_id("学识宝石")]
    assert rejected2 == []


def test_role_input_config_schema() -> None:
    # 只校验角色选项与加载函数，不写正式配置
    assert set(SUPPORT_ROLES := {"游走", "辅助"}) <= set(ROLE_OPTIONS)
