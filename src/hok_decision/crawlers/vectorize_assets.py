"""装备与技能的结构化 + 向量化。

输入：
  data/S45/normalized/items.json（官方装备表，des1 含属性/少数被动）
  data/S45/normalized/hero_skills.json（营地技能，含 skillData1-5 数值成长）
输出：
  data/S45/normalized/items_vector.json
  data/S45/normalized/skills_vector.json

说明：
- 装备被动：官方 item.json 只有极少数装备带被动文本；其余 passive 为空并在
  coverage 中标记 pending_source，待营地接口/公告数据补充后重跑。
- 向量为"数值属性 + 效果类型 one-hot"，可直接作为后续模型的输入特征。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hok_decision.config import GAME_VERSION, NORMALIZED_ROOT

_TAG_RE = re.compile(r"<[^>]+>")
_ATTR_RE = re.compile(r"\+([0-9.]+)%?\s*([\u4e00-\u9fa5]+)")

ITEM_ATTR_DIMS: list[str] = [
    "物理攻击", "法术攻击", "攻击速度", "暴击率", "暴击效果",
    "物理防御", "法术防御", "最大生命", "最大法力", "移动速度",
    "冷却缩减", "物理吸血", "法术吸血", "物理穿透", "法术穿透",
    "每5秒回血", "每5秒回蓝", "韧性", "护盾", "攻击范围",
]

ITEM_PASSIVE_DIMS: list[str] = [
    "减速", "灼烧", "强化普攻", "护盾", "回血", "移速", "冷却",
    "穿透", "复活", "无敌", "视野", "金币", "光环", "免伤",
    "反伤", "控制", "伤害加成", "攻速", "法强", "生命",
]

SKILL_EFFECT_DIMS: list[str] = [
    "伤害", "物理伤害", "法术伤害", "真实伤害", "控制", "减速",
    "眩晕", "击飞", "击退", "沉默", "位移", "回复", "护盾",
    "增益", "减益", "霸体", "免伤", "隐身", "攻速", "冷却",
]


def _legacy_item_ids_by_type(item_types: set[int]) -> set[int]:
    """从旧官方 item.json 按 item_type 提取打野/辅助装备 id。"""
    path = NORMALIZED_ROOT / "items.json"
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(item["item_id"]) for item in payload["data"] if item.get("item_type") in item_types
    }


_JUNGLE_ITEM_IDS = _legacy_item_ids_by_type({5})
_SUPPORT_ITEM_IDS = _legacy_item_ids_by_type({6, 7})


def _strip_html(text: str) -> str:
    return _TAG_RE.sub("", text or "").strip()


def _parse_attr_text(text: str) -> dict[str, float]:
    """解析 '+20物理攻击 / +10%攻击速度' 形式的属性文本。"""
    result: dict[str, float] = {}
    for match in _ATTR_RE.finditer(_strip_html(text)):
        value = float(match.group(1))
        name = match.group(2)
        if name in ITEM_ATTR_DIMS:
            result[name] = result.get(name, 0.0) + value
    return result


def _attr_vector(attrs: dict[str, float]) -> list[float]:
    return [round(attrs.get(dim, 0.0), 4) for dim in ITEM_ATTR_DIMS]


def _keyword_vector(text: str, dims: list[str]) -> list[float]:
    text = text or ""
    return [1.0 if keyword in text else 0.0 for keyword in dims]


def build_item_vectors() -> dict[str, Any]:
    """优先使用官方游戏内装备详情 equip_details.json；回退官方 item.json。"""
    detail_path = NORMALIZED_ROOT / "equip_details.json"
    if detail_path.exists():
        payload = json.loads(detail_path.read_text(encoding="utf-8"))
        # 旧官方 item.json 的基础属性兜底
        fallback_attrs: dict[int, dict[str, float]] = {}
        old_path = NORMALIZED_ROOT / "items.json"
        if old_path.exists():
            old_payload = json.loads(old_path.read_text(encoding="utf-8"))
            for old in old_payload["data"]:
                fallback_attrs[int(old["item_id"])] = _parse_attr_text(
                    old.get("des1", "")
                )
        rows: list[dict[str, Any]] = []
        for item in payload["data"]:
            equipment_id = int(item["equipment_id"])
            attrs = _parse_attr_text(item.get("attributes_text", ""))
            if not attrs:
                attrs = fallback_attrs.get(equipment_id, {})
            effects = item.get("effects_text", "")
            effect_kind = []
            if "主动" in effects:
                effect_kind.append("主动")
            if "被动" in effects:
                effect_kind.append("被动")
            sub_type = str(item.get("sub_type", ""))
            category = "normal"
            if "5" in sub_type or equipment_id in _JUNGLE_ITEM_IDS:
                category = "jungle"
            elif "6" in sub_type or "7" in sub_type or equipment_id in _SUPPORT_ITEM_IDS:
                category = "support"
            rows.append(
                {
                    "equipment_id": equipment_id,
                    "item_name": item["name"],
                    "item_type": int(item.get("type", 0)),
                    "sub_type": sub_type,
                    "category": category,
                    "price": float(item.get("price", 0)),
                    "total_price": float(item.get("price", 0)),
                    "attributes": attrs,
                    "attr_vector": _attr_vector(attrs),
                    "effects_text": effects,
                    "effect_kind": effect_kind,
                    "passive_vector": _keyword_vector(effects, ITEM_PASSIVE_DIMS),
                    "fit_heroes": item.get("fit_heroes", []),
                    "compound_ids": item.get("compound_ids", []),
                    "tips": item.get("tips", ""),
                }
            )
        with_effects = sum(1 for r in rows if r["effects_text"])
        return {
            "version": GAME_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "data/S45/normalized/equip_details.json (官方游戏内装备详情)",
            "attr_dims": ITEM_ATTR_DIMS,
            "passive_dims": ITEM_PASSIVE_DIMS,
            "count": len(rows),
            "coverage": {
                "total": len(rows),
                "with_effects_text": with_effects,
                "effects_missing": len(rows) - with_effects,
                "note": "官方游戏内详情已含属性与主动/被动效果",
            },
            "data": rows,
        }

    payload = json.loads(
        (NORMALIZED_ROOT / "items.json").read_text(encoding="utf-8")
    )
    items = payload["data"]
    rows: list[dict[str, Any]] = []
    passive_missing = 0
    for item in items:
        des = _strip_html(item.get("des1", ""))
        attrs = _parse_attr_text(des)
        # 被动文本：des 中"被动"之后的部分；属性数值部分去掉
        passive_text = ""
        if "被动" in des:
            passive_text = des[des.index("被动") :]
        elif "唯一" in des:
            passive_text = des[des.index("唯一") :]
        if not passive_text:
            passive_missing += 1
        rows.append(
            {
                "item_id": int(item["item_id"]),
                "item_name": item["item_name"],
                "item_type": int(item.get("item_type", 0)),
                "price": float(item.get("price", 0)),
                "total_price": float(item.get("total_price", 0)),
                "attributes": attrs,
                "attr_vector": _attr_vector(attrs),
                "passive_text": passive_text,
                "passive_vector": _keyword_vector(passive_text, ITEM_PASSIVE_DIMS),
            }
        )
    return {
        "version": GAME_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "data/S45/normalized/items.json (官方装备表)",
        "attr_dims": ITEM_ATTR_DIMS,
        "passive_dims": ITEM_PASSIVE_DIMS,
        "count": len(rows),
        "coverage": {
            "total": len(rows),
            "with_passive_text": len(rows) - passive_missing,
            "passive_missing": passive_missing,
            "note": "官方 item.json 不含完整被动描述；待营地装备接口/公告数据补充",
        },
        "data": rows,
    }


def _parse_skill_table(raw: str | None) -> dict[str, list[float]]:
    """skillData 形如 '基础伤害|55|66|77|88|99|110' → {名: [值...]}。"""
    result: dict[str, list[float]] = {}
    if not raw:
        return result
    parts = str(raw).split("|")
    name = parts[0].strip()
    values: list[float] = []
    for part in parts[1:]:
        try:
            values.append(float(part))
        except ValueError:
            continue
    if name and values:
        result[name] = values
    return result


def _skill_numeric_feature(
    tables: dict[str, list[float]], *, name_key: str
) -> dict[str, float | None]:
    """取某类数值表的 base / growth / min / max。"""
    for name, values in tables.items():
        if name_key in name:
            return {
                "base": values[0],
                "growth": values[-1] - values[0] if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
    return {"base": None, "growth": None, "min": None, "max": None}


def build_skill_vectors() -> dict[str, Any]:
    payload = json.loads(
        (NORMALIZED_ROOT / "hero_skills.json").read_text(encoding="utf-8")
    )
    skills = payload["data"]
    rows: list[dict[str, Any]] = []
    for skill in skills:
        raw = skill.get("data_tables") or {}
        tables: dict[str, list[float]] = {}
        for key in ("szSkillData1", "szSkillData2", "szSkillData3", "szSkillData4", "szSkillData5"):
            if raw.get(key):
                tables.update(_parse_skill_table(raw[key]))
        cooldown = skill.get("cooldown") or []
        damage = _skill_numeric_feature(tables, name_key="伤害")
        # 尽量补一个伤害值：描述中的数字为兜底
        if damage["base"] is None:
            numbers = re.findall(r"(\d+(?:\.\d+)?)", skill.get("description", ""))
            if numbers:
                damage["base"] = float(numbers[0])
                damage["min"] = damage["base"]
                damage["max"] = damage["base"]
                damage["growth"] = 0.0
        desc = skill.get("description", "")
        effect_text = desc + " " + " ".join(skill.get("labels") or [])
        rows.append(
            {
                "skill_id": skill.get("skill_id"),
                "ename": int(skill["ename"]),
                "cname": skill.get("cname", ""),
                "slot": int(skill.get("slot", 0)),
                "name": skill.get("name", ""),
                "type": skill.get("type"),
                "cooldown": cooldown,
                "cooldown_min": min(cooldown) if cooldown else None,
                "cooldown_max": max(cooldown) if cooldown else None,
                "damage": damage,
                "data_tables": tables,
                "effect_vector": _keyword_vector(effect_text, SKILL_EFFECT_DIMS),
                "labels": skill.get("labels") or [],
                "description": desc,
            }
        )
    return {
        "version": GAME_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "data/S45/normalized/hero_skills.json (王者营地技能)",
        "effect_dims": SKILL_EFFECT_DIMS,
        "count": len(rows),
        "coverage": {
            "total": len(rows),
            "with_cooldown": sum(1 for r in rows if r["cooldown_min"] is not None),
            "with_damage": sum(1 for r in rows if r["damage"]["base"] is not None),
            "note": "数值来自 skillData 成长表；无表字段时用描述首数字兜底并保留 null 语义",
        },
        "data": rows,
    }


def build_all() -> tuple[Path, Path]:
    items = build_item_vectors()
    skills = build_skill_vectors()
    items_path = NORMALIZED_ROOT / "items_vector.json"
    skills_path = NORMALIZED_ROOT / "skills_vector.json"
    items_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    skills_path.write_text(json.dumps(skills, ensure_ascii=False, indent=2), encoding="utf-8")
    return items_path, skills_path


if __name__ == "__main__":
    p1, p2 = build_all()
    print(f"已生成:\n  {p1}\n  {p2}")
