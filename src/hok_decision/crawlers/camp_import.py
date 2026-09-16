"""从本地 wzry-scraper（王者营地抓取）导入英雄基础数值与技能数据。

数据源目录：/Users/boqing/Downloads/wzry-scraper/data
数据特点：updateTime=20260827，覆盖 132 英雄基础数值与 538 条技能，属当前赛季二手一手抓取。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hok_decision.config import GAME_VERSION, NORMALIZED_ROOT
from hok_decision.schema.models import HeroStats, SkillInfo

DEFAULT_SOURCE_DIR = Path("/Users/boqing/Downloads/wzry-scraper/data")

_TAG_RE = re.compile(r"<[^>]+>")
_PCT_RE = re.compile(r"^([0-9.]+)%$")


def _strip_html(text: str) -> str:
    text = _TAG_RE.sub("", text)
    return text.strip()


def _num(value: Any) -> float | None:
    """把营地数值转 float；字符串百分比转小数。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = _PCT_RE.match(text)
    if match:
        return float(match.group(1)) / 100.0
    try:
        return float(text)
    except ValueError:
        return None


def _parse_camp_stats(entry: dict[str, Any], source: dict[str, Any]) -> HeroStats:
    attr = entry.get("attrInfo", {})
    base = {item["name"]: item.get("data") for item in attr.get("base", [])}
    attack = {item["name"]: item.get("data") for item in attr.get("attack", [])}
    defence = {item["name"]: item.get("data") for item in attr.get("defence", [])}

    return HeroStats(
        ename=int(entry["heroId"]),
        cname=entry.get("name", ""),
        base_hp=_num(base.get("最大生命")),
        base_mp=_num(base.get("最大法力")),
        base_physical_attack=_num(base.get("物理攻击")),
        base_magic_attack=_num(base.get("法术攻击")),
        base_physical_defense=_num(base.get("物理防御")),
        base_magic_defense=_num(base.get("法术防御")),
        base_attack_speed=_num(attack.get("攻速加成")),
        attack_range_label=str(attack.get("攻击范围")) if attack.get("攻击范围") else None,
        attack_range=None,  # 营地仅给近程/远程，数值待标定
        crit_rate=_num(attack.get("暴击几率")),
        crit_effect=_num(attack.get("暴击效果")),
        move_speed=_num(attack.get("移速")),
        hp_regen_5s=_num(defence.get("每五秒回血")),
        mp_regen_5s=_num(defence.get("每五秒回蓝")),
        source=source,
        verified=True,
    )


def _parse_camp_skills(
    ename: int, entry: dict[str, Any], source: dict[str, Any]
) -> list[SkillInfo]:
    skills: list[SkillInfo] = []
    for raw in entry.get("skillArray", []):
        i_skill_id = raw.get("iSkillId")
        skill_id = str(i_skill_id) if i_skill_id is not None else None
        try:
            slot = (int(i_skill_id) % 100) // 10 if i_skill_id is not None else 0
        except (TypeError, ValueError):
            slot = 0

        cooldown: list[float] = []
        cooldown_raw = raw.get("szSkillData3", "")
        if cooldown_raw:
            parts = str(cooldown_raw).split("|")
            if parts and "冷却" in parts[0]:
                for part in parts[1:]:
                    value = _num(part)
                    if value is not None:
                        cooldown.append(value)
        if not cooldown and raw.get("iCoolDown") is not None:
            cooldown = [float(raw["iCoolDown"])]

        data_tables: dict[str, str] = {}
        for key in ("szSkillData1", "szSkillData2", "szSkillData3"):
            if raw.get(key):
                data_tables[key] = str(raw[key])

        labels = [
            str(item["Text"]) for item in raw.get("gameLabels", []) if item.get("Text")
        ]

        skills.append(
            SkillInfo(
                ename=ename,
                cname=entry.get("name", ""),
                skill_id=skill_id,
                slot=slot,
                name=raw.get("szTitle", ""),
                type=raw.get("szType"),
                cooldown=cooldown,
                cooldown_raw=cooldown_raw,
                mana_cost=[],
                mana_cost_raw="",
                description=_strip_html(raw.get("szDesc", "")),
                control_effect=raw.get("szLabel"),
                data_tables=data_tables,
                labels=labels,
                source=source,
                verified=True,
            )
        )
    return skills


def _hero_list() -> list[dict[str, Any]]:
    heroes_path = NORMALIZED_ROOT / "heroes.json"
    payload = json.loads(heroes_path.read_text(encoding="utf-8"))
    return payload["data"]


def import_camp_data(
    source_dir: Path | None = None,
) -> tuple[Path, Path]:
    """导入营地数据，覆盖 hero_stats.json 与 hero_skills.json。"""
    source_dir = source_dir or DEFAULT_SOURCE_DIR
    stats_raw = json.loads(
        (source_dir / "wzry_heroes_stats.json").read_text(encoding="utf-8")
    )
    skills_raw = json.loads(
        (source_dir / "hero_skills.json").read_text(encoding="utf-8")
    )

    now = datetime.now(timezone.utc).isoformat()
    source = {
        "url": str(source_dir),
        "name": "王者营地本地抓取数据(wzry-scraper, updateTime=20260827)",
        "retrieved_at": now,
        "is_primary": False,
    }

    heroes = _hero_list()
    stats_data: list[dict[str, Any]] = []
    for hero in heroes:
        ename = int(hero["ename"])
        entry = stats_raw.get(str(ename))
        if entry is None:
            stats_data.append(
                HeroStats(
                    ename=ename,
                    cname=hero["cname"],
                    source={"url": str(source_dir), "name": "camp: missing"},
                    verified=False,
                ).model_dump()
            )
            continue
        stats_data.append(_parse_camp_stats(entry, source).model_dump())

    skills_data: list[dict[str, Any]] = []
    for hero in heroes:
        ename = int(hero["ename"])
        entry = skills_raw.get(str(ename))
        if entry is not None:
            skills_data.extend(
                skill.model_dump() for skill in _parse_camp_skills(ename, entry, source)
            )

    NORMALIZED_ROOT.mkdir(parents=True, exist_ok=True)
    stats_path = NORMALIZED_ROOT / "hero_stats.json"
    skills_path = NORMALIZED_ROOT / "hero_skills.json"

    stats_coverage = {
        "covered": sum(1 for s in stats_data if any(s.get(f) is not None for f in (
            "base_hp", "base_mp", "base_physical_attack", "base_magic_attack",
            "base_physical_defense", "base_magic_defense",
        ))),
        "total": len(stats_data),
        "verified": sum(1 for s in stats_data if s.get("verified")),
    }
    skills_coverage = {
        "covered": len({s["ename"] for s in skills_data}),
        "total": len(heroes),
        "skills": len(skills_data),
        "verified": sum(1 for s in skills_data if s.get("verified")),
    }

    stats_path.write_text(
        json.dumps(
            {
                "version": GAME_VERSION,
                "fetched_at": now,
                "source": source,
                "count": len(stats_data),
                "coverage": stats_coverage,
                "data": stats_data,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    skills_path.write_text(
        json.dumps(
            {
                "version": GAME_VERSION,
                "fetched_at": now,
                "source": source,
                "count": len(skills_data),
                "coverage": skills_coverage,
                "data": skills_data,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return stats_path, skills_path


if __name__ == "__main__":
    p1, p2 = import_camp_data()
    print(f"已导入营地数据:\n  {p1}\n  {p2}")
