"""英雄基础数值/技能落盘数据校验（离线，不访问网络）。"""

from __future__ import annotations

import json
from pathlib import Path

from hok_decision.config import GAME_VERSION, NORMALIZED_ROOT, RAW_ROOT
from hok_decision.schema.models import HeroStats, SkillInfo

HERO_LIST_PATH = RAW_ROOT / "herolist.json"
HERO_STATS_PATH = NORMALIZED_ROOT / "hero_stats.json"
HERO_SKILLS_PATH = NORMALIZED_ROOT / "hero_skills.json"


def _load_json(path: Path) -> dict:
    assert path.exists(), f"缺少数据文件: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _hero_ename_set() -> set[int]:
    heroes = json.loads(HERO_LIST_PATH.read_text(encoding="utf-8"))
    return {int(h["ename"]) for h in heroes}


def _print_coverage(kind: str, payload: dict) -> None:
    coverage = payload.get("coverage", {})
    print(
        f"[coverage] {kind}: {coverage.get('covered')}/{coverage.get('total')} "
        f"verified={coverage.get('verified')} missing={coverage.get('missing')}"
    )


def test_hero_stats_file_schema_and_alignment() -> None:
    payload = _load_json(HERO_STATS_PATH)
    assert payload["version"] == GAME_VERSION
    data = payload["data"]
    enames = _hero_ename_set()
    assert len(data) == len(enames) == 132
    assert {int(item["ename"]) for item in data} == enames

    for item in data:
        stats = HeroStats.model_validate(item)
        if stats.verified:
            assert stats.source.get("url"), (
                f"{stats.cname} verified 但缺少 source.url"
            )

    _print_coverage("hero_stats", payload)


def test_hero_skills_file_schema_and_alignment() -> None:
    payload = _load_json(HERO_SKILLS_PATH)
    assert payload["version"] == GAME_VERSION
    data = payload["data"]
    enames = _hero_ename_set()
    assert {int(skill["ename"]) for skill in data} == enames
    assert len(data) >= len(enames) * 3  # 至少被动+1/2/3 四个技能位

    for item in data:
        skill = SkillInfo.model_validate(item)
        if skill.verified:
            assert skill.source.get("url"), (
                f"{skill.cname} 技能 verified 但缺少 source.url"
            )

    _print_coverage("hero_skills", payload)


def test_coverage_matches_payload() -> None:
    stats_payload = _load_json(HERO_STATS_PATH)
    skills_payload = _load_json(HERO_SKILLS_PATH)

    stats_data = stats_payload["data"]
    stats_covered = sum(
        1
        for s in stats_data
        if any(
            s.get(field) is not None
            for field in (
                "base_hp",
                "base_mp",
                "base_physical_attack",
                "base_magic_attack",
                "base_physical_defense",
                "base_magic_defense",
                "base_attack_speed",
                "attack_range",
                "move_speed",
                "hp_regen_5s",
                "mp_regen_5s",
            )
        )
    )
    assert stats_payload["coverage"]["covered"] == stats_covered
    assert stats_payload["coverage"]["total"] == len(stats_data)

    skills_data = skills_payload["data"]
    skills_covered = len({int(s["ename"]) for s in skills_data})
    assert skills_payload["coverage"]["covered"] == skills_covered
    assert skills_payload["coverage"]["total"] == len(_hero_ename_set())
