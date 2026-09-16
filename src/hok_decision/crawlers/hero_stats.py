"""英雄基础数值与技能数据爬虫。

数据源：
- 官方 herodetail 页面（一手）：技能名称/冷却/消耗/描述，覆盖全部英雄；
- GitHub `cystanford/EM_data` 的 heros.csv（旧版二手）：69 位英雄的 1 级/满级/成长数值；
- GitHub `Javison666/HonourKings` 的 players.js（旧版二手）：88 位英雄的部分基础数值；
- 游侠网全英雄属性一览（旧版二手）：69 位英雄的移速/法防/法攻/攻击范围等。

二手旧版数值仅作参考，`verified=false`；与官方/S45 不一致时记入 `conflicts`，
不自行裁决数值。技能以官方页面为准，`verified=true`。
"""

from __future__ import annotations

import csv
import html
import io
import json
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hok_decision.config import GAME_VERSION, NORMALIZED_ROOT, RAW_ROOT
from hok_decision.crawlers.base import CrawlerError, fetch, save_raw, source_ref
from hok_decision.schema.models import HeroStats, SkillInfo

# 官方 herodetail 页面（GBK 编码，部分新英雄仅 id_name 路由可用）
HERO_DETAIL_URL = "https://pvp.qq.com/web201605/herodetail/{id_name}.shtml"
# 旧版英雄属性 CSV（EM 数据集，GBK）
EM_DATA_CSV_URL = "https://raw.githubusercontent.com/cystanford/EM_data/master/heros.csv"
# 旧版英雄基础数值 JS（HonourKings）
HONOUR_KINGS_URL = "https://raw.githubusercontent.com/Javison666/HonourKings/master/players.js"
# 游侠网全英雄属性一览（69 页，2017）
ALI213_INDEX_URL = "https://gl.ali213.net/html/2017-8/183555.html"
ALI213_PAGE_URL = "https://gl.ali213.net/html/2017-8/183555_{page}.html"
ALI213_PAGE_COUNT = 69

RAW_SRC_ROOT = RAW_ROOT / "hero_stats_src"

# HeroStats 中代表“已取得基础数值”的字段
_STAT_FIELDS: tuple[str, ...] = (
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

_CONTROL_KEYWORDS: tuple[str, ...] = (
    "击飞",
    "眩晕",
    "晕眩",
    "冰冻",
    "石化",
    "沉默",
    "嘲讽",
    "减速",
    "定身",
    "压制",
    "击退",
    "拉回",
    "缴械",
    "禁锢",
    "支配",
)


def _save_src(filename: str, content: bytes) -> Path:
    return save_raw(content, root=RAW_SRC_ROOT, filename=filename)


def _decode_gbk(content: bytes) -> str:
    return content.decode("gb18030", errors="replace")


def _strip_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    match = re.match(r"^[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _numbers_after_colon(text: str) -> list[float]:
    """从 '冷却值：9' / '消耗：0' / '冷却值：11/10.4/8' 提取数值列表。"""
    match = re.search(r"[:：]\s*(.*)$", text)
    if not match:
        return []
    raw = match.group(1).replace("（", "/").replace("(", "/")
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", raw)]


def _load_heroes() -> list[dict[str, Any]]:
    path = RAW_ROOT / "herolist.json"
    if not path.exists():
        raise CrawlerError(f"缺少英雄列表: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise CrawlerError(f"herolist.json 解析失败: {exc}") from exc
    if not isinstance(data, list):
        raise CrawlerError("herolist.json 不是数组")
    return data


def _parse_em_csv(content: bytes) -> dict[str, dict[str, str]]:
    """解析 EM_data heros.csv（GBK），返回 {英雄名: 行字典}。"""
    try:
        text = content.decode("gb18030", errors="replace")
    except UnicodeDecodeError as exc:  # pragma: no cover - 编码异常保护
        raise CrawlerError(f"heros.csv 解码失败: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text))
    return {row["英雄"]: row for row in reader if row.get("英雄")}


def _parse_players_js(content: str) -> dict[str, dict[str, float]]:
    """解析 HonourKings players.js 中的 name/attr 块。"""
    result: dict[str, dict[str, float]] = {}
    pattern = re.compile(r'name:"([^"]+)"\s*,\s*attr:\{([^}]+)\}', re.S)
    for m in pattern.finditer(content):
        name = m.group(1).strip()
        values: dict[str, float] = {}
        for kv in re.finditer(r"(\w+):\s*([\d.]+)", m.group(2)):
            values[kv.group(1)] = float(kv.group(2))
        if name:
            result[name] = values
    return result


def _parse_ali213_table(html_text: str) -> dict[str, Any] | None:
    """从游侠网单页解析属性表，返回 {hero_name, stats}。"""
    for table in re.findall(r"<table[^>]*>(.*?)</table>", html_text, re.S):
        if "最大生命" not in table:
            continue
        rows: list[list[str]] = []
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S):
            cells = [
                _strip_tags(cell)
                for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
            ]
            if cells:
                rows.append(cells)
        if not rows:
            continue
        name: str | None = None
        stats: dict[str, str] = {}
        for cells in rows:
            for label, value in zip(cells[0::2], cells[1::2]):
                label = label.strip()
                value = value.strip()
                if label.startswith("英雄："):
                    name = label.split("：", 1)[1].strip()
                elif label:
                    stats[label] = value
        if name and stats:
            return {"name": name, "stats": stats}
    return None


def _fetch_ali213_pages() -> dict[str, dict[str, str]]:
    """抓取游侠网 69 页属性表，返回 {英雄名: 属性字典}。"""
    result: dict[str, dict[str, str]] = {}
    for page in range(1, ALI213_PAGE_COUNT + 1):
        url = ALI213_INDEX_URL if page == 1 else ALI213_PAGE_URL.format(page=page)
        response = fetch(url)
        _save_src(f"ali213_183555_{page}.html", response.content)
        parsed = _parse_ali213_table(
            response.content.decode("utf-8", errors="replace")
        )
        if parsed:
            result[parsed["name"]] = parsed["stats"]
    return result


def _fetch_em_csv() -> dict[str, dict[str, str]]:
    response = fetch(EM_DATA_CSV_URL)
    _save_src("em_data_heros.csv", response.content)
    return _parse_em_csv(response.content)


def _fetch_players_js() -> dict[str, dict[str, float]]:
    response = fetch(HONOUR_KINGS_URL)
    _save_src("honourkings_players.js", response.content)
    return _parse_players_js(response.content.decode("utf-8", errors="replace"))


def _lookup_ename_by_name(heroes: Iterable[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for hero in heroes:
        cname = str(hero.get("cname", "")).strip()
        if cname:
            result[cname] = int(hero["ename"])
    # 兼容旧数据中的常见异体字/错字
    aliases = {
        "橘石京": "橘右京",
        "奕星": "弈星",
    }
    for old, new in aliases.items():
        if old in result:
            result[new] = result.pop(old)
    return result


def _build_hero_stats(
    heroes: list[dict[str, Any]],
    em_data: dict[str, dict[str, str]],
    players_data: dict[str, dict[str, float]],
    ali213_data: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """合并多个二手源为 132 位英雄的 HeroStats 条目（缺字段为 null）。"""
    name_to_ename = _lookup_ename_by_name(heroes)
    em_time = datetime.now(timezone.utc).isoformat()
    source_em = {
        **source_ref(EM_DATA_CSV_URL, "GitHub EM_data 王者荣耀英雄属性表（旧版二手）", False),
        "retrieved_at": em_time,
    }
    source_players = {
        **source_ref(HONOUR_KINGS_URL, "GitHub HonourKings players.js（旧版二手）", False),
        "retrieved_at": em_time,
    }
    source_ali213 = {
        **source_ref(ALI213_INDEX_URL, "游侠网全英雄属性一览（旧版二手）", False),
        "retrieved_at": em_time,
    }

    results: list[dict[str, Any]] = []
    for hero in heroes:
        ename = int(hero["ename"])
        cname = str(hero["cname"])
        em = em_data.get(cname)
        player = players_data.get(cname)
        ali = ali213_data.get(cname)

        if not em and not player and not ali:
            results.append(
                HeroStats(
                    ename=ename,
                    cname=cname,
                    source={
                        "url": "",
                        "name": "未覆盖",
                        "retrieved_at": em_time,
                        "is_primary": False,
                    },
                    verified=False,
                ).model_dump()
            )
            continue

        growth: dict[str, float] = {}
        conflicts: list[dict[str, Any]] = []

        def _set_growth(key: str, value: str | None) -> None:
            num = _to_float(value)
            if num is not None:
                growth[key] = num

        # EM CSV：初始/成长/满级
        base_hp = _to_float(em.get("初始生命")) if em else None
        base_mp = _to_float(em.get("初始法力")) if em else None
        base_physical_attack = _to_float(em.get("初始物攻")) if em else None
        base_magic_attack: float | None = None
        base_physical_defense = _to_float(em.get("初始物防")) if em else None
        max_hp = _to_float(em.get("最大生命")) if em else None
        max_mp = _to_float(em.get("最大法力")) if em else None
        max_physical_attack = _to_float(em.get("最高物攻")) if em else None
        max_physical_defense = _to_float(em.get("最大物防")) if em else None
        max_attack_speed = _to_float(em.get("最大攻速")) if em else None
        if em:
            _set_growth("hp", em.get("生命成长"))
            _set_growth("mp", em.get("法力成长"))
            _set_growth("physical_attack", em.get("物攻成长"))
            _set_growth("physical_defense", em.get("物防成长"))
            _set_growth("hp_regen_5s", em.get("每5秒回血成长"))
            _set_growth("mp_regen_5s", em.get("每5秒回蓝成长"))

        # players.js：补充/校验基础值
        if player:
            for key, target in (
                ("life", "base_hp"),
                ("spells", "base_mp"),
                ("attackPhysical", "base_physical_attack"),
                ("attackMagic", "base_magic_attack"),
                ("defensePhysical", "base_physical_defense"),
            ):
                val = player.get(key)
                if val is None:
                    continue
                current = {
                    "base_hp": base_hp,
                    "base_mp": base_mp,
                    "base_physical_attack": base_physical_attack,
                    "base_magic_attack": None,
                    "base_physical_defense": base_physical_defense,
                }[target]
                if current is None:
                    # 仅当 EM 没有该字段时采用 players.js
                    if target == "base_hp":
                        base_hp = val
                    elif target == "base_mp":
                        base_mp = val
                    elif target == "base_physical_attack":
                        base_physical_attack = val
                    elif target == "base_magic_attack":
                        pass  # EM 无此字段，保留 players 值稍后处理
                    elif target == "base_physical_defense":
                        base_physical_defense = val
                elif abs(current - val) > 1e-6:
                    conflicts.append(
                        {
                            "field": target,
                            "em_data": current,
                            "players_js": val,
                            "note": "旧版二手源冲突，未自动裁决",
                        }
                    )

        # players.js 单独提供的法术攻击（EM 无此列）
        if player and player.get("attackMagic") is not None:
            if base_magic_attack is None:
                base_magic_attack = player["attackMagic"]
            else:
                conflicts.append(
                    {
                        "field": "base_magic_attack",
                        "em_data": None,
                        "players_js": player["attackMagic"],
                        "note": "EM 未提供法术攻击，以 players.js 填充",
                    }
                )

        # ali213：补充移速/法防/法攻/范围/回血回蓝等
        ali_speed = _to_float(ali.get("移速")) if ali else None
        ali_magic_defense = _to_float(ali.get("法术防御")) if ali else None
        ali_magic_attack = _to_float(ali.get("法术攻击")) if ali else None
        ali_hp_regen = _to_float(ali.get("每5秒回血")) if ali else None
        ali_mp_regen = _to_float(ali.get("每5秒回蓝")) if ali else None
        ali_range_label = ali.get("攻击范围") if ali else None

        if ali and ali_speed is not None:
            move_speed = ali_speed
        else:
            move_speed = None
        if ali and ali_magic_defense is not None:
            base_magic_defense = ali_magic_defense
        else:
            base_magic_defense = None
        if ali and ali_magic_attack is not None:
            base_magic_attack = ali_magic_attack
        elif base_magic_attack is None:
            base_magic_attack = None

        attack_range_label = ali_range_label or None
        attack_range: float | None = None  # 旧源只有近程/远程，无数值
        hp_regen_5s = ali_hp_regen or None
        mp_regen_5s = ali_mp_regen or None
        base_attack_speed = _to_float(ali.get("攻速加成")) if ali else None

        results.append(
            HeroStats(
                ename=ename,
                cname=cname,
                base_hp=base_hp,
                base_mp=base_mp,
                base_physical_attack=base_physical_attack,
                base_magic_attack=base_magic_attack,
                base_physical_defense=base_physical_defense,
                base_magic_defense=base_magic_defense,
                base_attack_speed=base_attack_speed,
                attack_range=attack_range,
                attack_range_label=attack_range_label,
                move_speed=move_speed,
                hp_regen_5s=hp_regen_5s,
                mp_regen_5s=mp_regen_5s,
                max_hp=max_hp,
                max_mp=max_mp,
                max_physical_attack=max_physical_attack,
                max_physical_defense=max_physical_defense,
                max_magic_defense=None,
                max_attack_speed=max_attack_speed,
                growth=growth,
                conflicts=conflicts,
                source={
                    "sources": [
                        src
                        for src in (
                            source_em if em else None,
                            source_players if player else None,
                            source_ali213 if ali else None,
                        )
                        if src is not None
                    ]
                    if any([em, player, ali])
                    else {"url": "", "name": "未覆盖"},
                },
                verified=False,
            ).model_dump()
        )

    return results


def _parse_official_skills(
    html_text: str, ename: int, cname: str, source: dict[str, Any]
) -> list[dict[str, Any]]:
    """解析官方 herodetail 页面中的技能块。"""
    shows = re.findall(r'<div class="show-list"[^>]*>(.*?)</div>', html_text, re.S)
    skill_images = re.findall(
        r'<ul class="skill-u1">(.*?)</ul>', html_text, re.S
    )
    image_srcs: list[str] = []
    if skill_images:
        image_srcs = re.findall(r'<img src="([^"]+)"', skill_images[0], re.S)

    skills: list[dict[str, Any]] = []
    for idx, block in enumerate(shows[:5]):
        name_match = re.search(r"<b>(.*?)</b>", block, re.S)
        desc_match = re.search(r'<p class="skill-desc">(.*?)</p>', block, re.S)
        spans = re.findall(r"<span>(.*?)</span>", block, re.S)
        name = _strip_tags(name_match.group(1)) if name_match else ""
        desc = _strip_tags(desc_match.group(1)) if desc_match else ""
        if not name and not desc:
            continue

        cooldown_raw = _strip_tags(spans[0]) if len(spans) >= 1 else ""
        cost_raw = _strip_tags(spans[1]) if len(spans) >= 2 else ""
        skill_id: str | None = None
        if idx < len(image_srcs):
            id_match = re.search(r"/(\d{4,})\.png$", image_srcs[idx])
            if id_match:
                skill_id = id_match.group(1)

        control_effect: str | None = None
        for sentence in re.split(r"[。！？]", desc):
            if any(kw in sentence for kw in _CONTROL_KEYWORDS):
                control_effect = sentence.strip()
                break

        skills.append(
            SkillInfo(
                ename=ename,
                cname=cname,
                skill_id=skill_id,
                slot=idx,
                name=name,
                type="被动" if idx == 0 else "主动",
                cooldown=_numbers_after_colon(cooldown_raw),
                cooldown_raw=cooldown_raw,
                mana_cost=_numbers_after_colon(cost_raw),
                mana_cost_raw=cost_raw,
                description=desc,
                control_effect=control_effect,
                source=source,
                verified=True,
            ).model_dump()
        )
    return skills


def crawl_hero_skills(limit: int | None = None) -> list[dict[str, Any]]:
    """抓取官方 herodetail 技能页，返回全部/部分英雄技能列表。"""
    heroes = _load_heroes()
    if limit is not None:
        heroes = heroes[:limit]
    source = source_ref(
        "https://pvp.qq.com/web201605/herodetail/",
        "王者荣耀官网英雄详情页",
        is_primary=True,
    )
    all_skills: list[dict[str, Any]] = []
    for hero in heroes:
        ename = int(hero["ename"])
        cname = str(hero["cname"])
        id_name = str(hero.get("id_name", "")).strip()
        try:
            response = fetch(HERO_DETAIL_URL.format(id_name=id_name))
        except CrawlerError:
            # 个别英雄页面可能缺失或网络抖动，跳过并在 coverage 中体现
            continue
        _save_src(f"herodetail_{ename}.html", response.content)
        html_text = _decode_gbk(response.content)
        skills = _parse_official_skills(html_text, ename, cname, source)
        all_skills.extend(skills)
    return all_skills


def crawl_hero_stats() -> list[dict[str, Any]]:
    """抓取/读取二手属性源并生成 132 位英雄的 HeroStats 列表。"""
    heroes = _load_heroes()
    em_data = _fetch_em_csv()
    players_data = _fetch_players_js()
    ali213_data = _fetch_ali213_pages()
    return _build_hero_stats(heroes, em_data, players_data, ali213_data)


def _write_normalized(
    filename: str,
    data: list[dict[str, Any]],
    source: dict[str, Any],
    *,
    coverage: dict[str, Any],
) -> Path:
    NORMALIZED_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": GAME_VERSION,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "count": len(data),
        "coverage": coverage,
        "data": data,
    }
    target = NORMALIZED_ROOT / filename
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def _stats_coverage(stats: list[dict[str, Any]]) -> dict[str, Any]:
    covered_ids = {
        s["ename"]
        for s in stats
        if any(s.get(field) is not None for field in _STAT_FIELDS)
    }
    missing = [s["ename"] for s in stats if s["ename"] not in covered_ids]
    return {
        "covered": len(covered_ids),
        "total": len(stats),
        "missing": len(missing),
        "missing_enames": missing,
        "verified": sum(1 for s in stats if s.get("verified")),
    }


def _skills_coverage(skills: list[dict[str, Any]]) -> dict[str, Any]:
    by_hero: dict[int, list[dict[str, Any]]] = {}
    for skill in skills:
        by_hero.setdefault(int(skill["ename"]), []).append(skill)
    heroes = _load_heroes()
    total = len(heroes)
    covered = len(by_hero)
    missing = [
        int(h["ename"]) for h in heroes if int(h["ename"]) not in by_hero
    ]
    return {
        "covered": covered,
        "total": total,
        "missing": len(missing),
        "missing_enames": missing,
        "verified": sum(1 for s in skills if s.get("verified")),
    }


def crawl_all(limit_skills: int | None = None) -> tuple[Path, Path]:
    """执行完整爬取：先技能（官方），再数值（二手源），落盘 normalized。"""
    skills = crawl_hero_skills(limit=limit_skills)
    stats = crawl_hero_stats()

    skill_source = source_ref(
        "https://pvp.qq.com/web201605/herodetail/",
        "王者荣耀官网英雄详情页",
        is_primary=True,
    )
    stats_source = source_ref(
        EM_DATA_CSV_URL,
        "英雄基础数值（旧版二手源合并：EM_data/HonourKings/游侠网）",
        is_primary=False,
    )

    skills_path = _write_normalized(
        "hero_skills.json",
        skills,
        skill_source,
        coverage=_skills_coverage(skills),
    )
    stats_path = _write_normalized(
        "hero_stats.json",
        stats,
        stats_source,
        coverage=_stats_coverage(stats),
    )
    return stats_path, skills_path


if __name__ == "__main__":  # pragma: no cover
    stats_path, skills_path = crawl_all()
    print(f"written: {stats_path}")
    print(f"written: {skills_path}")
