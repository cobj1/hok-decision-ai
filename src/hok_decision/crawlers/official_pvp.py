"""官方资料库爬虫：英雄列表 / 装备 / 召唤师技能。

一手来源为 pvp.qq.com 的静态 JSON 端点。
注意：官方 JSON 只提供展示层字段，英雄基础战斗数值需后续补充
（见 data/{version}/normalized/hero_stats.json 的 schema 与覆盖机制）。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hok_decision.config import (
    GAME_VERSION,
    NORMALIZED_ROOT,
    OFFICIAL_HERO_LIST_URL,
    OFFICIAL_ITEM_LIST_URL,
    OFFICIAL_SUMMONER_URL,
    RAW_ROOT,
)
from hok_decision.crawlers.base import CrawlerError, fetch, save_raw, source_ref
from hok_decision.schema.models import DataManifest, HeroInfo, ItemInfo, SummonerSpell

_ENDPOINTS: dict[str, tuple[str, str]] = {
    "herolist.json": (OFFICIAL_HERO_LIST_URL, "王者荣耀官网英雄列表 JSON"),
    "item.json": (OFFICIAL_ITEM_LIST_URL, "王者荣耀官网装备列表 JSON"),
    "summoner.json": (OFFICIAL_SUMMONER_URL, "王者荣耀官网召唤师技能 JSON"),
}


def _normalize_roles(raw_roles: Any) -> list[int]:
    """官方 roles 字段为 '1|5' 形式的字符串，转为 int 列表。"""
    if isinstance(raw_roles, int):
        return [raw_roles]
    if isinstance(raw_roles, str):
        if not raw_roles.strip():
            return []
        parts = raw_roles.split("|")
        try:
            return [int(part) for part in parts if part.strip()]
        except ValueError as exc:  # 数据异常时保留原始信息而非中断
            raise CrawlerError(f"roles 字段无法解析: {raw_roles!r}") from exc
    return []


def crawl_official() -> DataManifest:
    """抓取三个官方 JSON 端点，落盘 raw + normalized。"""
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    NORMALIZED_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = DataManifest(version=GAME_VERSION)
    src = source_ref(
        "https://pvp.qq.com/web201605/", "王者荣耀官网资料库", is_primary=True
    )

    for filename, (url, source_name) in _ENDPOINTS.items():
        response = fetch(url)
        raw_path = save_raw(response.content, root=RAW_ROOT, filename=filename)
        manifest.files.append(str(raw_path.relative_to(RAW_ROOT.parent.parent)))
        manifest.sources.append(
            {**source_ref(url, source_name, is_primary=True), "file": filename}
        )

    try:
        heroes = json.loads((RAW_ROOT / "herolist.json").read_text(encoding="utf-8"))
        items = json.loads((RAW_ROOT / "item.json").read_text(encoding="utf-8"))
        summoners = json.loads(
            (RAW_ROOT / "summoner.json").read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as exc:
        raise CrawlerError(f"官方 JSON 解析失败: {exc}") from exc

    normalized_heroes = [
        HeroInfo(
            ename=int(h["ename"]),
            cname=h["cname"],
            id_name=h["id_name"],
            title=h["title"],
            hero_type=int(h.get("hero_type", 0)),
            roles=_normalize_roles(h.get("roles")),
        ).model_dump()
        for h in heroes
    ]
    normalized_items = [
        ItemInfo(
            item_id=int(i["item_id"]),
            item_name=i["item_name"],
            item_type=int(i.get("item_type", 0)),
            price=float(i.get("price", 0)),
            total_price=float(i.get("total_price", 0)),
            des1=i.get("des1", ""),
        ).model_dump()
        for i in items
    ]
    normalized_summoners = [
        SummonerSpell(
            summoner_id=int(s["summoner_id"]),
            summoner_name=s["summoner_name"],
            summoner_rank=s.get("summoner_rank", ""),
            summoner_description=s.get("summoner_description", ""),
        ).model_dump()
        for s in summoners
    ]

    out_files: dict[str, Any] = {
        "heroes.json": normalized_heroes,
        "items.json": normalized_items,
        "summoners.json": normalized_summoners,
    }
    for filename, data in out_files.items():
        target = NORMALIZED_ROOT / filename
        target.write_text(
            json.dumps(
                {
                    "version": GAME_VERSION,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "source": src,
                    "count": len(data),
                    "data": data,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest.files.append(str(target.relative_to(NORMALIZED_ROOT.parent.parent)))

    return manifest


def ensure_hero_stats_skeleton() -> Path:
    """生成英雄战斗数值的占位 schema 文件（仅当真实数据尚不存在）。"""
    target = NORMALIZED_ROOT / "hero_stats.json"
    if target.exists():
        return target
    skeleton = {
        "version": GAME_VERSION,
        "note": (
            "官方 JSON 不含英雄基础战斗数值。"
            "本文件为 schema 占位，数值需逐英雄补充并置 verified=true。"
            "补充后模拟器才能使用该英雄的真实数值；此前用 fixture 数值跑测试。"
        ),
        "data": [],
    }
    target.write_text(json.dumps(skeleton, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
