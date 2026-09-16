"""官方游戏内装备详情抓取（属性 + 主动/被动 + 适用英雄 + 合成）。

数据源：
  列表: https://wuji-1254960240.file.myqcloud.com/smoba_weapon/pages/p1.json
  详情: https://wuji-1254960240.file.myqcloud.com/smoba_weapon_detail/{equipment_id}.json
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hok_decision.config import GAME_VERSION, NORMALIZED_ROOT, RAW_ROOT
from hok_decision.crawlers.base import CrawlerError, fetch, save_raw

LIST_URL = "https://wuji-1254960240.file.myqcloud.com/smoba_weapon/pages/p1.json"
DETAIL_URL_TMPL = (
    "https://wuji-1254960240.file.myqcloud.com/smoba_weapon_detail/{equipment_id}.json"
)
DETAIL_RAW_DIR = RAW_ROOT / "equip_detail"


def _strip_html(text: str) -> str:
    import re

    text = re.sub(r"<[^>]+>", "", text or "")
    return text.strip()


def fetch_equip_details(*, delay_s: float = 0.12) -> tuple[Path, dict[str, Any]]:
    """抓取全量装备详情，返回 (归一化路径, 汇总数据)。"""
    list_response = fetch(LIST_URL)
    items = list_response.json()
    if not isinstance(items, list) or not items:
        raise CrawlerError(f"装备列表格式异常: {type(items)}")

    now = datetime.now(timezone.utc).isoformat()
    DETAIL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    details: list[dict[str, Any]] = []
    failed: list[int] = []

    for index, item in enumerate(items, start=1):
        equipment_id = item.get("equipment_id")
        if equipment_id is None:
            continue
        try:
            response = fetch(DETAIL_URL_TMPL.format(equipment_id=equipment_id), retries=2)
            payload = response.json()
        except CrawlerError as exc:
            failed.append(int(equipment_id))
            continue
        save_raw(response.content, root=DETAIL_RAW_DIR, filename=f"{equipment_id}.json")

        attrs = payload.get("attributes") or {}
        effects = attrs.get("sub", "")
        details.append(
            {
                "equipment_id": int(equipment_id),
                "name": item.get("name", ""),
                "price": float(item.get("price", 0)),
                "type": int(item.get("type", 1)),
                "sub_type": str(item.get("sub_type", "")),
                "icon": item.get("icon", ""),
                "attributes_text": _strip_html(attrs.get("attributes", "")),
                "effects_text": _strip_html(effects),
                "fit_heroes": [
                    {"hero_id": str(h.get("hero_id", "")), "name": h.get("desc", "")}
                    for h in (payload.get("fit_heroes") or {}).get("fit_heroes", [])
                ],
                "tips": _strip_html(payload.get("tips", "")),
                "compound_ids": [
                    int(c.get("equipment_id", 0))
                    for c in (item.get("compound_url") or {}).get("compound", [])
                    if c.get("equipment_id")
                ],
                "source": {
                    "url": DETAIL_URL_TMPL.format(equipment_id=equipment_id),
                    "name": "王者荣耀官方游戏内装备详情",
                    "retrieved_at": now,
                    "is_primary": True,
                },
            }
        )
        if index % 20 == 0:
            time.sleep(delay_s)

    summary = {
        "version": GAME_VERSION,
        "fetched_at": now,
        "list_url": LIST_URL,
        "count": len(details),
        "failed_ids": failed,
        "note": "装备详情为官方游戏内页面数据，含属性/主动被动/适用英雄/合成",
        "data": details,
    }
    target = NORMALIZED_ROOT / "equip_details.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return target, summary


if __name__ == "__main__":
    path, summary = fetch_equip_details()
    print(f"已抓取 {summary['count']} 件装备 -> {path}")
    if summary["failed_ids"]:
        print("失败:", summary["failed_ids"])
