"""资源点刷新时间表（当前收集的公开来源）。

把刷新规则写入 data/S45/normalized/map.json 的 resource_points。
数值来源为公开攻略，非官方确认，保留 source 溯源；后续以官方/实测校准。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hok_decision.config import GAME_VERSION, NORMALIZED_ROOT
from hok_decision.crawlers.base import source_ref

# 公开来源
_S23_URL = "https://www.9game.cn/wzry/8443363.html"
_ANCIENT_URL = "https://a.9game.cn/wzry/9627123.html"
_JUNGLE_URL = "https://a.9game.cn/wzry/11390256.html"


def _timing_source() -> dict[str, Any]:
    return source_ref(
        "https://www.9game.cn/wzry/8443363.html",
        "王者荣耀野怪刷新时间表（S23，含远古生物/野区，需随版本复核）",
        is_primary=False,
    )


def _base_timing(
    *,
    initial: float | None,
    respawn: float | None,
    rule: str,
    duration: float | None = None,
    active_until: float | None = None,
    possible: list[list[float]] | None = None,
    note: str = "",
) -> dict[str, Any]:
    return {
        "initial_spawn_s": initial,
        "respawn_s": respawn,
        "buff_duration_s": duration,
        "spawn_rule": rule,
        "possible_locations": possible,
        "active_until_s": active_until,
        "timing_notes": note,
    }


def _apply_timings(map_path: Path) -> None:
    data = json.loads(map_path.read_text(encoding="utf-8"))
    points = data["resource_points"]
    for rp in points:
        rid = rp["id"]
        if rp["kind"] == "tower":
            rp.update(_base_timing(initial=None, respawn=None, rule="static", note="防御塔常驻"))
        elif rid.startswith("small_"):
            rp.update(
                _base_timing(initial=30.0, respawn=80.0, rule="fixed", note="普通野怪：首次30s，间隔80s")
            )
        elif rid in {"red_buff", "blue_buff", "red_buff_red_side", "blue_buff_red_side"}:
            rp.update(
                _base_timing(
                    initial=30.0,
                    respawn=90.0,
                    rule="fixed",
                    duration=70.0,
                    note="红蓝Buff：首次30s，击杀后90s刷新，增益70s",
                )
            )
        elif rid in {"overlord", "tyrant"}:
            rp.update(
                _base_timing(
                    initial=120.0,
                    respawn=240.0,
                    rule="fixed",
                    note="暴君/主宰：开局2分钟，间隔4分钟（S23表；不同赛季有3/5分钟口径，待复核）",
                )
            )
        elif rid == "space_spirit":
            rp.update(
                _base_timing(
                    initial=60.0,
                    respawn=60.0,
                    rule="timed_window",
                    active_until=240.0,
                    note="空间之灵：首刷60s，前4分钟击杀后60s刷新；4分钟后停止",
                )
            )
        elif rid == "red_falcon":
            rp.update(
                _base_timing(
                    initial=30.0,
                    respawn=None,
                    rule="once",
                    note="红隼：30s首刷，全局仅一次",
                )
            )
        elif rid == "storm_dragon":
            # 从已定位的双龙坑取候选位置
            pos_map = {p["id"]: p.get("position") for p in points}
            possible = [pos_map.get("overlord"), pos_map.get("tyrant")]
            possible = [p for p in possible if p]
            rp.update(
                _base_timing(
                    initial=1200.0,
                    respawn=240.0,
                    rule="dynamic",
                    possible=possible,
                    note="风暴龙王：20分钟后在双龙坑随机刷新；实际位置需实时从小地图观测回填",
                )
            )
        rp.setdefault("timing_source", _timing_source())

    # 顶层口径说明
    data["resource_timing_note"] = (
        "刷新时间来自公开攻略（9game S23 表/远古生物/野区介绍），非官方确认，"
        "已按资源点记录 timing_source；后续需按当前赛季公告复核。"
    )
    data["fetched_at"] = datetime.now(timezone.utc).isoformat()
    map_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_resource_timings() -> Path:
    """把刷新时间写入正式 map.json，返回路径。"""
    map_path = NORMALIZED_ROOT / "map.json"
    if not map_path.exists():
        raise FileNotFoundError(f"缺少 map.json: {map_path}")
    _apply_timings(map_path)
    return map_path


if __name__ == "__main__":
    path = apply_resource_timings()
    print(f"已写入: {path}")
