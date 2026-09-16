"""命令行入口：hok-data crawl [official|map|hero_stats|camp|terrain|timings|all] / hok-data status。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from hok_decision.config import DATA_ROOT, GAME_VERSION, NORMALIZED_ROOT
from hok_decision.crawlers.base import CrawlerError
from hok_decision.crawlers.camp_import import import_camp_data
from hok_decision.crawlers.hero_stats import crawl_all as crawl_hero_stats_all
from hok_decision.crawlers.map_sources import crawl_map_all
from hok_decision.crawlers.official_pvp import crawl_official, ensure_hero_stats_skeleton
from hok_decision.crawlers.resource_timings import apply_resource_timings
from hok_decision.crawlers.terrain_assets import write_terrain_data
from hok_decision.crawlers.vectorize_assets import build_all as vectorize_assets
from hok_decision.rules.hard_rules import ROLE_OPTIONS, write_role_input


def _write_manifest(payload: dict[str, object], filename: str) -> None:
    NORMALIZED_ROOT.mkdir(parents=True, exist_ok=True)
    (NORMALIZED_ROOT / filename).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _manifest(version: str, files: list[Path]) -> dict[str, object]:
    return {
        "version": version,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "files": [str(path) for path in files],
    }


def _cmd_crawl(source: str) -> int:
    NORMALIZED_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        if source in {"official", "all"}:
            manifest = crawl_official()
            ensure_hero_stats_skeleton()
            _write_manifest(manifest.model_dump(), "manifest_official.json")
            print(
                f"[official] 完成: 英雄/装备/召唤师技能已入库 "
                f"(version={GAME_VERSION})"
            )
        if source in {"map", "all"}:
            outputs = crawl_map_all()
            apply_resource_timings()
            _write_manifest(_manifest(GAME_VERSION, outputs), "manifest_map.json")
            print(f"[map] 完成: {len(outputs)} 个文件已入库 (version={GAME_VERSION})")
        if source in {"timings", "all"}:
            path = apply_resource_timings()
            _write_manifest(_manifest(GAME_VERSION, [path]), "manifest_timings.json")
            print(f"[timings] 完成: 资源点刷新时间已写入 {path.name}")
        if source in {"hero_stats", "all"}:
            stats_path, skills_path = crawl_hero_stats_all()
            _write_manifest(
                _manifest(GAME_VERSION, [stats_path, skills_path]),
                "manifest_hero.json",
            )
            print(
                f"[hero_stats] 完成: {stats_path.name} + {skills_path.name} "
                f"已入库 (version={GAME_VERSION})"
            )
        if source in {"camp", "all"}:
            try:
                stats_path, skills_path = import_camp_data()
            except FileNotFoundError as exc:
                print(f"[camp] 跳过: 营地数据源不存在 ({exc})", file=sys.stderr)
            else:
                _write_manifest(
                    _manifest(GAME_VERSION, [stats_path, skills_path]),
                    "manifest_camp.json",
                )
                print(
                    f"[camp] 完成: 英雄数值/技能已从营地数据更新 "
                    f"(version={GAME_VERSION})"
                )
        if source in {"terrain", "all"}:
            outputs = write_terrain_data()
            _write_manifest(_manifest(GAME_VERSION, outputs), "manifest_terrain.json")
            print(f"[terrain] 完成: {len(outputs)} 个文件已入库 (version={GAME_VERSION})")
        if source in {"vectorize", "all"}:
            from hok_decision.crawlers.official_equip_detail import fetch_equip_details

            fetch_equip_details()
            items_path, skills_path = vectorize_assets()
            _write_manifest(
                _manifest(GAME_VERSION, [items_path, skills_path]),
                "manifest_vector.json",
            )
            print(f"[vectorize] 完成: 装备详情+装备/技能向量已生成")
    except CrawlerError as exc:
        print(f"爬取失败: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_status() -> int:
    print(f"数据根目录: {DATA_ROOT}")
    print(f"当前版本: {GAME_VERSION}")
    if not DATA_ROOT.exists():
        print("（尚无数据，请先运行 hok-data crawl all）")
        return 0
    for path in sorted(DATA_ROOT.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            print(f"  {path.relative_to(DATA_ROOT)}  {size} B")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hok-data", description="王者荣耀数据结构化工具")
    sub = parser.add_subparsers(dest="command", required=True)
    crawl_parser = sub.add_parser("crawl", help="爬取并结构化数据")
    crawl_parser.add_argument(
        "source",
        choices=["official", "map", "hero_stats", "camp", "terrain", "timings", "vectorize", "all"],
        help="数据源范围（all=全部，耗时较长）",
    )
    sub.add_parser("status", help="查看已入库数据")
    sub.add_parser("calibrate", help="校准价值公式权重（简化1v1模拟对局 logistic 拟合）")
    role_parser = sub.add_parser("set-role", help="模型启动前手动输入局内角色（辅助装硬规则）")
    role_parser.add_argument("role", choices=list(ROLE_OPTIONS), help="局内角色")
    args = parser.parse_args(argv)

    if args.command == "crawl":
        return _cmd_crawl(args.source)
    if args.command == "calibrate":
        from hok_decision.search.calibration import calibrate

        weights = calibrate()
        print(f"校准完成: {weights.to_dict()}")
        return 0
    if args.command == "set-role":
        write_role_input(args.role)
        print(f"局内角色已设置: {args.role}")
        return 0
    if args.command == "status":
        return _cmd_status()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
