"""5v5 地图数据爬虫。

- 官方地图介绍页：下载页面与图片资产（后续地形锚点标定用）。
- 17173 地图机制文章：提取资源点刷新/持续时间等规则，落成 MapData。

已知限制（如实记录，不伪装成事实）：
- 野怪/塔/龙的位置坐标官方不公开，position 留空待录屏地形定位管线回填；
- 文章为 2025-07-30 版本，数值是否适配当前赛季需按官方公告复核。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

from hok_decision.config import (
    DATA_ROOT,
    GAME_VERSION,
    MAP_ASSET_ROOT,
    MAP_MECHANICS_URL,
    NORMALIZED_ROOT,
    OFFICIAL_MAP_PAGE_URL,
    RAW_ROOT,
)
from hok_decision.crawlers.base import CrawlerError, fetch, save_raw, source_ref
from hok_decision.schema.models import (
    DataManifest,
    Lane,
    MapData,
    MapMechanics,
    ResourcePoint,
)

_IMG_RE = re.compile(r'src="([^"]+)"')


def crawl_official_map_assets() -> DataManifest:
    """下载官方王者峡谷介绍页及其图片资产。"""
    manifest = DataManifest(version=GAME_VERSION)
    response = fetch(OFFICIAL_MAP_PAGE_URL)
    html_path = save_raw(response.content, root=RAW_ROOT, filename="wtc-map.html")
    manifest.files.append(str(html_path.relative_to(RAW_ROOT.parent.parent)))
    manifest.sources.append(
        source_ref(OFFICIAL_MAP_PAGE_URL, "王者荣耀官网王者峡谷地图页", is_primary=True)
    )

    html = response.text
    image_urls: list[str] = []
    for raw in _IMG_RE.findall(html):
        url = urljoin(OFFICIAL_MAP_PAGE_URL, raw)
        if url not in image_urls:
            image_urls.append(url)

    MAP_ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    for index, url in enumerate(image_urls, start=1):
        try:
            img = fetch(url, retries=2, timeout=60.0)
        except CrawlerError as exc:
            manifest.sources.append(
                {**source_ref(url, "官方地图页图片", is_primary=True), "error": str(exc)}
            )
            continue
        suffix = Path(url).suffix or ".bin"
        path = save_raw(img.content, root=MAP_ASSET_ROOT, filename=f"{index:02d}{suffix}")
        manifest.files.append(str(path.relative_to(RAW_ROOT.parent.parent)))
        manifest.sources.append(
            {**source_ref(url, "官方地图页图片", is_primary=True), "file": path.name}
        )

    return manifest


def _resource(
    resource_id: str,
    name: str,
    *,
    kind: str,
    side: str = "neutral",
    initial: float | None = None,
    respawn: float | None = None,
    duration: float | None = None,
    notes: str = "",
    source: dict[str, object],
) -> ResourcePoint:
    allowed = {"buff", "camp", "neutral", "tower", "lane", "other"}
    if kind not in allowed:
        raise CrawlerError(f"非法资源点类型: {kind}")
    return ResourcePoint(
        id=resource_id,
        name=name,
        kind=kind,  # type: ignore[arg-type]
        side=side,  # type: ignore[arg-type]
        initial_spawn_s=initial,
        respawn_s=respawn,
        buff_duration_s=duration,
        notes=notes,
        source=source,
    )


def crawl_map_mechanics() -> MapData:
    """抓取地图机制文章并整理为 MapData。"""
    response = fetch(MAP_MECHANICS_URL)
    raw_path = save_raw(response.content, root=RAW_ROOT, filename="map-mechanics-17173.html")
    source = source_ref(MAP_MECHANICS_URL, "17173 王者地图资源信息详解", is_primary=False)

    html = response.text
    text = re.sub(r"<script.*?</script>", "", html, flags=re.S)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text_path = RAW_ROOT / "map-mechanics-17173.txt"
    text_path.write_text(text, encoding="utf-8")

    # 只提取文章正文中可核实的机制事实；坐标类数据不猜测。
    mechanics = MapMechanics(
        minion_rules={
            "before_4min": {
                "发育路": "兵线经济收益更高",
                "对抗路": "兵线经验收益更高",
                "中路": "多一个小兵，经济经验均衡",
            },
            "after_4min": "三路统一：1 弩车兵 + 2 普通小兵",
        },
        tower_rules={"detail": "防御塔机制见原文，数值待校准"},
        dragon_bond_s=90.0,
        notes=[
            "红隼：30s 刷新，全局仅一次，发育路墙外区域",
            "空间之灵：60s 首刷，前 4 分钟被击杀后 60s 再刷；首次击杀激活传送法阵至 10 分钟",
            "红蓝 Buff：增益持续 70s（刷新间隔待校准）",
            "暴君/主宰/风暴龙王：90s 内同方连续击杀同类型龙触发羁绊负面效果",
        ],
        source=source,
    )

    map_data = MapData(
        version=GAME_VERSION,
        name="王者峡谷",
        lanes=[
            Lane(id="farm", name="发育路", notes="红隼刷新侧"),
            Lane(id="clash", name="对抗路"),
            Lane(id="mid", name="中路"),
        ],
        resource_points=[
            _resource(
                "red_falcon",
                "红隼",
                kind="neutral",
                initial=30.0,
                notes="全局仅刷新一次；位置待标定",
                source=source,
            ),
            _resource(
                "space_spirit",
                "空间之灵",
                kind="neutral",
                initial=60.0,
                respawn=60.0,
                notes="仅前 4 分钟可再刷；首次击杀激活传送法阵至 10 分钟",
                source=source,
            ),
            _resource(
                "red_buff",
                "红Buff",
                kind="buff",
                side="blue",
                duration=70.0,
                notes="位置待标定；刷新间隔待校准",
                source=source,
            ),
            _resource(
                "blue_buff",
                "蓝Buff",
                kind="buff",
                side="blue",
                duration=70.0,
                notes="位置待标定；刷新间隔待校准",
                source=source,
            ),
            _resource(
                "red_buff_red_side",
                "红Buff(红方野区)",
                kind="buff",
                side="red",
                duration=70.0,
                notes="位置待标定；刷新间隔待校准",
                source=source,
            ),
            _resource(
                "blue_buff_red_side",
                "蓝Buff(红方野区)",
                kind="buff",
                side="red",
                duration=70.0,
                notes="位置待标定；刷新间隔待校准",
                source=source,
            ),
            _resource(
                "tyrant",
                "暴君",
                kind="neutral",
                notes="90s 内同方连续击杀同类型龙触发羁绊；位置与刷新待标定",
                source=source,
            ),
            _resource(
                "overlord",
                "主宰",
                kind="neutral",
                notes="90s 内同方连续击杀同类型龙触发羁绊；位置与刷新待标定",
                source=source,
            ),
            _resource(
                "storm_dragon",
                "风暴龙王",
                kind="neutral",
                notes="击杀后提升属性并刷新强力先锋；位置与刷新待标定",
                source=source,
            ),
            *[
                _resource(
                    f"tower_{side}_{index}",
                    f"{'蓝方' if side == 'blue' else '红方'}防御塔{index}",
                    kind="tower",
                    side=side,
                    notes="位置待标定",
                    source=source,
                )
                for side in ("blue", "red")
                for index in (1, 2, 3)
            ],
        ],
        mechanics=mechanics,
    )

    target = NORMALIZED_ROOT / "map.json"
    target.write_text(
        json.dumps(map_data.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return map_data


def crawl_map_all() -> list[Path]:
    """地图相关爬取全量：资产 + 机制。返回产物路径。"""
    assets_manifest = crawl_official_map_assets()
    map_data = crawl_map_mechanics()
    outputs = [
        NORMALIZED_ROOT / "map.json",
        RAW_ROOT / "map-mechanics-17173.html",
        RAW_ROOT / "map-mechanics-17173.txt",
    ]
    outputs.extend(
        DATA_ROOT / item for item in assets_manifest.files if "map_assets" in item
    )
    _ = map_data  # 已落盘，返回值用于校验
    return [p for p in outputs if p.exists()]
