"""王者峡谷地形图源下载与地形数据整理。

本模块只负责地形基建产物：
- `data/S45/raw/map_images/`：候选完整地图/小地图图源；
- `data/S45/normalized/terrain.json`：图片元数据 + 像素坐标系定义 + scale 假设；
- `data/S45/normalized/terrain_anchors.json`：资源点/地形锚点（无可靠来源时留空）；
- `data/S45/normalized/map_coordinates_notes.md`：坐标系研究笔记。

原则：不编造坐标；所有坐标必须可溯源。当前未发现带可信像素坐标的图源，
因此 terrain_anchors.json 中所有锚点 position=null、status=unverified。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hok_decision.config import GAME_VERSION, NORMALIZED_ROOT, RAW_ROOT
from hok_decision.crawlers.base import CrawlerError, fetch, save_raw

TERRAIN_IMAGE_DIR: Path = RAW_ROOT / "map_images"
TERRAIN_JSON: Path = NORMALIZED_ROOT / "terrain.json"
ANCHORS_JSON: Path = NORMALIZED_ROOT / "terrain_anchors.json"
NOTES_MD: Path = NORMALIZED_ROOT / "map_coordinates_notes.md"

SCALE_SOURCE_URL = "https://baijiahao.baidu.com/s?id=1620356316964077663"
SCALE_SOURCE_NAME = "百家号：王者荣耀：王者峡谷有多大？经实测，红蓝方泉水的距离只有100米"
RESOURCE_SOURCE_URL = "https://news.17173.com/z/pvp/content/07302025/102109377.shtml"

# 候选图源：只收录已验证可下载的 URL。source_level 用于区分来源可信度。
CANDIDATE_IMAGES: list[dict[str, Any]] = [
    {
        "id": "18183_monster_map",
        "file": "18183_monster_map_600x438.jpg",
        "url": "http://img.18183.com/uploads/allimg/170705/36-1FF51K001.jpg",
        "source_page": "http://www.18183.com/yxzjol/201707/888049.html",
        "source_title": "王者荣耀王者峡谷地图怪物分布与各BUFF作用详解",
        "source_level": "community_guide",
        "width": 600,
        "height": 438,
        "format": "jpeg",
        "notes": "疑似完整王者峡谷地图，带野怪/BUFF 分布标注；未做像素级锚点标注。",
    },
    {
        "id": "360kuai_kpl_map",
        "file": "360kuai_kpl_map_2000x1021.jpg",
        "url": "https://p0.ssl.img.360kuai.com/t01f2869bf5b2ace14f.jpg?size=2000x1021",
        "source_page": "https://www.360kuai.com/pc/9c94db9a996fee1d7?cota=3&kuai_so=1",
        "source_title": "红方的胜率碾压了蓝方?KPL的败方优先选蓝方,原来并不是福利",
        "source_level": "community_guide",
        "width": 2000,
        "height": 1021,
        "format": "jpeg",
        "notes": "高分辨率候选图，疑似对局/BP 用完整峡谷地图；需人工复核是否含 UI 偏移。",
    },
    {
        "id": "baidu_newbie_map",
        "file": "baidu_newbie_map_538x500.jpg",
        "url": "https://img0.baidu.com/it/u=2533584148,3392131871&fm=253&fmt=auto&app=138&f=JPEG?w=538&h=500",
        "source_page": "http://mbd.baidu.com/newspage/data/dtlandingsuper?nid=dt_5520797692466501742",
        "source_title": "王者荣耀新手必看:王者峡谷地图全解析",
        "source_level": "community_guide",
        "width": 538,
        "height": 500,
        "format": "jpeg",
        "notes": "接近正方形的峡谷全解析图，适合后续人工标定参考。",
    },
    {
        "id": "18183_s17_map",
        "file": "18183_s17_map_600x627.jpg",
        "url": "https://c-img.18183.com/images/2019/10/18/966a11f7886d02775ba2a08587213998.jpg@!18183",
        "source_page": "http://www.18183.com/yxzjol/201910/2432352.html",
        "source_title": "王者荣耀S17新赛季王者峡谷有哪些变化?王者峡谷优化调整一览!",
        "source_level": "community_guide",
        "width": 600,
        "height": 627,
        "format": "jpeg",
        "notes": "版本优化调整图，可能含局部/装饰性内容，标定前需人工确认完整地图边界。",
    },
    {
        "id": "9game_fullview_map",
        "file": "9game_fullview_map_500x281.png",
        "url": "https://image.9game.cn/2017/9/4/17944017.jpg",
        "source_page": "https://www.9game.cn/news/1896313.html",
        "source_title": "王者荣耀新版本地图全貌展示 最新王者峡谷效果预览",
        "source_level": "community_guide",
        "width": 500,
        "height": 281,
        "format": "png",
        "notes": "原标题为地图全貌展示；分辨率较低，仅作构图参考。",
    },
    {
        "id": "baike_wangzhexiagu",
        "file": "baike_wangzhexiagu_1200x696.jpg",
        "url": "https://bkimg.cdn.bcebos.com/pic/aa64034f78f0f7367a156ec80155b319eac4130b",
        "source_page": "https://baike.baidu.com/item/王者峡谷/22365564",
        "source_title": "百度百科-王者峡谷（《王者荣耀》5 VS 5 地图）概述图",
        "source_level": "encyclopedia",
        "width": 1200,
        "height": 696,
        "format": "jpeg",
        "notes": "百科概述图，需人工确认是否为完整俯视地图而非原画/概念图。",
    },
    {
        "id": "pvp_official_jungle_map",
        "file": "pvp_official_jungle_map_640x360.jpg",
        "url": "http://shp.qpic.cn/inewsnopic/449774817/b77962913ae12538bd4f083aa89a5431/0",
        "source_page": "https://pvp.qq.com/webplat/info/news_version3/15592/15748/15749/15753/m12014/201510/392243.shtml",
        "source_title": "主宰之怒！《王者荣耀》5V5野区深入分析",
        "source_level": "official",
        "width": 640,
        "height": 360,
        "format": "jpeg",
        "notes": "官方野区分析配图；文章注明红/蓝/黑/绿点为野怪分布，疑似完整峡谷地图，需人工确认无 UI 偏移。",
    },
]


def _retrieved_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def download_terrain_images(*, force: bool = False) -> list[Path]:
    """下载候选完整地图图源到 raw/map_images/，返回成功路径。"""
    TERRAIN_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for item in CANDIDATE_IMAGES:
        target = TERRAIN_IMAGE_DIR / item["file"]
        if target.exists() and target.stat().st_size > 0 and not force:
            outputs.append(target)
            continue
        try:
            response = fetch(item["url"], timeout=60.0, retries=2)
        except CrawlerError as exc:
            print(f"[terrain_assets] 跳过 {item['id']}: {exc}")
            continue
        path = save_raw(response.content, root=TERRAIN_IMAGE_DIR, filename=item["file"])
        outputs.append(path)
    return outputs


def _image_entries() -> list[dict[str, Any]]:
    now = _retrieved_at()
    entries: list[dict[str, Any]] = []
    for item in CANDIDATE_IMAGES:
        entries.append(
            {
                "id": item["id"],
                "file": f"data/S45/raw/map_images/{item['file']}",
                "url": item["url"],
                "source_page": item["source_page"],
                "source_title": item["source_title"],
                "source_level": item["source_level"],
                "width": item["width"],
                "height": item["height"],
                "format": item["format"],
                "retrieved_at": now,
                "notes": item["notes"],
            }
        )
    return entries


def build_terrain_json() -> dict[str, Any]:
    """构造 terrain.json（图片元数据 + 坐标系定义）。"""
    now = _retrieved_at()
    return {
        "version": GAME_VERSION,
        "name": "王者峡谷 5v5 地形锚点基建",
        "description": "王者峡谷完整地图/小地图候选图源、像素坐标系定义与地形锚点 schema。坐标标注遵循可溯源原则：无可靠来源时留空，不编造。",
        "images": _image_entries(),
        "coordinate_system": {
            "origin": "top-left",
            "axis": {"x": "right", "y": "down"},
            "unit": "px",
            "reference_image": "18183_monster_map",
            "scale": {
                "game_distance_to_meter": {
                    "ratio": 0.01,
                    "unit": "m per game-distance",
                    "derivation": "1800 游戏距离 ≈ 18 米 ⇒ 1 游戏距离 ≈ 0.01 米（即 100 距离 ≈ 1 米）",
                    "status": "unverified",
                    "source_url": SCALE_SOURCE_URL,
                    "source_name": SCALE_SOURCE_NAME,
                    "retrieved_at": now,
                },
                "fountain_distance_m": {
                    "value": "90-100",
                    "status": "unverified",
                    "source_url": SCALE_SOURCE_URL,
                    "source_name": SCALE_SOURCE_NAME,
                    "retrieved_at": now,
                },
                "pixel_to_meter": None,
                "notes": "当前没有官方像素-米换算或地图尺寸数值。百家号为二手实测，未证实；禁止据此推算坐标。后续应通过录屏/游戏内测距标定。",
            },
        },
        "anchor_schema": {
            "id": "锚点 id；资源点与 data/S45/normalized/map.json 的 resource_points[].id 对齐",
            "name": "显示名",
            "kind": "buff|camp|neutral|tower|other",
            "side": "blue|red|neutral",
            "position": "[x, y] 像素坐标，原点左上角；未标定为 null",
            "status": "verified|programmatic|unverified",
            "method": "manual|image_processing|web_source|none",
            "source_url": "坐标来源 URL；无来源则使用机制来源并注明待标定",
            "retrieved_at": "ISO 8601 UTC",
            "notes": "说明/待办",
        },
        "notes": [
            "官方未公开 5v5 王者峡谷的资源点像素坐标；网上未发现带明确坐标网格的可信图源。",
            "当前 terrain_anchors.json 全部锚点 position=null、status=unverified，等待人工标定或录屏地形定位管线回填。",
            "候选图仅用于人工标定/后续图像配准，不代表任何坐标已被证实。",
            "坐标系约定：像素原点为图片左上角，x 向右，y 向下；单位 px。",
        ],
    }


def build_terrain_anchors() -> dict[str, Any]:
    """构造 terrain_anchors.json。

    从 map.json 的 resource_points 派生全部资源点，另加泉水地形锚点。
    当前无可靠坐标来源，所有 position 保持 null。
    """
    now = _retrieved_at()
    map_path = NORMALIZED_ROOT / "map.json"
    map_data = json.loads(map_path.read_text(encoding="utf-8"))

    anchors: list[dict[str, Any]] = []
    for rp in map_data["resource_points"]:
        anchors.append(
            {
                "id": rp["id"],
                "name": rp["name"],
                "kind": rp["kind"],
                "side": rp["side"],
                "position": None,
                "status": "unverified",
                "method": "none",
                "source_url": RESOURCE_SOURCE_URL,
                "retrieved_at": now,
                "notes": rp.get("notes", "") + "；无可靠像素坐标来源，待人工标定。",
            }
        )

    anchors.extend(
        [
            {
                "id": "fountain_blue",
                "name": "蓝方泉水",
                "kind": "other",
                "side": "blue",
                "position": None,
                "status": "unverified",
                "method": "none",
                "source_url": SCALE_SOURCE_URL,
                "retrieved_at": now,
                "notes": "蓝方出生点/泉水；仅有红蓝泉水距离 90-100m 的二手实测，无像素坐标，待人工标定。",
            },
            {
                "id": "fountain_red",
                "name": "红方泉水",
                "kind": "other",
                "side": "red",
                "position": None,
                "status": "unverified",
                "method": "none",
                "source_url": SCALE_SOURCE_URL,
                "retrieved_at": now,
                "notes": "红方出生点/泉水；仅有红蓝泉水距离 90-100m 的二手实测，无像素坐标，待人工标定。",
            },
        ]
    )

    return {
        "version": GAME_VERSION,
        "name": "王者峡谷地形/资源锚点（像素坐标）",
        "reference_image": "18183_monster_map",
        "coordinate_system_ref": "terrain.json#coordinate_system",
        "anchors": anchors,
    }


def write_terrain_data() -> list[Path]:
    """下载图源并写出地形基建三个数据文件，返回产物路径。"""
    download_terrain_images()
    outputs = [
        TERRAIN_JSON,
        ANCHORS_JSON,
        NOTES_MD,
    ]
    TERRAIN_JSON.write_text(
        json.dumps(build_terrain_json(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ANCHORS_JSON.write_text(
        json.dumps(build_terrain_anchors(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not NOTES_MD.exists():
        NOTES_MD.write_text(
            "# 王者峡谷坐标系与比例尺研究笔记\n\n"
            "见 data/S45/raw/map_images 与 terrain.json。当前所有比例尺假设均未证实。\n",
            encoding="utf-8",
        )
    return outputs


if __name__ == "__main__":  # pragma: no cover
    for path in write_terrain_data():
        print(path)
