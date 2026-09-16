"""地图数据加载：从数据层读取 5v5 王者峡谷 MapData。"""

from __future__ import annotations

import json
from pathlib import Path

from hok_decision.config import GAME_VERSION, NORMALIZED_ROOT
from hok_decision.schema.models import MapData


class MapDataError(RuntimeError):
    """地图数据缺失或损坏。"""


def load_map(version: str = GAME_VERSION, *, data_root: Path | None = None) -> MapData:
    """加载指定版本的归一化地图数据。"""
    root = data_root or NORMALIZED_ROOT
    path = root / "map.json"
    if not path.exists():
        raise MapDataError(f"地图数据不存在: {path}（先运行爬虫）")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise MapDataError(f"地图数据解析失败: {path}: {exc}") from exc
    if payload.get("version") != version:
        raise MapDataError(
            f"地图数据版本不匹配: 数据={payload.get('version')}, 请求={version}"
        )
    return MapData.model_validate(payload)


def load_terrain_anchors(
    version: str = GAME_VERSION, *, data_root: Path | None = None
) -> dict[str, dict]:
    """加载正式地形锚点，返回 {anchor_id: anchor} 映射。"""
    root = data_root or NORMALIZED_ROOT
    path = root / "terrain_anchors.json"
    if not path.exists():
        raise MapDataError(f"地形锚点数据不存在: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise MapDataError(f"地形锚点解析失败: {path}: {exc}") from exc
    if payload.get("version") != version:
        raise MapDataError(
            f"地形锚点版本不匹配: 数据={payload.get('version')}, 请求={version}"
        )
    return {anchor["id"]: anchor for anchor in payload["anchors"]}
