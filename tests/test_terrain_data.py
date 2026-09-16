"""地形锚点基建数据测试。

验证：
- terrain.json / terrain_anchors.json 可解析且结构完整；
- 锚点 id 与 map.json 的 resource_points id 对齐；
- 有坐标的锚点必须可信、可溯源、且在参考图片边界内；
- unverified 锚点不允许伪造 position。
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NORMALIZED = PROJECT_ROOT / "data" / "S45" / "normalized"
RAW = PROJECT_ROOT / "data" / "S45" / "raw"


def _load_json(relative: str) -> dict:
    path = NORMALIZED / relative
    assert path.exists(), f"缺少数据文件: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_terrain_json_structure() -> None:
    terrain = _load_json("terrain.json")
    assert terrain["version"] == "S45"
    assert terrain["images"]
    assert terrain["coordinate_system"]["origin"] == "top-left"
    assert terrain["coordinate_system"]["axis"] == {"x": "right", "y": "down"}
    assert terrain["coordinate_system"]["unit"] == "px"
    assert "scale" in terrain["coordinate_system"]
    assert "anchor_schema" in terrain


def test_terrain_images_exist_and_metadata_sane() -> None:
    terrain = _load_json("terrain.json")
    seen: set[str] = set()
    for image in terrain["images"]:
        assert image["id"] not in seen
        seen.add(image["id"])
        assert image["width"] > 0
        assert image["height"] > 0
        assert image["url"].startswith(("http://", "https://", "file://"))
        assert image["retrieved_at"]
        # 文件路径以项目根为基准
        file_path = PROJECT_ROOT / image["file"]
        assert file_path.exists(), f"地图图片不存在: {file_path}"
        assert file_path.stat().st_size > 0


def test_anchor_ids_cover_map_resource_points() -> None:
    map_data = _load_json("map.json")
    anchors = _load_json("terrain_anchors.json")["anchors"]
    anchor_ids = {anchor["id"] for anchor in anchors}
    map_resource_ids = {rp["id"] for rp in map_data["resource_points"]}
    missing = map_resource_ids - anchor_ids
    assert not missing, f"map.json 中有锚点未覆盖: {sorted(missing)}"
    assert len(anchor_ids) == len(anchors), "锚点 id 重复"


def test_unverified_anchors_have_no_position() -> None:
    anchors = _load_json("terrain_anchors.json")["anchors"]
    for anchor in anchors:
        if anchor["status"] == "unverified":
            assert anchor["position"] is None, (
                f"unverified 锚点 {anchor['id']} 不应有伪造 position"
            )
            assert anchor["method"] == "none"
            assert anchor["source_url"]
            assert anchor["retrieved_at"]


def test_positioned_anchors_are_traceable_and_in_bounds() -> None:
    terrain = _load_json("terrain.json")
    anchors = _load_json("terrain_anchors.json")["anchors"]
    ref_id = terrain["coordinate_system"]["reference_image"]
    ref_image = next(img for img in terrain["images"] if img["id"] == ref_id)
    width, height = ref_image["width"], ref_image["height"]

    for anchor in anchors:
        pos = anchor["position"]
        if pos is None:
            continue
        assert anchor["status"] != "unverified", (
            f"已定位锚点 {anchor['id']} 不能仍标 unverified"
        )
        assert anchor["method"] != "none", (
            f"已定位锚点 {anchor['id']} 必须记录 method"
        )
        assert anchor["source_url"], f"已定位锚点 {anchor['id']} 必须记录 source_url"
        x, y = pos
        assert 0 <= x < width, f"{anchor['id']} x 超出参考图边界: {x}"
        assert 0 <= y < height, f"{anchor['id']} y 超出参考图边界: {y}"
