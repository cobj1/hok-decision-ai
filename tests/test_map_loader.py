"""地图数据加载测试。"""

from __future__ import annotations

import json

import pytest

from hok_decision.config import GAME_VERSION
from hok_decision.schema.models import Lane, MapData, MapMechanics, ResourcePoint
from hok_decision.simulator.map import MapDataError, load_map


def _write_fixture(tmp_path, *, version: str = GAME_VERSION) -> None:
    data = MapData(
        version=version,
        name="王者峡谷",
        lanes=[
            Lane(id="farm", name="发育路"),
            Lane(id="clash", name="对抗路"),
            Lane(id="mid", name="中路"),
        ],
        resource_points=[
            ResourcePoint(
                id="red_falcon",
                name="红隼",
                kind="neutral",
                initial_spawn_s=30.0,
            ),
            ResourcePoint(id="tower_blue_1", name="蓝方防御塔1", kind="tower", side="blue"),
        ],
        mechanics=MapMechanics(dragon_bond_s=90.0),
    )
    (tmp_path / "map.json").write_text(
        json.dumps(data.model_dump(), ensure_ascii=False), encoding="utf-8"
    )


def test_load_map_ok(tmp_path) -> None:
    _write_fixture(tmp_path)
    data = load_map(data_root=tmp_path)
    assert data.name == "王者峡谷"
    assert len(data.lanes) == 3
    falcon = next(p for p in data.resource_points if p.id == "red_falcon")
    assert falcon.initial_spawn_s == 30.0
    assert data.mechanics is not None and data.mechanics.dragon_bond_s == 90.0


def test_load_map_version_mismatch(tmp_path) -> None:
    _write_fixture(tmp_path, version="S1")
    with pytest.raises(MapDataError, match="版本不匹配"):
        load_map(data_root=tmp_path)


def test_load_map_missing(tmp_path) -> None:
    with pytest.raises(MapDataError, match="不存在"):
        load_map(data_root=tmp_path)
