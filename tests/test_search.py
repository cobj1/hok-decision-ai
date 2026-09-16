"""P2 搜索骨架测试。"""

from __future__ import annotations

import pytest

from hok_decision.schema.models import DamageConfig, MapData, ResourcePoint
from hok_decision.search.actions import generate_candidates
from hok_decision.search.planner import SearchState, beam_search, terminal_value
from hok_decision.simulator.units import Hero


def _state() -> SearchState:
    blue = Hero(
        unit_id="blue_1",
        name="蓝方",
        side="blue",
        position=(50.0, 0.0),
        hp=1000.0,
        max_hp=1000.0,
        physical_attack=100.0,
        physical_defense=0.0,
        attack_range=60.0,
        attack_interval=1.0,
        move_speed=100.0,
    )
    red = Hero(
        unit_id="red_1",
        name="红方",
        side="red",
        position=(400.0, 0.0),
        hp=1000.0,
        max_hp=1000.0,
        physical_attack=0.0,
        physical_defense=0.0,
        attack_range=50.0,
        attack_interval=1.0,
        move_speed=100.0,
    )
    return SearchState(heroes={"blue_1": blue, "red_1": red})


def _map_data() -> MapData:
    return MapData(
        version="S45",
        name="test",
        lanes=[],
        resource_points=[
            ResourcePoint(
                id="tower_blue_1",
                name="蓝方塔",
                kind="tower",
                side="blue",
                position=(0.0, 0.0),
                spawn_rule="static",
            ),
            ResourcePoint(
                id="tower_red_1",
                name="红方塔",
                kind="tower",
                side="red",
                position=(500.0, 0.0),
                spawn_rule="static",
            ),
            ResourcePoint(
                id="small_blue_1",
                name="蓝方野",
                kind="camp",
                side="blue",
                position=(100.0, 0.0),
                spawn_rule="fixed",
            ),
            ResourcePoint(
                id="small_red_1",
                name="红方野",
                kind="camp",
                side="red",
                position=(400.0, 0.0),
                spawn_rule="fixed",
            ),
        ],
    )


def _anchors() -> dict:
    return {
        "fountain_blue": {"position": (0.0, 0.0)},
        "fountain_red": {"position": (500.0, 0.0)},
    }


def test_candidate_generation_bounded_and_unique() -> None:
    candidates = generate_candidates(_state(), _map_data(), _anchors())
    assert 1 <= len(candidates) <= 8
    positions = [a.target_position for a in candidates]
    assert len(positions) == len(set(positions))


def test_beam_search_prefers_attack_weak_nearby_enemy() -> None:
    state = _state()
    best = beam_search(
        state,
        map_data=_map_data(),
        anchors=_anchors(),
        horizon_s=16.0,
        beam_width=5,
        action_duration_s=4.0,
    )
    assert best is not None
    assert best.actions[0].kind == "attack_enemy"


def test_beam_search_deterministic() -> None:
    first = beam_search(_state(), map_data=_map_data(), anchors=_anchors())
    second = beam_search(_state(), map_data=_map_data(), anchors=_anchors())
    assert first is not None and second is not None
    assert first.value == second.value
    assert [a.label for a in first.actions] == [a.label for a in second.actions]


def test_beam_search_with_decision_output() -> None:
    best = beam_search(
        _state(),
        map_data=_map_data(),
        anchors=_anchors(),
        with_decision_output=True,
    )
    assert best is not None
    output = best.decision_output
    assert output is not None
    assert len(output.goal_vector) == 6
    assert output.target_position is not None
    assert 0.0 <= output.difficulty <= 1.0
    assert output.description


def test_terminal_value_encourages_kills_and_captures() -> None:
    state = _state()
    base = terminal_value(state)
    state.captured_resources.add("small_blue_1")
    assert terminal_value(state) > base
    state.heroes["red_1"].alive = False
    assert terminal_value(state) > base
