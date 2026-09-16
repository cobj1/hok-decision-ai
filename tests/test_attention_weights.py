"""注意力权重评估器测试（无 torch 环境回退路径）。"""

from __future__ import annotations

from pathlib import Path

from hok_decision.search.attention_weights import (
    AttentionWeightedValue,
    VALUE_DIMS,
    value_features_from_state,
)
from hok_decision.search.planner import SearchState
from hok_decision.simulator.units import Hero


def _state() -> SearchState:
    blue = Hero(
        unit_id="blue_105",
        name="我方",
        side="blue",
        position=(50.0, 350.0),
        hp=900.0,
        max_hp=1000.0,
    )
    red = Hero(
        unit_id="red_105",
        name="敌方",
        side="red",
        position=(380.0, 40.0),
        hp=800.0,
        max_hp=1000.0,
    )
    state = SearchState(heroes={"blue_105": blue, "red_105": red})
    state.gold["blue_105"] = 6000.0
    state.gold["red_105"] = 4000.0
    state.captured_resources.add("tyrant")
    state.captured_resources.add("blue_buff_1")
    state.tower_hp["tower_red_1"] = 0.5
    return state


def test_value_features_shape_and_semantics() -> None:
    features = value_features_from_state(_state())
    assert len(features) == len(VALUE_DIMS) == 10
    assert features[0] == 0.9 - 0.8
    assert features[1] == 0.2  # (6000-4000)/10000
    assert features[5] == 1.0  # tyrant
    assert features[4] == 1.0  # blue_buff
    assert features[8] == 0.5  # 红方塔损


def test_attention_value_fallback_is_linear_and_deterministic() -> None:
    state = _state()
    evaluator = AttentionWeightedValue(
        context=[0.1] * 13,
        model_path=Path("E:/definitely_missing/model.pt"),
        base_weights=[1.0] * 10,
    )
    features = value_features_from_state(state)
    expected = sum(features)
    assert evaluator.evaluate(state) == expected
    assert evaluator.evaluate(state) == evaluator.evaluate(state)
