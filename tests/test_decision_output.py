"""决策输出层测试。"""

from __future__ import annotations

from hok_decision.search.decision_output import (
    BENEFIT_DIMS,
    DecisionOutput,
    build_decision_output,
    estimate_difficulty,
    relative_position,
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
    return SearchState(heroes={"blue_105": blue, "red_105": red})


def test_build_decision_output_vector() -> None:
    before = _state()
    after = _state()
    after.heroes["red_105"].alive = False  # 击杀
    after.gold["blue_105"] = 500.0
    after.captured_resources.add("overlord")
    output = build_decision_output(
        initial_state=before,
        final_state=after,
        target_position=(285.0, 275.0),
        description="打龙后推进",
    )
    assert isinstance(output, DecisionOutput)
    assert len(output.goal_vector) == len(BENEFIT_DIMS) == 6
    assert output.goal_vector[0] > 0  # 击杀
    assert output.goal_vector[1] > 0  # 经济
    assert output.goal_vector[3] > 0  # 龙
    assert 0.0 <= output.difficulty <= 1.0
    assert output.target_position == (285.0, 275.0)


def test_relative_position_and_difficulty() -> None:
    state = _state()
    rel = relative_position((380.0, 40.0), (50.0, 350.0), self_side="blue")
    assert rel["lane"] == "上路"
    assert rel["side"] == "敌方半区"
    difficulty = estimate_difficulty(target=(380.0, 40.0), state=state)
    assert 0.0 <= difficulty <= 1.0
