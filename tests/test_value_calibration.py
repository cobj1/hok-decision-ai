"""价值模型与校准测试。"""

from __future__ import annotations

import json

import numpy as np

from hok_decision.config import NORMALIZED_ROOT
from hok_decision.search.calibration import generate_calibration_data, fit_logistic
from hok_decision.search.planner import SearchState, extract_features, terminal_value
from hok_decision.search.value import ValueWeights, load_weights
from hok_decision.simulator.units import Hero


def _state() -> SearchState:
    blue = Hero(
        unit_id="blue_1",
        name="蓝方",
        side="blue",
        position=(0.0, 0.0),
        hp=500.0,
        max_hp=1000.0,
        physical_attack=100.0,
        physical_defense=0.0,
    )
    red = Hero(
        unit_id="red_1",
        name="红方",
        side="red",
        position=(100.0, 0.0),
        hp=250.0,
        max_hp=1000.0,
        physical_attack=100.0,
        physical_defense=0.0,
    )
    state = SearchState(heroes={"blue_1": blue, "red_1": red})
    state.captured_resources.add("camp_1")
    return state


def test_extract_features() -> None:
    state = _state()
    hp_diff, resource, kill_diff = extract_features(state)
    assert hp_diff == 0.25
    assert resource == 1.0
    assert kill_diff == 0.0
    state.heroes["red_1"].alive = False
    assert extract_features(state)[2] == 1.0


def test_terminal_value_uses_calibrated_weights() -> None:
    weights = load_weights()
    state = _state()
    hp_diff, resource, kill_diff = extract_features(state)
    assert terminal_value(state, weights=weights) == weights.score(
        hp_diff, resource, kill_diff
    )


def test_calibrated_weights_file() -> None:
    path = NORMALIZED_ROOT / "value_weights.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = payload["weights"]
    assert weights["w_hp"] > 0
    assert weights["w_kill"] > 0
    assert payload["metrics"]["accuracy"] > 0.7


def test_calibration_dataset_and_fit() -> None:
    dataset = generate_calibration_data(episodes=5)
    assert dataset.X.shape[0] == dataset.y.shape[0]
    assert dataset.X.shape[0] > 5
    assert dataset.X.shape[1] == 3
    theta, metrics = fit_logistic(dataset, iterations=200)
    assert theta.shape == (4,)
    assert 0.0 <= metrics["accuracy"] <= 1.0
