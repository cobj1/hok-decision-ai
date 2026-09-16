"""价值公式校准。

流程：
1. 在简化 1v1 地图上生成大量随机对局（蓝方策略 = 随机攻击/打资源，红方默认追击）；
2. 记录每个决策点的特征（hp_diff / resource / kill_diff）与最终胜负；
3. 用 logistic 回归拟合特征权重，得到校准后的 ValueWeights；
4. 写入 data/S45/normalized/value_weights.json，planner 自动加载。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from hok_decision.schema.models import DamageConfig, MapData, ResourcePoint
from hok_decision.search.actions import MacroAction
from hok_decision.search.planner import SearchState, extract_features, rollout_decision
from hok_decision.search.value import ValueWeights, save_weights
from hok_decision.simulator.units import Hero

_DECISION_S = 4.0
_DT = 0.25
_MAX_EPISODE_S = 40.0
_DEFAULT_EPISODES = 400


@dataclass
class CalibrationDataset:
    X: np.ndarray  # (n, 3)
    y: np.ndarray  # (n,) 0/1
    episodes: int


def _make_map() -> MapData:
    return MapData(
        version="S45",
        name="calibration",
        lanes=[],
        resource_points=[
            ResourcePoint(
                id="camp_blue",
                name="蓝方野",
                kind="camp",
                side="blue",
                position=(100.0, 0.0),
                spawn_rule="fixed",
            ),
            ResourcePoint(
                id="camp_red",
                name="红方野",
                kind="camp",
                side="red",
                position=(400.0, 0.0),
                spawn_rule="fixed",
            ),
        ],
    )


def _random_hero(rng: random.Random, side: str) -> Hero:
    x = rng.uniform(0.0, 220.0) if side == "blue" else rng.uniform(280.0, 500.0)
    hp = rng.uniform(500.0, 1500.0)
    attack = rng.uniform(50.0, 150.0)
    defense = rng.uniform(0.0, 100.0)
    return Hero(
        unit_id=f"{side}_1",
        name="蓝方" if side == "blue" else "红方",
        side=side,
        position=(x, 0.0),
        hp=hp,
        max_hp=hp,
        physical_attack=attack,
        physical_defense=defense,
        magic_defense=defense,
        attack_range=60.0,
        attack_interval=0.8,
        move_speed=100.0,
        crit_rate=0.0,
        crit_multiplier=2.0,
    )


def generate_calibration_data(
    episodes: int = _DEFAULT_EPISODES, *, seed: int = 20260828
) -> CalibrationDataset:
    """生成校准数据：每个决策点一条 (特征, 胜负) 样本。"""
    rng = random.Random(seed)
    map_data = _make_map()
    cfg = DamageConfig(enemy_damage_mode="expected")
    rows: list[tuple[float, float, float, float]] = []

    for _ in range(episodes):
        blue = _random_hero(rng, "blue")
        red = _random_hero(rng, "red")
        state = SearchState(heroes={blue.unit_id: blue, red.unit_id: red})

        while state.game_time < _MAX_EPISODE_S and blue.alive and red.alive:
            hp_diff, resource, kill_diff = extract_features(state)
            # 蓝方策略：随机攻击或打最近资源，制造不同资源/血量分布
            if rng.random() < 0.6 or not map_data.resource_points:
                action = MacroAction(
                    kind="attack_enemy",
                    target_id=red.unit_id,
                    target_position=red.position,
                    duration_s=_DECISION_S,
                )
            else:
                rp = min(
                    map_data.resource_points,
                    key=lambda r: (
                        (blue.position[0] - r.position[0]) ** 2
                        + (blue.position[1] - r.position[1]) ** 2
                    ),
                )
                action = MacroAction(
                    kind="capture_resource",
                    target_id=rp.id,
                    target_position=rp.position,
                    duration_s=_DECISION_S,
                )

            rollout_decision(
                state,
                action,
                map_data=map_data,
                duration_s=_DECISION_S,
                damage_config=cfg,
                dt=_DT,
            )
            blue_won = _outcome(state)
            rows.append((hp_diff, resource, kill_diff, blue_won))

        # 终局样本：让 kill_diff 在数据中有变化（击杀后 +1/-1）
        hp_diff, resource, kill_diff = extract_features(state)
        rows.append((hp_diff, resource, kill_diff, _outcome(state)))

    X = np.asarray([row[:3] for row in rows], dtype=float)
    y = np.asarray([row[3] for row in rows], dtype=float)
    return CalibrationDataset(X=X, y=y, episodes=episodes)


def _outcome(state: SearchState) -> float:
    blues = [h for h in state.heroes.values() if h.side == "blue"]
    reds = [h for h in state.heroes.values() if h.side == "red"]
    blue_alive = any(h.alive for h in blues)
    red_alive = any(h.alive for h in reds)
    if blue_alive and not red_alive:
        return 1.0
    if red_alive and not blue_alive:
        return 0.0
    blue_frac = sum(h.hp / h.max_hp for h in blues if h.max_hp > 0)
    red_frac = sum(h.hp / h.max_hp for h in reds if h.max_hp > 0)
    return 1.0 if blue_frac >= red_frac else 0.0


def fit_logistic(
    dataset: CalibrationDataset,
    *,
    learning_rate: float = 0.05,
    iterations: int = 3000,
    l2: float = 1e-4,
) -> tuple[np.ndarray, dict[str, float]]:
    """拟合 logistic 回归，返回 (theta=[w_hp,w_resource,w_kill,bias], metrics)。"""
    n = dataset.X.shape[0]
    X = np.column_stack([dataset.X, np.ones(n)])
    y = dataset.y
    rng = np.random.default_rng(20260828)
    theta = rng.normal(0.0, 0.1, size=X.shape[1])

    for _ in range(iterations):
        logits = X @ theta
        preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        grad = X.T @ (preds - y) / n + l2 * theta
        theta -= learning_rate * grad

    logits = X @ theta
    preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    labels = (preds >= 0.5).astype(float)
    accuracy = float(np.mean(labels == y))
    log_loss = float(-np.mean(y * np.log(np.clip(preds, 1e-9, 1.0)) + (1 - y) * np.log(np.clip(1 - preds, 1e-9, 1.0))))
    metrics = {"accuracy": accuracy, "log_loss": log_loss, "samples": int(n)}
    return theta, metrics


def calibrate(episodes: int = _DEFAULT_EPISODES) -> ValueWeights:
    """执行校准并落盘 value_weights.json。"""
    dataset = generate_calibration_data(episodes)
    theta, metrics = fit_logistic(dataset)
    weights = ValueWeights(
        w_hp=float(theta[0]),
        w_resource=float(theta[1]),
        w_kill=float(theta[2]),
        bias=float(theta[3]),
    )
    save_weights(
        weights,
        source={
            "url": "simulator://calibration",
            "name": "简化1v1模拟对局 logistic 校准",
            "retrieved_at": "",
            "is_primary": False,
        },
        metrics={"episodes": episodes, **metrics},
    )
    return weights


if __name__ == "__main__":
    result = calibrate()
    print("校准完成:", result.to_dict())
