"""对局水平前置分层模块（按段位分档）。

八档（每档都需要独立训练数据）：
  0: 1200以下
  1: 1200档
  2: 1500档
  3: 1800档
  4: 2100档
  5: 2300档
  6: 2500+档
  7: 职业

输入优先级：
  1. is_professional=True → 职业档
  2. rank_score（段位分）→ 按阈值分档
  3. manual_level 手动指定
  4. 无分数时用阵容胜率/经济曲线做启发式估档（仅兜底）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from hok_decision.config import NORMALIZED_ROOT

LEVELS = ("1200以下", "1200", "1500", "1800", "2100", "2300", "2500+", "职业")
RANK_THRESHOLDS = (1200.0, 1500.0, 1800.0, 2100.0, 2300.0, 2500.0)

_T_RANK_SCORE = {"T0": 5.0, "T1": 4.0, "T2": 3.0, "T3": 2.0, "T4": 1.0}


@dataclass(frozen=True)
class LevelResult:
    level_id: int
    level_name: str
    score: float
    features: dict[str, float]


def _load_hero_meta() -> dict[int, dict[str, Any]]:
    path = NORMALIZED_ROOT / "hero_meta.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {int(row["ename"]): row for row in payload["data"]}
    except (json.JSONDecodeError, KeyError, OSError):
        return {}


def _heuristic_score(
    hero_enames: list[int], game_time_s: float, avg_gold: float, avg_deaths: float
) -> dict[str, float]:
    meta = _load_hero_meta()
    known = 0
    win_sum = ban_sum = show_sum = tier_sum = 0.0
    for ename in hero_enames:
        row = meta.get(int(ename))
        if row is None:
            continue
        known += 1
        win_sum += float(row.get("win_rate", 0.5))
        ban_sum += float(row.get("ban_rate", 0.0))
        show_sum += float(row.get("show_rate", 0.0))
        tier_sum += _T_RANK_SCORE.get(str(row.get("t_rank", "")), 3.0)
    known = max(1, known)
    expected_gold = 2000.0 + game_time_s * 8.0
    economy_ratio = avg_gold / max(1.0, expected_gold)
    return {
        "avg_win": round(win_sum / known, 4),
        "avg_ban": round(ban_sum / known, 4),
        "avg_show": round(show_sum / known, 4),
        "avg_tier": round(tier_sum / known, 2),
        "economy_ratio": round(economy_ratio, 4),
        "avg_deaths": float(avg_deaths),
    }


def classify_level(
    *,
    hero_enames: list[int] | None = None,
    game_time_s: float = 0.0,
    avg_gold: float = 0.0,
    avg_deaths: float = 0.0,
    rank_score: float | None = None,
    is_professional: bool = False,
    manual_level: int | None = None,
) -> LevelResult:
    """按段位分档。rank_score 为段位分（如巅峰分）。"""
    hero_enames = hero_enames or []
    if is_professional:
        return LevelResult(7, LEVELS[7], 7.0, {"rank_score": float(rank_score or 0.0)})
    if manual_level is not None:
        level_id = max(0, min(len(LEVELS) - 1, int(manual_level)))
        return LevelResult(level_id, LEVELS[level_id], float(level_id), {})
    if rank_score is not None:
        level_id = 0
        for threshold in RANK_THRESHOLDS:
            if rank_score >= threshold:
                level_id += 1
            else:
                break
        return LevelResult(
            level_id, LEVELS[level_id], float(level_id), {"rank_score": float(rank_score)}
        )

    # 兜底启发式：把阵容/经济特征映射到 0~7 档
    feat = _heuristic_score(hero_enames, game_time_s, avg_gold, avg_deaths)
    score = (
        (feat["avg_win"] - 0.48) * 20.0
        + feat["avg_ban"] * 3.0
        + feat["avg_show"] * 2.0
        + (feat["avg_tier"] - 3.0) * 0.8
        + (feat["economy_ratio"] - 1.0) * 3.0
        + max(-2.0, 0.0 - feat["avg_deaths"] * 0.4)
    )
    level_id = int(max(0, min(7, round((score + 2.4) / 0.8))))
    feat["score"] = round(score, 4)
    return LevelResult(level_id, LEVELS[level_id], round(score, 4), feat)
