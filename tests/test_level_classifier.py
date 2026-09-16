"""对局水平前置分层测试（八档：<1200/1200/1500/1800/2100/2300/2500+/职业）。"""

from __future__ import annotations

from hok_decision.model.level_classifier import LEVELS, classify_level


def test_rank_score_thresholds() -> None:
    cases = {
        1100: 0,
        1200: 1,
        1499: 1,
        1500: 2,
        1799: 2,
        1800: 3,
        2099: 3,
        2100: 4,
        2299: 4,
        2300: 5,
        2499: 5,
        2500: 6,
        3000: 6,
    }
    for score, expected in cases.items():
        result = classify_level(rank_score=float(score))
        assert result.level_id == expected, (score, result.level_name)
        assert result.level_name == LEVELS[expected]


def test_professional_level() -> None:
    result = classify_level(rank_score=2600.0, is_professional=True)
    assert result.level_id == 7 and result.level_name == "职业"


def test_manual_override_still_works() -> None:
    result = classify_level(manual_level=6)
    assert result.level_id == 6 and result.level_name == "2500+"


def test_heuristic_fallback_returns_valid_level() -> None:
    result = classify_level(hero_enames=[105], game_time_s=600, avg_gold=6000)
    assert 0 <= result.level_id < len(LEVELS)
