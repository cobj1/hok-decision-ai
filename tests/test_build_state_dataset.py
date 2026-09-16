"""状态帧 → 预训练语料构建器测试。"""

from __future__ import annotations

import json

from hok_decision.model.state_tokenizer import SEP_TOKEN, tokenize_state

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_state_dataset import build_state_tokens  # noqa: E402


def _frames() -> list[dict]:
    def hero(side: str, i: int) -> dict:
        return {
            "hero_ename": 100 + i,
            "side": side,
            "hp_frac": 0.7,
            "mp_frac": 0.5,
            "gold": 2000.0 + i * 100,
            "position": [50.0 + i * 30, 200.0],
        }

    heroes = [hero("self", 0)]
    heroes += [hero("ally", i) for i in range(1, 5)]
    heroes += [hero("enemy", i) for i in range(5, 10)]
    return [{"frame_id": "f1", "game_time_s": 300.0, "heroes": heroes}]


def test_build_state_tokens() -> None:
    rows = build_state_tokens(_frames())
    assert len(rows) == 1
    tokens = rows[0]["tokens"]
    assert len(tokens) == 83
    assert tokens[-1] == SEP_TOKEN
    assert tokens == tokenize_state(game_time_s=300.0, heroes=_frames()[0]["heroes"])


def test_side_counts_enforced(tmp_path) -> None:
    import pytest

    frames = _frames()
    frames[0]["heroes"][1]["side"] = "self"  # 变成 2 个 self
    with pytest.raises(AssertionError):
        build_state_tokens(frames)
