"""5v5 状态 tokenizer 测试。"""

from __future__ import annotations

import hok_decision.model.state_tokenizer as st
from hok_decision.model.state_tokenizer import tokenize_state


def _heroes() -> list[dict]:
    return [
        {
            "hero_ename": 105,
            "side": "self",
            "hp_frac": 0.8,
            "mp_frac": 0.6,
            "gold": 3000.0,
            "position": [52.0, 350.0],
        },
        *[
            {
                "hero_ename": 106 + i,
                "side": "ally",
                "hp_frac": 0.7,
                "mp_frac": 0.5,
                "gold": 2500.0,
                "position": [100.0 + i * 10, 200.0],
            }
            for i in range(4)
        ],
        *[
            {
                "hero_ename": 200 + i,
                "side": "enemy",
                "hp_frac": 0.75,
                "mp_frac": 0.55,
                "gold": 2600.0,
                "position": [300.0, 100.0 + i * 10],
            }
            for i in range(5)
        ],
    ]


def test_tokenize_state_length_and_bounds() -> None:
    tokens = tokenize_state(game_time_s=600.0, heroes=_heroes())
    assert len(tokens) == 2 + 10 * st.HERO_TOKENS_PER_HERO + 1
    assert all(0 <= token < st.VOCAB_SIZE for token in tokens)
    assert tokens[0] == st.CLS_TOKEN and tokens[-1] == st.SEP_TOKEN


def test_unknown_fields_encoded_as_unknown_token() -> None:
    heroes = [
        {
            "hero_ename": None,
            "side": "self",
            "hp_frac": None,
            "mp_frac": None,
            "gold": None,
            "position": None,
        }
    ]
    tokens = tokenize_state(game_time_s=0.0, heroes=heroes)
    assert tokens.count(st.UNKNOWN_TOKEN) == 6  # 英雄ID + HP + MP + GOLD + X + Y
    assert all(token < st.VOCAB_SIZE for token in tokens)


def test_only_active_heroes_encoded() -> None:
    tokens = tokenize_state(game_time_s=0.0, heroes=_heroes())
    active_tokens = [
        token for token in tokens if st.ACTIVE_BASE <= token < st.ACTIVE_BASE + st.ACTIVE_COUNT
    ]
    assert active_tokens == list(range(st.ACTIVE_BASE, st.ACTIVE_BASE + 10))


def test_action_registry_lazy(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(st, "NORMALIZED_ROOT", tmp_path)
    first = st.register_action("recall")
    second = st.register_action("recall")
    third = st.register_action("attack")
    assert first == second == st.ACTION_BASE
    assert third == st.ACTION_BASE + 1
    registry = st.load_action_registry()
    assert registry == {"recall": st.ACTION_BASE, "attack": st.ACTION_BASE + 1}
