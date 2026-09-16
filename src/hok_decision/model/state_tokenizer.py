"""单帧 5v5 状态 tokenizer（只激活局内 10 名英雄的信息与经济）。

设计约定（用户确认）：
- 只编码当前对局中**活跃的 10 名英雄**（我方 5 + 对方 5），不铺全量英雄表；
- 动作词表不在本模块预定义，等真实训练实例出现后动态识别注册；
- 装备/技能/地图 token 为后续扩展位，当前不激活。

Token 布局（VOCAB_SIZE 见 build_vocab_size）：
  PAD=0, CLS=1, SEP=2
  时间桶 TIME_BASE..（12 桶，0~1500s）
  阵营 SIDE_BASE..（self=0 / ally=1 / enemy=2）
  英雄全局ID HERO_BASE..（按 heroes.json 顺序索引 0..131）
  局内活跃序号 ACTIVE_BASE..（0..9）
  血量/蓝量/经济 HP/MP/GOLD_BASE..（各 10 桶，0~1 / 0~15000）
  位置 X/Y_BASE..（各 20 桶，0~416 / 0~424 小地图像素系）
  动作占位 ACTION_BASE（先留 64 个槽位，训练实例到来后注册）
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from hok_decision.config import NORMALIZED_ROOT

PAD_TOKEN = 0
CLS_TOKEN = 1
SEP_TOKEN = 2

TIME_BASE = 10
TIME_BUCKETS = 12
TIME_SCALE = 1500.0

SIDE_BASE = 30
SIDE_IDS = {"self": 0, "ally": 1, "enemy": 2}

HERO_BASE = 33
HERO_COUNT = 132

ACTIVE_BASE = 165
ACTIVE_COUNT = 10

HP_BASE = 175
MP_BASE = 185
GOLD_BASE = 195
STAT_BUCKETS = 10
GOLD_SCALE = 15000.0

X_BASE = 205
Y_BASE = 225
POS_BUCKETS = 20
X_SCALE = 416.0
Y_SCALE = 424.0

ACTION_BASE = 245
ACTION_SLOTS = 64

MASK_TOKEN = ACTION_BASE + ACTION_SLOTS
UNKNOWN_TOKEN = MASK_TOKEN + 1
VOCAB_SIZE = UNKNOWN_TOKEN + 1

HERO_TOKENS_PER_HERO = 8  # SIDE + ACTIVE + HERO + HP + MP + GOLD + X + Y


def _bucket_or_unknown(value: float | None, scale: float, buckets: int, base: int) -> int:
    if value is None:
        return UNKNOWN_TOKEN
    return base + _bucket(float(value), scale, buckets)


def mask_state_tokens(
    tokens: list[int],
    *,
    rng: Any,
    mask_ratio: float = 0.15,
) -> tuple[list[int], list[int]]:
    """MLM 掩码：返回 (masked_tokens, labels)。

    labels 中未掩码位置为 -100（torch CrossEntropy ignore_index）。
    80% 替换为 MASK，10% 随机替换，10% 保持原 token 但仍要求预测。
    """
    masked = list(tokens)
    labels = [-100] * len(tokens)
    for index in range(1, len(tokens) - 1):  # 保留 CLS/SEP，其余可掩
        if rng.random() >= mask_ratio:
            continue
        original = tokens[index]
        labels[index] = original
        roll = rng.random()
        if roll < 0.8:
            masked[index] = MASK_TOKEN
        elif roll < 0.9:
            masked[index] = rng.randint(3, MASK_TOKEN - 1)
        # else: 原 token 不变，但标签参与 loss
    return masked, labels


@lru_cache(maxsize=1)
def _hero_order() -> dict[int, int]:
    payload = json.loads(
        (NORMALIZED_ROOT / "heroes.json").read_text(encoding="utf-8")
    )
    return {int(h["ename"]): index for index, h in enumerate(payload["data"])}


def build_vocab_size() -> int:
    return VOCAB_SIZE


def _bucket(value: float, scale: float, buckets: int) -> int:
    return max(0, min(buckets - 1, int(value / scale * buckets)))


def tokenize_state(
    *,
    game_time_s: float,
    heroes: list[dict[str, Any]],
) -> list[int]:
    """把单帧 5v5 状态编码为 token 列表。

    heroes 顺序：self, allies(4), enemies(5)，共 10 名；
    每名 hero 字段（除 side 外均可缺省，缺省→UNKNOWN token）：
      hero_ename: int | None
      side ∈ {self,ally,enemy}
      hp_frac / mp_frac: float | None（辅助特征，识别不到给 None）
      gold: float | None（辅助特征）
      position: [x,y] | None（小地图主特征；缺失给 None）
    """
    hero_order = _hero_order()
    tokens = [CLS_TOKEN, TIME_BASE + _bucket(game_time_s, TIME_SCALE, TIME_BUCKETS)]
    for active_index, hero in enumerate(heroes[:ACTIVE_COUNT]):
        side = SIDE_BASE + SIDE_IDS[str(hero["side"])]
        ename = hero.get("hero_ename")
        hero_id = (
            HERO_BASE + hero_order.get(int(ename), 0)
            if ename is not None
            else UNKNOWN_TOKEN
        )
        active = ACTIVE_BASE + active_index
        hp = _bucket_or_unknown(
            hero.get("hp_frac"), 1.0, STAT_BUCKETS, HP_BASE
        )
        mp = _bucket_or_unknown(
            hero.get("mp_frac"), 1.0, STAT_BUCKETS, MP_BASE
        )
        gold = _bucket_or_unknown(
            hero.get("gold"), GOLD_SCALE, STAT_BUCKETS, GOLD_BASE
        )
        position = hero.get("position")
        if position is None:
            xt = UNKNOWN_TOKEN
            yt = UNKNOWN_TOKEN
        else:
            x, y = position
            xt = _bucket_or_unknown(float(x), X_SCALE, POS_BUCKETS, X_BASE)
            yt = _bucket_or_unknown(float(y), Y_SCALE, POS_BUCKETS, Y_BASE)
        tokens.extend([side, active, hero_id, hp, mp, gold, xt, yt])
    tokens.append(SEP_TOKEN)
    return tokens


def register_action(action_name: str) -> int:
    """动作词表延迟注册：真实训练实例出现一个动作，注册一个 id。"""
    registry_path = NORMALIZED_ROOT / "action_registry.json"
    registry: dict[str, int] = {}
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            registry = {}
    if action_name in registry:
        return int(registry[action_name])
    if len(registry) >= ACTION_SLOTS:
        raise ValueError(f"动作槽位已满({ACTION_SLOTS})，无法注册 {action_name}")
    token = ACTION_BASE + len(registry)
    registry[action_name] = token
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return token


def load_action_registry() -> dict[str, int]:
    path = NORMALIZED_ROOT / "action_registry.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in payload.items()}
    except (json.JSONDecodeError, OSError):
        return {}
