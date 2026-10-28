r"""严格 TD 训练注意力二阶权重（Win torch，模型只留 Win）。

目标：
  V(s_t) ← r(t→t+Δ) + γ^Δ · V_target(s_{t+Δ})
  r 为即时事件收益（击杀/塔/龙/经济差值，游戏内固定价值，非手写长期权重）
  Δ ∈ {5,10,15,20,30,60,120,180,300} 多区间兼顾
  V_target 为慢更新目标网络（每 200 步同步）

数据：dataset_final_v4.jsonl（作者视角映射，同局连续片段内配对）

运行:
    E:\py312\python.exe E:\project\hok-decision-ai\scripts\win_train_attention_td.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hok_decision.data.splits import load_splits, split_of  # noqa: E402
from hok_decision.model.attention_value import (  # noqa: E402
    CONTEXT_FEATURES,
    VALUE_FEATURES,
    AttentionValueModel,
)

HOKDATA_ROOT = Path(os.environ.get("HOKDATA_ROOT", str(Path(__file__).resolve().parents[1].parent / "hokdata")))
DATASET = HOKDATA_ROOT / "dataset"
SEGMENT_GAP_S = float(os.environ.get("BENEFIT_SEGMENT_GAP_S", "8.0"))
INTERVALS = (5, 10, 15, 20, 30, 60, 120, 180, 300)
LEVEL_MAP = {"1500+": 2, "1800+": 3, "2100+": 4, "2300+": 5, "2500+": 6, "赛事": 7}
GAMMA = 0.99
REWARD_SCALE = 3000.0
# 即时事件收益（游戏内固定价值，单位=金币）
REWARD = {"kill": 100.0, "death": -100.0, "tower": 300.0, "dragon": 400.0, "gold": 1.0}


def _load_sb2b() -> dict[tuple[str, str], dict]:
    table: dict[tuple[str, str], dict] = {}
    for p in DATASET.glob("*.sb2b.json"):
        payload = json.loads(p.read_text(encoding="utf-8"))
        bv = payload.get("bv", p.stem.split(".sb2b")[0])
        for entry in payload.get("scoreboard", []):
            if entry.get("frame"):
                table[(str(bv), str(entry["frame"]))] = entry.get("state") or {}
    return table


SB2B = _load_sb2b()


def _clamp_gold(value: float) -> float:
    # OCR 脏值防护：gold 超过合理区间按 0 处理
    return value if 0.0 <= value <= 15000.0 else 0.0


def _mean(values: list) -> float:
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def _context_features(frame: dict) -> tuple[list[float], list[float]]:
    raw = _raw_author(frame)
    kills_self = raw["kills_self"]
    kills_enemy = raw["kills_enemy"]

    time_norm = float(frame.get("game_time_s") or 0.0) / 1500.0
    level_id = LEVEL_MAP.get(str(frame.get("level_id", "")), 3)
    ally_hp = _mean(frame.get("ally_hp") or [])
    enemy_hp = _mean(frame.get("enemy_hp") or [])

    context = [0.0] * len(CONTEXT_FEATURES)
    context[CONTEXT_FEATURES.index("time_norm")] = time_norm
    context[CONTEXT_FEATURES.index("econ_sat")] = min(1.0, time_norm)
    context[CONTEXT_FEATURES.index("enemy_threat")] = 0.5
    context[CONTEXT_FEATURES.index("own_hp_frac")] = ally_hp if ally_hp else 0.7
    context[CONTEXT_FEATURES.index("tower_adv")] = raw["towers"] / 10.0
    context[CONTEXT_FEATURES.index("dragon_avail")] = 1.0 if raw["dragons"] > 0 or float(frame.get("game_time_s") or 0.0) >= 120.0 else 0.0
    context[CONTEXT_FEATURES.index("team_aa_power")] = 0.8
    context[CONTEXT_FEATURES.index("team_caster_power")] = 0.8
    context[CONTEXT_FEATURES.index("build_attack_power")] = 0.8
    context[CONTEXT_FEATURES.index("build_defense_power")] = 0.8
    context[CONTEXT_FEATURES.index("build_utility_power")] = 0.8
    context[CONTEXT_FEATURES.index("skill_power")] = 0.8
    context[CONTEXT_FEATURES.index("match_level")] = level_id / 7.0

    features = [0.0] * len(VALUE_FEATURES)
    features[VALUE_FEATURES.index("kill_diff")] = (kills_self - kills_enemy) / 10.0
    features[VALUE_FEATURES.index("economy_diff")] = raw["gold"] / 10000.0
    features[VALUE_FEATURES.index("tower_diff")] = raw["towers"] / 10.0
    features[VALUE_FEATURES.index("overlord_diff")] = raw["dragons"] / 5.0
    if ally_hp and enemy_hp:
        features[VALUE_FEATURES.index("hp_diff")] = ally_hp - enemy_hp
    return context, features


def _raw_author(frame: dict) -> dict:
    """作者视角的原始 scoreboard 值（优先逐帧 sb2b，回退 battle_state）。"""
    camp = str(frame.get("author_camp", "blue"))
    state = SB2B.get((str(frame.get("video_bv", "")), str(frame.get("frame_id", ""))))
    if state:
        blue = state.get("blue") or {}
        red = state.get("red") or {}
        def num(v):
            return float(v) if isinstance(v, (int, float)) else 0.0
        if camp == "blue":
            self_k, enemy_k = num(blue.get("kills")), num(red.get("kills"))
            self_t, enemy_t = num(blue.get("towers")), num(red.get("towers"))
            self_d, enemy_d = num(blue.get("dragons")), num(red.get("dragons"))
            self_g = _clamp_gold(num(blue.get("gold")))
        else:
            self_k, enemy_k = num(red.get("kills")), num(blue.get("kills"))
            self_t, enemy_t = num(red.get("towers")), num(blue.get("towers"))
            self_d, enemy_d = num(red.get("dragons")), num(blue.get("dragons"))
            self_g = _clamp_gold(num(red.get("gold")))
        return {
            "kills_self": min(40.0, max(0.0, self_k)),
            "kills_enemy": min(40.0, max(0.0, enemy_k)),
            "towers": min(9.0, max(-9.0, self_t - enemy_t)),
            "dragons": min(3.0, max(-3.0, self_d - enemy_d)),
            "gold": self_g,
        }
    bs = (frame.get("battle_state") or {}).get("base") or {}
    kb = bs.get("kills_blue")
    kr = bs.get("kills_red")
    if camp == "red":
        kb, kr = kr, kb
    def num2(v):
        return float(v) if isinstance(v, (int, float)) else 0.0
    return {
        "kills_self": min(40.0, max(0.0, num2(kb))),
        "kills_enemy": min(40.0, max(0.0, num2(kr))),
        "towers": min(9.0, max(-9.0, num2(bs.get("towers")))),
        "dragons": min(3.0, max(-3.0, num2(bs.get("dragons")))),
        "gold": _clamp_gold(num2(bs.get("gold"))),
    }


def _reward_between(a: dict, b: dict) -> float:
    d_self = b["kills_self"] - a["kills_self"]
    d_enemy = b["kills_enemy"] - a["kills_enemy"]
    d_tower = b["towers"] - a["towers"]
    d_dragon = b["dragons"] - a["dragons"]
    d_gold = max(-3000.0, min(3000.0, b["gold"] - a["gold"]))
    r = (
        d_self * REWARD["kill"]
        + d_enemy * REWARD["death"]
        + d_tower * REWARD["tower"]
        + d_dragon * REWARD["dragon"]
        + d_gold * REWARD["gold"]
    )
    return max(-3.0, min(3.0, r / REWARD_SCALE))


def _segments(frames: list[dict]) -> list[list[dict]]:
    """按时间连续性切分：间隔>8s、时间倒退或 scene_break 视为新片段。"""
    frames = sorted(frames, key=lambda f: float(f.get("game_time_s") or 0.0))
    segments: list[list[dict]] = []
    current: list[dict] = []
    prev_t: float | None = None
    for frame in frames:
        t = float(frame.get("game_time_s") or 0.0)
        if current and (prev_t is None or t < prev_t or t - prev_t > SEGMENT_GAP_S):
            segments.append(current)
            current = []
        if frame.get("frame_quality") == "scene_break" and current:
            segments.append(current)
            current = []
        current.append(frame)
        prev_t = t
    if current:
        segments.append(current)
    return segments


def _build_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    frames_by_video: dict[str, list[dict]] = defaultdict(list)
    jsonl = DATASET / "dataset_final_v4.jsonl"
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            continue
        frames_by_video[str(frame.get("video_bv"))].append(frame)

    contexts: list[list[float]] = []
    features: list[list[float]] = []
    next_contexts: list[list[float]] = []
    next_features: list[list[float]] = []
    rewards: list[float] = []
    discounts: list[float] = []
    interval_ids: list[int] = []
    bvs: list[str] = []
    for frames in frames_by_video.values():
        for segment in _segments(frames):
            by_time = {round(float(f.get("game_time_s", 0)), 1): f for f in segment}
            times = sorted(by_time.keys())
            for t in times:
                frame_t = by_time[t]
                c_t, f_t = _context_features(frame_t)
                raw_t = _raw_author(frame_t)
                for di, delta in enumerate(INTERVALS):
                    target_t = round(t + delta, 1)
                    if target_t not in by_time:
                        continue
                    frame_next = by_time[target_t]
                    c_n, f_n = _context_features(frame_next)
                    r = _reward_between(raw_t, _raw_author(frame_next))
                    contexts.append(c_t)
                    features.append(f_t)
                    next_contexts.append(c_n)
                    next_features.append(f_n)
                    rewards.append(r)
                    discounts.append(GAMMA ** delta)
                    interval_ids.append(di)
                    bvs.append(str(frame_t.get("video_bv")))
    return (
        torch.tensor(contexts, dtype=torch.float32),
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(next_contexts, dtype=torch.float32),
        torch.tensor(next_features, dtype=torch.float32),
        torch.tensor(rewards, dtype=torch.float32),
        torch.tensor(discounts, dtype=torch.float32),
        interval_ids,
        bvs,
    )


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch={torch.__version__} device={device}")
    c, f, cn, fn, r, disc, interval_ids, bvs = _build_data()
    print(f"samples={c.shape[0]} reward_mean={r.mean().item():.4f}")
    print("interval samples:", {INTERVALS[k]: v for k, v in sorted(Counter(interval_ids).items())})

    game_splits = load_splits(HOKDATA_ROOT / "splits.json")
    train_idx = [i for i, bv in enumerate(bvs) if split_of(game_splits, bv) == "train"]
    val_idx = [i for i, bv in enumerate(bvs) if split_of(game_splits, bv) == "val"]
    if not train_idx or not val_idx:
        raise SystemExit(f"split failed: train={len(train_idx)} val={len(val_idx)}")
    print(
        f"game split: train_games={len(set(bvs[i] for i in train_idx))} "
        f"val_games={len(set(bvs[i] for i in val_idx))} "
        f"train_samples={len(train_idx)} val_samples={len(val_idx)}"
    )

    c, f, cn, fn, r, disc = c.to(device), f.to(device), cn.to(device), fn.to(device), r.to(device), disc.to(device)
    ids = torch.tensor(interval_ids, device=device)
    tr = torch.tensor(train_idx, dtype=torch.long, device=device)
    va = torch.tensor(val_idx, dtype=torch.long, device=device)

    model = AttentionValueModel(base_weights=[1.0] * 10).to(device)
    target_model = AttentionValueModel(base_weights=[1.0] * 10).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()

    for step in range(4000):
        idx = tr[torch.randint(0, len(tr), (256,), device=device)]
        with torch.no_grad():
            v_next = target_model(cn[idx], fn[idx])
            target = r[idx] + disc[idx] * v_next
        opt.zero_grad()
        out = model(c[idx], f[idx])
        loss = loss_fn(out, target)
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            target_model.load_state_dict(model.state_dict())
        if (step + 1) % 500 == 0:
            print(f"step {step+1}/4000 loss={loss.item():.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        pred = model(c[va], f[va])
        with torch.no_grad():
            target = r[va] + disc[va] * target_model(cn[va], fn[va])
        ss_res = ((target - pred) ** 2).sum().item()
        ss_tot = ((target - target.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / (ss_tot + 1e-9)
        mse = ss_res / len(va)
        print(f"val MSE={mse:.6f} R2={r2:.4f}")
        per_interval = {}
        for di, delta in enumerate(INTERVALS):
            mask = ids[va] == di
            if mask.sum() == 0:
                continue
            pr = pred[mask]
            tg = target[mask]
            r2i = 1 - ((tg - pr) ** 2).sum().item() / (((tg - tg.mean()) ** 2).sum().item() + 1e-9)
            per_interval[str(delta)] = round(r2i, 4)
        print("per-interval R2:", per_interval)
        artifact = PROJECT_ROOT / "data" / "S45" / "models" / "attention_value_td.pt"
        model.save_artifacts(
            artifact,
            {"mse": mse, "r2": r2, "per_interval_r2": per_interval, "samples": c.shape[0]},
        )
        print("saved:", artifact)
    print("TD_TRAIN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
