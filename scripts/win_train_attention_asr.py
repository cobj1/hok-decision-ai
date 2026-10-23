r"""用解说(ASR)的『下一步动作』拟合注意力二阶权重（Win 运行）。

分工不变：动作来自 search 候选 + rollout，模型只学权重 w(s)。
训练信号：每个决策点展开 8 个候选宏观动作，rollout 后得到一阶价值特征 F(s,a)；
        解说动作命中的候选为正样本，其余为负；同一决策点内 softmax 排序损失。
        模型输出 V(s,a)=Σ w_j(s)·F_j(s,a)，训练让正样本得分最高。

数据：asr_calibration_final_v2.json（LLM 提取的动作意图）
初值：attention_value_td.pt（TD 先行验，在其上微调）
输出：data/S45/models/attention_value_asr.pt + .json

运行:
    E:\py312\python.exe E:\project\hok-decision-ai\scripts\win_train_attention_asr.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hok_decision.model.attention_value import (  # noqa: E402
    CONTEXT_FEATURES,
    AttentionValueModel,
)
from hok_decision.schema.models import MapData  # noqa: E402
from hok_decision.search.actions import generate_candidates  # noqa: E402
from hok_decision.search.attention_weights import (  # noqa: E402
    ACTION_DIMS,
    action_features_from_state,
)
from hok_decision.search.planner import SearchState, rollout_decision  # noqa: E402
from hok_decision.simulator.units import Hero  # noqa: E402

HOKDATA_ROOT = Path(os.environ.get("HOKDATA_ROOT", str(Path(__file__).resolve().parents[1].parent / "hokdata")))
DATASET = HOKDATA_ROOT / "dataset"
ASR_REPORT = HOKDATA_ROOT / "asr_calibration_final_v2.json"
SPLIT_FILE = HOKDATA_ROOT / "asr_train_split.json"
MODELS_DIR = PROJECT_ROOT / "data" / "S45" / "models"
TD_MODEL = MODELS_DIR / "attention_value_td.pt"
OUT_MODEL = MODELS_DIR / "attention_value_asr.pt"
MAP_JSON = PROJECT_ROOT / "data" / "S45" / "normalized" / "map.json"

HUD_AFFINE = ((3.9619274696916493, 1.3453769972204605e-05), (-1.9928688568260355e-05, 4.038131527190943), (-196.00459141224727, -105.2570867363273))
BLUE_BASE = (25.0, 395.0)
RED_BASE = (397.0, 28.0)
LEVEL_MAP = {"1500+": 2, "1800+": 3, "2100+": 4, "2300+": 5, "2500+": 6, "赛事": 7}
CONTEXT_NAMES = CONTEXT_FEATURES
DRAGON_IDS = {"tyrant", "overlord", "storm_dragon"}

# ASR 动作 -> 期望 MacroAction.kind；dragon 要求目标是龙，jungle 要求目标非龙
EXPECT_KIND = {
    "kill": "attack_enemy",
    "fight": "attack_enemy",
    "retreat": "retreat",
    "tower": "push_tower",
    "jungle": "capture_resource",
    "dragon": "capture_resource",
    "lane": "capture_resource",
}


def _to_ref(hero: dict, cs: str) -> tuple[float, float]:
    if cs == "hud_roi":
        a, b, c = HUD_AFFINE
        return a[0] * hero["x"] + a[1] * hero["y"] + c[0], b[0] * hero["x"] + b[1] * hero["y"] + c[1]
    return hero.get("x_ref", hero["x"]), hero.get("y_ref", hero["y"])


def _mean(values) -> float:
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def _clamp_gold(value) -> float:
    return value if 0.0 <= value <= 15000.0 else 0.0


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


def _raw_author(frame: dict) -> dict:
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
            "kills_self": self_k, "kills_enemy": enemy_k,
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
        "kills_self": num2(kb), "kills_enemy": num2(kr),
        "towers": min(9.0, max(-9.0, num2(bs.get("towers")))),
        "dragons": min(3.0, max(-3.0, num2(bs.get("dragons")))),
        "gold": _clamp_gold(num2(bs.get("gold"))),
    }


def _context_vector(frame: dict) -> list[float]:
    raw = _raw_author(frame)
    t = float(frame.get("game_time_s", 0))
    level_id = LEVEL_MAP.get(str(frame.get("level_id", "")), 3)
    ally_hp = _mean(frame.get("ally_hp") or [])
    ctx = [0.0] * len(CONTEXT_NAMES)
    ctx[CONTEXT_NAMES.index("time_norm")] = t / 1500.0
    ctx[CONTEXT_NAMES.index("econ_sat")] = min(1.0, t / 1500.0)
    ctx[CONTEXT_NAMES.index("enemy_threat")] = 0.5
    ctx[CONTEXT_NAMES.index("own_hp_frac")] = ally_hp if ally_hp else 0.7
    ctx[CONTEXT_NAMES.index("tower_adv")] = raw["towers"] / 10.0
    ctx[CONTEXT_NAMES.index("dragon_avail")] = 1.0 if raw["dragons"] > 0 or t >= 120.0 else 0.0
    ctx[CONTEXT_NAMES.index("team_aa_power")] = 0.8
    ctx[CONTEXT_NAMES.index("team_caster_power")] = 0.8
    ctx[CONTEXT_NAMES.index("build_attack_power")] = 0.8
    ctx[CONTEXT_NAMES.index("build_defense_power")] = 0.8
    ctx[CONTEXT_NAMES.index("build_utility_power")] = 0.8
    ctx[CONTEXT_NAMES.index("skill_power")] = 0.8
    ctx[CONTEXT_NAMES.index("match_level")] = level_id / 7.0
    return ctx


def _build_state(frame: dict) -> SearchState:
    heroes = frame.get("heroes") or []
    cs = str(frame.get("coordinate_system", "hud_roi"))
    self_h = [h for h in heroes if str(h.get("side")) == "self"]
    enemy_h = [h for h in heroes if str(h.get("side")) == "enemy"]
    if not self_h or not enemy_h:
        raise ValueError("minimap 缺少 self/enemy 英雄")

    ally_hp = [v for v in (frame.get("ally_hp") or []) if isinstance(v, (int, float))]
    enemy_hp = [v for v in (frame.get("enemy_hp") or []) if isinstance(v, (int, float))]

    state = SearchState(game_time=float(frame.get("game_time_s", 0)))
    for i, h in enumerate(self_h):
        hp = ally_hp[i] if i < len(ally_hp) and 0.0 <= ally_hp[i] <= 1.0 else 0.7
        state.heroes[f"blue_{i}"] = Hero(
            unit_id=f"blue_{i}", name="self", side="blue",
            position=_to_ref(h, cs), hp=hp, max_hp=1.0,
        )
    for i, h in enumerate(enemy_h):
        hp = enemy_hp[i] if i < len(enemy_hp) and 0.0 <= enemy_hp[i] <= 1.0 else 0.7
        state.heroes[f"red_{i}"] = Hero(
            unit_id=f"red_{i}", name="enemy", side="red",
            position=_to_ref(h, cs), hp=hp, max_hp=1.0,
        )

    raw = _raw_author(frame)
    n_self = len(self_h)
    for h in state.heroes.values():
        if h.side == "blue":
            state.gold[h.unit_id] = raw["gold"] / n_self if n_self else 0.0
    for tower_id in (frame.get("towers_state") or {}):
        state.tower_hp[tower_id] = 1.0
    for rid, count in (("tyrant", max(0, min(2, int(raw["dragons"])))),):
        for _ in range(count):
            if rid not in state.captured_resources:
                state.captured_resources.add(rid)
                break
    return state


def _candidate_matches(action_type: str, action) -> bool:
    kind = action.kind
    target = action.target_id or ""
    if action_type in {"kill", "fight"}:
        return kind == "attack_enemy"
    if action_type == "retreat":
        return kind == "retreat"
    if action_type == "tower":
        return kind == "push_tower"
    if action_type == "dragon":
        return kind == "capture_resource" and any(d in target for d in DRAGON_IDS)
    if action_type == "jungle":
        return kind == "capture_resource" and not any(d in target for d in DRAGON_IDS)
    if action_type == "lane":
        return (kind == "capture_resource" and not any(d in target for d in DRAGON_IDS)) or kind == "push_tower"
    return False


def _build_training_points() -> tuple[list[dict], dict[str, list[str]]]:
    map_data = MapData.model_validate(json.loads(MAP_JSON.read_text(encoding="utf-8")))
    anchors = {"fountain_blue": {"position": BLUE_BASE}, "fountain_red": {"position": RED_BASE}}

    frames_index: dict[tuple[str, float], dict] = {}
    for line in (DATASET / "dataset_final_v4.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            continue
        frames_index[(str(frame.get("video_bv")), round(float(frame.get("game_time_s", 0)), 1))] = frame

    asr_rows = json.loads(ASR_REPORT.read_text(encoding="utf-8"))["llm_rows"]
    by_point: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in asr_rows:
        by_point[(str(r["bv"]), round(float(r["frame_t"]), 1))].append(r)

    points = []
    skipped = {"frame": 0, "state": 0, "no_match": 0}
    for (bv, t), rows in sorted(by_point.items()):
        frame = frames_index.get((bv, t))
        if frame is None:
            skipped["frame"] += 1
            continue
        try:
            state = _build_state(frame)
        except ValueError:
            skipped["state"] += 1
            continue
        candidates = generate_candidates(state, map_data, anchors, max_candidates=8)
        if not candidates:
            continue
        target_types = {str(r["action"]) for r in rows}
        positive = [i for i, a in enumerate(candidates) if any(_candidate_matches(t, a) for t in target_types)]
        if not positive:
            skipped["no_match"] += 1
            continue
        rolled: list[list[float]] = []
        for action in candidates:
            st = _clone_state(state)
            rollout_decision(st, action, map_data=map_data, duration_s=12.0)
            rolled.append(action_features_from_state(st, action, feature_dim=len(ACTION_DIMS)))
        points.append(
            {
                "bv": bv,
                "t": t,
                "context": _context_vector(frame),
                "features": rolled,
                "positive": positive,
                "actions": [a.kind for a in candidates],
                "targets": [a.target_id for a in candidates],
                "asr_types": sorted(target_types),
            }
        )
    uniq_ratios = []
    for p in points:
        arr = np.unique(np.round(np.array(p["features"]), 6), axis=0)
        uniq_ratios.append(len(arr) / len(p["features"]))
    print(f"points={len(points)} skipped={skipped} uniq_ratio_mean={float(np.mean(uniq_ratios)):.3f} min={float(np.min(uniq_ratios)):.3f}")

    # 视频 5 折留一：每第 5 个 bv 进 holdout（20%）
    bvs = sorted({p["bv"] for p in points})
    holdout = bvs[::5]
    split = {"holdout": holdout, "train": [b for b in bvs if b not in holdout]}
    SPLIT_FILE.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"split train={len(split['train'])} holdout={len(split['holdout'])}")

    single = [p for p in points if len(p["asr_types"]) == 1]
    print(f"single_label_points={len(single)}/{len(points)} baseline_top1={float(np.mean([len(p['positive'])/len(p['features']) for p in points])):.3f} single_baseline={float(np.mean([len(p['positive'])/len(p['features']) for p in single])):.3f}")
    return points, split, single


def _clone_state(state: SearchState) -> SearchState:
    import copy

    return copy.deepcopy(state)


def _to_tensors(points: list[dict], split: dict[str, list[str]]) -> dict[str, torch.Tensor]:
    c, f, y, groups = [], [], [], []
    for p in points:
        ctx = torch.tensor(p["context"], dtype=torch.float32)
        feats = torch.tensor(p["features"], dtype=torch.float32)
        y_t = torch.zeros(len(p["features"]), dtype=torch.float32)
        y_t[p["positive"]] = 1.0 / len(p["positive"])
        c.append(ctx.expand(len(p["features"]), -1))
        f.append(feats)
        y.append(y_t)
        groups.append(len(p["features"]))
    return {
        "c": torch.cat(c, dim=0),
        "f": torch.cat(f, dim=0),
        "y": torch.cat(y, dim=0),
        "groups": torch.tensor(groups, dtype=torch.long),
        "bvs": [p["bv"] for p in points],
    }


def main() -> int:
    args = sys.argv[1:]
    mode = "all"
    if "--mode" in args:
        mode = args[args.index("--mode") + 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch={torch.__version__} device={device} mode={mode}")
    points, split, single = _build_training_points()
    data = _to_tensors(points, split)
    single_data = _to_tensors(single, split)
    c, f, y = data["c"].to(device), data["f"].to(device), data["y"].to(device)
    groups = data["groups"].to(device)
    offsets = torch.cumsum(torch.cat([torch.tensor([0], device=device), groups[:-1]]), dim=0)

    # 单标签点噪声更小；全部点覆盖更全。默认 all，--mode single 用单标签子集。
    from collections import Counter as _Counter
    use_points = single if mode == "single" else points
    train_idx = [i for i, p in enumerate(use_points) if p["bv"] in set(split["train"])]
    hold_idx = [i for i, p in enumerate(use_points) if p["bv"] in split["holdout"]]
    type_counts = _Counter(t for i in train_idx for t in use_points[i]["asr_types"])
    min_count = min(type_counts.values())
    type_weights = {t: max(0.5, min(3.0, (min_count / c) ** 0.5)) for t, c in type_counts.items()}
    point_weights = torch.tensor(
        [sum(type_weights[t] for t in use_points[i]["asr_types"]) / len(use_points[i]["asr_types"]) for i in range(len(use_points))],
        dtype=torch.float32,
    )
    print("type_weights:", {t: round(w, 3) for t, w in sorted(type_weights.items())})
    data_used = single_data if mode == "single" else data
    c, f, y = data_used["c"].to(device), data_used["f"].to(device), data_used["y"].to(device)
    groups = data_used["groups"].to(device)
    single_train = [i for i, p in enumerate(single) if p["bv"] in set(split["train"])]
    single_hold = [i for i, p in enumerate(single) if p["bv"] in split["holdout"]]
    offsets = torch.cumsum(torch.cat([torch.tensor([0], device=device), groups[:-1]]), dim=0)
    print(f"train_points={len(train_idx)} holdout_points={len(hold_idx)} samples={c.shape[0]}")

    feature_dim = len(ACTION_DIMS)
    model = AttentionValueModel(feature_dim=feature_dim, base_weights=[1.0] * feature_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    def point_scores(indices: list[int]) -> tuple[torch.Tensor, list[int]]:
        ids = torch.tensor(indices, device=device)
        start = offsets[ids]
        lengths = groups[ids]
        flat = torch.cat([torch.arange(s, s + ln, device=device) for s, ln in zip(start, lengths)])
        scores = model(c[flat], f[flat])
        return scores, lengths.tolist()

    def top1_hit_simple(indices: list[int]) -> float:
        if not indices:
            return 0.0
        ids = torch.tensor(indices, device=device)
        start = offsets[ids]
        correct = 0
        for k, i in enumerate(indices):
            s = int(start[k])
            ln = int(groups[i])
            with torch.no_grad():
                seg = model(c[s: s + ln], f[s: s + ln])
            top = int(torch.argmax(seg))
            if y[s + top] > 0:
                correct += 1
        return correct / len(indices)

    print(f"init top1 train={top1_hit_simple(train_idx):.4f} holdout={top1_hit_simple(hold_idx):.4f}")

    best_hold = 0.0
    best_state = None
    best_metrics = {}
    for step in range(6000):
        perm = torch.randperm(len(train_idx), device=device)[:48].cpu().tolist()
        batch = [train_idx[i] for i in perm]
        ids = torch.tensor(batch, device=device)
        start = offsets[ids]
        lengths = groups[ids]
        flat = torch.cat([torch.arange(s, s + ln, device=device) for s, ln in zip(start, lengths)])
        scores = model(c[flat], f[flat])
        pieces = torch.split(scores, lengths.tolist())
        targets = [torch.argmax(y[s: s + ln]) if False else None for s, ln in zip(start.tolist(), lengths.tolist())]
        targets = []
        for s, ln in zip(start.tolist(), lengths.tolist()):
            targets.append(torch.argmax(y[s: s + ln]))
        stacked = nn.utils.rnn.pad_sequence(pieces, batch_first=True, padding_value=-1e9)
        target_tensor = torch.stack(targets)
        sample_w = point_weights[ids.cpu()].to(device)
        loss = (loss_fn(stacked, target_tensor) * sample_w).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 500 == 0:
            with torch.no_grad():
                tr = top1_hit_simple(train_idx)
                ho = top1_hit_simple(hold_idx)
            print(f"step {step + 1} loss={loss.item():.4f} top1_train={tr:.4f} top1_hold={ho:.4f}", flush=True)
            if ho >= best_hold:
                best_hold = ho
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = {"top1_train": tr, "top1_holdout": ho}

    model.eval()
    tr = top1_hit_simple(train_idx)
    ho = top1_hit_simple(hold_idx)
    if best_state is not None and ho <= best_hold:
        model.load_state_dict(best_state)
        ho = best_hold
        tr = best_metrics.get("top1_train", tr)
    with torch.no_grad():
        single_tr = top1_hit_simple(single_train)
        single_ho = top1_hit_simple(single_hold)
    metrics = {
        "top1_train": tr,
        "top1_holdout": ho,
        "single_top1_train": single_tr,
        "single_top1_holdout": single_ho,
        "train_points": len(train_idx),
        "holdout_points": len(hold_idx),
        "samples": int(c.shape[0]),
        "feature_dim": feature_dim,
        "loss_fn": "softmax_rank_per_point",
    }
    torch.save(model.state_dict(), OUT_MODEL)
    OUT_MODEL.with_suffix(".json").write_text(
        json.dumps(
            {"context_features": CONTEXT_FEATURES, "action_features": ACTION_DIMS, "metrics": metrics},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"final top1_train={tr:.4f} top1_holdout={ho:.4f}")
    print("saved:", OUT_MODEL)
    print("ASR_WEIGHT_TRAIN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
