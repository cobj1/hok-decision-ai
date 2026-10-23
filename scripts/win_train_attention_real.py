r"""用 E:\hokdata\dataset\dataset_final_v4.jsonl 训练注意力二阶权重（Win torch，模型只留 Win）。

数据适配：
  - 通过 meta.json 取各视频 winner_side，只保留 蓝方/红方 已知的帧；
  - 只保留 battle_state 有非零信息（击杀/塔/龙/经济）的帧；
  - 特征为蓝方视角：kill_diff / economy_diff / tower_diff / dragon；
  - 标签：winner_side==蓝方 → 1，否则 0。

运行:
    E:\py312\python.exe E:\project\hok-decision-ai\scripts\win_train_attention_real.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import glob
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hok_decision.model.attention_value import (  # noqa: E402
    CONTEXT_FEATURES,
    VALUE_FEATURES,
    AttentionValueModel,
)

HOKDATA_ROOT = Path(os.environ.get("HOKDATA_ROOT", str(Path(__file__).resolve().parents[1].parent / "hokdata")))
DATASET = HOKDATA_ROOT / "dataset"
LEVEL_MAP = {"1500+": 2, "1800+": 3, "2100+": 4, "2300+": 5, "2500+": 6, "赛事": 7}


def _base_key(key: str) -> str:
    return str(key).split("#", 1)[0]


def _fallback_split(key: str, train: float = 0.70, val: float = 0.15) -> str:
    """splits.json 缺失时的确定性按局切分，禁止随机切分。"""
    bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < train:
        return "train"
    if bucket < train + val:
        return "val"
    return "test"


def _load_winners() -> dict[str, str]:
    winners: dict[str, str] = {}
    for meta in glob.glob(str(DATASET / "*.meta.json")):
        bv = Path(meta).name[:-10]  # 去掉 .meta.json
        raw = Path(meta).read_bytes()
        payload = None
        for enc in ("utf-8", "gbk"):
            try:
                payload = json.loads(raw.decode(enc))
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        if payload:
            winners[bv] = str((payload.get("win") or {}).get("winner_side", ""))
    return winners


def _mean(values: list) -> float:
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def _build_dataset() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    winners = _load_winners()
    context_rows: list[list[float]] = []
    feature_rows: list[list[float]] = []
    labels: list[float] = []
    bvs: list[str] = []
    report: dict[str, int] = {}
    jsonl = DATASET / "dataset_final_v4.jsonl"

    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            continue
        win_label = frame.get("win_label") or {}
        winner = str(win_label.get("winner_side", ""))
        winner = "蓝方" if winner == "blue" else ("红方" if winner == "red" else winner)
        author_camp = str(frame.get("author_camp", ""))
        author_win = win_label.get("author_win")
        if author_win is not None:
            label = 1.0 if bool(author_win) else 0.0
        elif winner in ("蓝方", "红方") and author_camp in ("blue", "red"):
            label = 1.0 if winner == ("蓝方" if author_camp == "blue" else "红方") else 0.0
        else:
            continue
        bs = (frame.get("battle_state") or {}).get("base") or {}
        kills_blue = bs.get("kills_blue")
        kills_red = bs.get("kills_red")
        if author_camp == "red":
            kills_blue, kills_red = kills_red, kills_blue  # 转成作者视角
        towers = bs.get("towers")
        dragons = bs.get("dragons")
        gold = bs.get("gold")
        numeric = [v for v in (kills_blue, kills_red, towers, dragons, gold) if isinstance(v, (int, float))]
        if not numeric or all(v == 0 for v in numeric):
            continue

        time_norm = float(frame.get("game_time_s", 0)) / 1500.0
        level_id = LEVEL_MAP.get(str(frame.get("level_id", "")), 3)
        ally_hp = _mean(frame.get("ally_hp") or [])
        enemy_hp = _mean(frame.get("enemy_hp") or [])

        context = [0.0] * len(CONTEXT_FEATURES)
        context[CONTEXT_FEATURES.index("time_norm")] = time_norm
        context[CONTEXT_FEATURES.index("econ_sat")] = min(1.0, time_norm)
        context[CONTEXT_FEATURES.index("enemy_threat")] = 0.5
        context[CONTEXT_FEATURES.index("own_hp_frac")] = ally_hp if ally_hp > 0 else 0.7
        context[CONTEXT_FEATURES.index("tower_adv")] = 0.0
        context[CONTEXT_FEATURES.index("dragon_avail")] = 1.0 if float(frame.get("game_time_s", 0)) >= 120.0 else 0.0
        context[CONTEXT_FEATURES.index("team_aa_power")] = 0.8
        context[CONTEXT_FEATURES.index("team_caster_power")] = 0.8
        context[CONTEXT_FEATURES.index("build_attack_power")] = 0.8
        context[CONTEXT_FEATURES.index("build_defense_power")] = 0.8
        context[CONTEXT_FEATURES.index("build_utility_power")] = 0.8
        context[CONTEXT_FEATURES.index("skill_power")] = 0.8
        context[CONTEXT_FEATURES.index("match_level")] = level_id / 7.0

        features = [0.0] * len(VALUE_FEATURES)
        if isinstance(kills_blue, (int, float)) and isinstance(kills_red, (int, float)):
            features[VALUE_FEATURES.index("kill_diff")] = (kills_blue - kills_red) / 10.0
        if isinstance(gold, (int, float)):
            features[VALUE_FEATURES.index("economy_diff")] = gold / 10000.0
        if isinstance(towers, (int, float)):
            features[VALUE_FEATURES.index("tower_diff")] = towers / 10.0
        if isinstance(dragons, (int, float)):
            features[VALUE_FEATURES.index("overlord_diff")] = dragons / 5.0
        if ally_hp and enemy_hp:
            features[VALUE_FEATURES.index("hp_diff")] = ally_hp - enemy_hp

        context_rows.append(context)
        feature_rows.append(features)
        labels.append(label)
        bv = str(frame.get("video_bv"))
        bvs.append(bv)
        report[bv] = report.get(bv, 0) + 1

    c = torch.tensor(context_rows, dtype=torch.float32)
    f = torch.tensor(feature_rows, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    return c, f, y, [{"bv": k, "frames": v} for k, v in sorted(report.items())], bvs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--no-save", action="store_true", help="冒烟测试不覆盖正式模型")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch={torch.__version__} device={device}")
    c, f, y, report, bvs = _build_dataset()
    print(f"samples={c.shape[0]} labels_pos={y.sum().item()}/{y.shape[0]}")

    split_path = HOKDATA_ROOT / "splits.json"
    game_splits: dict[str, str] = {}
    if split_path.exists():
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        game_splits = payload.get("tencent_rt", {})
        base_splits = payload.get("tencent_rt_base", {})
        if base_splits:
            game_splits = {**base_splits, **game_splits}
    if game_splits:
        def split_of(bv: str) -> str | None:
            return game_splits.get(bv) or game_splits.get(_base_key(bv))
    else:
        print("splits.json 缺失：使用确定性 hash 按基础局切分")
        def split_of(bv: str) -> str | None:
            return _fallback_split(_base_key(bv))

    train_idx = [i for i, bv in enumerate(bvs) if split_of(bv) == "train"]
    val_idx = [i for i, bv in enumerate(bvs) if split_of(bv) == "val"]
    test_idx = [i for i, bv in enumerate(bvs) if split_of(bv) == "test"]
    if not train_idx or not val_idx or not test_idx:
        raise SystemExit(f"split failed: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(
        f"game split: train_games={len(set(bvs[i] for i in train_idx))} "
        f"val_games={len(set(bvs[i] for i in val_idx))} "
        f"test_games={len(set(bvs[i] for i in test_idx))} "
        f"train_frames={len(train_idx)} val_frames={len(val_idx)} test_frames={len(test_idx)}"
    )

    c, f, y = c.to(device), f.to(device), y.to(device)
    tr = torch.tensor(train_idx, dtype=torch.long, device=device)
    va = torch.tensor(val_idx, dtype=torch.long, device=device)
    te = torch.tensor(test_idx, dtype=torch.long, device=device)
    model = AttentionValueModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    accs: list[float] = []
    baselines: list[float] = []
    test_accs: list[float] = []
    test_bases: list[float] = []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        model = AttentionValueModel().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for step in range(args.steps):
            idx = tr[torch.randint(0, len(tr), (64,), device=device)]
            opt.zero_grad()
            out = model(c[idx], f[idx])
            loss = loss_fn(out, y[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(model(c[va], f[va])) > 0.5).float()
            acc = (pred == y[va]).float().mean().item()
            baseline = max(y[va].mean().item(), 1 - y[va].mean().item())
            pred_test = (torch.sigmoid(model(c[te], f[te])) > 0.5).float()
            test_acc = (pred_test == y[te]).float().mean().item()
            test_base = max(y[te].mean().item(), 1 - y[te].mean().item())
        accs.append(acc)
        baselines.append(baseline)
        test_accs.append(test_acc)
        test_bases.append(test_base)
        print(
            f"seed {seed}: val acc={acc:.4f} baseline={baseline:.4f} | "
            f"test acc={test_acc:.4f} baseline={test_base:.4f}",
            flush=True,
        )
    mean_acc = sum(accs) / len(accs)
    mean_base = sum(baselines) / len(baselines)
    mean_test_acc = sum(test_accs) / len(test_accs)
    mean_test_base = sum(test_bases) / len(test_bases)
    print(f"mean val accuracy: {mean_acc:.4f} | majority baseline: {mean_base:.4f}")
    print(f"mean test accuracy: {mean_test_acc:.4f} | majority baseline: {mean_test_base:.4f}")
    artifact = PROJECT_ROOT / "data" / "S45" / "models" / "attention_value_real.pt"
    metrics = {
        "val_accuracy": mean_acc,
        "majority_baseline": mean_base,
        "test_accuracy": mean_test_acc,
        "test_majority_baseline": mean_test_base,
        "samples": c.shape[0],
        "train_games": len(set(bvs[i] for i in train_idx)),
        "val_games": len(set(bvs[i] for i in val_idx)),
        "test_games": len(set(bvs[i] for i in test_idx)),
        "seeds": accs,
        "test_seeds": test_accs,
        "steps": args.steps,
        "levels": report,
    }
    if args.no_save:
        print("no-save: skip artifact", metrics)
    else:
        model.save_artifacts(artifact, metrics)
        print("saved:", artifact)
    print("REAL_TRAIN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
