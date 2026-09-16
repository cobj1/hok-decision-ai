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

import json
import glob
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

DATASET = Path(r"E:\hokdata\dataset")
LEVEL_MAP = {"1500+": 2, "1800+": 3, "2100+": 4, "2300+": 5, "2500+": 6, "赛事": 7}


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
        bv = frame.get("video_bv")
        report[bv] = report.get(bv, 0) + 1

    c = torch.tensor(context_rows, dtype=torch.float32)
    f = torch.tensor(feature_rows, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    return c, f, y, [{"bv": k, "frames": v} for k, v in sorted(report.items())]


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch={torch.__version__} device={device}")
    c, f, y, report = _build_dataset()
    print(f"samples={c.shape[0]} labels_pos={y.sum().item()}/{y.shape[0]}")
    for row in report:
        print(row)

    c, f, y = c.to(device), f.to(device), y.to(device)
    model = AttentionValueModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    accs: list[float] = []
    baselines: list[float] = []
    for seed in range(5):
        torch.manual_seed(seed)
        model = AttentionValueModel().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        perm = torch.randperm(c.shape[0], device=device)
        n_train = max(1, int(c.shape[0] * 0.8))
        tr, va = perm[:n_train], perm[n_train:]
        for step in range(1500):
            idx = tr[torch.randint(0, len(tr), (64,), device=device)]
            opt.zero_grad()
            out = model(c[idx], f[idx])
            loss = loss_fn(out, y[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out = model(c[va], f[va])
            pred = (torch.sigmoid(out) > 0.5).float()
            acc = (pred == y[va]).float().mean().item()
            baseline = max(y[va].mean().item(), 1 - y[va].mean().item())
        accs.append(acc)
        baselines.append(baseline)
        print(f"seed {seed}: acc={acc:.4f} baseline={baseline:.4f}", flush=True)
    mean_acc = sum(accs) / len(accs)
    mean_base = sum(baselines) / len(baselines)
    print(f"mean val accuracy: {mean_acc:.4f} | majority baseline: {mean_base:.4f}")
    artifact = PROJECT_ROOT / "data" / "S45" / "models" / "attention_value_real.pt"
    model.save_artifacts(
        artifact,
        {"val_accuracy": mean_acc, "majority_baseline": mean_base, "samples": c.shape[0], "seeds": accs, "levels": report},
    )
    print("saved:", artifact)
    print("REAL_TRAIN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
