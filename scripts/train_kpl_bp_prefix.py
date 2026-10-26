r"""KPL BP 前缀 -> 胜负：选秀阶段决策价值基线。

数据：<hokdata>/external/kpl/battles.jsonl（2026 官方 API，20 步 BP）
样本：每局 20 个前缀（开局 -> 第 19 步后），共 ~26k
特征：[蓝pick | 红pick | 蓝ban | 红ban] one-hot + action_index + year
切分：按 match_id 70/15/15。

输出：
  <hokdata>/models/kpl_bp_prefix_value.pt
  <hokdata>/models/kpl_bp_prefix_metrics.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

HOKDATA = Path(__file__).resolve().parents[2] / "hokdata"
BATTLES = HOKDATA / "external" / "kpl" / "battles.jsonl"
HEROES = PROJECT_ROOT / "data" / "S45" / "normalized" / "heroes.json"
OUT_DIR = HOKDATA / "models"


def _split(key: str) -> str:
    bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < 0.70:
        return "train"
    if bucket < 0.85:
        return "val"
    return "test"


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pairs = sorted(zip(scores.tolist(), labels.tolist()))
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return 0.5
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        rank_sum += ((i + j) / 2 + 1) * sum(y for _, y in pairs[i : j + 1])
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> int:
    hero_order = {
        int(row["ename"]): idx
        for idx, row in enumerate(json.loads(HEROES.read_text(encoding="utf-8"))["data"])
    }
    h = len(hero_order)
    rows = []
    games = 0
    for line in BATTLES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        game = json.loads(line)
        bp = game.get("bp") or []
        if len(bp) != 20:
            continue
        games += 1
        year = int(str(game.get("league_id", ""))[:4]) if game.get("league_id") else 2026
        blue_picks: set[int] = set()
        red_picks: set[int] = set()
        blue_bans: set[int] = set()
        red_bans: set[int] = set()
        for i, action in enumerate(bp):
            vec = np.zeros(h * 4 + 2, dtype=np.float32)
            for hero in blue_picks:
                vec[hero] = 1.0
            for hero in red_picks:
                vec[h + hero] = 1.0
            for hero in blue_bans:
                vec[2 * h + hero] = 1.0
            for hero in red_bans:
                vec[3 * h + hero] = 1.0
            vec[-2] = i / 19.0
            vec[-1] = year / 2026.0
            rows.append(
                {
                    "match_id": str(game.get("match_id")),
                    "battle_id": game.get("battle_id"),
                    "action_index": i,
                    "label": 1.0 if game.get("win_camp") == 1 else 0.0,
                    "features": vec,
                }
            )
            hero = hero_order.get(int(action.get("hero_id"))) if action.get("hero_id") else None
            if hero is None:
                continue
            if action.get("camp") == 1 and action.get("action") == "pick":
                blue_picks.add(hero)
            elif action.get("camp") == 2 and action.get("action") == "pick":
                red_picks.add(hero)
            elif action.get("camp") == 1 and action.get("action") == "ban":
                blue_bans.add(hero)
            elif action.get("camp") == 2 and action.get("action") == "ban":
                red_bans.add(hero)

    if not rows:
        print("no BP rows; run fetch_kpl_battles.py first")
        return 1
    x = np.stack([r["features"] for r in rows])
    y = np.asarray([r["label"] for r in rows], dtype=np.float32)
    splits = np.asarray([_split(r["match_id"]) for r in rows])
    print(f"games={games} prefixes={len(rows)} features={x.shape[1]} splits={dict(Counter(splits))}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.tensor(x, device=device)
    yt = torch.tensor(y, device=device)
    tr = torch.tensor(np.where(splits == "train")[0], device=device)
    va = torch.tensor(np.where(splits == "val")[0], device=device)
    te = torch.tensor(np.where(splits == "test")[0], device=device)

    model = nn.Linear(x.shape[1], 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc, best_state = -1.0, None
    for epoch in range(400):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xt[tr]).squeeze(-1), yt[tr])
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_p = torch.sigmoid(model(xt[va]).squeeze(-1)).cpu().numpy()
            val_auc = _auc(val_p, y[np.where(splits == "val")[0]])
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(xt[te]).squeeze(-1)).cpu().numpy()
    labels = y[np.where(splits == "test")[0]]
    action_idx = np.asarray([r["action_index"] for r in rows])[np.where(splits == "test")[0]]
    buckets = {}
    for name, lo, hi in (("0-4", 0, 4), ("5-9", 5, 9), ("10-14", 10, 14), ("15-19", 15, 19)):
        mask = (action_idx >= lo) & (action_idx <= hi)
        if mask.any():
            buckets[name] = {
                "rows": int(mask.sum()),
                "auc": round(_auc(probs[mask], labels[mask]), 4),
                "accuracy": round(float(((probs[mask] > 0.5).astype(np.float32) == labels[mask]).mean()), 4),
            }
    metrics = {
        "games": games,
        "prefixes": len(rows),
        "splits": dict(Counter(splits)),
        "val_auc": round(best_auc, 4),
        "test_auc": round(_auc(probs, labels), 4),
        "test_accuracy": round(float(((probs > 0.5).astype(np.float32) == labels).mean()), 4),
        "test_majority": round(float(max(labels.mean(), 1 - labels.mean())), 4),
        "by_action_bucket": buckets,
        "note": "BP prefix value baseline; composition signal is weak but increases with more picks/bans",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "hero_order": hero_order, "hero_count": h,
         "features": "blue_pick|red_pick|blue_ban|red_ban|action_index|year_norm"},
        OUT_DIR / "kpl_bp_prefix_value.pt",
    )
    (OUT_DIR / "kpl_bp_prefix_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
