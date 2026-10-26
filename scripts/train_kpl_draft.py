r"""KPL 5v5 人类对局：BP/阵容 -> 胜负 基线。

数据：<hokdata>/external/kpl/composition.jsonl（职业档，7k+ 局）
特征：[蓝pick | 红pick | 蓝ban | 红ban] one-hot + has_bans + year_norm
切分：按 match_id 70/15/15，避免同场多局泄漏。

输出：
  <hokdata>/models/kpl_draft_value.pt
  <hokdata>/models/kpl_draft_metrics.json

运行（需 torch）：
  D:/work/aiKings/WZRYagnt/venv/Scripts/python.exe scripts/train_kpl_draft.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

HOKDATA = Path(__file__).resolve().parents[2] / "hokdata"
COMPOSITION = HOKDATA / "external" / "kpl" / "composition.jsonl"
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
    if not COMPOSITION.exists():
        print(f"missing {COMPOSITION}")
        return 1
    hero_order = {
        int(row["ename"]): idx
        for idx, row in enumerate(json.loads(HEROES.read_text(encoding="utf-8"))["data"])
    }
    hero_count = len(hero_order)

    rows = []
    unknown = set()
    for line in COMPOSITION.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        year = row.get("year") or 2024
        vec = np.zeros(hero_count * 4 + 2, dtype=np.float32)
        for hero in row.get("blue_picks") or []:
            idx = hero_order.get(int(hero))
            if idx is None:
                unknown.add(hero)
                continue
            vec[idx] = 1.0
        for hero in row.get("red_picks") or []:
            idx = hero_order.get(int(hero))
            if idx is None:
                unknown.add(hero)
                continue
            vec[hero_count + idx] = 1.0
        for hero in row.get("blue_bans") or []:
            idx = hero_order.get(int(hero))
            if idx is None:
                unknown.add(hero)
                continue
            vec[hero_count * 2 + idx] = 1.0
        for hero in row.get("red_bans") or []:
            idx = hero_order.get(int(hero))
            if idx is None:
                unknown.add(hero)
                continue
            vec[hero_count * 3 + idx] = 1.0
        has_bans = 1.0 if (row.get("blue_bans") or row.get("red_bans")) else 0.0
        vec[-2] = has_bans
        vec[-1] = float(year) / 2026.0
        rows.append(
            {
                "match_id": str(row.get("match_id")),
                "year": year,
                "features": vec,
                "label": 1.0 if row.get("blue_win") else 0.0,
                "has_bans": bool(has_bans),
            }
        )

    if not rows:
        print("no rows")
        return 1
    x = np.stack([r["features"] for r in rows])
    y = np.asarray([r["label"] for r in rows], dtype=np.float32)
    splits = np.asarray([_split(r["match_id"]) for r in rows])
    print(f"rows={len(rows)} heroes={hero_count} unknown_heroes={len(unknown)}")
    print("splits", {s: int((splits == s).sum()) for s in ("train", "val", "test")})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.tensor(x, device=device)
    yt = torch.tensor(y, device=device)
    tr = torch.tensor(np.where(splits == "train")[0], device=device)
    va = torch.tensor(np.where(splits == "val")[0], device=device)
    te = torch.tensor(np.where(splits == "test")[0], device=device)

    def evaluate(model, idx):
        model.eval()
        with torch.no_grad():
            logits = model(xt[idx]).squeeze(-1)
            probs = torch.sigmoid(logits)
        labels = yt[idx].cpu().numpy()
        preds = (probs.cpu().numpy() > 0.5).astype(np.float32)
        return {
            "auc": round(_auc(probs.cpu().numpy(), labels), 4),
            "accuracy": round(float((preds == labels).mean()), 4),
            "majority_baseline": round(float(max(labels.mean(), 1 - labels.mean())), 4),
            "rows": int(len(labels)),
        }

    results = {}
    for variant in ("picks_only", "picks_bans"):
        train_x, val_x = xt[tr].clone(), xt[va].clone()
        if variant == "picks_only":
            # 把 ban 段与 has_bans 置零，所有年份都可用
            train_x[:, hero_count * 2 : hero_count * 4 + 1] = 0.0
            val_x[:, hero_count * 2 : hero_count * 4 + 1] = 0.0
            train_idx, val_idx = tr, va
        else:
            train_idx, val_idx = tr, va
        model = nn.Linear(xt.shape[1], 1).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        best_val, best_state = -1.0, None
        for epoch in range(400):
            model.train()
            opt.zero_grad()
            logits = model(train_x).squeeze(-1)
            loss = loss_fn(logits, yt[train_idx])
            loss.backward()
            opt.step()
            if (epoch + 1) % 10 == 0:
                val_logits = model(val_x).squeeze(-1)
                val_auc = _auc(torch.sigmoid(val_logits).detach().cpu().numpy(), yt[val_idx].cpu().numpy())
                if val_auc > best_val:
                    best_val = val_auc
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if best_state:
            model.load_state_dict(best_state)
        # 测试集要对齐 variant 的特征处理
        test_x = xt[te].clone()
        if variant == "picks_only":
            test_x[:, hero_count * 2 : hero_count * 4 + 1] = 0.0
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(test_x).squeeze(-1)).cpu().numpy()
        labels = yt[te].cpu().numpy()
        results[variant] = {
            "val_auc": round(best_val, 4),
            "test_auc": round(_auc(probs, labels), 4),
            "test_accuracy": round(float(((probs > 0.5).astype(np.float32) == labels).mean()), 4),
            "test_majority": round(float(max(labels.mean(), 1 - labels.mean())), 4),
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hero_order": hero_order,
                "hero_count": hero_count,
                "variant": variant,
                "features": "blue_pick|red_pick|blue_ban|red_ban|has_bans|year_norm",
            },
            OUT_DIR / f"kpl_draft_value_{variant}.pt",
        )
        print(variant, results[variant])

    metrics = {
        "rows": len(rows),
        "matches": len({r["match_id"] for r in rows}),
        "hero_count": hero_count,
        "unknown_heroes": sorted(unknown),
        "years": sorted({r["year"] for r in rows}),
        "has_bans_rows": sum(1 for r in rows if r["has_bans"]),
        "splits": {s: int((splits == s).sum()) for s in ("train", "val", "test")},
        "results": results,
        "note": "composition-only baseline; post-game KPL human data; group split by match_id",
    }
    (OUT_DIR / "kpl_draft_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", OUT_DIR / "kpl_draft_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
