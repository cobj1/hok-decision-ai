r"""LoL Kaggle 逐分钟快照 -> 胜负价值基线（跨游戏预训练验证）。

数据：<hokdata>/external/lol_kaggle/snapshots.npz（333k 快照，27 特征，5/10/15 分钟）
切分：按 match_id 70/15/15，避免同局多分钟泄漏。

输出：
  <hokdata>/models/lol_snapshot_value.pt
  <hokdata>/models/lol_snapshot_metrics.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HOKDATA = Path(__file__).resolve().parents[2] / "hokdata"
NPZ = HOKDATA / "external" / "lol_kaggle" / "snapshots.npz"
META = HOKDATA / "external" / "lol_kaggle" / "snapshot_meta.jsonl"
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


class ValueNet(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main() -> int:
    data = np.load(NPZ)
    x = data["features"].astype(np.float32)
    y = data["labels"].astype(np.float32)
    minutes = data["minutes"].astype(np.int16)
    names = [str(v) for v in data["feature_names"]]
    match_ids = [json.loads(line)["match_id"] for line in META.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(match_ids) != len(x):
        print("meta/feature length mismatch", len(match_ids), len(x))
        return 1

    splits = np.asarray([_split(m) for m in match_ids])
    train = np.where(splits == "train")[0]
    val = np.where(splits == "val")[0]
    test = np.where(splits == "test")[0]
    print(f"rows={len(x)} dim={x.shape[1]} splits train/val/test={len(train)}/{len(val)}/{len(test)}")
    print("minutes", {int(m): int((minutes == m).sum()) for m in np.unique(minutes)})

    mean = x[train].mean(axis=0, keepdims=True)
    std = x[train].std(axis=0, keepdims=True) + 1e-6
    xz = (x - mean) / std
    xt = torch.tensor(xz)
    yt = torch.tensor(y)
    tr = torch.tensor(train, dtype=torch.long)
    va = torch.tensor(val, dtype=torch.long)
    te = torch.tensor(test, dtype=torch.long)
    mt = minutes

    model = ValueNet(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc, best_state = -1.0, None
    for epoch in range(30):
        model.train()
        perm = tr[torch.randperm(len(tr))]
        for start in range(0, len(perm), 4096):
            idx = perm[start : start + 4096]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_p = torch.sigmoid(model(xt[va])).numpy()
        val_auc = _auc(val_p, y[val])
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch+1} val_auc={val_auc:.4f}", flush=True)
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(xt[te])).numpy()
    labels = y[test]
    metrics = {
        "rows": int(len(x)),
        "matches": len(set(match_ids)),
        "features": names,
        "split_rows": {"train": len(train), "val": len(val), "test": len(test)},
        "overall": {
            "val_auc": round(best_auc, 4),
            "test_auc": round(_auc(probs, labels), 4),
            "test_accuracy": round(float(((probs > 0.5).astype(np.float32) == labels).mean()), 4),
            "majority_baseline": round(float(max(labels.mean(), 1 - labels.mean())), 4),
        },
        "by_minute": {},
    }
    for minute in (5, 10, 15):
        mask = (mt[test] == minute)
        if mask.any():
            metrics["by_minute"][str(minute)] = {
                "rows": int(mask.sum()),
                "test_auc": round(_auc(probs[mask], labels[mask]), 4),
                "test_accuracy": round(float(((probs[mask] > 0.5).astype(np.float32) == labels[mask]).mean()), 4),
                "majority_baseline": round(float(max(labels[mask].mean(), 1 - labels[mask].mean())), 4),
            }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "mean": mean, "std": std, "features": names},
        OUT_DIR / "lol_snapshot_value.pt",
    )
    (OUT_DIR / "lol_snapshot_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
