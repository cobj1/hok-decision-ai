#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 Dota2 语义数据上训练"局面 → 语义动作"先验，并映射成 HoK 的动作先验。

回答两个问题
------------
1. **只用局面特征，能预测"人类该做什么"到什么程度？**
   这是行为克隆的**天花板参照**：如果局面→动作的可预测性就很低，
   那么上视觉特征也救不了；反之说明"局面"这一层是值得投入的。
2. **能否给 BCNet 的 10 类动作头一个合理初值？**
   Dota2 的 7 类语义动作有 4 类与 HoK 一一对应（move / skill / recall / none），
   其余靠映射矩阵展开；映射不上的 3 类（attack / buy / restore）保持 0。

训练协议（与项目既有口径一致）
------------------------------
- **按 match_id 切分**，不按行切；70/15/15；同一局绝不跨 split。
- 标准化统计量**只用 train**。
- 类别不平衡用逆频率加权。
- 报告 majority 基线、先验基线、模型三项，避免"看起来不错"。

用法
----
    cd D:/work/aiKings/hok-decision-ai
    ../WZRYagnt/venv/Scripts/python.exe scripts/train_dota2_action_prior.py \
        --data ../hokdata/external/dota2_dumps/semantic_pilot.jsonl \
        --out-model data/S45/models/dota2_action_prior.pt \
        --out-report data/S45/models/dota2_action_prior_report.json \
        --out-prior ../hokdata/action_prior.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

CLASSES = ["advance", "retreat", "roam", "hold", "cast", "recall", "dead"]
FEATURES = [
    "x_norm", "y_norm", "hp_ratio", "mana_ratio", "level", "networth",
    "gold_diff", "kill_diff", "n_enemy_unseen", "alive",
    "dist_own_base", "dist_enemy_base", "t_min",
]

# Dota2 语义动作 -> HoK ACT_NAMES（move/skill1/skill2/skill3/attack/buy/recall/summoner/restore/none）
HOK_ACT = ["move", "skill1", "skill2", "skill3", "attack", "buy",
           "recall", "summoner", "restore", "none"]
MAP = {
    "advance": {"move": 1.0},
    "retreat": {"move": 1.0},
    "roam": {"move": 1.0},
    # Dota 的 combat log 不区分技能位，均匀摊到 3 个技能 + 召唤师技能
    "cast": {"skill1": 0.25, "skill2": 0.25, "skill3": 0.25, "summoner": 0.25},
    "recall": {"recall": 1.0},
    "hold": {"none": 1.0},
    # 阵亡不可行动：归到 none，而不是 move（否则会教出"死了还在走"）
    "dead": {"none": 1.0},
}


def load_rows(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """返回 X, y, match_ids, meta。缺字段的行直接跳过并计数。"""
    xs, ys, mids = [], [], []
    skipped = Counter()
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        st = r["state"]
        act = r.get("action")
        if act not in CLASSES:
            skipped["unknown_action"] += 1
            continue
        try:
            hp = st["hp_ratio"]
            mana = st["mana_ratio"]
            lvl = st["level"]
            nw = st["networth"]
            x, y = st["x_norm"], st["y_norm"]
        except KeyError:
            skipped["missing_field"] += 1
            continue
        if hp is None or mana is None or lvl is None or nw is None:
            skipped["null_field"] += 1
            continue
        team = r["team"]
        # 己方基地在 (-1,-1) 或 (1,1) 角；敌方在对角
        sgn = 1.0 if (team == 0) != (x < 0) else -1.0
        own = ((x - sgn) ** 2 + (y - sgn) ** 2) ** 0.5
        enemy = ((x + sgn) ** 2 + (y + sgn) ** 2) ** 0.5
        xs.append([
            x, y, hp, mana, float(lvl), float(nw) / 1000.0,
            st["gold_diff"] / 1000.0, float(st["kill_diff"]),
            1.0 if st["alive"] else 0.0, float(st["n_enemy_unseen"]),
            own, enemy, float(r.get("t") or 0.0) / 60.0,
        ])
        ys.append(CLASSES.index(act))
        mids.append(r["match_id"])
    X = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    meta = {"rows": len(y), "skipped": dict(skipped)}
    return X, y, np.asarray(mids), meta


def split_by_match(mids: np.ndarray, seed: int = 0) -> np.ndarray:
    """0=train / 1=val / 2=test，**按局**做 70/15/15 的秩切分。

    不用 md5 哈希定桶：55 局时哈希分布很不均匀（实测会切出 44/3/8，
    val 只有 3 局，模型选择完全噪声）。按排序后的局序号切，
    能保证 val/test 的局数接近目标比例。
    """
    uniq = np.array(sorted(set(mids.tolist())))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(uniq))
    uniq = uniq[order]
    n = len(uniq)
    n_tr = max(1, int(round(n * 0.70)))
    n_va = max(1, int(round(n * 0.15)))
    tr_set = set(uniq[:n_tr].tolist())
    va_set = set(uniq[n_tr:n_tr + n_va].tolist())
    out = np.full(len(mids), 2, dtype=np.int64)
    for i, m in enumerate(mids):
        if m in tr_set:
            out[i] = 0
        elif m in va_set:
            out[i] = 1
    return out


class MLP(nn.Module):
    def __init__(self, d_in: int, n_cls: int, hidden: int = 64, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, x):
        return self.net(x)


def metrics(y_true, y_pred, n_cls):
    acc = float((y_true == y_pred).mean())
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per = []
    for c in range(n_cls):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per.append({"class": CLASSES[c], "precision": round(float(prec), 4),
                    "recall": round(float(rec), 4), "f1": round(float(f1), 4),
                    "support": int(cm[c, :].sum())})
    macro_f1 = float(np.mean([p["f1"] for p in per]))
    return {"acc": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "per_class": per, "confusion": cm.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Dota2 语义动作先验训练")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-model", required=True)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--out-prior", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--class-weight", choices=["inv", "none"], default="inv",
                    help="inv=逆频率加权（偏 macro-F1）；none=不加权（偏 accuracy）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    X, y, mids, meta = load_rows(args.data)
    print(f"载入 {meta['rows']} 行，跳过 {meta['skipped']}，耗时 {time.time()-t0:.1f}s")
    print("类别分布:", {CLASSES[k]: v for k, v in sorted(Counter(y).items())})

    part = split_by_match(mids)
    tr, va, te = part == 0, part == 1, part == 2
    print(f"按局切分: train {len(set(mids[tr]))} 局 / val {len(set(mids[va]))} 局 / "
          f"test {len(set(mids[te]))} 局（行数 {tr.sum()}/{va.sum()}/{te.sum()}）")
    if len(set(mids[tr])) < 3 or len(set(mids[te])) < 1:
        sys.exit("局数太少，无法按局切分")

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd

    Xt = torch.from_numpy(Xn)
    yt = torch.from_numpy(y)

    cnt = np.bincount(y[tr], minlength=len(CLASSES)).astype(np.float64)
    if args.class_weight == "inv":
        w = torch.tensor((cnt.sum() / (len(CLASSES) * np.maximum(cnt, 1))), dtype=torch.float32)
    else:
        w = None
    print("类别权重:", "逆频率" if w is not None else "无（纯 CE）")

    model = MLP(X.shape[1], len(CLASSES))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss(weight=w)

    idx = np.where(tr)[0]
    best = {"macro_f1": -1.0}
    for ep in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(idx)
        tot = 0.0
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            xb, yb = Xt[b], yt[b]
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        model.eval()
        with torch.no_grad():
            pv = model(Xt[va]).argmax(-1).numpy()
        m = metrics(y[va], pv, len(CLASSES))
        print(f"  ep{ep:02d} loss={tot/max(len(idx),1):.4f} val_acc={m['acc']:.4f} "
              f"val_macroF1={m['macro_f1']:.4f}")
        if m["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": m["macro_f1"], "state": {k: v.clone() for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        pt_va = model(Xt[va]).argmax(-1).numpy()
        pt_te = model(Xt[te]).argmax(-1).numpy()
    mva = metrics(y[va], pt_va, len(CLASSES))
    mte = metrics(y[te], pt_te, len(CLASSES))

    majority = int(np.bincount(y[tr]).argmax())
    base_acc = float((y[te] == majority).mean())
    prior_te = np.bincount(y[tr], minlength=len(CLASSES)) / len(y[tr])
    prior_pred_acc = float((y[te] == int(prior_te.argmax())).mean())

    print(f"\nval  acc={mva['acc']:.4f}  macroF1={mva['macro_f1']:.4f}")
    print(f"test acc={mte['acc']:.4f}  macroF1={mte['macro_f1']:.4f}")
    print(f"基线 majority({CLASSES[majority]}) acc={base_acc:.4f}；先验 argmax acc={prior_pred_acc:.4f}")
    print(f"相对 majority 提升 {(mte['acc']-base_acc)*100:+.1f} pp")

    # 映射到 HoK 10 类
    p_dota = prior_te
    hok = np.zeros(len(HOK_ACT))
    for i, c in enumerate(CLASSES):
        for name, wgt in MAP[c].items():
            hok[HOK_ACT.index(name)] += p_dota[i] * wgt
    hok = hok / hok.sum()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_model)) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "features": FEATURES,
                "classes": CLASSES, "mu": mu.tolist(), "sd": sd.tolist(),
                "split": "by_match_id 70/15/15", "val_acc": mva["acc"],
                "val_macro_f1": mva["macro_f1"], "test_acc": mte["acc"]}, args.out_model)

    report = {"data": args.data, "rows": meta["rows"], "skipped": meta["skipped"],
              "matches": {"train": len(set(mids[tr])), "val": len(set(mids[va])),
                          "test": len(set(mids[te]))},
              "class_prior_train": {CLASSES[i]: round(float(p), 4) for i, p in enumerate(prior_te)},
              "baselines": {"majority_class": CLASSES[majority],
                            "majority_acc": round(base_acc, 4),
                            "prior_argmax_acc": round(prior_pred_acc, 4)},
              "class_weight": args.class_weight, "val": mva, "test": mte,
              "delta_vs_majority_pp": round((mte["acc"] - base_acc) * 100, 2),
              "features": FEATURES, "epochs": args.epochs, "seed": args.seed}
    json.dump(report, open(args.out_report, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    prior = {
        "source": "Dota2 semantic_pilot（CC BY-NC-SA 4.0 衍生，仅研究用）",
        "method": "state-only MLP(test acc 见 metrics) 的类别先验，经映射矩阵投影到 HoK 10 类",
        "hok_act_names": HOK_ACT,
        "dota_class_prior": {CLASSES[i]: round(float(p), 4) for i, p in enumerate(p_dota)},
        "mapping": MAP,
        "hok_prior": {HOK_ACT[j]: round(float(p), 5) for j, p in enumerate(hok)},
        "usage": ("作为 BCNet act 头 bias 的初值：bias_j = log(prior_j + eps)；"
                  "或作为样本加权/阈值校准的参照。不要把它当作'人类在王者里的动作分布'，"
                  "它是 Dota2 语义动作经 4 类一对多映射后的先验，attack/buy/restore 恒为 0。"),
        "caveat": ("cast 在 Dota 里不分技能位，这里均匀摊到 skill1/2/3/summoner，"
                   "所以这 4 类的相对大小没有信息量；有信息量的是"
                   "move 总量 / skill 总量 / none 总量 / recall 总量的比例。"),
        "metrics": {"test_acc": mte["acc"], "test_macro_f1": mte["macro_f1"],
                    "majority_acc": round(base_acc, 4)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prior)) or ".", exist_ok=True)
    json.dump(prior, open(args.out_prior, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"\nHoK 先验: " + "  ".join(f"{k}={v:.3f}" for k, v in prior["hok_prior"].items()))
    print(f"模型 -> {args.out_model}\n报告 -> {args.out_report}\n先验 -> {args.out_prior}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
