# -*- coding: utf-8 -*-
"""KPL BP 到底有没有信号：把"阵容组合"换成"单英雄边际强度"再测一次（只读）。

为什么需要这个脚本
------------------
已有结论是「KPL 阵容 BP 的 test AUC 0.5272 / 0.5319，几乎无信号」。但 KPL 语料是
**8740 局、131 个英雄、唯一阵容 8737/8740** —— 任何以"10 元组合"为特征的模型都
不可能泛化（每个组合平均只出现 1 次）。所以低 AUC 有两种完全不同的解释：

  A. BP 真的没有信息量；
  B. 特征参数化错了（组合级 one-hot 不可学），单英雄边际效应其实有信号。

本脚本把特征换成 131 维多热：某英雄在蓝方记 +1、在红方记 -1、未上场记 0。
这样学到的权重就是「该英雄相对另一方的强度」，参数量 131，样本 8740 局，
足以估出稳定权重。若 AUC 仍 ≈0.5，则 A 成立；若明显 >0.5，则 B 成立。

用法::

  venv/Scripts/python.exe -X utf8 scripts/probe_kpl_bp_signal.py
  venv/Scripts/python.exe -X utf8 scripts/probe_kpl_bp_signal.py --json runs/eval/kpl_bp_signal.json

口径：按 battle_id 做 <battle_id> 分组切分（每局只有一个样本，天然无跨局泄漏）；
      AUC 用 5 折 CV 的 out-of-fold 预测算，另给 2000 次 bootstrap 的 95% CI。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
COMPOSITION = WORKSPACE / "hokdata" / "external" / "kpl" / "composition.jsonl"


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        picks = (d.get("blue_picks") or []) + (d.get("red_picks") or [])
        if len(picks) != 10 or d.get("blue_win") is None:
            continue
        rows.append(d)
    return rows


def build_matrix(rows: list[dict]):
    heroes = sorted({h for r in rows for h in (r["blue_picks"] + r["red_picks"])})
    index = {h: i for i, h in enumerate(heroes)}
    X = np.zeros((len(rows), len(heroes)), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int8)
    for i, r in enumerate(rows):
        for h in r["blue_picks"]:
            X[i, index[h]] += 1.0
        for h in r["red_picks"]:
            X[i, index[h]] -= 1.0
        y[i] = 1 if r["blue_win"] else 0
    return heroes, X, y


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 2e-3,
                  iters: int = 1500, lr: float = 0.6):
    """手写 L2-logistic（不依赖 sklearn）。

    特征列已按 sqrt(var)+eps 缩放，所以固定学习率就足够稳定；131 维 × 近万行
    梯度下降在毫秒级，没有必要引入 scipy/sklearn。
    """
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    for _ in range(iters):
        z = np.clip(X @ w + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w)
        b -= lr * float(g.mean())
    return w, b


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC（不依赖 sklearn 的 tie 处理）。"""
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    sorted_score = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_score[j + 1] == sorted_score[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "runs" / "eval" / "kpl_bp_signal.json"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261026)
    args = ap.parse_args()

    rows = load_rows(COMPOSITION)
    heroes, X, y = build_matrix(rows)
    scale = np.sqrt((X ** 2).mean(axis=0)) + 1e-6
    Xs = X / scale
    print(f"[data] {len(rows)} 局 / {len(heroes)} 英雄；蓝方胜率 {y.mean():.3f}")
    base = max(y.mean(), 1 - y.mean())
    print(f"[base] 多数类准确率 {base:.4f}")

    # 按 battle_id 分组切分：每局只有一个样本，等价于按局切分。
    ids = [str(r.get("battle_id")) for r in rows]
    rnd = random.Random(args.seed)
    order = list(range(len(rows)))
    rnd.shuffle(order)
    fold_of = {idx: n % args.folds for n, idx in enumerate(order)}

    oof = np.zeros(len(rows), dtype=np.float64)
    for f in range(args.folds):
        tr = [i for i in range(len(rows)) if fold_of[i] != f]
        te = [i for i in range(len(rows)) if fold_of[i] == f]
        if not te:
            continue
        model_w, model_b = _fit_logistic(Xs[tr], y[tr].astype(np.float64))
        oof[te] = Xs[te] @ model_w + model_b

    a = auc(y, oof)
    print(f"[cv] 单英雄边际强度 out-of-fold AUC = {a:.4f}")

    # bootstrap CI（按局重采样）
    rnd2 = random.Random(args.seed + 1)
    boots = []
    for _ in range(args.boot):
        idx = [rnd2.randrange(len(rows)) for _ in range(len(rows))]
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        boots.append(auc(yy, oof[idx]))
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
    print(f"[cv] 95% CI = [{lo:.4f}, {hi:.4f}]（{len(boots)} 次重采样）")

    # 全量拟合，给出最弱/最强英雄（仅诊断，不作因果解读）
    full_w, full_b = _fit_logistic(Xs, y.astype(np.float64))
    w = full_w / scale                 # 换回原尺度，权重可直接按英雄比较
    top = np.argsort(-w)[:5]
    bot = np.argsort(w)[:5]
    print("[w] 权重最高:", [(heroes[i], round(float(w[i]), 3)) for i in top])
    print("[w] 权重最低:", [(heroes[i], round(float(w[i]), 3)) for i in bot])

    # 判定必须看 CI，不能只看点估计：CI 下界 > 0.5 意味着"信号与 0.5 可靠可分"，
    # 但"可分"不等于"够用"——LoL 局面快照 AUC 0.77 才是可用档位。
    if lo <= 0.5:
        verdict = "无可靠信号（CI 跨 0.5）"
    elif a >= 0.60:
        verdict = "有可用信号"
    elif a >= 0.55:
        verdict = "信号真实但偏弱，需与局面特征联合才可能有用"
    else:
        verdict = "信号真实但幅度极小（AUC<0.55），单独使用无决策价值"
    print(f"[verdict] {verdict}")

    out = {
        "games": len(rows), "heroes": len(heroes), "blue_win_rate": round(float(y.mean()), 4),
        "majority_baseline": round(float(base), 4),
        "cv_auc": round(a, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "folds": args.folds, "boot": len(boots),
        "feature": "131 维多热：蓝方 +1 / 红方 -1 / 未上场 0",
        "top_weights": [(heroes[i], round(float(w[i]), 3)) for i in top],
        "bottom_weights": [(heroes[i], round(float(w[i]), 3)) for i in bot],
        "note_weights": "极端权重（|w|>2）多为低频英雄，勿当英雄强度结论；CV AUC 已排除泄漏",
        "verdict": verdict,
    }
    p = Path(args.json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[json] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
