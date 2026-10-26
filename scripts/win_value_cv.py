r"""价值/胜负线性模型的交叉验证与校准（S2-2）。

在现有 `win_train_linear_baseline.py` 的线性 logistic 基线上补齐：
  1) 按基础局分组的分层 K 折（同局不同视角永不同折）；
  2) 网格超参选择（折内 val AUC 的 均值-标准差）；
  3) Platt / temperature scaling 校准（嵌套折外评估）；
  4) 按局 bootstrap 的特征系数稳定性；
  5) 按局置换标签的显著性检验；
  6) 全程只评一次 test。

约束：test（14 个基础局）不参与任何选择；不覆盖 value_linear_*.json。

运行：
    HOKDATA_ROOT=D:/work/aiKings/hokdata \
    D:/work/aiKings/WZRYagnt/venv/Scripts/python.exe scripts/win_value_cv.py

说明：训练器是 Adam 的 numpy 复刻（与 torch.optim.Adam 同语义：L2 权重衰减进梯度、
偏差校正）。在 1500 步/同 minibatch 序列下与 torch 版本的验证集概率最大差 3.5e-5、
val acc 完全一致；numpy 版快约 35 倍，便于网格 + bootstrap + 置换检验在同一命令内跑完。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HOKDATA_ROOT", str(PROJECT_ROOT.parent / "hokdata"))

from hok_decision.model.attention_value import CONTEXT_FEATURES, VALUE_FEATURES  # noqa: E402

HOKDATA_ROOT = Path(os.environ["HOKDATA_ROOT"])
FEATURES = CONTEXT_FEATURES + VALUE_FEATURES
SIGMA_FLOOR = 1e-6


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def _load_train_module():
    path = PROJECT_ROOT / "scripts" / "win_train_attention_real.py"
    spec = importlib.util.spec_from_file_location("win_train_attention_real", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_data():
    module = _load_train_module()
    c, f, y, _report, bvs = module._build_dataset()
    x = torch.cat([c, f], dim=1).numpy().astype(np.float64)
    labels = y.numpy().astype(np.float64)
    splits = json.loads((HOKDATA_ROOT / "splits.json").read_text(encoding="utf-8"))
    game_splits = {**splits.get("tencent_rt_base", {}), **splits.get("tencent_rt", {})}

    def split_of(bv: str) -> str | None:
        return game_splits.get(bv) or game_splits.get(str(bv).split("#", 1)[0])

    groups = np.array([str(bv).split("#", 1)[0] for bv in bvs])
    split = np.array([split_of(bv) or "unknown" for bv in bvs], dtype=object)
    return x, labels, groups, split, list(bvs)


def group_index(groups: np.ndarray) -> dict[str, np.ndarray]:
    table: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        table.setdefault(g, []).append(i)
    return {g: np.asarray(v, dtype=np.int64) for g, v in table.items()}


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """秩和法 AUC，等值用平均秩（numpy 实现，无 sklearn/scipy）。"""
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    uniq, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    last_cum = np.cumsum(counts).astype(np.float64)
    avg_rank = last_cum - (counts - 1) / 2.0
    ranks = avg_rank[inv]
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def logloss(labels: np.ndarray, prob: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-12, 1 - 1e-12)
    y = np.asarray(labels, dtype=np.float64)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(labels: np.ndarray, prob: np.ndarray) -> float:
    return float(((np.asarray(prob) - np.asarray(labels)) ** 2).mean())


def ece(labels: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    """等宽 10 桶 ECE。"""
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    n = len(p)
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        total += abs(y[mask].mean() - p[mask].mean()) * mask.sum() / n
    return float(total)


def accuracy(labels: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> float:
    return float(((prob > threshold).astype(np.float64) == np.asarray(labels)).mean())


def majority_rate(labels: np.ndarray) -> float:
    rate = float(np.asarray(labels).mean())
    return max(rate, 1.0 - rate)


def fit_logistic_1d(z: np.ndarray, labels: np.ndarray, ridge: float = 1e-6,
                    iters: int = 100) -> tuple[float, float]:
    """对一维得分 z 再拟合 logistic：p = sigmoid(a*z + b)。Newton 法（凸问题）。"""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    design = np.stack([z, np.ones_like(z)], axis=1)
    theta = np.array([1.0, 0.0])
    for _ in range(iters):
        p = _sigmoid(design @ theta)
        grad = design.T @ (p - y) / len(y) + ridge * theta
        weight = p * (1.0 - p) + 1e-9
        hess = (design * weight[:, None]).T @ design / len(y) + ridge * np.eye(2)
        step = np.linalg.solve(hess, grad)
        theta = theta - step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(theta[0]), float(theta[1])


def fit_temperature(z: np.ndarray, labels: np.ndarray) -> float:
    """温度缩放：p = sigmoid(z / T)，对 u=logT 做 Newton。"""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    u = 0.0
    for _ in range(200):
        t = np.exp(u)
        p = _sigmoid(z / t)
        grad = float((-(p - y) * (z / t)).mean())
        h = float((p * (1 - p) * (z / t) ** 2).mean()) + 1e-9
        step = grad / h
        step = float(np.clip(step, -0.5, 0.5))
        u -= step
        u = float(np.clip(u, -3.0, 3.0))
        if abs(step) < 1e-9:
            break
    return float(np.exp(u))


# --------------------------------------------------------------------------- #
# 训练器（Adam 的 numpy 复刻，与 torch.optim.Adam 同语义）
# --------------------------------------------------------------------------- #
def train_logistic(
    x: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    *,
    lr: float,
    weight_decay: float,
    steps: int,
    seed: int,
    batch: int = 64,
    standardize: bool = True,
) -> dict:
    tr = np.asarray(train_idx, dtype=np.int64)
    mu = x[tr].mean(axis=0) if standardize else np.zeros(x.shape[1])
    sd = x[tr].std(axis=0) if standardize else np.ones(x.shape[1])
    sd = np.where(sd < SIGMA_FLOOR, SIGMA_FLOOR, sd)
    xs = (x - mu) / sd

    rng = np.random.default_rng(seed)
    n_feat = x.shape[1]
    w = np.zeros(n_feat, dtype=np.float64)
    b = 0.0
    mw = np.zeros(n_feat)
    vw = np.zeros(n_feat)
    mb = 0.0
    vb = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    size = min(batch, len(tr))
    for t in range(1, steps + 1):
        pick = tr[rng.integers(0, len(tr), size)]
        xb = xs[pick]
        yb = labels[pick]
        p = _sigmoid(xb @ w + b)
        g = (p - yb) / size
        gw = xb.T @ g + weight_decay * w
        gb = float(g.sum())
        mw = beta1 * mw + (1 - beta1) * gw
        vw = beta2 * vw + (1 - beta2) * gw * gw
        mb = beta1 * mb + (1 - beta1) * gb
        vb = beta2 * vb + (1 - beta2) * gb * gb
        c1 = 1 - beta1 ** t
        c2 = 1 - beta2 ** t
        w = w - lr * (mw / c1) / (np.sqrt(vw / c2) + eps)
        b = b - lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
    return {"w": w, "b": float(b), "mu": mu, "sd": sd}


def predict_logit(model: dict, x: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
    sub = x if idx is None else x[np.asarray(idx, dtype=np.int64)]
    return ((sub - model["mu"]) / model["sd"]) @ model["w"] + model["b"]


def eval_bundle(labels: np.ndarray, logit: np.ndarray) -> dict:
    prob = _sigmoid(logit)
    return {
        "acc": accuracy(labels, prob),
        "auc": auc_score(labels, logit),
        "logloss": logloss(labels, prob),
        "brier": brier(labels, prob),
        "ece": ece(labels, prob, 10),
        "n": int(len(labels)),
        "pos_rate": float(np.asarray(labels).mean()),
        "majority": majority_rate(labels),
    }


def group_eval(labels: np.ndarray, group_ids: np.ndarray, logit: np.ndarray) -> dict:
    """按局聚合（帧概率取均值）后再算准确率：14 个基础局下更接近真实决策单位。"""
    uniq = sorted(set(group_ids.tolist()))
    prob = _sigmoid(logit)
    probs, ys, sizes = [], [], []
    for g in uniq:
        mask = group_ids == g
        probs.append(float(prob[mask].mean()))
        ys.append(float(labels[mask].mean()))
        sizes.append(int(mask.sum()))
    probs = np.array(probs)
    ys = np.array(ys)
    pred = (probs > 0.5).astype(np.float64)
    truth = (ys >= 0.5).astype(np.float64)
    return {
        "groups": len(uniq),
        "group_acc": float((pred == truth).mean()),
        "group_auc": auc_score(truth, probs),
        "group_logloss": logloss(truth, probs),
        "mean_frames": float(np.mean(sizes)),
        "detail": [
            {"group": g, "frames": s, "label": float(t), "prob": float(p)}
            for g, s, t, p in zip(uniq, sizes, truth, probs)
        ],
    }


# --------------------------------------------------------------------------- #
# 折
# --------------------------------------------------------------------------- #
def stratified_group_folds(group_labels: dict[str, int], group_sizes: dict[str, int],
                           k: int, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    folds: list[list[str]] = [[] for _ in range(k)]
    load = [0] * k
    for cls in sorted(set(group_labels.values())):
        members = [g for g, c in group_labels.items() if c == cls]
        members.sort()                      # 先固定顺序，保证同 seed 可复现
        rng.shuffle(members)
        for g in members:                   # 按当前帧数最少优先，兼顾类别比例与折大小
            target = int(np.argmin(load))
            folds[target].append(g)
            load[target] += group_sizes[g]
    return [sorted(f) for f in folds]


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20261024)
    parser.add_argument("--lrs", default="0.003,0.01,0.03")
    parser.add_argument("--wds", default="0,0.003,0.01,0.03,0.1")
    parser.add_argument("--steps-grid", default="300,800,1500")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--final-seeds", type=int, default=3)
    parser.add_argument("--test-boot", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--out", default=str(HOKDATA_ROOT / "value_cv_report.json"))
    parser.add_argument("--model-out", default=str(HOKDATA_ROOT / "value_cv_model.json"))
    parser.add_argument("--fast", action="store_true", help="冒烟：缩小网格/重采样次数")
    args = parser.parse_args()

    t_start = time.time()
    lrs = [float(v) for v in args.lrs.split(",")]
    wds = [float(v) for v in args.wds.split(",")]
    steps_grid = [int(v) for v in args.steps_grid.split(",")]
    if args.fast:
        lrs, wds, steps_grid = lrs[:2], wds[:2], steps_grid[:2]
        args.bootstrap, args.permutations = 5, 5
        args.test_boot, args.final_seeds = 200, 1

    x, labels, groups, split, bvs = load_data()
    gidx = group_index(groups)
    dev_groups = sorted({g for g, s in zip(groups, split) if s in ("train", "val")})
    test_groups = sorted({g for g, s in zip(groups, split) if s == "test"})
    dev_idx = np.concatenate([gidx[g] for g in dev_groups])
    test_idx = np.concatenate([gidx[g] for g in test_groups])
    n_feat = x.shape[1]

    group_labels = {g: int(round(float(labels[gidx[g]].mean()))) for g in dev_groups}
    group_sizes = {g: len(gidx[g]) for g in dev_groups}
    folds = stratified_group_folds(group_labels, group_sizes, args.folds, args.fold_seed)
    fold_frames = [[int(i) for g in fold for i in gidx[g]] for fold in folds]

    const_cols = [FEATURES[j] for j in range(n_feat) if float(x[dev_idx][:, j].std()) < SIGMA_FLOOR]

    print(f"samples={len(x)} features={n_feat} (const={len(const_cols)})")
    print(f"dev: {len(dev_groups)} 基础局 / {len(dev_idx)} 帧 | test: {len(test_groups)} 基础局 / {len(test_idx)} 帧")

    fold_desc = []
    for fi, fold in enumerate(folds):
        va = np.array(fold_frames[fi], dtype=np.int64)
        tr = np.array([i for j, f in enumerate(fold_frames) if j != fi for i in f], dtype=np.int64)
        fold_desc.append({
            "fold": fi,
            "val_groups": len(fold),
            "val_frames": int(len(va)),
            "train_frames": int(len(tr)),
            "val_pos_rate": float(labels[va].mean()),
            "val_majority": majority_rate(labels[va]),
            "val_groups_pos": int(sum(group_labels[g] for g in fold)),
        })
        print(f"  fold {fi}: {len(fold)} 局 / val {len(va)} 帧 / pos {fold_desc[-1]['val_pos_rate']:.3f} "
              f"| train {len(tr)} 帧")

    # ---------------- 2) 网格搜索 ----------------
    print("grid search ...", flush=True)
    grid_rows = []
    t_grid = time.time()
    for lr in lrs:
        for wd in wds:
            for steps in steps_grid:
                per_fold = []
                for fi in range(len(folds)):
                    va = np.array(fold_frames[fi], dtype=np.int64)
                    tr = np.array([i for j, f in enumerate(fold_frames) if j != fi for i in f],
                                  dtype=np.int64)
                    model = train_logistic(x, labels, tr, lr=lr, weight_decay=wd,
                                           steps=steps, seed=1000 + fi, batch=args.batch)
                    logit = predict_logit(model, x, va)
                    per_fold.append(eval_bundle(labels[va], logit))
                aucs = np.array([m["auc"] for m in per_fold])
                accs = np.array([m["acc"] for m in per_fold])
                lls = np.array([m["logloss"] for m in per_fold])
                row = {
                    "lr": lr,
                    "weight_decay": wd,
                    "steps": steps,
                    "mean_auc": float(aucs.mean()),
                    "std_auc": float(aucs.std(ddof=0)),
                    "score": float(aucs.mean() - aucs.std(ddof=0)),
                    "mean_acc": float(accs.mean()),
                    "mean_logloss": float(lls.mean()),
                    "per_fold": per_fold,
                }
                grid_rows.append(row)
    grid_rows.sort(key=lambda r: (-r["score"], -r["mean_auc"]))
    best = grid_rows[0]
    print(f"  {len(grid_rows)} 配置 / {time.time() - t_grid:.1f}s")
    print(f"  best: lr={best['lr']} wd={best['weight_decay']} steps={best['steps']} "
          f"AUC={best['mean_auc']:.4f}-{best['std_auc']:.4f}={best['score']:.4f} acc={best['mean_acc']:.4f}")

    # ---------------- 选定配置的折外预测（OOF） ----------------
    oof_logit = np.full(len(x), np.nan)
    cv_folds = []
    for fi in range(len(folds)):
        va = np.array(fold_frames[fi], dtype=np.int64)
        tr = np.array([i for j, f in enumerate(fold_frames) if j != fi for i in f], dtype=np.int64)
        model = train_logistic(x, labels, tr, lr=best["lr"], weight_decay=best["weight_decay"],
                               steps=best["steps"], seed=1000 + fi, batch=args.batch)
        logit = predict_logit(model, x, va)
        oof_logit[va] = logit
        bundle = eval_bundle(labels[va], logit)
        bundle["fold"] = fi
        bundle["val_groups"] = len(folds[fi])
        bundle["group_level"] = group_eval(labels[va], groups[va], logit)
        cv_folds.append(bundle)
    oof_mask = ~np.isnan(oof_logit)
    oof_labels = labels[oof_mask]
    cv_oof = eval_bundle(oof_labels, oof_logit[oof_mask])
    cv_oof["group_level"] = group_eval(oof_labels, groups[oof_mask], oof_logit[oof_mask])
    print(f"  OOF: acc={cv_oof['acc']:.4f} auc={cv_oof['auc']:.4f} logloss={cv_oof['logloss']:.4f} "
          f"| 按局 acc={cv_oof['group_level']['group_acc']:.4f} "
          f"auc={cv_oof['group_level']['group_auc']:.4f}")

    # ---------------- 3) 校准（嵌套：其余折拟合 → 本折评估） ----------------
    print("calibration ...", flush=True)
    calib_rows = []
    for fi in range(len(folds)):
        va = np.array(fold_frames[fi], dtype=np.int64)
        fit_mask = oof_mask & ~np.isin(np.arange(len(x)), va)
        z_fit, y_fit = oof_logit[fit_mask], labels[fit_mask]
        z_ev, y_ev = oof_logit[va], labels[va]
        a, c = fit_logistic_1d(z_fit, y_fit)
        temp = fit_temperature(z_fit, y_fit)
        variants = {
            "raw": z_ev,
            "platt": a * z_ev + c,
            "temperature": z_ev / temp,
        }
        row = {"fold": fi, "platt_a": a, "platt_b": c, "temperature": temp}
        for name, z in variants.items():
            prob = _sigmoid(z)
            slope, _intercept = fit_logistic_1d(z, y_ev)
            row[name] = {
                "brier": brier(y_ev, prob),
                "ece": ece(y_ev, prob, 10),
                "logloss": logloss(y_ev, prob),
                "acc": accuracy(y_ev, prob),
                "slope_on_val": slope,
            }
        calib_rows.append(row)

    def _avg(key: str, field: str) -> float:
        return float(np.mean([r[key][field] for r in calib_rows]))

    calib_summary = {
        name: {
            "brier": _avg(name, "brier"),
            "ece": _avg(name, "ece"),
            "logloss": _avg(name, "logloss"),
            "acc": _avg(name, "acc"),
            "slope_mean": _avg(name, "slope_on_val"),
            "slope_std": float(np.std([r[name]["slope_on_val"] for r in calib_rows])),
        }
        for name in ("raw", "platt", "temperature")
    }
    platt_a, platt_b = fit_logistic_1d(oof_logit[oof_mask], oof_labels)
    temp_all = fit_temperature(oof_logit[oof_mask], oof_labels)
    print(f"  raw brier={calib_summary['raw']['brier']:.4f} ece={calib_summary['raw']['ece']:.4f} "
          f"slope={calib_summary['raw']['slope_mean']:.3f}")
    print(f"  platt brier={calib_summary['platt']['brier']:.4f} ece={calib_summary['platt']['ece']:.4f} "
          f"slope={calib_summary['platt']['slope_mean']:.3f} (a={platt_a:.3f}, b={platt_b:.3f})")
    print(f"  temp  brier={calib_summary['temperature']['brier']:.4f} "
          f"ece={calib_summary['temperature']['ece']:.4f} "
          f"slope={calib_summary['temperature']['slope_mean']:.3f} (T={temp_all:.3f})")

    # ---------------- 4) 特征稳定性（按局 bootstrap） ----------------
    print(f"bootstrap coefficients x{args.bootstrap} ...", flush=True)
    ref = train_logistic(x, labels, dev_idx, lr=best["lr"], weight_decay=best["weight_decay"],
                         steps=best["steps"], seed=0, batch=args.batch)
    boot_w = []
    rng = np.random.default_rng(7)
    for bi in range(args.bootstrap):
        pick = rng.choice(len(dev_groups), size=len(dev_groups), replace=True)
        idx = np.concatenate([gidx[dev_groups[p]] for p in pick])
        m = train_logistic(x, labels, idx, lr=best["lr"], weight_decay=best["weight_decay"],
                           steps=best["steps"], seed=100 + bi, batch=args.batch)
        boot_w.append(m["w"])
    boot_w = np.array(boot_w)
    ref_sign = np.sign(ref["w"])
    boot_sign = np.sign(boot_w)
    stability = []
    for j, name in enumerate(FEATURES):
        agree = float((boot_sign[:, j] == ref_sign[j]).mean()) if ref_sign[j] != 0 else 0.0
        stability.append({
            "feature": name,
            "weight": float(ref["w"][j]),
            "boot_mean": float(boot_w[:, j].mean()),
            "boot_std": float(boot_w[:, j].std(ddof=1)) if args.bootstrap > 1 else 0.0,
            "sign_consistency": agree,
            "constant_in_data": name in const_cols,
            "unstable": bool(agree < 0.9),
        })
    unstable = [s["feature"] for s in stability if s["unstable"] and not s["constant_in_data"]]
    print(f"  unstable(非恒定, sign_consistency<0.9): {unstable}")

    # ---------------- 5) 置换检验（按局打乱标签） ----------------
    print(f"permutation test x{args.permutations} ...", flush=True)
    dev_group_arr = np.array(dev_groups, dtype=object)
    perm_accs = []
    perm_aucs = []
    for pi in range(args.permutations):
        rng_p = np.random.default_rng(90000 + pi)
        perm_labels = group_labels.copy()
        shuffled = rng_p.permutation(dev_group_arr)
        for src, dst in zip(dev_group_arr, shuffled):
            perm_labels[dst] = group_labels[src]
        y_perm = labels.copy()
        for g in dev_groups:
            y_perm[gidx[g]] = perm_labels[g]
        m = train_logistic(x, y_perm, dev_idx, lr=best["lr"], weight_decay=best["weight_decay"],
                           steps=best["steps"], seed=500 + pi, batch=args.batch)
        logit = predict_logit(m, x, test_idx)
        perm_accs.append(accuracy(labels[test_idx], _sigmoid(logit)))
        perm_aucs.append(auc_score(labels[test_idx], logit))
    perm_accs_arr = np.array(perm_accs)

    # ---------------- 6) 最终模型：dev 重训，只评一次 test ----------------
    print("final fit on dev ...", flush=True)
    seeds_info = []
    seed0_logit = None
    for si in range(args.final_seeds):
        m = train_logistic(x, labels, dev_idx, lr=best["lr"], weight_decay=best["weight_decay"],
                           steps=best["steps"], seed=si, batch=args.batch)
        dev_logit = predict_logit(m, x, dev_idx)
        te_logit = predict_logit(m, x, test_idx)
        row = {
            "seed": si,
            "dev": eval_bundle(labels[dev_idx], dev_logit),
            "test": eval_bundle(labels[test_idx], te_logit),
            "model": m,
        }
        seeds_info.append(row)
        if si == 0:
            seed0_logit = te_logit
        print(f"  seed {si}: dev acc={row['dev']['acc']:.4f} auc={row['dev']['auc']:.4f} | "
              f"test acc={row['test']['acc']:.4f} auc={row['test']['auc']:.4f} "
              f"ll={row['test']['logloss']:.4f}")
    primary = seeds_info[0]

    # 按局 bootstrap 的 test 置信区间（对 seed 0 预测）
    te_labels = labels[test_idx]
    offset = {}
    pos = 0
    for g in test_groups:
        n = len(gidx[g])
        offset[g] = np.arange(pos, pos + n)
        pos += n
    rng_b = np.random.default_rng(20261024)
    b_acc, b_auc, b_delta = [], [], []
    for _ in range(args.test_boot):
        pick = rng_b.integers(0, len(test_groups), len(test_groups))
        sel = np.concatenate([offset[test_groups[p]] for p in pick])
        yb = te_labels[sel]
        zb = seed0_logit[sel]
        acc_v = accuracy(yb, _sigmoid(zb))
        b_acc.append(acc_v)
        b_auc.append(auc_score(yb, zb))
        b_delta.append(acc_v - majority_rate(yb))

    def _ci(arr):
        arr = np.asarray([v for v in arr if not np.isnan(v)], dtype=np.float64)
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    test_groups_eval = group_eval(te_labels, groups[test_idx], seed0_logit)
    g_probs = np.array([row["prob"] for row in test_groups_eval["detail"]])
    g_truth = np.array([row["label"] for row in test_groups_eval["detail"]])
    b_gacc = []
    b_gdelta = []
    for _ in range(args.test_boot):
        pick = rng_b.integers(0, len(g_truth), len(g_truth))
        acc_v = float(((g_probs[pick] > 0.5).astype(float) == g_truth[pick]).mean())
        b_gacc.append(acc_v)
        b_gdelta.append(acc_v - majority_rate(g_truth[pick]))

    test_ci = {
        "acc_ci95": _ci(b_acc),
        "auc_ci95": _ci(b_auc),
        "acc_minus_majority_ci95": _ci(b_delta),
        "group_acc_ci95": _ci(b_gacc),
        "group_acc_minus_majority_ci95": _ci(b_gdelta),
        "boot": args.test_boot,
        "note": "按基础局（14 个）有放回重采样；帧级指标按帧数加权，按局指标每局等权",
    }

    test_logit = seed0_logit
    test_prob_raw = _sigmoid(test_logit)
    cal_variants = {
        "raw": test_prob_raw,
        "platt": _sigmoid(platt_a * test_logit + platt_b),
        "temperature": _sigmoid(test_logit / temp_all),
    }
    test_calib = {}
    for name, prob in cal_variants.items():
        z = np.log(np.clip(prob, 1e-12, 1 - 1e-12) / np.clip(1 - prob, 1e-12, 1 - 1e-12))
        slope, _ = fit_logistic_1d(z, te_labels)
        test_calib[name] = {
            "brier": brier(te_labels, prob),
            "ece": ece(te_labels, prob, 10),
            "logloss": logloss(te_labels, prob),
            "acc": accuracy(te_labels, prob),
            "slope": slope,
        }
    print(f"  test calib raw ece={test_calib['raw']['ece']:.4f} platt ece={test_calib['platt']['ece']:.4f}")
    print(f"  test 按局 acc={test_groups_eval['group_acc']:.4f} "
          f"({int(test_groups_eval['group_acc'] * test_groups_eval['groups'])}/{test_groups_eval['groups']}) "
          f"ci={test_ci['group_acc_ci95']}")

    # 置换检验 p 值（用 seed 0 test acc 作观测值）
    observed_acc = primary["test"]["acc"]
    p_value = float((1 + int((perm_accs_arr >= observed_acc).sum())) / (1 + len(perm_accs_arr)))
    perm_block = {
        "n": args.permutations,
        "observed_test_acc": observed_acc,
        "null_mean": float(perm_accs_arr.mean()),
        "null_std": float(perm_accs_arr.std(ddof=1)) if len(perm_accs_arr) > 1 else 0.0,
        "null_q95": float(np.percentile(perm_accs_arr, 95)),
        "null_max": float(perm_accs_arr.max()),
        "p_value": p_value,
        "null_accs": [float(v) for v in perm_accs],
        "null_auc_mean": float(np.nanmean(perm_aucs)),
    }
    print(f"  permutation p={p_value:.4f} null_mean={perm_block['null_mean']:.4f} "
          f"q95={perm_block['null_q95']:.4f}")

    # ---------------- 落盘 ----------------
    legacy_report = json.loads((HOKDATA_ROOT / "value_linear_report.json").read_text(encoding="utf-8"))
    comparison = {
        "cv_selected_test": {
            "acc": primary["test"]["acc"],
            "auc": primary["test"]["auc"],
            "logloss": primary["test"]["logloss"],
            "brier": primary["test"]["brier"],
            "ece": primary["test"]["ece"],
            "group_acc": test_groups_eval["group_acc"],
            "group_auc": test_groups_eval["group_auc"],
        },
        "legacy_linear_test": {
            "acc": legacy_report["mean_test_acc"],
            "majority": legacy_report["mean_test_majority"],
            "val_acc": legacy_report["mean_val_acc"],
            "val_majority": legacy_report["mean_val_majority"],
            "source": "hokdata/value_linear_report.json",
            "setting": {
                "lr": 0.01,
                "weight_decay": legacy_report["weight_decay"],
                "steps": legacy_report["steps"],
                "seeds": len(legacy_report["seeds"]),
            },
        },
        "test_majority": majority_rate(te_labels),
        "test_frames": int(len(te_labels)),
        "test_groups": len(test_groups),
    }

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol": {
            "grouping": "game_id.split('#')[0]（同局不同视角永不同折）",
            "dev": "tencent_rt_base/tencent_rt 的 train+val",
            "test": "仅最终评估一次，不参与任何选择",
            "folds": args.folds,
            "fold_seed": args.fold_seed,
            "stratify_by": "基础局内多数标签",
            "selection": "折内 val AUC 的 mean - std",
            "trainer": "Adam(numpy) 复刻，batch=64，L2 进梯度，偏差校正",
            "normalization": "每折/每次重采样只用参与训练的帧的均值/方差",
        },
        "data": {
            "samples": int(len(x)),
            "features": FEATURES,
            "n_features": n_feat,
            "constant_features": const_cols,
            "usable_features": [f for f in FEATURES if f not in const_cols],
            "dev_groups": len(dev_groups),
            "dev_frames": int(len(dev_idx)),
            "test_groups": len(test_groups),
            "test_frames": int(len(test_idx)),
            "dev_pos_rate": float(labels[dev_idx].mean()),
            "test_pos_rate": float(labels[test_idx].mean()),
            "bvs": len(bvs),
            "folds": fold_desc,
            "frames_per_group": {g: len(gidx[g]) for g in dev_groups + test_groups},
        },
        "grid": {
            "n_configs": len(grid_rows),
            "configs": grid_rows,
            "selected": {k: best[k] for k in ("lr", "weight_decay", "steps", "mean_auc",
                                              "std_auc", "score", "mean_acc", "mean_logloss")},
        },
        "cv": {
            "config": {k: best[k] for k in ("lr", "weight_decay", "steps")},
            "per_fold": cv_folds,
            "mean_acc": float(np.mean([f["acc"] for f in cv_folds])),
            "mean_auc": float(np.mean([f["auc"] for f in cv_folds])),
            "mean_logloss": float(np.mean([f["logloss"] for f in cv_folds])),
            "mean_majority": float(np.mean([f["majority"] for f in cv_folds])),
            "oof": cv_oof,
        },
        "calibration": {
            "nested_mean": calib_summary,
            "per_fold": calib_rows,
            "platt_final": {"a": platt_a, "b": platt_b, "fitted_on": "dev OOF logits"},
            "temperature_final": {"T": temp_all, "fitted_on": "dev OOF logits"},
        },
        "feature_stability": {
            "bootstrap": args.bootstrap,
            "resample": "按基础局（dev 83 局）有放回",
            "threshold": "sign_consistency < 0.9 → unstable",
            "features": stability,
            "unstable": unstable,
        },
        "permutation": perm_block,
        "final_test": {
            "config": {k: best[k] for k in ("lr", "weight_decay", "steps")},
            "seeds": [
                {"seed": r["seed"], "dev": r["dev"], "test": r["test"]} for r in seeds_info
            ],
            "primary": {
                "seed": 0,
                **primary["test"],
                "dev": primary["dev"],
                "group_level": test_groups_eval,
                "logit": [float(v) for v in test_logit],
            },
            "ci95": test_ci,
            "calibration": test_calib,
            "comparison": comparison,
        },
    }

    out = Path(args.out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    model_payload = {
        "model": "linear_logistic_cv",
        "features": FEATURES,
        "mean": [float(v) for v in primary["model"]["mu"]],
        "sigma": [float(v) for v in primary["model"]["sd"]],
        "weights": [float(v) for v in primary["model"]["w"]],
        "bias": float(primary["model"]["b"]),
        "platt": {"a": platt_a, "b": platt_b, "applied_to": "logit",
                  "note": "p = sigmoid(a * logit + b)，a/b 由 dev 折外预测拟合"},
        "temperature": temp_all,
        "selected_by": "cv_auc_mean_minus_std",
        "cv": {
            "lr": best["lr"],
            "weight_decay": best["weight_decay"],
            "steps": best["steps"],
            "folds": args.folds,
            "mean_val_auc": best["mean_auc"],
            "std_val_auc": best["std_auc"],
            "mean_val_acc": best["mean_acc"],
            "oof_auc": cv_oof["auc"],
            "oof_acc": cv_oof["acc"],
        },
        "val_acc": primary["dev"]["acc"],
        "val_majority": primary["dev"]["majority"],
        "test_acc": primary["test"]["acc"],
        "test_auc": primary["test"]["auc"],
        "test_majority": comparison["test_majority"],
        "test_acc_ci95": test_ci["acc_ci95"],
        "test_group_acc": test_groups_eval["group_acc"],
        "test_group_acc_ci95": test_ci["group_acc_ci95"],
    }
    model_out = Path(args.model_out)
    model_out.write_text(json.dumps(model_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"elapsed={time.time() - t_start:.1f}s")
    print("saved:", out)
    print("model:", model_out)
    print("CV_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
