# -*- coding: utf-8 -*-
"""人类宏观走位：1s 短期预测 / 8 向方向分类（§17 的下一步，只读）。

背景
----
`train_human_macro_baseline.py` 做的是 **5s 位移回归**，结论是「岭回归 MAE 42.94px、
20px 内只有 1.03%，不足以直接做走位建议」，并写明下一步应是
「把任务改成 1s 短期预测/方向分类，先做可用信号」。

本脚本就做这件事，口径与基线脚本一致（同 26 维特征、同按 battle_id 切分），
只改两处：

  1. 预测跨度 5s → 可配置（默认 1.0s）；
  2. 评估从"像素误差"换成**方向准确率**——8 个 45° 扇区，只统计真值确实移动了
     的样本（位移 < 2px 视为静止，单列）。

对照三条基线：
  zero    : 预测不动（方向准确率无定义，只报 MAE）
  persist : 用 t-1s→t 的位移方向当预测（1s 跨度下这个基线很强）
  ridge   : 同特征线性回归后把预测向量量化到 8 向

用法::

  venv/Scripts/python.exe -X utf8 scripts/probe_human_macro_dir.py
  venv/Scripts/python.exe -X utf8 scripts/probe_human_macro_dir.py --horizon 1.0 --json runs/eval/macro_dir.json

注意：语料是 1Hz（`kpl_5v5_macro_multi.jsonl`），所以 horizon=1.0 等价于"预测下一帧"，
而 horizon=2.0 才是"隔一帧"。两种都跑一遍才看得出跨度的影响。
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOKDATA = ROOT.parent / "hokdata"
INPUT = HOKDATA / "human_demo" / "kpl_5v5_macro_multi.jsonl"
OUT_DEFAULT = ROOT / "runs" / "eval" / "macro_dir.json"
MAP_MAX = math.sqrt(416.0 ** 2 + 424.0 ** 2)
PAST_S = 5.0
STILL_PX = 2.0
LANES = ("top", "mid", "bottom")
HALVES = ("blue_half", "river", "red_half")
SECTORS = 8


def _split(key: str) -> str:
    bucket = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if bucket < 0.70 else ("val" if bucket < 0.85 else "test")


def _one_hot(value, options):
    return [1.0 if value == option else 0.0 for option in options]


def _nearest(times: list[float], t: float, tol: float = 0.6):
    if not times:
        return None
    i = bisect.bisect_left(times, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(times):
            if best is None or abs(times[j] - t) < abs(times[best] - t):
                best = j
    if best is None or abs(times[best] - t) > tol:
        return None
    return best


def _sector(dx: float, dy: float) -> int | None:
    """8 向编号（0=东，逆时针每 45° 一档）；位移过小返回 None。"""
    if math.hypot(dx, dy) < STILL_PX:
        return None
    ang = math.atan2(dy, dx)                    # 画布坐标 y 向下
    return int(round(ang / (2 * math.pi / SECTORS))) % SECTORS


def build(horizon: float, past_s: float = PAST_S):
    by_game = defaultdict(list)
    for line in INPUT.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        game = row.get("game") or row.get("battle_id") or row.get("file")
        if game is None:
            continue
        by_game[str(game)].append(row)

    x, y_reg, y_sec, past_sec, groups = [], [], [], [], []
    for game, rows in by_game.items():
        rows.sort(key=lambda r: float(r["t"]))
        times = [float(r["t"]) for r in rows]
        centroids = [(r.get("blue_centroid"), r.get("red_centroid")) for r in rows]
        t0, t1 = times[0], times[-1]
        span = max(1.0, t1 - t0)
        for idx, row in enumerate(rows):
            bc, rc = centroids[idx]
            if not bc or not rc:
                continue
            fi = _nearest(times, times[idx] + horizon)
            pi = _nearest(times, times[idx] - past_s)
            if fi is None:
                continue
            fbc, frc = centroids[fi]
            if not fbc or not frc:
                continue
            pbc, prc = centroids[pi] if pi is not None else (None, None)
            dist = math.hypot(bc[0] - rc[0], bc[1] - rc[1]) / MAP_MAX
            feat = [bc[0] / 416.0, bc[1] / 424.0, rc[0] / 416.0, rc[1] / 424.0, dist,
                    (row.get("blue_spread") or 0.0) / MAP_MAX,
                    (row.get("red_spread") or 0.0) / MAP_MAX,
                    (row.get("n_blue") or 0) / 5.0, (row.get("n_red") or 0) / 5.0]
            feat += _one_hot(row.get("blue_lane"), LANES)
            feat += _one_hot(row.get("red_lane"), LANES)
            feat += _one_hot(row.get("blue_half"), HALVES)
            feat += _one_hot(row.get("red_half"), HALVES)
            feat.append((times[idx] - t0) / span)
            feat += [(bc[0] - pbc[0]) / 100.0 if pbc else 0.0,
                     (bc[1] - pbc[1]) / 100.0 if pbc else 0.0,
                     (rc[0] - prc[0]) / 100.0 if prc else 0.0,
                     (rc[1] - prc[1]) / 100.0 if prc else 0.0]
            # 只评蓝队方向：红队质心由 5 个红点平均，检测噪声更大（§17 已记录）
            tdx, tdy = fbc[0] - bc[0], fbc[1] - bc[1]
            pdx = (bc[0] - pbc[0]) if pbc else 0.0
            pdy = (bc[1] - pbc[1]) if pbc else 0.0
            x.append(feat)
            y_reg.append([tdx, tdy])
            y_sec.append(_sector(tdx, tdy))
            past_sec.append(_sector(pdx, pdy))
            groups.append(game)
    return (np.asarray(x, dtype=np.float64), np.asarray(y_reg, dtype=np.float64),
            np.asarray(y_sec, dtype=object), np.asarray(past_sec, dtype=object),
            np.asarray(groups))


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    xb = np.column_stack([np.ones(len(x)), x])
    reg = np.eye(xb.shape[1]) * lam
    reg[0, 0] = 0.0
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y)


def predict_ridge(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ w


def dir_acc(truth, pred) -> tuple[float, int]:
    """只在真值确实移动的样本上算；pred 为 None 的样本跳过。"""
    hit = tot = 0
    for a, b in zip(truth, pred):
        if a is None:
            continue
        tot += 1
        hit += int(b is not None and int(a) == int(b))
    return (hit / tot if tot else float("nan")), tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=float, nargs="+", default=[1.0, 2.0, 5.0])
    ap.add_argument("--past", type=float, nargs="+", default=None,
                    help="persist 基线用的历史窗口；默认与 horizon 相同（同窗口才公平）")
    ap.add_argument("--json", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    past_list = args.past or [None] * len(args.horizon)
    if len(past_list) != len(args.horizon):
        past_list = [None] * len(args.horizon)

    results = {}
    for h, p_override in zip(args.horizon, past_list):
        past_used = float(p_override) if p_override is not None else float(h)
        x, y_reg, y_sec, past_sec, groups = build(h, past_used)
        if len(x) < 200:
            print(f"[skip] horizon={h}s 只有 {len(x)} 样本")
            continue
        split = np.asarray([_split(g) for g in groups])
        tr, te = split == "train", split == "test"
        w = fit_ridge(x[tr], y_reg[tr])
        pred = predict_ridge(w, x[te])
        sec_pred = [_sector(float(p[0]), float(p[1])) for p in pred]
        acc_ridge, n_eval = dir_acc(y_sec[te], sec_pred)
        acc_persist, _ = dir_acc(y_sec[te], past_sec[te])
        acc_zero, _ = dir_acc(y_sec[te], [None] * int(te.sum()))
        mae = float(np.abs(y_reg[te] - pred).mean())
        still = float(np.mean([s is None for s in y_sec[te]]))
        # 多数类（训练集最常见的非静默扇区）
        common = Counter([s for s in y_sec[tr] if s is not None]).most_common(1)
        maj, maj_n = (common[0][0], common[0][1]) if common else (None, 0)
        maj_share = acc_maj = float("nan")
        if maj is not None and n_eval:
            maj_share = maj_n / max(1, sum(1 for s in y_sec[tr] if s is not None))
            acc_maj = float(np.mean([1 if s == maj else 0 for s in y_sec[te] if s is not None]))
        results[f"{h}s"] = {
            "samples": int(len(x)), "test_samples": int(te.sum()),
            "evaluable": n_eval, "still_rate": round(still, 4),
            "past_window_s": past_used,
            "mae_px": round(mae, 2),
            "dir_acc_ridge": round(acc_ridge, 4),
            "dir_acc_persist": round(acc_persist, 4),
            "dir_acc_majority": round(acc_maj, 4) if acc_maj == acc_maj else None,
            "majority_share": round(maj_share, 4) if maj_share == maj_share else None,
        }
        r = results[f"{h}s"]
        print(f"[{h}s / past {past_used:g}s] 样本 {r['samples']} / test {r['test_samples']}（可评 {n_eval}，"
              f"静止率 {still:.1%}）")
        print(f"       MAE {mae:.2f}px | 方向准确率 ridge {acc_ridge:.1%} / "
              f"persist {acc_persist:.1%} / 多数类 {r['dir_acc_majority']}")

    out = {"corpus": str(INPUT), "past_window_s": PAST_S, "still_px": STILL_PX,
           "sectors": SECTORS, "by_horizon": results}
    p = Path(args.json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[json] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
