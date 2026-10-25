r"""人类宏观走位基线（KPL VOD，按局切分）。

任务：由 t 时刻状态（位置/队形/人数/车道/半区 + t-5→t 的历史位移）
预测 t→t+5 的蓝/红队质心位移（dx,dy，416x424 像素）。
不使用当前 t→t+1 位移，避免标签泄漏。

对比：
  zero   : 预测未来不移动
  persist: 用过去 5s 位移预测未来 5s
  ridge  : 线性岭回归（多输出）

输出：
  hokdata/models/human_macro_baseline.json
  hokdata/models/human_macro_baseline.npz
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HOKDATA = Path(__file__).resolve().parents[2] / "hokdata"
INPUT = HOKDATA / "human_demo" / "kpl_5v5_macro_multi.jsonl"
OUT_JSON = HOKDATA / "models" / "human_macro_baseline.json"
OUT_NPZ = HOKDATA / "models" / "human_macro_baseline.npz"
MAP_MAX = math.sqrt(416.0**2 + 424.0**2)
HORIZON_S = 5.0
PAST_S = 5.0
LANES = ("top", "mid", "bottom")
HALVES = ("blue_half", "river", "red_half")

FEATURES = [
    "blue_cx", "blue_cy", "red_cx", "red_cy", "centroid_dist",
    "blue_spread", "red_spread", "n_blue", "n_red",
    "blue_lane_top", "blue_lane_mid", "blue_lane_bottom",
    "red_lane_top", "red_lane_mid", "red_lane_bottom",
    "blue_half_blue", "blue_half_river", "blue_half_red",
    "red_half_blue", "red_half_river", "red_half_red",
    "time_norm",
    "past_blue_dx", "past_blue_dy", "past_red_dx", "past_red_dy",
]
TARGETS = ["blue_dx", "blue_dy", "red_dx", "red_dy"]


def _split(key: str) -> str:
    bucket = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if bucket < 0.70 else ("val" if bucket < 0.85 else "test")


def _one_hot(value, options) -> list[float]:
    return [1.0 if value == option else 0.0 for option in options]


def _nearest(rows: list[dict], times: list[float], t: float, tol: float = 0.6) -> dict | None:
    if not times:
        return None
    i = bisect.bisect_left(times, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(rows):
            dt = abs(times[j] - t)
            if dt <= tol and (best is None or dt < best[0]):
                best = (dt, rows[j])
    return best[1] if best else None


def _build() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    by_game: dict[str, list[dict]] = defaultdict(list)
    for line in INPUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_game[str(row.get("battle_id"))].append(row)
    x, y, groups = [], [], []
    for game, rows in by_game.items():
        rows.sort(key=lambda row: float(row["t"]))
        times = [float(row["t"]) for row in rows]
        t0, t1 = times[0], times[-1]
        span = max(1.0, t1 - t0)
        for row in rows:
            bc = row.get("blue_centroid")
            rc = row.get("red_centroid")
            if not bc or not rc:
                continue
            t = float(row["t"])
            future = _nearest(rows, times, t + HORIZON_S)
            past = _nearest(rows, times, t - PAST_S)
            if future is None:
                continue
            fbc, frc = future.get("blue_centroid"), future.get("red_centroid")
            if not fbc or not frc:
                continue
            pbc = past.get("blue_centroid") if past else None
            prc = past.get("red_centroid") if past else None
            dist = math.hypot(bc[0] - rc[0], bc[1] - rc[1]) / MAP_MAX
            feature = [
                bc[0] / 416.0, bc[1] / 424.0, rc[0] / 416.0, rc[1] / 424.0, dist,
                (row.get("blue_spread") or 0.0) / MAP_MAX,
                (row.get("red_spread") or 0.0) / MAP_MAX,
                (row.get("n_blue") or 0) / 5.0, (row.get("n_red") or 0) / 5.0,
            ]
            feature += _one_hot(row.get("blue_lane"), LANES)
            feature += _one_hot(row.get("red_lane"), LANES)
            feature += _one_hot(row.get("blue_half"), HALVES)
            feature += _one_hot(row.get("red_half"), HALVES)
            feature.append((t - t0) / span)
            feature += [
                (bc[0] - pbc[0]) / 100.0 if pbc else 0.0,
                (bc[1] - pbc[1]) / 100.0 if pbc else 0.0,
                (rc[0] - prc[0]) / 100.0 if prc else 0.0,
                (rc[1] - prc[1]) / 100.0 if prc else 0.0,
            ]
            target = [fbc[0] - bc[0], fbc[1] - bc[1], frc[0] - rc[0], frc[1] - rc[1]]
            x.append(feature)
            y.append(target)
            groups.append(game)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(groups), list(groups)


def _fit_ridge(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    xb = np.column_stack([np.ones(len(x)), x])
    reg = np.eye(xb.shape[1]) * lam
    reg[0, 0] = 0.0
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = np.abs(y - pred)
    return {
        "mae": float(err.mean()),
        "mae_per_target": {name: float(err[:, i].mean()) for i, name in enumerate(TARGETS)},
        "rmse": float(np.sqrt(((y - pred) ** 2).mean())),
        "within_20px_pct": float((np.linalg.norm(y - pred, axis=1) <= 20.0).mean() * 100.0),
        "direction_sign_acc": {
            name: float(((np.sign(pred[:, i]) == np.sign(y[:, i])) | (np.abs(y[:, i]) < 1.0)).mean())
            for i, name in enumerate(TARGETS)
        },
    }


def main() -> int:
    x, y, game_arr, groups = _build()
    split = np.array([_split(game) for game in groups])
    tr, va, te = split == "train", split == "val", split == "test"
    if not tr.any() or not va.any() or not te.any():
        raise SystemExit(f"split failed: {np.bincount(split.astype(int))}")
    mean = x[tr].mean(axis=0)
    std = x[tr].std(axis=0)
    std[std < 1e-9] = 1.0
    xz = (x - mean) / std
    weights = _fit_ridge(xz[tr], y[tr], lam=1.0)
    pred = np.column_stack([np.ones(te.sum()), xz[te]]) @ weights

    zero = np.zeros_like(y[te])
    persist_idx = [FEATURES.index("past_blue_dx"), FEATURES.index("past_blue_dy"),
                   FEATURES.index("past_red_dx"), FEATURES.index("past_red_dy")]
    persist = x[te][:, persist_idx] * 100.0

    report = {
        "input": str(INPUT),
        "samples": int(len(x)),
        "games": int(len(set(groups))),
        "train_samples": int(tr.sum()),
        "val_samples": int(va.sum()),
        "test_samples": int(te.sum()),
        "train_games": sorted(set(np.asarray(groups)[tr].tolist())),
        "val_games": sorted(set(np.asarray(groups)[va].tolist())),
        "test_games": sorted(set(np.asarray(groups)[te].tolist())),
        "horizon_s": HORIZON_S,
        "past_s": PAST_S,
        "features": FEATURES,
        "targets": TARGETS,
        "ridge_lambda": 1.0,
        "test": {
            "ridge": _metrics(y[te], pred),
            "zero": _metrics(y[te], zero),
            "persistence": _metrics(y[te], persist),
        },
        "note": "human macro movement only; no skill/attack labels; 120s windows may be short",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        OUT_NPZ,
        mean=mean, std=std, weights=weights,
        features=np.asarray(FEATURES), targets=np.asarray(TARGETS),
    )
    for name in ("ridge", "zero", "persistence"):
        m = report["test"][name]
        print(f"{name}: MAE={m['mae']:.2f}px within20={m['within_20px_pct']:.1f}% "
              f"sign={m['direction_sign_acc']}")
    print("saved:", OUT_JSON, OUT_NPZ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
