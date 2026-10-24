r"""KPL 人类宏观示范 -> 特征包（numpy）。

输入：<hokdata>/human_demo/kpl_5v5_macro.jsonl
输出：
  <hokdata>/human_demo/kpl_5v5_macro.npz
  <hokdata>/human_demo/kpl_5v5_macro_features.json

注意：当前只有 1 局职业 VOD 的 400s 窗口；只能作为人类宏观走位样例，
不足以训练生产模型。多局 VOD 可按同一脚本追加。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

HOKDATA = Path(__file__).resolve().parents[2] / "hokdata"
MAP_MAX = (416.0**2 + 424.0**2) ** 0.5
MACROS = ["unknown", "hold", "right", "right_up", "up", "left_up", "left", "left_down", "down", "right_down"]
MACRO_ID = {name: index for index, name in enumerate(MACROS)}

FEATURES = [
    "blue_cx", "blue_cy", "red_cx", "red_cy", "centroid_dist",
    "blue_spread", "red_spread", "n_blue", "n_red",
    "blue_dx", "blue_dy", "red_dx", "red_dy",
    "blue_lane_top", "blue_lane_mid", "blue_lane_bottom",
    "red_lane_top", "red_lane_mid", "red_lane_bottom",
    "blue_half_blue", "blue_half_river", "blue_half_red",
    "red_half_blue", "red_half_river", "red_half_red",
    "time_norm",
]

LANES = ("top", "mid", "bottom")
HALVES = ("blue_half", "river", "red_half")


def _one_hot(value, options) -> list[float]:
    return [1.0 if value == option else 0.0 for option in options]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(HOKDATA / "human_demo" / "kpl_5v5_macro.jsonl"))
    parser.add_argument("--out", default=str(HOKDATA / "human_demo" / "kpl_5v5_macro.npz"))
    parser.add_argument("--meta", default=str(HOKDATA / "human_demo" / "kpl_5v5_macro_features.json"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("empty human macro dataset")
    rows.sort(key=lambda row: (row.get("battle_id"), row["t"]))
    t0 = min(row["t"] for row in rows)
    t1 = max(row["t"] for row in rows)
    span = max(1.0, t1 - t0)

    x, y_blue, y_red, mask_blue, mask_red, times = [], [], [], [], [], []
    for row in rows:
        bc = row.get("blue_centroid")
        rc = row.get("red_centroid")
        n_blue = int(row.get("n_blue") or 0)
        n_red = int(row.get("n_red") or 0)
        bd = row.get("blue_delta") or [0.0, 0.0]
        rd = row.get("red_delta") or [0.0, 0.0]
        bspread = row.get("blue_spread") or 0.0
        rspread = row.get("red_spread") or 0.0
        dist = ((bc[0] - rc[0]) ** 2 + (bc[1] - rc[1]) ** 2) ** 0.5 if bc and rc else 0.0
        feature = [
            (bc[0] / 416.0) if bc else 0.0,
            (bc[1] / 424.0) if bc else 0.0,
            (rc[0] / 416.0) if rc else 0.0,
            (rc[1] / 424.0) if rc else 0.0,
            dist / MAP_MAX,
            bspread / MAP_MAX,
            rspread / MAP_MAX,
            n_blue / 5.0,
            n_red / 5.0,
            max(-1.0, min(1.0, float(bd[0]) / 50.0)),
            max(-1.0, min(1.0, float(bd[1]) / 50.0)),
            max(-1.0, min(1.0, float(rd[0]) / 50.0)),
            max(-1.0, min(1.0, float(rd[1]) / 50.0)),
        ]
        feature += _one_hot(row.get("blue_lane"), LANES)
        feature += _one_hot(row.get("red_lane"), LANES)
        feature += _one_hot(row.get("blue_half"), HALVES)
        feature += _one_hot(row.get("red_half"), HALVES)
        feature.append(float(row["t"] - t0) / span)
        x.append(feature)
        y_blue.append(MACRO_ID.get(str(row.get("blue_macro")), MACRO_ID["unknown"]))
        y_red.append(MACRO_ID.get(str(row.get("red_macro")), MACRO_ID["unknown"]))
        mask_blue.append(1.0 if bc and n_blue >= 1 else 0.0)
        mask_red.append(1.0 if rc and n_red >= 1 else 0.0)
        times.append(float(row["t"]))

    x_arr = np.asarray(x, dtype=np.float32)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        x=x_arr,
        y_blue=np.asarray(y_blue, dtype=np.int64),
        y_red=np.asarray(y_red, dtype=np.int64),
        mask_blue=np.asarray(mask_blue, dtype=np.float32),
        mask_red=np.asarray(mask_red, dtype=np.float32),
        t=np.asarray(times, dtype=np.float32),
        macros=np.asarray(MACROS),
        battle_ids=np.asarray([str(row.get("battle_id")) for row in rows]),
    )
    meta_path = Path(args.meta).resolve()
    meta = {
        "input": str(args.input),
        "output": str(out),
        "rows": int(x_arr.shape[0]),
        "features": FEATURES,
        "macros": MACROS,
        "blue_macro_counts": {MACROS[k]: v for k, v in Counter(y_blue).items()},
        "red_macro_counts": {MACROS[k]: v for k, v in Counter(y_red).items()},
        "mask_blue": int(sum(mask_blue)),
        "mask_red": int(sum(mask_red)),
        "window_s": [t0, t1],
        "split": "sample_only_1_game",
        "note": "人类宏观走位样例；不足以训练生产模型，多局 VOD 可按同一脚本追加",
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = HOKDATA / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("human_demo", {}).setdefault("kpl_vod_macro", {}).update(
            {
                "features": str(out.relative_to(HOKDATA)).replace("\\", "/"),
                "features_meta": str(meta_path.relative_to(HOKDATA)).replace("\\", "/"),
                "feature_rows": int(x_arr.shape[0]),
                "macro_labels": MACROS,
            }
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"rows={x_arr.shape[0]} features={x_arr.shape[1]} blue_mask={sum(mask_blue)} red_mask={sum(mask_red)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
