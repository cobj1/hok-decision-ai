r"""把批量 KPL VOD 轨迹聚合成人类移动示范数据集（团队质心+位移+胜负）。

输入：<kpl_vod>/tracks/*.jsonl（batch_kpl_vod_yolo.py 产物）
输出：
  <hokdata>/external/human_demo/kpl_batch_bc.npz
  <hokdata>/external/human_demo/kpl_batch_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _centroid(items):
    if not items:
        return None
    return [sum(x["pos"][0] for x in items) / len(items), sum(x["pos"][1] for x in items) / len(items)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2] / "hokdata"))
    args = parser.parse_args()
    root = Path(args.root)
    tracks = root / "external" / "kpl_vod" / "tracks"
    out_dir = root / "external" / "human_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    states, actions, labels, meta = [], [], [], []
    games = 0
    frames_total = 0
    skipped = 0
    for path in sorted(tracks.glob("*.jsonl")):
        frames = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not frames:
            continue
        mean_det = sum(f.get("n_blue", 0) + f.get("n_red", 0) for f in frames) / len(frames)
        if mean_det < 2.0:
            skipped += 1
            continue
        # 胜负从 batch_progress 或文件名外的 queue 查；这里用轨迹中的 win_camp（如果有）
        win_camp = next((f.get("win_camp") for f in frames if f.get("win_camp") is not None), None)
        prev = None
        used = 0
        for f in frames:
            bc, rc = _centroid(f.get("blue") or []), _centroid(f.get("red") or [])
            if not bc or not rc:
                continue
            row = [f["t"], bc[0], bc[1], rc[0], rc[1], f.get("n_blue", 0), f.get("n_red", 0)]
            states.append(row)
            if prev:
                actions.append([bc[0] - prev[0], bc[1] - prev[1], rc[0] - prev[2], rc[1] - prev[3]])
                labels.append(1 if win_camp == 2 else 0)
                meta.append({"game": path.stem, "t": f["t"], "win_camp": win_camp})
                used += 1
            prev = [bc[0], bc[1], rc[0], rc[1]]
        games += 1
        frames_total += used

    if not actions:
        print("no tracks with valid frames")
        return 1
    x = np.asarray(actions, dtype=np.float32)
    # state 与 action 对齐（去掉每个游戏第一帧）
    npz = out_dir / "kpl_batch_bc.npz"
    np.savez_compressed(
        npz,
        actions=x,
        labels=np.asarray(labels, dtype=np.int8),
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
        action_names=np.asarray(["blue_dx", "blue_dy", "red_dx", "red_dy"]),
    )
    summary = {
        "games": games,
        "skipped_low_detection": skipped,
        "samples": len(actions),
        "winner_balance": {str(k): int(np.sum(np.asarray(labels) == k)) for k in (0, 1)},
        "action_mean": [round(float(v), 2) for v in x.mean(axis=0)],
        "note": "team-level human macro movement from KPL VOD; no skill/attack or hero IDs",
    }
    (out_dir / "kpl_batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
