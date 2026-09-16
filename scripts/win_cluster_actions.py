r"""决策点细致聚类（动作原型发现，Win 运行）。

K=16，标准化后 k-means；输出每簇质心与可解释统计，供人工命名/注册动作。

运行:
    E:\py312\python.exe E:\project\hok-decision-ai\scripts\win_cluster_actions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT_NPZ = Path(r"E:\hokdata\decision_features.npz")
META_JSON = Path(r"E:\hokdata\decision_features.json")
OUT_CLUSTERS = Path(r"E:\hokdata\action_clusters.json")
OUT_ASSIGN = Path(r"E:\hokdata\action_cluster_assign.npz")
K = 16
SEED = 20260905


def kmeans(x: np.ndarray, k: int, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    centers = [x[rng.integers(0, x.shape[0])]]
    for _ in range(k - 1):
        d2 = np.min([((x - c) ** 2).sum(axis=1) for c in centers], axis=0)
        probs = d2 / (d2.sum() + 1e-12)
        centers.append(x[rng.choice(x.shape[0], p=probs)])
    centers = np.asarray(centers)
    labels = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(iters):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = d.argmin(axis=1)
        new_centers = np.stack([x[labels == j].mean(axis=0) if (labels == j).any() else centers[j] for j in range(k)])
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return labels, centers


def main() -> int:
    data = np.load(OUT_NPZ)
    x = data["features"].astype(np.float32)
    meta = json.loads(META_JSON.read_text(encoding="utf-8"))
    names = meta["feature_names"]
    idx = {name: i for i, name in enumerate(names)}

    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-8
    z = (x - mu) / sd
    labels, centers_z = kmeans(z, K)
    centers = centers_z * sd + mu

    print("cluster sizes:")
    counts = [(labels == j).sum() for j in range(K)]
    for j, cnt in sorted(enumerate(counts), key=lambda p: -p[1]):
        print(f"  cluster_{j}: {cnt}")

    def val(j, feature):
        return float(centers[j, idx[feature]])

    report = {"k": K, "feature_names": names, "clusters": []}
    for j, cnt in sorted(enumerate(counts), key=lambda p: -p[1]):
        cluster = {
            "cluster_id": j,
            "size": int(cnt),
            "center": {
                "time_norm": round(val(j, "time_norm"), 3),
                "level_norm": round(val(j, "level_norm"), 3),
                "self_xy": [round(val(j, "self_x_norm"), 3), round(val(j, "self_y_norm"), 3)],
                "enemy_xy": [round(val(j, "enemy_x_norm"), 3), round(val(j, "enemy_y_norm"), 3)],
                "self_in_enemy_half": round(val(j, "self_in_enemy_half"), 3),
                "lanes_self": [round(val(j, f"self_lane_{i}"), 3) for i in range(3)],
                "lanes_enemy": [round(val(j, f"enemy_lane_{i}"), 3) for i in range(3)],
                "aggression": [round(val(j, "self_aggression"), 3), round(val(j, "enemy_aggression"), 3)],
                "speed": [round(val(j, "self_speed_mean"), 3), round(val(j, "enemy_speed_mean"), 3)],
                "jungle_proximity": [round(val(j, "jungle_proximity_self"), 3), round(val(j, "jungle_proximity_enemy"), 3)],
                "recall": [round(val(j, "self_recall_count"), 3), round(val(j, "enemy_recall_count"), 3)],
                "kill_diff": round(val(j, "kill_diff_norm"), 3),
                "economy": round(val(j, "economy_norm"), 3),
                "tower": round(val(j, "tower_norm"), 3),
                "dragon": round(val(j, "dragon_norm"), 3),
                "hp_diff": round(val(j, "hp_diff"), 3),
                "roster_known": round(val(j, "roster_known"), 3),
            },
        }
        report["clusters"].append(cluster)

    OUT_CLUSTERS.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(OUT_ASSIGN, labels=labels, k=K)
    print(f"saved -> {OUT_CLUSTERS}")
    print("CLUSTER_ACTIONS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
