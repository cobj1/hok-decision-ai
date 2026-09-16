r"""决策点 → 收益向量标签（self 视角，多 Δ）。

每个决策点 x 的动作收益向量：
  [击杀收益, 经济收益, 推塔收益, 龙收益, 生存收益, 推进收益]
其中每维 = 该维度在 t→t+Δ 的变化（作者视角，缺省=0+mask）。

Δ ∈ {5,10,15,20,30,60,120,180,300}；另对 Δ=20s 收益向量聚类做动作原型。

运行:
    E:\py312\python.exe E:\project\hok-decision-ai\scripts\win_build_benefit_labels.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATASET = Path(r"E:\hokdata\dataset")
OUT_LABELS = Path(r"E:\hokdata\decision_benefit_labels.npz")
OUT_META = Path(r"E:\hokdata\decision_benefit_labels.json")
OUT_CLUSTERS = Path(r"E:\hokdata\action_benefit_clusters.json")

INTERVALS = (5, 10, 15, 20, 30, 60, 120, 180, 300)
BENEFIT_NAMES = ["kill", "economy", "tower", "dragon", "survival", "advance"]
K = 16
SEED = 20260905

HUD_AFFINE = ((3.9619274696916493, 1.3453769972204605e-05), (-1.9928688568260355e-05, 4.038131527190943), (-196.00459141224727, -105.2570867363273))
BLUE_BASE = (25.0, 395.0)
RED_BASE = (397.0, 28.0)
MAP_MAX = (416.0**2 + 424.0**2) ** 0.5


def _to_ref(h: dict, cs: str) -> tuple[float, float]:
    if cs == "hud_roi":
        a, b, c = HUD_AFFINE
        return (a[0] * h["x"] + a[1] * h["y"] + c[0], b[0] * h["x"] + b[1] * h["y"] + c[1])
    return h.get("x_ref", h["x"]), h.get("y_ref", h["y"])


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _load_sb2b() -> dict[tuple[str, str], dict]:
    """加载 .sb2b.json 逐帧 scoreboard：{(bv, frame_id): state}。"""
    table: dict[tuple[str, str], dict] = {}
    for p in DATASET.glob("*.sb2b.json"):
        payload = json.loads(p.read_text(encoding="utf-8"))
        bv = payload.get("bv", p.stem.split(".sb2b")[0])
        for entry in payload.get("scoreboard", []):
            frame_id = entry.get("frame")
            if frame_id:
                table[(str(bv), str(frame_id))] = entry.get("state") or {}
    return table


SB2B = _load_sb2b()


def _raw(frame: dict) -> dict:
    camp = str(frame.get("author_camp", "blue"))
    key = (str(frame.get("video_bv", "")), str(frame.get("frame_id", "")))
    state = SB2B.get(key)
    if state:
        blue = state.get("blue") or {}
        red = state.get("red") or {}
        def num(v):
            return float(v) if isinstance(v, (int, float)) else None
        self_k, enemy_k = (num(blue.get("kills")), num(red.get("kills"))) if camp == "blue" else (num(red.get("kills")), num(blue.get("kills")))
        self_t, enemy_t = (num(blue.get("towers")), num(red.get("towers"))) if camp == "blue" else (num(red.get("towers")), num(blue.get("towers")))
        self_d, enemy_d = (num(blue.get("dragons")), num(red.get("dragons"))) if camp == "blue" else (num(red.get("dragons")), num(blue.get("dragons")))
        self_g, enemy_g = (num(blue.get("gold")), num(red.get("gold"))) if camp == "blue" else (num(red.get("gold")), num(blue.get("gold")))
        hp = [v for v in (frame.get("ally_hp") or []) if isinstance(v, (int, float))]
        return {
            "kill": ((self_k - enemy_k) / 10.0) if self_k is not None and enemy_k is not None else None,
            "economy": ((self_g - enemy_g) / 10000.0) if self_g is not None and enemy_g is not None else None,
            "tower": ((self_t - enemy_t) / 10.0) if self_t is not None and enemy_t is not None else None,
            "dragon": ((self_d - enemy_d) / 5.0) if self_d is not None and enemy_d is not None else None,
            "survival": _mean(hp) if hp else None,
        }
    bs = (frame.get("battle_state") or {}).get("base") or {}
    kb = bs.get("kills_blue")
    kr = bs.get("kills_red")
    if camp == "red":
        kb, kr = kr, kb
    def num2(v):
        return float(v) if isinstance(v, (int, float)) else None
    hp = [v for v in (frame.get("ally_hp") or []) if isinstance(v, (int, float))]
    return {
        "kill": ((num2(kb) - num2(kr)) / 10.0) if num2(kb) is not None and num2(kr) is not None else None,
        "economy": (num2(bs.get("gold")) / 10000.0) if num2(bs.get("gold")) is not None else None,
        "tower": (num2(bs.get("towers")) / 10.0) if num2(bs.get("towers")) is not None else None,
        "dragon": (num2(bs.get("dragons")) / 5.0) if num2(bs.get("dragons")) is not None else None,
        "survival": _mean(hp) if hp else None,
    }


def _advance(frame: dict) -> float | None:
    cs = frame.get("coordinate_system", "hud_roi")
    self_pts = [_to_ref(h, cs) for h in frame.get("heroes", []) if h.get("side") == "self"]
    enemy_pts = [_to_ref(h, cs) for h in frame.get("heroes", []) if h.get("side") == "enemy"]
    if not self_pts or not enemy_pts:
        return None
    sc = (_mean([p[0] for p in self_pts]), _mean([p[1] for p in self_pts]))
    ec = (_mean([p[0] for p in enemy_pts]), _mean([p[1] for p in enemy_pts]))
    self_progress = 1.0 - ((sc[0] - RED_BASE[0]) ** 2 + (sc[1] - RED_BASE[1]) ** 2) ** 0.5 / MAP_MAX
    enemy_progress = 1.0 - ((ec[0] - BLUE_BASE[0]) ** 2 + (ec[1] - BLUE_BASE[1]) ** 2) ** 0.5 / MAP_MAX
    return self_progress - enemy_progress


def _segments(frames: list[dict]) -> list[list[dict]]:
    frames = sorted(frames, key=lambda f: float(f.get("game_time_s", 0)))
    segs = []
    cur = []
    prev_t = None
    for fr in frames:
        t = float(fr.get("game_time_s", 0))
        if cur and (prev_t is None or t < prev_t or t - prev_t > 8.0):
            segs.append(cur)
            cur = []
        if fr.get("frame_quality") == "scene_break" and cur:
            segs.append(cur)
            cur = []
        cur.append(fr)
        prev_t = t
    if cur:
        segs.append(cur)
    return segs


def _benefit_between(a: dict, b: dict) -> tuple[list[float], list[float]]:
    ra, rb = _raw(a), _raw(b)
    vec, mask = [], []
    for name in BENEFIT_NAMES:
        if name == "advance":
            va, vb = _advance(a), _advance(b)
        else:
            va, vb = ra.get(name), rb.get(name)
        if va is not None and vb is not None:
            vec.append(vb - va)
            mask.append(1.0)
        else:
            vec.append(0.0)
            mask.append(0.0)
    return vec, mask


def kmeans(x: np.ndarray, k: int, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    centers = [x[rng.integers(0, x.shape[0])]]
    for _ in range(k - 1):
        d2 = np.min([((x - c) ** 2).sum(axis=1) for c in centers], axis=0)
        centers.append(x[rng.choice(x.shape[0], p=d2 / (d2.sum() + 1e-12))])
    centers = np.asarray(centers)
    labels = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(iters):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = d.argmin(axis=1)
        new = np.stack([x[labels == j].mean(axis=0) if (labels == j).any() else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            centers = new
            break
        centers = new
    return labels, centers


def main() -> int:
    frames_by_video: dict[str, list[dict]] = defaultdict(list)
    for line in (DATASET / "dataset_final_v4.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            fr = json.loads(line)
        except json.JSONDecodeError:
            continue
        frames_by_video[str(fr.get("video_bv"))].append(fr)

    points: list[dict] = []
    for frames in frames_by_video.values():
        for seg in _segments(frames):
            by_time = {round(float(f.get("game_time_s", 0)), 1): f for f in seg}
            times = sorted(by_time.keys())
            for t in times:
                a = by_time[t]
                row = {"t": t, "bv": a.get("video_bv"), "benefits": {}, "masks": {}}
                for delta in INTERVALS:
                    tt = round(t + delta, 1)
                    if tt not in by_time:
                        continue
                    vec, mask = _benefit_between(a, by_time[tt])
                    row["benefits"][str(delta)] = vec
                    row["masks"][str(delta)] = mask
                if row["benefits"]:
                    points.append(row)

    labels20 = np.asarray([p["benefits"].get("20", [0.0] * len(BENEFIT_NAMES)) for p in points], dtype=np.float32)
    mask20 = np.asarray([p["masks"].get("20", [0.0] * len(BENEFIT_NAMES)) for p in points], dtype=np.float32)
    np.savez_compressed(OUT_LABELS, points_json=json.dumps(points), labels20=labels20, mask20=mask20)
    OUT_META.write_text(
        json.dumps(
            {
                "benefit_names": BENEFIT_NAMES,
                "intervals": INTERVALS,
                "decision_points": len(points),
                "note": "self视角收益向量；survival=己方血量均值变化；advance=己方推进-敌方推进",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"decision points: {len(points)}; labels20 shape {labels20.shape}")

    # 对 Δ=20s 收益向量聚类（仅用完整观测样本；每维 1%/99% 截尾防 OCR 脏值）
    full = mask20.sum(axis=1) == len(BENEFIT_NAMES)
    x = labels20[full]
    clipped = np.clip(
        x,
        np.percentile(x, 1, axis=0),
        np.percentile(x, 99, axis=0),
    )
    z = (clipped - clipped.mean(axis=0, keepdims=True)) / (clipped.std(axis=0, keepdims=True) + 1e-8)
    if len(x) >= K:
        labels, centers = kmeans(z, K)
        centers = centers * clipped.std(axis=0) + clipped.mean(axis=0)
        report = {"k": K, "clusters": []}
        for j in range(K):
            cnt = int((labels == j).sum())
            report["clusters"].append(
                {
                    "cluster_id": j,
                    "size": cnt,
                    "benefit_center": {name: round(float(centers[j, i]), 4) for i, name in enumerate(BENEFIT_NAMES)},
                }
            )
        OUT_CLUSTERS.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved clusters -> {OUT_CLUSTERS}")
    print("BUILD_BENEFIT_LABELS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
