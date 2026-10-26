r"""从 KPL 公开 VOD 抽 HoK 5v5 小地图位置轨迹（近似）。

方法：固定 KPL 导播小地图 ROI -> Hough 圆检测英雄头像 -> 圆环蓝/红分类 -> 映射到 416x424。
局限：头像重叠时会漏检；不包含操作/技能标签。需用带 cv2 的 Python 运行。

用法：
  venv/Scripts/python.exe scripts/extract_kpl_vod_track.py \
    --video <vod.mp4> --start 1200 --end 1260 --out track.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROI = (5, 6, 153, 144)  # 864x486 KPL 导播布局
MAP_W, MAP_H = 416.0, 424.0


def _detect(frame: np.ndarray) -> tuple[list[list[float]], list[list[float]], list[dict]]:
    x0, y0, x1, y1 = ROI
    roi = frame[y0:y1, x0:x1]
    gray = cv2.medianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 3)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=7, param1=70, param2=12, minRadius=7, maxRadius=13
    )
    if circles is None:
        return [], [], []
    candidates = []
    for x, y, r in circles[0]:
        x, y, r = int(x), int(y), int(r)
        center = hsv[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4].reshape(-1, 3).mean(axis=0)
        if center[2] < 125 or center[1] < 45 or 115 <= center[0] <= 130:
            continue
        ring = []
        for a in np.linspace(0, 2 * np.pi, 48, endpoint=False):
            xx = int(x + (r - 1) * np.cos(a))
            yy = int(y + (r - 1) * np.sin(a))
            if 0 <= xx < roi.shape[1] and 0 <= yy < roi.shape[0]:
                ring.append(hsv[yy, xx])
        pts = np.array(ring)
        b = float(((pts[:, 0] >= 90) & (pts[:, 0] <= 140) & (pts[:, 1] > 70) & (pts[:, 2] > 50)).mean())
        rr = float((((pts[:, 0] <= 12) | (pts[:, 0] >= 168)) & (pts[:, 1] > 70) & (pts[:, 2] > 50)).mean())
        candidates.append((x, y, r, b, rr))
    blues = sorted([c for c in candidates if c[3] > c[4] and c[3] > 0.3], key=lambda c: c[3], reverse=True)[:5]
    reds = sorted([c for c in candidates if c[4] >= c[3] and c[4] > 0.3], key=lambda c: c[4], reverse=True)[:5]
    blue = [[round(x / (x1 - x0) * MAP_W, 1), round(y / (y1 - y0) * MAP_H, 1)] for x, y, r, b, rr in blues]
    red = [[round(x / (x1 - x0) * MAP_W, 1), round(y / (y1 - y0) * MAP_H, 1)] for x, y, r, b, rr in reds]
    raw = [{"box": [x, y, r], "ring_blue": round(b, 3), "ring_red": round(rr, 3)} for x, y, r, b, rr in candidates]
    return blue, red, raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--fps-sample", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    step = max(1, int(fps / args.fps_sample))
    rows = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = args.start + i / fps
        if t > args.end:
            break
        if i % step == 0:
            blue, red, raw = _detect(frame)
            rows.append(
                {
                    "t": round(t, 1),
                    "blue": blue,
                    "red": red,
                    "n_blue": len(blue),
                    "n_red": len(red),
                    "raw_candidates": len(raw),
                }
            )
        i += 1
    cap.release()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    total = max(1, len(rows))
    print(
        f"frames={len(rows)} blue5={sum(1 for r in rows if r['n_blue'] == 5)} "
        f"red5={sum(1 for r in rows if r['n_red'] == 5)} "
        f"mean_blue={sum(r['n_blue'] for r in rows) / total:.2f} "
        f"mean_red={sum(r['n_red'] for r in rows) / total:.2f} -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
