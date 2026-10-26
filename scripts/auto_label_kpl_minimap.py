r"""用公开 HoK 小地图检测模型自动标注 KPL VOD 帧（蓝/红英雄位置）。

模型：kangkang520-dev/wzry_newai_3.0 的 best_perfect.onnx
  - 输入：1920x1080 游戏画面左上 341x341 小地图区域
  - 类别：green_/blue_/red_ + 英雄ID；这里只保留队伍侧，输出 0=blue_hero,1=red_hero
  - 坐标：KPL 864x486 取左上 341x341，再映射到 341 归一化

输出：
  <out>/images/*.png
  <out>/labels/*.txt  (YOLO: cls cx cy w h)
  <out>/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2

MODEL = Path(__file__).resolve().parents[2] / "hokdata" / "external" / "minimap_model" / "best_perfect.onnx"
MAP_LIMIT = 160  # 341 裁剪内的小地图区域上限（KPL 864x486）
ROIS = [(5, 5, 158, 150), (5, 5, 110, 105), (0, 0, 341, 341)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=float, default=960)
    parser.add_argument("--end", type=float, default=1800)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(MODEL))
    out = Path(args.out)
    images = out / "images"
    labels = out / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    step = max(1, int(round(fps)))
    saved = 0
    scanned = 0
    stats = Counter()
    conf_sum = Counter()
    i = 0
    while saved < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t = args.start + i / fps
        if t > args.end:
            break
        if i % step == 0:
            scanned += 1
            best = None
            for x0, y0, x1, y1 in ROIS:
                crop = cv2.resize(frame[y0:y1, x0:x1], (341, 341))
                result = model.predict(crop, conf=args.conf, verbose=False)[0]
                boxes = []
                for box in result.boxes:
                    name = model.names.get(int(box.cls[0]), "")
                    m = re.match(r"(blue|red)_(\d+)$", name)
                    if not m:
                        continue
                    bx1, by1, bx2, by2 = [float(v) for v in box.xyxy[0].tolist()]
                    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                    side = m.group(1)
                    boxes.append((0 if side == "blue" else 1, cx, cy, bx2 - bx1, by2 - by1, float(box.conf[0])))
                score = len(boxes) + sum(b[5] for b in boxes) / 10.0
                if best is None or score > best[0]:
                    best = (score, crop, boxes)
            if best is None:
                continue
            _, crop, boxes = best
            # 去重：同侧中心距 <12px 或跨侧 <10px，保留置信度高者
            kept = []
            for b in sorted(boxes, key=lambda z: z[5], reverse=True):
                duplicate = False
                for k in kept:
                    d = ((b[1] - k[1]) ** 2 + (b[2] - k[2]) ** 2) ** 0.5
                    if d < (12 if b[0] == k[0] else 10):
                        duplicate = True
                        break
                if not duplicate:
                    kept.append(b)
            boxes = kept
            if len(boxes) < 2:
                continue
            name = f"kpl_{int(t):06d}"
            cv2.imwrite(str(images / f"{name}.png"), crop)
            lines = []
            for cls, cx, cy, w, h, conf in boxes:
                lines.append(f"{cls} {cx/341:.6f} {cy/341:.6f} {w/341:.6f} {h/341:.6f}")
                stats["blue" if cls == 0 else "red"] += 1
                conf_sum["blue" if cls == 0 else "red"] += conf
            (labels / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
            saved += 1
        i += 1
    cap.release()

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "kangkang520-dev/wzry_newai_3.0 best_perfect.onnx",
        "video": args.video,
        "window_s": [args.start, args.end],
        "scanned_frames": scanned,
        "saved_frames": saved,
        "classes": ["blue_hero", "red_hero"],
        "boxes": dict(stats),
        "mean_conf": {k: round(v / max(1, stats[k]), 3) for k, v in conf_sum.items()},
        "note": "auto-labeled with public detector; side-level labels, IDs not trusted; needs spot check",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
