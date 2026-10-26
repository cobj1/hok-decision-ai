r"""批量处理 KPL VOD：按 VOD 时长选对局窗口 -> 小地图 YOLO -> 可选 BP 名单约束 ID。

输出：
  tracks/<battle_id>.jsonl
  batch_progress.jsonl / batch_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2

ROIS = [(5, 5, 158, 150), (5, 5, 110, 105), (0, 0, 341, 341)]
CLASS_SIDE = {"blue_hero": "blue", "red_hero": "red"}
WINDOW_FRACS = (0.65, 0.50, 0.80)


def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _probe_duration(vod: str) -> float | None:
    cmd = [sys.executable, "-m", "yt_dlp", "--skip-download", "--print", "duration", vod]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
        return float(out[-1]) if out and out[-1] not in ("NA", "None") else None
    except Exception:
        return None


def _download(vod: str, start: float, end: float, out: Path, ffmpeg: str) -> bool:
    cmd = [sys.executable, "-m", "yt_dlp", "-f", "hd-0", "--ffmpeg-location", ffmpeg,
           "--download-sections", f"*{int(start)}-{int(end)}", "-o", str(out), vod]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return result.returncode == 0 and out.exists() and out.stat().st_size > 0
    except subprocess.TimeoutExpired:
        return False


def _detect(model, frame):
    best = None
    for x0, y0, x1, y1 in ROIS:
        crop = cv2.resize(frame[y0:y1, x0:x1], (341, 341))
        result = model.predict(crop, conf=0.35, verbose=False)[0]
        boxes = []
        for box in result.boxes:
            side = CLASS_SIDE.get(model.names.get(int(box.cls[0]), ""))
            if side is None:
                continue
            x1b, y1b, x2b, y2b = [float(v) for v in box.xyxy[0].tolist()]
            boxes.append({"side": side,
                          "pos": [round((x1b + x2b) / 2 / 341 * 416, 1), round((y1b + y2b) / 2 / 341 * 424, 1)],
                          "box": [round(x1b, 1), round(y1b, 1), round(x2b, 1), round(y2b, 1)],
                          "conf": round(float(box.conf[0]), 3)})
        score = len(boxes) + sum(b["conf"] for b in boxes) / 10.0
        if best is None or score > best[0]:
            best = (score, crop, boxes)
    return best[1], best[2]


def _add_ids(id_model, crop, boxes, roster):
    if id_model is None or not roster:
        return boxes
    result = id_model.predict(crop, conf=0.25, verbose=False)[0]
    ids = []
    for box in result.boxes:
        m = re.match(r"(blue|red)_(\d+)$", id_model.names.get(int(box.cls[0]), ""))
        if not m:
            continue
        side, hid = m.group(1), int(m.group(2))
        if hid not in roster.get(side, set()):
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        ids.append({"side": side, "hid": hid, "pos": [(x1 + x2) / 2, (y1 + y2) / 2], "conf": float(box.conf[0])})
    for b in boxes:
        best = None
        for item in ids:
            if item["side"] != b["side"]:
                continue
            cx = (b["box"][0] + b["box"][2]) / 2
            cy = (b["box"][1] + b["box"][3]) / 2
            dist = ((item["pos"][0] - cx) ** 2 + (item["pos"][1] - cy) ** 2) ** 0.5
            if dist <= 18 and (best is None or item["conf"] > best["conf"]):
                best = item
        if best:
            b["hero_id"] = best["hid"]
            b["id_conf"] = round(best["conf"], 3)
    return boxes


def _infer(model, id_model, video: Path, start: float, roster, battle: str, win_camp):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % max(1, int(round(fps))) == 0:
            crop, boxes = _detect(model, frame)
            boxes = _add_ids(id_model, crop, boxes, roster)
            frames.append({"t": round(start + i / fps, 1),
                           "blue": [b for b in boxes if b["side"] == "blue"],
                           "red": [b for b in boxes if b["side"] == "red"],
                           "n_blue": sum(1 for b in boxes if b["side"] == "blue"),
                           "n_red": sum(1 for b in boxes if b["side"] == "red"),
                           "win_camp": win_camp, "battle_id": battle})
        i += 1
    cap.release()
    return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2] / "hokdata" / "external" / "kpl_vod"))
    parser.add_argument("--model", default="")
    parser.add_argument("--id-model", default="")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base = Path(args.root)
    queue = [json.loads(line) for line in (base / "queue.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    queue = queue[args.offset: args.offset + args.limit]
    tracks = base / "tracks"
    tracks.mkdir(parents=True, exist_ok=True)

    rosters = {}
    battle_file = base.parent / "kpl" / "battles.jsonl"
    if battle_file.exists():
        for line in battle_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rosters[r["battle_id"]] = {
                "blue": {p["hero_id"] for p in r["players"] if p["camp"] == 1},
                "red": {p["hero_id"] for p in r["players"] if p["camp"] == 2},
            }

    from ultralytics import YOLO

    root = Path(__file__).resolve().parents[2] / "hokdata"
    model_path = args.model or str(root / "models" / "kpl_minimap_yolo" / "weights" / "best.pt")
    model = YOLO(model_path)
    id_model = YOLO(args.id_model) if args.id_model else None
    ffmpeg = _ffmpeg()
    print("model:", model_path, "id_model:", args.id_model or None)

    progress = []
    for row in queue:
        battle = row["battle_id"]
        out_track = tracks / f"{battle}.jsonl"
        if out_track.exists() and not args.force:
            progress.append({"battle_id": battle, "status": "skipped_exists"})
            continue
        vod_dur = _probe_duration(row["vod"]) or row.get("duration_s") or 0
        roster = rosters.get(battle)
        best_frames = []
        best_score = -1
        chosen = None
        for frac in WINDOW_FRACS:
            start = max(0, vod_dur * frac)
            end = start + 60
            video = base / f"_tmp_{battle}.mp4"
            if not _download(row["vod"], start, end, video, ffmpeg):
                continue
            frames = _infer(model, id_model, video, start, roster, battle, row.get("win_camp"))
            video.unlink(missing_ok=True)
            if not frames:
                continue
            score = sum(f["n_blue"] + f["n_red"] for f in frames) / len(frames)
            if score > best_score:
                best_score, best_frames, chosen = score, frames, (start, end)
            if score >= 2.5:
                break
        if not best_frames:
            progress.append({"battle_id": battle, "status": "download_or_infer_failed"})
            continue
        out_track.write_text("\n".join(json.dumps(f, ensure_ascii=False) for f in best_frames) + "\n", encoding="utf-8")
        ids = sum(1 for f in best_frames for side in ("blue", "red") for b in f[side] if "hero_id" in b)
        progress.append({"battle_id": battle, "status": "ok", "frames": len(best_frames),
                         "window": [round(chosen[0], 1), round(chosen[1], 1)],
                         "mean_blue": round(sum(f["n_blue"] for f in best_frames) / len(best_frames), 2),
                         "mean_red": round(sum(f["n_red"] for f in best_frames) / len(best_frames), 2),
                         "id_boxes": ids, "win_camp": row.get("win_camp")})
        print(progress[-1], flush=True)

    with (base / "batch_progress.jsonl").open("a", encoding="utf-8") as fh:
        for row in progress:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"requested": len(queue), "ok": sum(1 for p in progress if p["status"] == "ok"),
               "status": dict((s, sum(1 for p in progress if p["status"] == s)) for s in {p["status"] for p in progress}),
               "model": model_path, "id_model": args.id_model or None}
    (base / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
