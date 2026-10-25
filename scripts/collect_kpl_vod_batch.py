r"""KPL VOD 自适应窗口批量采集（人类 5v5 宏观示范）。

流程：
  1. 每场先探 4 个 20s 窗口（默认 700/1100/1500/1900）；
  2. 选覆盖最高的窗口，下载其前后 600s；
  3. 抽取 1Hz 小地图轨迹，计算 any 覆盖率；
  4. 达标的局构建 per-game macro JSONL，并追加到 selected_games.json；
  5. 重新合并所有 selected 局 -> macro multi + npz + manifest。

需要：yt-dlp、ffmpeg、带 cv2 的 Python（--extract-python）。
代理从 HTTP_PROXY/HTTPS_PROXY 读取。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HOKDATA = Path(__file__).resolve().parents[2] / "hokdata"
BATTLES = HOKDATA / "external" / "kpl" / "battles.jsonl"
SEGMENTS = HOKDATA / "external" / "kpl_vod" / "segments"
HUMAN = HOKDATA / "human_demo"
INDEX = HUMAN / "selected_games.json"
EXTRACTOR = Path(__file__).resolve().parent / "extract_kpl_vod_track.py"
BUILDER = Path(__file__).resolve().parent / "build_kpl_vod_human_demo.py"
FEATURES = Path(__file__).resolve().parent / "build_human_macro_features.py"
DEFAULT_FFMPEG = Path(r"D:/app/JianyingPro/10.8.0.14162")

EXISTING = [
    ("48251408_30_1768371100", "kpl_5v5_macro.jsonl", 0.0),
    ("48251408_36_1768384447", "kpl_5v5_macro_48251408_36.jsonl", 1000.0),
    ("48251408_41_1768456943", "kpl_5v5_macro_48251408_41.jsonl", 1000.0),
    ("48251408_64_1768629795", "kpl_5v5_macro_48251408_64_1768629795.jsonl", 1000.0),
    ("836780560_4_1768565853", "kpl_5v5_macro_836780560_4_1768565853.jsonl", 1000.0),
    ("1272988176_3_1768716402", "kpl_5v5_macro_1272988176_3_1768716402.jsonl", 1000.0),
    ("836780560_11_1768727340", "kpl_5v5_macro_836780560_11_1768727340.jsonl", 1000.0),
    ("48251408_18_1769061734", "kpl_5v5_macro_48251408_18_1769061734.jsonl", 1000.0),
]


def _load_battles() -> dict[str, dict]:
    rows = {}
    for line in BATTLES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row.get("battle_id"))] = row
    return rows


def _run(cmd: list[str]) -> int:
    print("run:", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def _download(video: str, out: Path, start: float, end: float, proxy: str | None, ffmpeg_dir: Path) -> bool:
    if out.exists():
        return True
    cmd = [
        sys.executable, "-m", "yt_dlp", "--no-warnings", "-f", "hd-0",
        "--ffmpeg-location", str(ffmpeg_dir),
        "--download-sections", f"*{start}-{end}", "-o", str(out), video,
    ]
    if proxy:
        cmd[3:3] = ["--proxy", proxy]
    return _run(cmd) == 0 and out.exists()


def _extract(mp4: Path, track: Path, seconds: float, python: str) -> bool:
    if track.exists():
        return True
    return _run([
        python, str(EXTRACTOR), "--video", str(mp4), "--start", "0",
        "--end", str(seconds), "--fps-sample", "1.0", "--out", str(track),
    ]) == 0 and track.exists()


def _coverage(track: Path) -> dict:
    rows = [json.loads(line) for line in track.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {"frames": 0, "blue_any": 0.0, "red_any": 0.0, "blue5": 0.0, "red5": 0.0,
                "mean_blue": 0.0, "mean_red": 0.0}
    n = len(rows)
    nb = [r.get("n_blue", 0) for r in rows]
    nr = [r.get("n_red", 0) for r in rows]
    return {
        "frames": n,
        "blue_any": 100.0 * sum(1 for x in nb if x > 0) / n,
        "red_any": 100.0 * sum(1 for x in nr if x > 0) / n,
        "blue5": 100.0 * sum(1 for x in nb if x >= 5) / n,
        "red5": 100.0 * sum(1 for x in nr if x >= 5) / n,
        "mean_blue": sum(nb) / n,
        "mean_red": sum(nr) / n,
    }


def _build_meta(battle: dict, track: Path, start: float, end: float) -> Path:
    cov = _coverage(track)
    meta = {
        "battle_id": battle["battle_id"],
        "match_id": battle.get("match_id"),
        "win_camp": battle.get("win_camp"),
        "duration_s": battle.get("duration_s"),
        "vod": (battle.get("videos") or [None])[0],
        "sample_window_s": [start, end],
        "fps_sample": 1.0,
        "frames": cov["frames"],
        "segment_start_s": start,
        "detection": {
            "mean_blue": cov["mean_blue"], "mean_red": cov["mean_red"],
            "blue5_frames": round(cov["blue5"] * cov["frames"] / 100),
            "red5_frames": round(cov["red5"] * cov["frames"] / 100),
        },
        "coordinate_frame": "416x424 ref (approx; KPL minimap ROI 5,6-153,144)",
        "limitations": ["overlapping portraits cause missed detections",
                        "no action/skill labels", "broadcast layout fixed to 864x486"],
    }
    path = SEGMENTS / f"{battle['battle_id']}.meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _load_index() -> list[dict]:
    if INDEX.exists():
        return json.loads(INDEX.read_text(encoding="utf-8"))
    return [{"battle_id": bid, "file": f"human_demo/{name}", "offset": off} for bid, name, off in EXISTING]


def _combine(index: list[dict]) -> dict:
    from collections import Counter, defaultdict

    rows = []
    for entry in index:
        path = HOKDATA / entry["file"]
        if not path.exists():
            print("missing per-game", path)
            continue
        offset = float(entry.get("offset", 0.0))
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["t"] = round(float(row["t"]) + offset, 1)
            if offset:
                row["segment_start_s"] = offset
            rows.append(row)
    rows.sort(key=lambda row: (str(row.get("battle_id")), float(row["t"])))
    out = HUMAN / "kpl_5v5_macro_multi.jsonl"
    out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    per = defaultdict(list)
    for row in rows:
        per[row["battle_id"]].append(row)
    games = {}
    for bid, rs in per.items():
        n = len(rs)
        games[bid] = {
            "battle_id": bid, "match_id": rs[0].get("match_id"), "frames": n,
            "winner_side": rs[0].get("winner_side"),
            "blue_team": rs[0].get("blue_team"), "red_team": rs[0].get("red_team"),
            "t_range": [min(r["t"] for r in rs), max(r["t"] for r in rs)],
            "coverage_pct": {
                "blue_any": round(100 * sum(1 for r in rs if r.get("blue_centroid")) / n, 2),
                "red_any": round(100 * sum(1 for r in rs if r.get("red_centroid")) / n, 2),
                "blue5": round(100 * sum(1 for r in rs if r.get("n_blue", 0) >= 5) / n, 2),
                "red5": round(100 * sum(1 for r in rs if r.get("n_red", 0) >= 5) / n, 2),
            },
        }
    summary = {
        "source": "KPL official VOD (human professional 5v5)",
        "games": games, "total_frames": len(rows), "total_games": len(games),
        "selection": "adaptive 600s window chosen by 20s probes; any>=threshold",
        "note": "movement-only labels; approximate minimap positions; 1 Hz",
    }
    (HUMAN / "kpl_5v5_macro_multi_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-good", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--probe-offsets", default="700,1100,1500,1900")
    parser.add_argument("--probe-seconds", type=float, default=20.0)
    parser.add_argument("--window-seconds", type=float, default=600.0)
    parser.add_argument("--min-any", type=float, default=85.0)
    parser.add_argument("--extract-python", default=sys.executable)
    parser.add_argument("--ffmpeg-dir", default=str(DEFAULT_FFMPEG))
    args = parser.parse_args()

    SEGMENTS.mkdir(parents=True, exist_ok=True)
    HUMAN.mkdir(parents=True, exist_ok=True)
    proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    ffmpeg_dir = Path(args.ffmpeg_dir)
    offsets = [float(x) for x in args.probe_offsets.split(",") if x.strip()]
    index = _load_index()
    known = {row["battle_id"] for row in index}
    battles = _load_battles()

    candidates = []
    seen_matches = set()
    for bid, battle in battles.items():
        if bid in known or not battle.get("videos"):
            continue
        mid = str(battle.get("match_id"))
        if mid in seen_matches:
            continue
        seen_matches.add(mid)
        candidates.append(battle)
        if len(candidates) >= args.max_attempts * 2:
            break

    good = 0
    attempts = 0
    for battle in candidates:
        if good >= args.target_good or attempts >= args.max_attempts:
            break
        attempts += 1
        bid = str(battle["battle_id"])
        video = (battle.get("videos") or [None])[0]
        if not video:
            continue
        probes = []
        for offset in offsets:
            mp4 = SEGMENTS / f"probe_{bid}_{int(offset)}.mp4"
            track = SEGMENTS / f"probe_{bid}_{int(offset)}.track.jsonl"
            if not _download(video, mp4, offset, offset + args.probe_seconds, proxy, ffmpeg_dir):
                continue
            if not _extract(mp4, track, args.probe_seconds, args.extract_python):
                continue
            cov = _coverage(track)
            probes.append((min(cov["blue_any"], cov["red_any"]), offset, cov))
        if not probes:
            print(f"[skip] {bid}: no probes")
            continue
        score, best_offset, best_cov = max(probes, key=lambda item: item[0])
        print(f"[probe] {bid}: best {best_offset:.0f}s score={score:.1f} cov={best_cov}")
        if score < args.min_any:
            print(f"[skip] {bid}: best any {score:.1f} < {args.min_any}")
            continue
        start = max(0.0, best_offset - 30.0)
        end = start + args.window_seconds
        mp4 = SEGMENTS / f"{bid}.mp4"
        track = SEGMENTS / f"{bid}.track.jsonl"
        if not _download(video, mp4, start, end, proxy, ffmpeg_dir):
            print(f"[fail] {bid}: final download")
            continue
        if not _extract(mp4, track, args.window_seconds, args.extract_python):
            print(f"[fail] {bid}: final extract")
            continue
        cov = _coverage(track)
        if min(cov["blue_any"], cov["red_any"]) < args.min_any:
            print(f"[skip] {bid}: final cov {cov}")
            continue
        meta = _build_meta(battle, track, start, end)
        out = HUMAN / f"kpl_5v5_macro_{bid}.jsonl"
        summary = HUMAN / f"kpl_5v5_macro_{bid}_summary.json"
        rc = _run([
            sys.executable, str(BUILDER), "--track", str(track), "--meta", str(meta),
            "--out", str(out), "--summary", str(summary),
        ])
        if rc != 0 or not out.exists():
            print(f"[fail] {bid}: builder")
            continue
        index.append({"battle_id": bid, "file": f"human_demo/{out.name}", "offset": start})
        INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        good += 1
        print(f"[good] {bid}: start={start:.0f}s frames={cov['frames']} cov={cov}")

    summary = _combine(index)
    _run([sys.executable, str(FEATURES),
          "--input", str(HUMAN / "kpl_5v5_macro_multi.jsonl"),
          "--out", str(HUMAN / "kpl_5v5_macro_multi.npz"),
          "--meta", str(HUMAN / "kpl_5v5_macro_multi_features.json")])
    print(f"added_good={good} attempts={attempts} total_games={summary['total_games']} frames={summary['total_frames']}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
