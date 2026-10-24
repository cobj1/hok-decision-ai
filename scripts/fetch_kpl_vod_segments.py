r"""下载 KPL VOD 片段并抽取人类 5v5 小地图轨迹。

依赖：yt-dlp + ffmpeg（通过代理访问 v.qq.com）。
输出：
  <out-dir>/<battle_id>.mp4
  <out-dir>/<battle_id>.track.jsonl
  <out-dir>/segments_meta.jsonl

用法：
  HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 \
  python scripts/fetch_kpl_vod_segments.py --battle-ids ID1 ID2 --start 1000 --end 1600
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
DEFAULT_FFMPEG = Path(r"D:/app/JianyingPro/10.8.0.14162")
EXTRACTOR = Path(__file__).resolve().parent / "extract_kpl_vod_track.py"


def _load_battles(path: Path) -> dict[str, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result[str(row.get("battle_id"))] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battle-ids", nargs="+", required=True)
    parser.add_argument("--start", type=float, default=1000.0)
    parser.add_argument("--end", type=float, default=1600.0)
    parser.add_argument("--out-dir", default=str(HOKDATA / "external" / "kpl_vod" / "segments"))
    parser.add_argument("--battles", default=str(BATTLES))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--extract-python", default=sys.executable,
                        help="带 cv2 的 Python；默认当前解释器")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    battles = _load_battles(Path(args.battles))
    proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    ffmpeg_dir = Path(os.environ.get("YTDLP_FFMPEG", str(DEFAULT_FFMPEG)))
    meta_rows = []

    for battle_id in args.battle_ids:
        battle = battles.get(str(battle_id))
        if not battle:
            print(f"[skip] unknown battle {battle_id}")
            continue
        videos = battle.get("videos") or []
        if not videos:
            print(f"[skip] no video {battle_id}")
            continue
        video = videos[0]
        mp4 = out_dir / f"{battle_id}.mp4"
        track = out_dir / f"{battle_id}.track.jsonl"
        if not args.skip_download and not mp4.exists():
            cmd = [
                sys.executable, "-m", "yt_dlp",
                "--no-warnings", "-f", "hd-0",
                "--ffmpeg-location", str(ffmpeg_dir),
                "--download-sections", f"*{args.start}-{args.end}",
                "-o", str(mp4), video,
            ]
            if proxy:
                cmd[3:3] = ["--proxy", proxy]
            print("download:", " ".join(cmd))
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"[fail] download {battle_id} rc={result.returncode}")
                continue
        if not mp4.exists():
            print(f"[fail] missing {mp4}")
            continue
        if not args.skip_extract and not track.exists():
            cmd = [
                args.extract_python, str(EXTRACTOR),
                "--video", str(mp4), "--start", "0", "--end", str(args.end - args.start),
                "--fps-sample", "1.0", "--out", str(track),
            ]
            print("extract:", " ".join(cmd))
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"[fail] extract {battle_id} rc={result.returncode}")
                continue
        meta_rows.append(
            {
                "battle_id": battle_id,
                "match_id": battle.get("match_id"),
                "win_camp": battle.get("win_camp"),
                "duration_s": battle.get("duration_s"),
                "teams": battle.get("teams"),
                "bp": battle.get("bp"),
                "video": video,
                "segment_start_s": args.start,
                "segment_end_s": args.end,
                "mp4": str(mp4),
                "track": str(track),
            }
        )
        print(f"[ok] {battle_id} -> {mp4.name} / {track.name}")

    meta_path = out_dir / "segments_meta.jsonl"
    meta_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in meta_rows) + "\n",
        encoding="utf-8",
    )
    print(f"done {len(meta_rows)} segments -> {meta_path}")
    return 0 if meta_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
