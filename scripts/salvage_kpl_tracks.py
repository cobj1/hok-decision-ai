r"""从已下载 KPL 600s 轨迹中滑窗选最佳 120s，救回高质量人类示范。

不下载视频；只对 segments/*.track.jsonl 做滑窗覆盖率评估，达标局构建 per-game
macro JSONL，并合并到 human_demo/kpl_5v5_macro_multi。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
COLLECT = Path(__file__).resolve().parent / "collect_kpl_vod_batch.py"


def _load_collect():
    spec = importlib.util.spec_from_file_location("collect_kpl_vod_batch", COLLECT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _best_window(rows: list[dict], width: int, step: int) -> tuple[float, int, float, float] | None:
    if len(rows) < width:
        return None
    best = None
    for start in range(0, len(rows) - width + 1, max(1, step)):
        win = rows[start:start + width]
        blue_any = 100.0 * sum(1 for row in win if row.get("n_blue", 0) > 0) / width
        red_any = 100.0 * sum(1 for row in win if row.get("n_red", 0) > 0) / width
        score = min(blue_any, red_any)
        if best is None or score > best[0]:
            best = (score, start, blue_any, red_any)
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--min-any", type=float, default=85.0)
    parser.add_argument("--max-games", type=int, default=30)
    args = parser.parse_args()

    mod = _load_collect()
    battles = mod._load_battles()
    index = mod._load_index()
    known = {row["battle_id"] for row in index}
    added = 0

    for track in sorted(mod.SEGMENTS.glob("*.track.jsonl")):
        bid = track.name[: -len(".track.jsonl")]
        if bid in known or bid.startswith("probe_") or bid.endswith(".win"):
            continue
        battle = battles.get(bid)
        if not battle:
            continue
        rows = [json.loads(line) for line in track.read_text(encoding="utf-8").splitlines() if line.strip()]
        best = _best_window(rows, args.window, args.step)
        if best is None:
            continue
        score, start, blue_any, red_any = best
        if score < args.min_any:
            print(f"[skip] {bid} best={start} score={score:.1f}")
            continue
        win = rows[start:start + args.window]
        seg_start = 1000.0
        meta_path = mod.SEGMENTS / f"{bid}.meta.json"
        if meta_path.exists():
            try:
                seg_start = float(json.loads(meta_path.read_text(encoding="utf-8")).get("segment_start_s", 1000.0))
            except Exception:
                pass
        filtered = mod.SEGMENTS / f"{bid}.win{args.window}.track.jsonl"
        filtered.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in win) + "\n", encoding="utf-8")
        meta = {
            "battle_id": bid,
            "match_id": battle.get("match_id"),
            "win_camp": battle.get("win_camp"),
            "duration_s": battle.get("duration_s"),
            "vod": (battle.get("videos") or [None])[0],
            "sample_window_s": [seg_start + win[0]["t"], seg_start + win[-1]["t"]],
            "fps_sample": 1.0,
            "frames": len(win),
            "segment_start_s": seg_start,
            "detection": {
                "mean_blue": sum(r["n_blue"] for r in win) / len(win),
                "mean_red": sum(r["n_red"] for r in win) / len(win),
                "blue5_frames": sum(1 for r in win if r["n_blue"] >= 5),
                "red5_frames": sum(1 for r in win if r["n_red"] >= 5),
            },
            "coordinate_frame": "416x424 ref (approx; KPL minimap ROI 5,6-153,144)",
            "limitations": ["overlapping portraits cause missed detections",
                            "no action/skill labels", "broadcast layout fixed to 864x486"],
        }
        meta_out = mod.SEGMENTS / f"{bid}.win{args.window}.meta.json"
        meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        out = mod.HUMAN / f"kpl_5v5_macro_{bid}.jsonl"
        summary = mod.HUMAN / f"kpl_5v5_macro_{bid}_summary.json"
        rc = mod._run([
            sys.executable, str(mod.BUILDER), "--track", str(filtered), "--meta", str(meta_out),
            "--out", str(out), "--summary", str(summary),
        ])
        if rc != 0 or not out.exists():
            print(f"[fail] {bid} builder")
            continue
        index.append({"battle_id": bid, "file": f"human_demo/{out.name}", "offset": seg_start})
        known.add(bid)
        added += 1
        print(f"[good] {bid} window_start={win[0]['t']} score={score:.1f} frames={len(win)} -> {out.name}")
        if added >= args.max_games:
            break

    mod.INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = mod._combine(index)
    mod._run([
        sys.executable, str(mod.FEATURES),
        "--input", str(mod.HUMAN / "kpl_5v5_macro_multi.jsonl"),
        "--out", str(mod.HUMAN / "kpl_5v5_macro_multi.npz"),
        "--meta", str(mod.HUMAN / "kpl_5v5_macro_multi_features.json"),
    ])

    manifest_path = mod.HOKDATA / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hd = manifest.setdefault("human_demo", {})
        hd.setdefault("kpl_vod", {}).update({
            "processed_games": summary["total_games"],
            "macro_games": summary["total_games"],
            "macro_frames": summary["total_frames"],
        })
        hd["kpl_vod_macro"] = {
            "file": "human_demo/kpl_5v5_macro_multi.jsonl",
            "summary": "human_demo/kpl_5v5_macro_multi_summary.json",
            "features": "human_demo/kpl_5v5_macro_multi.npz",
            "features_meta": "human_demo/kpl_5v5_macro_multi_features.json",
            "frames": summary["total_frames"],
            "games": summary["total_games"],
            "source": "KPL VOD",
            "coverage_pct": {bid: game["coverage_pct"] for bid, game in summary["games"].items()},
            "winner_side": {bid: game["winner_side"] for bid, game in summary["games"].items()},
            "limitations": ["human macro movement only; no skill/attack labels",
                            "1 Hz sampling from broadcast VOD",
                            "overlapping portraits cause missing heroes",
                            "camp 1/2 assumed blue/red"],
            "selection": summary["selection"],
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"added={added} total_games={summary['total_games']} frames={summary['total_frames']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
