r"""KPL VOD 小地图轨迹 -> 人类 5v5 宏观示范数据。

来源：KPL 官方 VOD（人类职业对局），小地图 Hough 圆检测 + 蓝/红环分类。
输入：
  <hokdata>/external/kpl_vod/track_1100_1500.jsonl
  <hokdata>/external/kpl_vod/track_sample_meta.json
  <hokdata>/external/kpl/battles.jsonl
输出：
  <hokdata>/human_demo/kpl_5v5_macro.jsonl
  <hokdata>/human_demo/kpl_5v5_macro_summary.json

说明：这是人类职业对局的**宏观走位**示范，不含技能/攻击动作；坐标 416x424。
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HOKDATA = Path(__file__).resolve().parents[2] / "hokdata"
MAP_MAX = math.sqrt(416.0**2 + 424.0**2)


def _mean_point(points: list[list[float]]) -> list[float] | None:
    if not points:
        return None
    return [sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)]


def _spread(points: list[list[float]], center: list[float] | None) -> float | None:
    if not points or center is None:
        return None
    return sum(math.hypot(p[0] - center[0], p[1] - center[1]) for p in points) / len(points)


def _lane(y: float | None) -> str | None:
    if y is None:
        return None
    return "top" if y < 140 else ("mid" if y < 280 else "bottom")


def _half(x: float | None) -> str | None:
    if x is None:
        return None
    return "blue_half" if x < 180 else ("red_half" if x > 240 else "river")


def _delta(a: list[float] | None, b: list[float] | None) -> list[float] | None:
    if a is None or b is None:
        return None
    return [round(b[0] - a[0], 2), round(b[1] - a[1], 2)]


def _macro_action(delta: list[float] | None) -> str:
    if delta is None:
        return "unknown"
    dx, dy = delta
    dist = math.hypot(dx, dy)
    if dist < 2.0:
        return "hold"
    angle = math.degrees(math.atan2(-dy, dx))  # 0=右，逆时针
    if angle < 0:
        angle += 360
    bins = ["right", "right_up", "up", "left_up", "left", "left_down", "down", "right_down"]
    return bins[int((angle + 22.5) // 45) % 8]


def _load_battle(battle_id: str, path: Path) -> dict | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("battle_id")) == battle_id:
            return row
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default=str(HOKDATA / "external" / "kpl_vod" / "track_1100_1500.jsonl"))
    parser.add_argument("--meta", default=str(HOKDATA / "external" / "kpl_vod" / "track_sample_meta.json"))
    parser.add_argument("--battles", default=str(HOKDATA / "external" / "kpl" / "battles.jsonl"))
    parser.add_argument("--out", default=str(HOKDATA / "human_demo" / "kpl_5v5_macro.jsonl"))
    parser.add_argument("--summary", default=str(HOKDATA / "human_demo" / "kpl_5v5_macro_summary.json"))
    args = parser.parse_args()

    track_path = Path(args.track)
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    battle = _load_battle(str(meta.get("battle_id")), Path(args.battles))
    rows = [json.loads(line) for line in track_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: row["t"])

    winner_side = None
    win_camp = meta.get("win_camp") or (battle or {}).get("win_camp")
    if win_camp in (1, 2):
        winner_side = "blue" if int(win_camp) == 1 else "red"
    teams = (battle or {}).get("teams") or {}
    picks = (battle or {}).get("bp") or []
    heroes_by_camp = {"blue": [], "red": []}
    for item in picks:
        if item.get("action") != "pick":
            continue
        side = "blue" if int(item.get("camp", 0)) == 1 else "red"
        if item.get("hero_id") is not None:
            heroes_by_camp[side].append(item["hero_id"])

    out_rows = []
    prev_blue_center = None
    prev_red_center = None
    coverage = Counter()
    macro_counts = Counter()
    for row in rows:
        blue = row.get("blue") or []
        red = row.get("red") or []
        blue_center = _mean_point(blue)
        red_center = _mean_point(red)
        blue_delta = _delta(prev_blue_center, blue_center)
        red_delta = _delta(prev_red_center, red_center)
        blue_action = _macro_action(blue_delta)
        red_action = _macro_action(red_delta)
        macro_counts[blue_action] += 1
        if len(blue) >= 5:
            coverage["blue5"] += 1
        if len(red) >= 5:
            coverage["red5"] += 1
        if blue_center:
            coverage["blue_any"] += 1
        if red_center:
            coverage["red_any"] += 1

        out_rows.append(
            {
                "source": "kpl_vod",
                "battle_id": meta.get("battle_id"),
                "match_id": meta.get("match_id"),
                "vod": meta.get("vod"),
                "t": row["t"],
                "coordinate_system": "ref_416x424",
                "blue": blue,
                "red": red,
                "n_blue": len(blue),
                "n_red": len(red),
                "blue_centroid": [round(v, 2) for v in blue_center] if blue_center else None,
                "red_centroid": [round(v, 2) for v in red_center] if red_center else None,
                "blue_spread": round(_spread(blue, blue_center), 2) if _spread(blue, blue_center) is not None else None,
                "red_spread": round(_spread(red, red_center), 2) if _spread(red, red_center) is not None else None,
                "blue_lane": _lane(blue_center[1] if blue_center else None),
                "red_lane": _lane(red_center[1] if red_center else None),
                "blue_half": _half(blue_center[0] if blue_center else None),
                "red_half": _half(red_center[0] if red_center else None),
                "blue_delta": blue_delta,
                "red_delta": red_delta,
                "blue_macro": blue_action,
                "red_macro": red_action,
                "winner_side": winner_side,
                "blue_team": (teams.get("1") or {}).get("team_abbreviation"),
                "red_team": (teams.get("2") or {}).get("team_abbreviation"),
                "blue_heroes": heroes_by_camp["blue"],
                "red_heroes": heroes_by_camp["red"],
                "limitations": [
                    "positions only; no skill/attack labels",
                    "overlapping minimap portraits cause missed detections",
                    "hero identity per dot not tracked",
                ],
            }
        )
        prev_blue_center = blue_center
        prev_red_center = red_center

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in out_rows) + "\n", encoding="utf-8")
    summary_path = Path(args.summary).resolve()
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "KPL official VOD (human professional 5v5)",
        "battle_id": meta.get("battle_id"),
        "match_id": meta.get("match_id"),
        "vod": meta.get("vod"),
        "frames": len(out_rows),
        "window_s": [out_rows[0]["t"], out_rows[-1]["t"]] if out_rows else None,
        "coverage": dict(coverage),
        "coverage_pct": {k: round(100.0 * v / max(1, len(out_rows)), 2) for k, v in coverage.items()},
        "mean_n_blue": round(statistics.mean(r["n_blue"] for r in out_rows), 3) if out_rows else 0,
        "mean_n_red": round(statistics.mean(r["n_red"] for r in out_rows), 3) if out_rows else 0,
        "blue_macro_counts": dict(macro_counts),
        "winner_side": winner_side,
        "output": str(out),
        "limitations": [
            "human macro movement only; no action/skill labels",
            "1 Hz sampling from broadcast VOD",
            "overlapping portraits cause missing heroes",
            "camp 1/2 assumed blue/red (KPL API convention)",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = HOKDATA / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("human_demo", {})["kpl_vod_macro"] = {
            "file": str(out.relative_to(HOKDATA)).replace("\\", "/"),
            "summary": str(summary_path.relative_to(HOKDATA)).replace("\\", "/"),
            "frames": len(out_rows),
            "source": "KPL VOD",
            "winner_side": winner_side,
            "coverage_pct": summary["coverage_pct"],
            "limitations": summary["limitations"],
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"frames={len(out_rows)} coverage={summary['coverage_pct']} -> {out}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
