r"""构建感知黄金集候选：选帧 + 模型预标注（待人工核验）。

输入：
  <wzryagnt>/data/matches/<session>/frames/frame_*.png
  <wzryagnt>/data/matches/<session>/states.jsonl
  <hokdata>/splits.json

输出：
  <hokdata>/golden/images/            硬链接（失败则复制）
  <hokdata>/golden/prefill/*.txt      YOLO 预标注
  <hokdata>/golden/candidates.jsonl
  <hokdata>/golden/summary.json

场景：death / teamfight / lane_clear / other；录制中没有面板与结算帧，panel/end 记为缺口。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

CLASS_MAP = {
    "enemy_hero": 0,
    "ally_hero": 1,
    "enemy_minion": 2,
    "ally_minion": 3,
    "enemy_turret": 4,
    "ally_turret": 5,
    "enemy_crystal": 6,
    "ally_crystal": 7,
    "neutral_monster": 8,
    "dragon": 8,
}
CLASS_NAMES = [
    "enemy_hero", "ally_hero", "enemy_minion", "ally_minion", "enemy_turret",
    "ally_turret", "enemy_crystal", "ally_crystal", "neutral_monster", "hook_aim", "skill_effect",
]
FRAME_RE = re.compile(r"frame_(\d+)\.png$")


def _load_states(path: Path) -> dict[int, dict]:
    states = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        states[int(row.get("frame_id", -1))] = row
    return states


def _nearest_state(states: dict[int, dict], frame_id: int, radius: int = 5) -> tuple[dict | None, int | None]:
    for offset in range(radius + 1):
        for candidate in (frame_id - offset, frame_id + offset):
            state = states.get(candidate)
            if state and state.get("units"):
                return state, candidate - frame_id
    return states.get(frame_id), 0 if frame_id in states else None


def _scenario(state: dict) -> tuple[str, str]:
    behavior = state.get("behavior") or {}
    label = str(behavior.get("label") or state.get("label") or "")
    units = state.get("units") or []
    counts = Counter(u.get("cls") for u in units)
    if state.get("dead") or label == "dead":
        return "death", "dead flag / behavior"
    if counts.get("enemy_hero", 0) >= 1 and counts.get("ally_hero", 0) >= 1:
        return "teamfight", f"enemy_hero={counts.get('enemy_hero')} ally_hero={counts.get('ally_hero')}"
    if label in {"attacking_enemy", "being_attacked"}:
        return "teamfight", f"behavior={label}"
    if label == "clearing_minions" or counts.get("ally_minion", 0) + counts.get("enemy_minion", 0) >= 3:
        return "lane_clear", f"behavior={label} minions={counts.get('ally_minion', 0) + counts.get('enemy_minion', 0)}"
    return "other", f"behavior={label}"


def _prefill(state: dict) -> list[str]:
    lines = []
    for unit in state.get("units") or []:
        cls = CLASS_MAP.get(unit.get("cls"))
        box = unit.get("screen") or []
        if cls is None or len(box) != 4:
            continue
        x, y, w, h = (float(v) for v in box)
        if w <= 0 or h <= 0:
            continue
        cx = min(1.0, max(0.0, x + w / 2))
        cy = min(1.0, max(0.0, y + h / 2))
        w = min(1.0, max(0.0, w))
        h = min(1.0, max(0.0, h))
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wzryagnt", default=str(Path(__file__).resolve().parents[2] / "WZRYagnt"))
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2] / "hokdata"))
    parser.add_argument("--split", default="test", choices=["test", "val", "all"],
                        help="只从指定 split 取帧；默认 test，保证 golden 是留出集")
    parser.add_argument("--images-subdir", default="images",
                        help="相对 golden/ 的图片子目录；重建留出集时用 images_test 避免旧帧污染")
    parser.add_argument("--prefill-subdir", default="prefill",
                        help="相对 golden/ 的预标注子目录")
    args = parser.parse_args()

    wzry = Path(args.wzryagnt)
    root = Path(args.root)
    golden = root / "golden"
    images_dir = golden / args.images_subdir
    prefill_dir = golden / args.prefill_subdir
    images_dir.mkdir(parents=True, exist_ok=True)
    prefill_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    split_path = root / "splits.json"
    if split_path.exists():
        splits = json.loads(split_path.read_text(encoding="utf-8")).get("wzry", {})
    if args.split != "all" and not splits:
        raise SystemExit("splits.json 缺失：golden 必须从指定留出集取帧")
    selected_sessions = {session for session, split in splits.items() if split == args.split}
    if args.split != "all" and not selected_sessions:
        raise SystemExit(f"split={args.split} 没有可用 session")

    rows = []
    scenario_counts = Counter()
    class_counts = Counter()
    per_session = Counter()
    missing_state = 0
    duplicates_skipped = 0
    seen_hashes: set[str] = set()
    state_cache: dict[str, dict[int, dict]] = {}
    for img in sorted((wzry / "data" / "matches").glob("*/frames/frame_*.png")):
        match = FRAME_RE.search(img.name)
        if not match:
            continue
        frame_id = int(match.group(1))
        session = img.parent.parent.name
        if args.split != "all" and session not in selected_sessions:
            continue
        digest = hashlib.md5(img.read_bytes()).hexdigest()
        if digest in seen_hashes:
            duplicates_skipped += 1
            continue
        seen_hashes.add(digest)
        if session not in state_cache:
            state_cache[session] = _load_states(wzry / "data" / "matches" / session / "states.jsonl")
        state, state_offset = _nearest_state(state_cache[session], frame_id)
        if state is None or state_offset is None:
            missing_state += 1
            continue
        scenario, reason = _scenario(state)
        scenario_counts[scenario] += 1
        per_session[session] += 1

        out_name = f"{session}__{img.name}"
        out_img = images_dir / out_name
        if not out_img.exists():
            try:
                os.link(img, out_img)
            except OSError:
                shutil.copy2(img, out_img)
        lines = _prefill(state)
        for line in lines:
            class_counts[int(line.split()[0])] += 1
        (prefill_dir / f"{out_name[:-4]}.txt").write_text("\n".join(lines), encoding="utf-8")

        rows.append(
            {
                "image": str(out_img),
                "session": session,
                "frame_id": frame_id,
                "state_frame_id": state.get("frame_id"),
                "state_offset": state_offset,
                "wall_t": state.get("t"),
                "game_time_s": state.get("game_time_s"),
                "scenario": scenario,
                "scenario_reason": reason,
                "split": splits.get(session),
                "prefill_boxes": len(lines),
                "prefill_source": "yolo_units",
                "verified": False,
                "ground_truth": None,
            }
        )

    rows.sort(key=lambda r: (r["session"], r["frame_id"]))
    (golden / "candidates.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frames": len(rows),
        "sessions": len(per_session),
        "split_filter": args.split,
        "images_subdir": args.images_subdir,
        "prefill_subdir": args.prefill_subdir,
        "selected_sessions": sorted(selected_sessions) if args.split != "all" else None,
        "duplicates_skipped": duplicates_skipped,
        "missing_state_frames": missing_state,
        "scenario_counts": dict(scenario_counts),
        "scenario_gaps": [
            name for name in ("death", "lane_clear", "panel_or_loading", "end_screen")
            if not scenario_counts.get(name)
        ],
        "prefill_boxes_by_class": {CLASS_NAMES[k]: v for k, v in sorted(class_counts.items())},
        "per_session": dict(per_session),
        "class_names": CLASS_NAMES,
        "annotation_protocol": {
            "format": "YOLO: class cx cy w h (normalized)",
            "classes": CLASS_NAMES,
            "review": "人工核验时把 verified 改为 true 并写入 ground_truth（同 YOLO 行格式）",
        },
        "gaps": [
            f"{len(rows)} 帧来自 {len(per_session)} 局（split={args.split}，已按 MD5 去重 {duplicates_skipped} 张）",
            "录制中没有面板/结算/加载帧：panel/end 场景为 0，需要补录",
            "预标注来自 YOLO units，不是人工真值；verified 全为 false",
            "留出帧数量有限，未达到 200–500 帧目标，需要继续采集 test split 的关键帧",
        ],
    }
    (golden / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        manifest["golden"] = {
            "frames": len(rows),
            "sessions": len(per_session),
            "scenarios": dict(scenario_counts),
            "verified": 0,
            "split_filter": args.split,
            "images_subdir": args.images_subdir,
            "prefill_subdir": args.prefill_subdir,
            "duplicates_skipped": duplicates_skipped,
            "gaps": summary["scenario_gaps"],
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"golden frames: {len(rows)} from {len(per_session)} sessions")
    print("scenarios:", dict(scenario_counts))
    print("prefill boxes:", sum(class_counts.values()))
    print("output:", golden)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
