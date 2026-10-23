r"""构建本地 hokdata 数据根（替代原机器 E:\hokdata）。

输入：
  data/S45/training/merged/tencent_rt_frames.jsonl  （胜负/时间/scoreboard）
  data/S45/training/merged/wzry_frames.jsonl        （WZRYagnt v2：位置/动作/奖励）

输出（默认 D:/work/aiKings/hokdata）：
  dataset/dataset_final_v4.jsonl   腾讯 100 局，补 win_label，供 win_* 脚本
  dataset/*.meta.json              每局 winner_side
  dataset/meta.json                汇总
  behavior/dataset_behavior_v2.jsonl
  splits.json                      按 game/session 的 70/15/15 切分
  manifest.json

运行：
  python scripts/build_hokdata.py [--root D:/work/aiKings/hokdata]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hok_decision.config import GAME_VERSION  # noqa: E402

MERGED = PROJECT_ROOT / "data" / GAME_VERSION / "training" / "merged"


def _split_of(key: str, train: float = 0.70, val: float = 0.15) -> str:
    """确定性 hash 切分；调用方必须传基础局 key（不带 #2 后缀）。

    实际比例受样本量影响，以 manifest 中统计值为准。
    """
    bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < train:
        return "train"
    if bucket < train + val:
        return "val"
    return "test"


def _base_key(key: str) -> str:
    """把重复局后缀（#2）归一为基础局 key。"""
    return str(key).split("#", 1)[0]


def _stratified_base_splits(strata: dict[str, list[str]]) -> dict[str, str]:
    """按层（winner_side × author_camp）分配基础局，保证每层 70/15/15。

    同一基础局的所有 #2 视角必须落在同一 split；层内用 md5 排序保证可复现。
    """
    result: dict[str, str] = {}
    for key in sorted(strata):
        bases = sorted(strata[key], key=lambda base: hashlib.md5(base.encode("utf-8")).hexdigest())
        n = len(bases)
        if n >= 3:
            n_val = max(1, int(round(n * 0.15)))
            n_test = max(1, int(round(n * 0.15)))
        else:
            n_val = n_test = 0
        n_train = max(1, n - n_val - n_test)
        while n_train + n_val + n_test > n:
            if n_test > 0:
                n_test -= 1
            else:
                n_val -= 1
        for index, base in enumerate(bases):
            if index < n_train:
                result[base] = "train"
            elif index < n_train + n_val:
                result[base] = "val"
            else:
                result[base] = "test"
    return result


def _build_tencent(root: Path) -> dict:
    src = MERGED / "tencent_rt_frames.jsonl"
    dataset_dir = root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    out = dataset_dir / "dataset_final_v4.jsonl"

    games: dict[str, dict] = {}
    frames = 0
    with src.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            bv = str(frame["video_bv"])
            base = _base_key(frame.get("source_game_id") or bv)
            winner = frame["winner_side"]
            author_win = frame["author_camp"] == winner
            frame["win_label"] = {"winner_side": winner, "author_win": author_win}
            fout.write(json.dumps(frame, ensure_ascii=False) + "\n")
            frames += 1
            entry = games.setdefault(
                bv,
                {
                    "bv": bv,
                    "base_game": base,
                    "source_game_id": frame.get("source_game_id"),
                    "winner_side": winner,
                    "author_camp": frame["author_camp"],
                    "frames": 0,
                    "split": None,
                },
            )
            entry["frames"] += 1

    base_info: dict[str, dict] = {}
    for entry in games.values():
        info = base_info.setdefault(
            entry["base_game"],
            {"winner_side": entry["winner_side"], "camps": set()},
        )
        info["camps"].add(entry["author_camp"])
    strata: dict[str, list[str]] = {}
    for base, info in base_info.items():
        camp_key = "mixed" if len(info["camps"]) > 1 else next(iter(info["camps"]))
        strata.setdefault(f"{info['winner_side']}|{camp_key}", []).append(base)
    base_splits = _stratified_base_splits(strata)
    for entry in games.values():
        entry["split"] = base_splits[entry["base_game"]]

    for base, split in base_splits.items():
        if base not in base_info:
            raise SystemExit(f"split leakage: unknown base {base}")
        assigned = {entry["split"] for entry in games.values() if entry["base_game"] == base}
        if assigned != {split}:
            raise SystemExit(f"split leakage: base game {base} -> {assigned}")

    for bv, entry in games.items():
        meta = {
            "bv": bv,
            "win": {"winner_side": entry["winner_side"]},
            "author_camp": entry["author_camp"],
            "source": "tencent_rt_hok_sample100",
            "split": entry["split"],
        }
        (dataset_dir / f"{bv}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (dataset_dir / "meta.json").write_text(
        json.dumps(
            {
                "version": GAME_VERSION,
                "source": "tencent_rt_hok_sample100",
                "games": games,
                "base_games": len(base_splits),
                "frame_counts": dict(Counter(entry["split"] for entry in games.values())),
                "base_game_counts": dict(Counter(base_splits.values())),
                "strata": {key: len(value) for key, value in sorted(strata.items())},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "file": str(out),
        "games": len(games),
        "base_games": len(base_splits),
        "frames": frames,
        "splits": dict(Counter(entry["split"] for entry in games.values())),
        "base_splits": dict(Counter(base_splits.values())),
    }


def _build_behavior(root: Path) -> dict:
    src = MERGED / "wzry_frames.jsonl"
    behavior_dir = root / "behavior"
    behavior_dir.mkdir(parents=True, exist_ok=True)
    out = behavior_dir / "dataset_behavior_v2.jsonl"
    shutil.copy2(src, out)

    sessions = {}
    frames = 0
    with out.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            frame = json.loads(line)
            bv = str(frame["video_bv"])
            frames += 1
            entry = sessions.setdefault(bv, {"session": bv, "frames": 0, "split": _split_of(bv)})
            entry["frames"] += 1
    (behavior_dir / "meta.json").write_text(
        json.dumps({"version": GAME_VERSION, "source": "WZRYagnt_v2", "sessions": sessions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"file": str(out), "sessions": len(sessions), "frames": frames, "splits": dict(Counter(e["split"] for e in sessions.values()))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(PROJECT_ROOT.parent / "hokdata"))
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("transcripts/pool", "golden", "external"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    tencent = _build_tencent(root)
    behavior = _build_behavior(root)

    tencent_meta = json.loads((root / "dataset" / "meta.json").read_text(encoding="utf-8"))
    behavior_meta = json.loads((root / "behavior" / "meta.json").read_text(encoding="utf-8"))
    tencent_games = tencent_meta["games"]
    base_splits: dict[str, str] = {}
    for entry in tencent_games.values():
        base_splits[entry["base_game"]] = entry["split"]
    splits = {
        "tencent_rt": {entry["bv"]: entry["split"] for entry in tencent_games.values()},
        "tencent_rt_base": base_splits,
        "wzry": {entry["session"]: entry["split"] for entry in behavior_meta["sessions"].values()},
    }
    (root / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = root / "manifest.json"
    existing_manifest: dict = {}
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_manifest = {}

    manifest = {
        **existing_manifest,
        "hok_version": GAME_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "datasets": {"tencent_rt": tencent, "wzry_behavior": behavior},
        "split_policy": {
            "method": "stratified group split by winner_side × author_camp; 70/15/15 per stratum",
            "group_key": "tencent source_game_id 或 video_bv 去掉 # 后缀；wzry 按 session",
            "note": "同局 #2 视角强制同 split；实际比例见 datasets.*.splits / base_splits",
        },
        "capabilities": {
            "winner_label": "tencent_rt 100%",
            "game_time": "tencent_rt 100%",
            "hero_identity": "tencent_rt 100%",
            "scoreboard": "tencent_rt 100%",
            "positions": "wzry_behavior 84.1%",
            "action": "wzry_behavior 100%（selfplay join）",
            "reward": "wzry_behavior 99.99%",
            "human_action": "0（待录制）",
            "level_id": "0（待补段位）",
        },
        "usage": {
            "HOKDATA_ROOT": str(root),
            "win_scripts": "HOKDATA_ROOT=<root> python scripts/win_build_decision_features.py",
            "behavior": "读 behavior/dataset_behavior_v2.jsonl，按 splits.json 的 wzry 切分",
        },
        "missing": [
            "人类 swipe/专家动作标签",
            "段位/水平分档",
            "同一帧同时具备位置+动作+胜负",
            "S45 当前版本的全量对局（腾讯样本为 2019 旧版）",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"hokdata root: {root}")
    print(f"tencent: {tencent['games']} games / {tencent['frames']} frames -> {tencent['file']}")
    print(f"behavior: {behavior['sessions']} sessions / {behavior['frames']} frames -> {behavior['file']}")
    print(f"splits: {root / 'splits.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
