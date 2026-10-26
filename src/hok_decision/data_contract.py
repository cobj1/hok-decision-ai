# -*- coding: utf-8 -*-
"""数据集契约审计（DATA-03）。

只做静态检查，不训练、不改数据。要回答的问题：
  - split 是否覆盖全部会话，是否有同一会话跨 split（泄漏）；
  - frame_id / session 等 join key 是否唯一、是否缺失；
  - 标签字段是否真有值、分布是否退化；
  - 数值特征是否全空或零方差（这类特征进模型只会浪费容量）。

用法：
    python scripts/audit_dataset.py --data ../../hokdata/dataset/dataset_final_v4.jsonl \
        --splits ../../hokdata/splits.json --namespace tencent_rt
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SPLITS = {"train", "val", "test"}
NUMERIC_FEATURES = (
    "game_time_s", "session_elapsed_s",
    "self_state.hp", "self_state.mp", "self_state.gold", "self_state.level",
    "ui.hp", "ui.mp", "ui.gold", "ui.level",
    "ally_hp", "enemy_hp", "ally_hp_min", "enemy_hp_min",
)
LABEL_FIELDS = ("label", "action.type", "win_label", "phase", "dead")
JOIN_KEYS = ("frame_id", "source_game_id", "video_bv", "session")


def load_splits(path: str | Path, namespace: str | None = None) -> dict[str, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("splits.json 顶层必须是对象")
    table = raw
    if namespace is not None:
        table = raw.get(namespace)
        if not isinstance(table, dict):
            raise ValueError(f"splits.json 缺少命名空间: {namespace}")
    elif all(isinstance(v, dict) for v in raw.values()):
        raise ValueError("splits.json 是多命名空间结构，必须指定 namespace")
    bad = {k: v for k, v in table.items() if v not in SPLITS}
    if bad:
        raise ValueError(f"非法 split 值: {list(bad.items())[:5]}")
    return {str(k): str(v) for k, v in table.items()}


def _get(row: dict, dotted: str) -> Any:
    cur: Any = row
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _resolve_game(row: dict) -> str | None:
    for key in ("source_game_id", "video_bv", "session"):
        value = row.get(key)
        if value:
            return str(value)
    return None


def _stats(values: list[float]) -> dict[str, Any]:
    uniq = len(set(values))
    return {
        "n": len(values),
        "unique": uniq,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "zero_variance": bool(values) and uniq == 1,
    }


def audit_dataset(path: str | Path, splits: dict[str, str],
                  max_rows: int = 0) -> dict[str, Any]:
    rows = 0
    games: Counter[str] = Counter()
    split_games: dict[str, set[str]] = defaultdict(set)
    split_rows: Counter[str] = Counter()
    frame_ids: Counter[str] = Counter()
    labels: dict[str, Counter[str]] = {k: Counter() for k in LABEL_FIELDS}
    features: dict[str, list[float]] = {k: [] for k in NUMERIC_FEATURES}
    missing_join = Counter()
    missing_label = Counter()
    unsplit = set()

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows += 1
            game = _resolve_game(row)
            if game:
                games[game] += 1
                split = splits.get(game)
                if split:
                    split_games[split].add(game)
                    split_rows[split] += 1
                else:
                    unsplit.add(game)
            else:
                missing_join["source_game_id/video_bv/session"] += 1

            fid = row.get("frame_id")
            if fid:
                frame_ids[str(fid)] += 1
            else:
                missing_join["frame_id"] += 1

            for field in LABEL_FIELDS:
                value = _get(row, field)
                if value is None or value == "":
                    missing_label[field] += 1
                elif isinstance(value, (str, int, float, bool)):
                    labels[field][str(value)] += 1
                elif isinstance(value, dict) and "type" in value:
                    labels[field][str(value["type"])] += 1

            for field in NUMERIC_FEATURES:
                value = _get(row, field)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    features[field].append(float(value))

            if max_rows and rows >= max_rows:
                break

    dup_frames = {k: v for k, v in frame_ids.items() if v > 1}
    overlap = {}
    names = sorted(split_games)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = split_games[a] & split_games[b]
            if shared:
                overlap[f"{a}&{b}"] = sorted(shared)[:10]

    feature_report = {k: _stats(v) for k, v in features.items()}
    all_null = [k for k, v in feature_report.items() if v["n"] == 0]
    zero_var = [k for k, v in feature_report.items() if v["zero_variance"]]
    errors = []
    warnings = []
    if missing_join["frame_id"]:
        errors.append(f"{missing_join['frame_id']} 行缺 frame_id")
    if dup_frames:
        errors.append(f"frame_id 重复 {len(dup_frames)} 个（示例 {list(dup_frames)[:3]}）")
    if overlap:
        errors.append(f"split 间会话重叠: {overlap}")
    if unsplit:
        warnings.append(f"{len(unsplit)} 个会话不在 split 中")
    if all_null:
        warnings.append(f"特征全空: {all_null}")
    if zero_var:
        warnings.append(f"特征零方差: {zero_var}")
    empty_labels = [k for k, c in labels.items() if not c]
    if empty_labels:
        warnings.append(f"标签全空: {empty_labels}")

    return {
        "path": str(path),
        "rows": rows,
        "games": len(games),
        "split_games": {k: len(v) for k, v in sorted(split_games.items())},
        "split_rows": dict(split_rows),
        "unsplit_games": sorted(unsplit),
        "duplicate_frame_ids": len(dup_frames),
        "missing_join": dict(missing_join),
        "missing_label": dict(missing_label),
        "labels": {k: dict(c.most_common(10)) for k, c in labels.items()},
        "features": feature_report,
        "errors": errors,
        "warnings": warnings,
    }
