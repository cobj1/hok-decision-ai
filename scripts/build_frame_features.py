# -*- coding: utf-8 -*-
"""行为数据 -> 固定维特征/标签（DATA-03）。

输入：`hokdata/behavior/dataset_behavior_v2.jsonl`（每帧一条）
输出：带 session split 的 JSONL：
    {"frame_id","session","split","x":[...],"mask":[0/1...],
     "y_action","y_reason","y_label","quality_tier"}

设计约束：
  - 只用**当前确实有值**的字段；缺失写 0 且 mask=0，不假装观测到；
  - split 取 `hokdata/splits.json` 的 session 映射，不按帧随机切（防泄漏）；
  - 特征名与顺序写进 summary，训练侧只认这个顺序；
  - 零方差特征在 summary 里标出，训练前应剔除。
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FEATURES = (
    "hp_frac", "mp_frac", "ui_hp", "ui_mp", "dead",
    "phase_in_match", "visible_blue", "visible_red",
    "enemy_towers_remaining", "kills_self",
    "reward_total", "reward_delta", "action_age_s",
    "has_ally_hp", "has_enemy_hp", "quality_ok",
)


def _num(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def extract(row: dict) -> tuple[list[float], list[int]]:
    ui = row.get("ui") or {}
    self_state = row.get("self_state") or {}
    towers = row.get("towers") or {}
    reward = row.get("reward") or {}
    ally_hp = row.get("ally_hp") or []
    enemy_hp = row.get("enemy_hp") or []
    raw = {
        "hp_frac": _num(self_state.get("hp_frac")),
        "mp_frac": _num(self_state.get("mp_frac")),
        "ui_hp": _num(ui.get("hp")),
        "ui_mp": _num(ui.get("mp")),
        "dead": 1.0 if self_state.get("dead") is True else 0.0,
        "phase_in_match": 1.0 if row.get("phase") == "in_match" else 0.0,
        "visible_blue": _num(towers.get("visible_blue")),
        "visible_red": _num(towers.get("visible_red")),
        "enemy_towers_remaining": _num(towers.get("enemy_towers_remaining_est")),
        "kills_self": _num(towers.get("kills_self")),
        "reward_total": _num(reward.get("total")),
        "reward_delta": _num(reward.get("delta")),
        "action_age_s": _num((row.get("action") or {}).get("age_s")),
        "has_ally_hp": 1.0 if ally_hp else 0.0,
        "has_enemy_hp": 1.0 if enemy_hp else 0.0,
        "quality_ok": 1.0 if (row.get("frame_quality") == "ok"
                              and row.get("perception_ok", True)) else 0.0,
    }
    x, mask = [], []
    for name in FEATURES:
        value = raw[name]
        x.append(0.0 if value is None else float(value))
        mask.append(0 if value is None else 1)
    return x, mask


def load_splits(path: str | Path, namespace: str) -> dict[str, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    table = raw.get(namespace)
    if not isinstance(table, dict):
        raise ValueError(f"splits.json 缺少命名空间 {namespace}")
    return {str(k): str(v) for k, v in table.items()}


def build(data_path: str | Path, splits: dict[str, str],
          out_path: str | Path, summary_path: str | Path | None = None) -> dict[str, Any]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "rows": 0, "skipped_unsplit": 0, "bad_lines": 0,
        "splits": Counter(), "actions": Counter(), "labels": Counter(),
        "mask_hits": Counter(),
    }
    mins = {name: None for name in FEATURES}
    maxs = {name: None for name in FEATURES}
    with open(data_path, encoding="utf-8", errors="replace") as src, \
            open(out, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_lines"] += 1
                continue
            session = str(row.get("session") or "")
            split = splits.get(session)
            if split is None:
                stats["skipped_unsplit"] += 1
                continue
            x, mask = extract(row)
            action = row.get("action") or {}
            rec = {
                "frame_id": row.get("frame_id"),
                "session": session,
                "split": split,
                "x": x,
                "mask": mask,
                "y_action": str(action.get("type") or "none"),
                "y_reason": action.get("reason"),
                "y_label": row.get("label"),
                "quality_tier": row.get("quality_tier"),
            }
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["rows"] += 1
            stats["splits"][split] += 1
            stats["actions"][rec["y_action"]] += 1
            stats["labels"][str(rec["y_label"])] += 1
            for i, name in enumerate(FEATURES):
                if mask[i]:
                    stats["mask_hits"][name] += 1
                    v = x[i]
                    mins[name] = v if mins[name] is None else min(mins[name], v)
                    maxs[name] = v if maxs[name] is None else max(maxs[name], v)
    n = max(1, stats["rows"])
    zero_var = [name for name in FEATURES
                if mins[name] is not None and mins[name] == maxs[name]]
    summary = {
        "data": str(data_path),
        "out": str(out),
        "rows": stats["rows"],
        "skipped_unsplit": stats["skipped_unsplit"],
        "bad_lines": stats["bad_lines"],
        "splits": dict(stats["splits"]),
        "actions": dict(stats["actions"].most_common()),
        "labels": dict(stats["labels"].most_common(20)),
        "feature_names": list(FEATURES),
        "mask_rate": {k: round(v / n, 4) for k, v in stats["mask_hits"].items()},
        "zero_variance": zero_var,
    }
    if summary_path:
        Path(summary_path).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def register_manifest(summary: dict, manifest_path: str | Path) -> bool:
    """把产物登记进数据根清单；清单不存在则跳过。"""
    p = Path(manifest_path)
    if not p.exists():
        return False
    try:
        manifest = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(manifest, dict):
        return False
    arts = manifest.setdefault("artifacts", {})
    arts["frame_features"] = "behavior/features_v1.jsonl"
    arts["frame_features_summary"] = "behavior/features_v1_summary.json"
    manifest["frame_feature_build"] = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rows": summary["rows"],
        "splits": summary["splits"],
        "feature_names": summary["feature_names"],
        "zero_variance": summary["zero_variance"],
    }
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "hokdata" / "behavior"
                                          / "dataset_behavior_v2.jsonl"))
    ap.add_argument("--splits", default=str(ROOT.parent / "hokdata" / "splits.json"))
    ap.add_argument("--namespace", default="wzry")
    ap.add_argument("--out", default=str(ROOT.parent / "hokdata" / "behavior"
                                         / "features_v1.jsonl"))
    ap.add_argument("--summary", default=str(ROOT.parent / "hokdata" / "behavior"
                                             / "features_v1_summary.json"))
    ap.add_argument("--manifest", default=str(ROOT.parent / "hokdata" / "manifest.json"))
    a = ap.parse_args()
    summary = build(a.data, load_splits(a.splits, a.namespace), a.out, a.summary)
    register_manifest(summary, a.manifest)
    print(json.dumps({k: summary[k] for k in
                      ("rows", "skipped_unsplit", "splits", "zero_variance")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
