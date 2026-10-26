# -*- coding: utf-8 -*-
"""黄金集评估就绪度（EVAL-01）。

评估闭环的前置条件不是"有图"，而是"有可核对的标签"：
  - ground_truth 非空、verified=true；
  - train/val/test 至少各有样本（否则只能当 demo）；
  - 图片文件真实存在。

本脚本只出报告与门禁，不训练、不推断。当前 `hokdata/golden` 的 71 条全部
是 test + verified=false，因此 strict 会失败 —— 这是事实，不是脚本 bug。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT.parent / "hokdata" / "golden" / "candidates.jsonl"


def readiness(path: str | Path) -> dict[str, Any]:
    total = verified = labelled = missing_img = 0
    splits = Counter()
    scenarios = Counter()
    actions = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            splits[str(row.get("split") or "?")] += 1
            scenarios[str(row.get("scenario") or "?")] += 1
            if row.get("verified") is True:
                verified += 1
            gt = row.get("ground_truth")
            if isinstance(gt, dict) and gt:
                labelled += 1
                key = gt.get("action") or gt.get("type") or gt.get("intent")
                if key:
                    actions[str(key)] += 1
            image = row.get("image")
            if image and not Path(image).exists():
                missing_img += 1
    errors = []
    if total == 0:
        errors.append("黄金集为空")
    if labelled == 0:
        errors.append("ground_truth 全空")
    if verified == 0:
        errors.append("verified 全为 false")
    if missing_img:
        errors.append(f"{missing_img} 条图片不存在")
    missing_splits = [s for s in ("train", "val", "test") if not splits.get(s)]
    if missing_splits:
        errors.append(f"缺少 split: {missing_splits}")
    return {
        "path": str(path),
        "total": total,
        "verified": verified,
        "labelled": labelled,
        "missing_images": missing_img,
        "splits": dict(splits),
        "scenarios": dict(scenarios),
        "actions": dict(actions),
        "errors": errors,
        "ready": not errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    rep = readiness(a.candidates)
    text = json.dumps(rep, ensure_ascii=False, indent=2)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0 if (rep["ready"] or not a.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
