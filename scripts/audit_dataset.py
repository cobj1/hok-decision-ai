# -*- coding: utf-8 -*-
"""数据集契约审计入口（DATA-03）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hok_decision.data_contract import audit_dataset, load_splits  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", default=str(ROOT.parent / "hokdata" / "splits.json"))
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    splits = load_splits(a.splits, a.namespace)
    report = audit_dataset(a.data, splits, a.max_rows)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
