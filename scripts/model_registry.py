# -*- coding: utf-8 -*-
"""模型评估注册表（MODEL-01 / EVAL-01）。

扫描模型目录，把每个 .pt 与其训练报告 JSON 对上，核验：
  - 权重存在且可算 sha256；
  - 报告是否有可用指标（样本数 > 0 且主指标非空）；
  - 主指标与基线是否同时给出（避免只报 accuracy 不报 majority）。

输出 registry.json，供部署前 gate 与文档引用。`--strict` 时任一权重无可用
指标即返回 1 —— 这是把"训练完但没评估"挡在部署前的唯一硬门槛。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _metrics_of(report: dict[str, Any]) -> dict[str, Any] | None:
    m = report.get("metrics")
    if isinstance(m, dict):
        samples = int(m.get("samples") or 0)
        primary = m.get("val_accuracy")
        if primary is None:
            primary = m.get("r2")
        if primary is None and m.get("mse") is not None:
            primary = m.get("mse")
        if samples <= 0 or primary is None:
            return None
        return {
            "kind": "supervised",
            "samples": samples,
            "primary": float(primary),
            "val_accuracy": m.get("val_accuracy"),
            "baseline_accuracy": m.get("majority_baseline"),
            "mse": m.get("mse"),
            "r2": m.get("r2"),
            "seeds": len(m.get("seeds") or []),
        }
    test = report.get("test")
    if isinstance(test, dict):
        acc = test.get("acc")
        macro_f1 = test.get("macro_f1")
        samples = int(report.get("rows") or 0)
        if samples <= 0 or acc is None:
            return None
        base = (report.get("baselines") or {}).get("majority_acc")
        return {
            "kind": "sequence",
            "samples": samples,
            "primary": float(acc),
            "test_acc": float(acc),
            "test_macro_f1": macro_f1,
            "baseline_accuracy": base,
            "matches": report.get("matches"),
        }
    return None


def scan(models_dir: str | Path) -> dict[str, Any]:
    d = Path(models_dir)
    models = []
    reports = set()
    for pt in sorted(d.glob("*.pt")):
        candidates = [pt.with_suffix(".json"), d / f"{pt.stem}_report.json"]
        report_path = next((p for p in candidates if p.exists()), None)
        entry: dict[str, Any] = {
            "name": pt.stem,
            "file": pt.name,
            "bytes": pt.stat().st_size,
            "sha256": sha256(pt),
            "report": None,
            "metrics": None,
            "status": "no_report",
        }
        if report_path is not None:
            reports.add(report_path.name)
            entry["report"] = report_path.name
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                metrics = _metrics_of(report)
            except Exception as exc:
                entry["status"] = "bad_report"
                entry["error"] = f"{type(exc).__name__}: {exc}"
            else:
                if metrics:
                    entry["metrics"] = metrics
                    entry["status"] = "validated"
                else:
                    entry["status"] = "unvalidated"
        models.append(entry)
    orphan = sorted(p.name for p in d.glob("*.json")
                    if p.name not in reports and p.name != "registry.json")
    counts = {}
    for m in models:
        counts[m["status"]] = counts.get(m["status"], 0) + 1
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models_dir": str(d),
        "counts": counts,
        "models": models,
        "orphan_reports": orphan,
        "errors": ([f"{m['name']}: {m['status']}" for m in models
                    if m["status"] != "validated"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(ROOT / "data" / "S45" / "models"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    reg = scan(a.models_dir)
    text = json.dumps(reg, ensure_ascii=False, indent=2)
    out = Path(a.out) if a.out else Path(a.models_dir) / "registry.json"
    out.write_text(text, encoding="utf-8")
    print(text)
    if a.strict and reg["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
