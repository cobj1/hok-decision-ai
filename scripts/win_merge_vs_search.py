import json
import os
import re
from collections import Counter
from pathlib import Path

HOKDATA_ROOT = Path(os.environ.get("HOKDATA_ROOT", str(Path(__file__).resolve().parents[1].parent / "hokdata")))
parts = sorted(
    p
    for p in HOKDATA_ROOT.glob("asr_vs_search_part_*.json")
    if re.fullmatch(r"asr_vs_search_part_\d+\.json", p.name)
)
rows = []
meta = {"missing_frames": 0, "state_errors": 0, "points": 0}
for p in parts:
    d = json.loads(p.read_text(encoding="utf-8"))
    rows.extend(d.get("rows", []))
    meta["missing_frames"] += d.get("missing_frames", 0)
    meta["state_errors"] += d.get("state_errors", 0)
    meta["points"] += d.get("points_in_chunk", 0)
rows.sort(key=lambda r: (r["bv"], r["frame_t"]))
by_action = {}
for k, v in Counter(r["action"] for r in rows).items():
    sel = [r for r in rows if r["action"] == k]
    by_action[k] = {"n": v, "hits": sum(r["hit"] for r in sel),
                    "rate": round(sum(r["hit"] for r in sel) / max(1, len(sel)), 4)}
final = {
    "rows": rows,
    "total": len(rows),
    "hits": sum(r["hit"] for r in rows),
    "hit_rate": round(sum(r["hit"] for r in rows) / max(1, len(rows)), 4),
    "by_action": by_action,
    "meta": meta,
}
out = HOKDATA_ROOT / "asr_vs_search_final.json"
out.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
print("parts", len(parts), "rows", final["total"], "hits", final["hits"], "rate", final["hit_rate"])
print(json.dumps(final["by_action"], ensure_ascii=False, indent=2))
print("meta", meta)
print("saved", out)
