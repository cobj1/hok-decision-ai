#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""核验 Dota 2 全事件转储（.jsonl.zst）的结构与事件规模。

用途：确认某批 dump 是否真的包含「逐秒十人坐标 + 技能释放 + 可见性」，
并给出可用于数据规模估算的实测事件数（README 的宣传值需要被验证，不能直接引）。

依赖：zstandard（`pip install zstandard`）。本机隔离环境：
    C:/Users/obj/.workbuddy/binaries/python/envs/aikings/Scripts/python.exe

用法：
    python scripts/inspect_dota2_dump.py ../hokdata/external/dota2_dumps/matches/*.jsonl.zst
    python scripts/inspect_dota2_dump.py <dir> --json out.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    sys.exit("需要 zstandard：pip install zstandard（或用隔离 venv 运行）")


def iter_rows(path: str):
    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            buf = b""
            while True:
                chunk = reader.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        yield json.loads(line)
            if buf.strip():
                yield json.loads(buf)


def inspect(path: str) -> dict:
    streams = Counter()
    cl_types = Counter()
    slots = set()
    chars = set()
    st_pos = 0
    st_rows = 0
    first_t = last_t = None
    meta = None
    abil = 0
    for r in iter_rows(path):
        e = r.get("e")
        streams[e] += 1
        t = r.get("t")
        if isinstance(t, (int, float)):
            first_t = t if first_t is None else min(first_t, t)
            last_t = t if last_t is None else max(last_t, t)
        if e == "meta":
            meta = meta or {k: r.get(k) for k in ("v", "game_start", "tick_at_horn") if k in r}
            for key in ("players", "heroes"):
                if isinstance(r.get(key), (list, dict)):
                    chars.add(key)
        elif e == "st":
            st_rows += 1
            d = r.get("d") or {}
            if "x" in d and "y" in d:
                st_pos += 1
            for k in ("player", "slot", "hero", "name"):
                if k in r:
                    slots.add(str(r[k]))
        elif e == "cl":
            pb = r.get("pb") or {}
            tp = pb.get("type")
            if tp:
                cl_types[tp] += 1
            if tp == "DOTA_COMBATLOG_ABILITY":
                abil += 1
    return {
        "file": os.path.basename(path),
        "size": os.path.getsize(path),
        "streams": dict(streams),
        "cl_types": dict(cl_types),
        "ability_casts": abil,
        "st_rows": st_rows,
        "st_with_xy": st_pos,
        "duration_s": round((last_t - first_t), 1) if first_t is not None and last_t is not None else None,
        "meta": meta,
        "unit_slots": len(slots) if slots else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="核验 Dota2 dump 的结构与事件规模")
    ap.add_argument("paths", nargs="+", help="文件或目录（目录会递归取 *.jsonl.zst）")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args()

    files: list[str] = []
    for p in args.paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.jsonl.zst"), recursive=True))
        else:
            files += glob.glob(p)
    files = sorted(set(files))
    if not files:
        sys.exit("没有匹配的 *.jsonl.zst")

    results = []
    agg = defaultdict(int)
    for f in files:
        try:
            r = inspect(f)
        except Exception as exc:  # 单文件坏不拖累整批
            print(f"[跳过] {os.path.basename(f)}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        results.append(r)
        for k, v in r["streams"].items():
            agg[f"stream:{k}"] += v
        agg["ability_casts"] += r["ability_casts"]
        agg["st_rows"] += r["st_rows"]
        agg["st_with_xy"] += r["st_with_xy"]
        print(f"{r['file']:>18}  {r['size']/1048576:6.2f}MB  "
              f"时长 {r['duration_s']:>7}s  st {r['st_rows']:>6}(xy {r['st_with_xy']:>6})  "
              f"技能 {r['ability_casts']:>5}  cl {r['streams'].get('cl', 0):>7}")

    n = len(results)
    print(f"\n合计 {n} 局  {sum(r['size'] for r in results)/1048576:.1f} MB")
    print(f"  技能释放 {agg['ability_casts']}（均 {agg['ability_casts']/max(n,1):.0f}/局）")
    print(f"  st 行 {agg['st_rows']}（均 {agg['st_rows']/max(n,1):.0f}/局），含 xy {agg['st_with_xy']}"
          f"（{100*agg['st_with_xy']/max(agg['st_rows'],1):.1f}%）")
    for k in sorted(agg):
        if k.startswith("stream:"):
            print(f"  {k[7:]:>12} {agg[k]}（均 {agg[k]/max(n,1):.0f}/局）")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"files": results, "aggregate": dict(agg)}, fh, ensure_ascii=False, indent=1)
        print(f"\n明细写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
