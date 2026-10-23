#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 Dota 2 全事件转储转成跨游戏决策先验数据集。

产出：每行 = 一个玩家在一秒上的 `(state, semantic_action, Δstate, win)`。
用途：给本项目的"人类 5v5 决策先验"用（`docs/数据专题/人类示范数据源检索.md` 路线 A）。
**不做动作坐标迁移**——Dota 的点击/朝向与王者的摇杆无关，只借"局面 → 该干什么"的语义。

设计要点
--------
1. **不用原始按键**：`.dem` 只有命令流、没有逐帧输入，所以动作是**语义类**：
   advance（推进）/ retreat（撤退）/ roam（转线）/ hold（驻守）/ cast（开技能）/
   recall（TP 回城）/ dead（阵亡）。
2. **`st` 是稀疏增量流**：某 pid 只在"有变化"时出现，各 pid 更新时刻不同。
   直接按原始时间戳当帧会得到"每帧只有 5–6 个玩家"的假象，必须
   **按 1 Hz 网格 + 每 pid 前向填充**。
3. **方向数据驱动**：不写死"天辉在左下"。用前 200 秒两队 x 均值定符号，
   得到"己方 → 敌方基地"单位向量，再用位移投影分 advance / retreat。
4. **`cl` 事件先按 npc 名收集**，读完 meta 后再换算成 pid（顺序不能反）。
5. **按局质量闸门**：`st_with_xy/st_rows > 0.5` 且 `ability_casts > 50`，
   否则剔除（实测约 1/10 对局为暂停/观战异常，见 SOURCE.md）。

依赖：zstandard。本机隔离环境：
    C:/Users/obj/.workbuddy/binaries/python/envs/aikings/Scripts/python.exe
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    sys.exit("需要 zstandard：pip install zstandard（或用隔离 venv 运行）")

MAP_HALF = 8600.0           # Dota 2 地图半幅（实测 x/y ∈ [-8609, +8649]）
COORD_OFFSET = 16384.0      # 实测坐标带 +2^14 偏移（原始 x+16384，范围 0..32768）
ACTION_WINDOW_S = 5.0        # 判动作的观察窗
DELTA_S = 30.0               # 状态变化量的时间差
UNSEEN_S = 10.0              # 敌人多久没上报位置算"看不到"
MOVE_EPS = 150.0             # 位移小于此值算 hold
DIR_EPS = 0.35               # 方向投影阈值
MIN_XY_RATIO = 0.5           # 质量闸门
MIN_ABILITIES = 50

CARRY_FIELDS = ("hp", "maxhp", "mana", "maxmana", "Level", "NetWorth", "Kills",
                "Deaths", "LastHitCount", "DenyCount", "CommandsIssued")


def iter_rows(path: str):
    with open(path, "rb") as fh:
        with zstd.ZstdDecompressor().stream_reader(fh) as reader:
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


def load_match_index(csv_path: str) -> dict:
    out = {}
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            mid = row.get("match_id")
            if not mid:
                continue
            out[mid] = {
                "radiant_win": str(row.get("radiant_win", "")).lower() in ("t", "true", "1"),
                "duration_s": row.get("duration_s"),
                "league": row.get("league"),
                "tier": row.get("tier"),
                "patch": row.get("major_patch"),
            }
    return out


def parse_match(path: str):
    """读一局，返回 (pid_hero, 每 pid 快照历史, 按名字的事件, 统计)。

    `st` 是稀疏增量流 → 保留每个 pid 的 (t, snapshot) 历史，供后续前向填充。
    `cl` 事件只能拿到 npc 名 → 先按名字收集，meta 读完后再换算 pid。
    """
    pid_hero = {}
    cur = defaultdict(dict)
    down = defaultdict(list)          # pid -> [(t, snapshot)]
    last_pos = {}
    last_seen = {}
    by_name = defaultdict(lambda: defaultdict(list))
    stats = {"st_rows": 0, "st_xy": 0, "abilities": 0, "tps": 0}

    for r in iter_rows(path):
        e = r.get("e")
        t = r.get("t")
        if e == "meta":
            for k, v in (r.get("heroes") or {}).items():
                pid_hero[int(k)] = int(v)
        elif e == "st":
            pid = r.get("pid")
            d = r.get("d") or {}
            stats["st_rows"] += 1
            if pid is None:
                continue
            pid = int(pid)
            if "x" in d and "y" in d:
                last_pos[pid] = (float(d["x"]), float(d["y"]))
                last_seen[pid] = float(t)
                stats["st_xy"] += 1
            for k in CARRY_FIELDS:
                if k in d:
                    cur[pid][k] = d[k]
            snap = dict(cur[pid])
            if pid in last_pos:
                snap["x"], snap["y"] = last_pos[pid]
            snap["_seen"] = last_seen.get(pid)
            down[pid].append((float(t), snap))
        elif e == "cl":
            pb = r.get("pb") or {}
            tp = pb.get("type") or ""
            att = r.get("att") or ""
            tgt = r.get("tgt") or ""
            inf = r.get("inf") or ""
            if tp == "DOTA_COMBATLOG_ABILITY" and att.startswith("npc_dota_hero_"):
                by_name[att]["cast"].append(float(t))
                stats["abilities"] += 1
            elif tp == "DOTA_COMBATLOG_DEATH" and tgt.startswith("npc_dota_hero_"):
                by_name[tgt]["death"].append(float(t))
            elif "tpscroll" in inf or "item_travel_boots" in inf:
                if att.startswith("npc_dota_hero_"):
                    by_name[att]["tp"].append(float(t))
                    stats["tps"] += 1
    return pid_hero, down, by_name, stats


def _last_at(series, t):
    """二分取 <= t 的最近一条快照（前向填充）。"""
    lo, hi = 0, len(series)
    while lo < hi:
        mid = (lo + hi) // 2
        if series[mid][0] <= t:
            lo = mid + 1
        else:
            hi = mid
    return series[lo - 1][1] if lo else None


def _team(pid: int) -> int:
    return 0 if pid < 5 else 1


def analyse(path: str, hero_map: dict, index: dict, delta_s: float = DELTA_S):
    mid = os.path.basename(path).split(".")[0]
    pid_hero, down, by_name, stats = parse_match(path)

    ratio = stats["st_xy"] / max(stats["st_rows"], 1)
    meta = {"match_id": mid, "st_rows": stats["st_rows"], "st_xy": stats["st_xy"],
            "xy_ratio": round(ratio, 3), "abilities": stats["abilities"], "tps": stats["tps"],
            "usable": ratio > MIN_XY_RATIO and stats["abilities"] > MIN_ABILITIES}
    if not meta["usable"]:
        return [], meta
    if not down:
        meta["usable"] = False
        meta["skip"] = "无 st 快照"
        return [], meta

    # npc 名 -> pid
    id2name = {v["id"]: v["name"] for v in hero_map.values() if "id" in v and "name" in v}
    name2pid = {}
    for pid, hid in pid_hero.items():
        nm = id2name.get(hid)
        if nm:
            name2pid[nm] = pid
    events = defaultdict(lambda: defaultdict(list))
    for nm, kinds in by_name.items():
        pid = name2pid.get(nm)
        if pid is None:
            continue
        for kind, ts_ in kinds.items():
            events[pid][kind].extend(ts_)

    t_start = min(v[0][0] for v in down.values())
    t_end = max(v[-1][0] for v in down.values())
    ts = [t_start + i for i in range(int(t_end - t_start) + 1)]
    if len(ts) < 60:
        meta["usable"] = False
        meta["skip"] = "时间轴过短"
        return [], meta

    def snap_at(t):
        out = {}
        for pid, series in down.items():
            s_ = _last_at(series, t)
            if s_ is not None:
                out[pid] = s_
        return out

    # 方向符号：前 200 秒两队 x 均值
    sx = defaultdict(list)
    for t in ts[:200]:
        for pid, s in snap_at(t).items():
            if "x" in s:
                sx[_team(pid)].append(s["x"])
    radiant_left = (sum(sx[0]) / max(len(sx[0]), 1)) < (sum(sx[1]) / max(len(sx[1]), 1))
    sign = {0: (1.0 if radiant_left else -1.0), 1: (-1.0 if radiant_left else 1.0)}

    rows = []
    for t in ts:
        snap = snap_at(t)
        if len(snap) < 8:
            continue
        nw = {p: (s.get("NetWorth") or 0.0) for p, s in snap.items()}
        kl = {p: (s.get("Kills") or 0.0) for p, s in snap.items()}
        gold_diff = (sum(v for p, v in nw.items() if _team(p) == 0)
                     - sum(v for p, v in nw.items() if _team(p) == 1))
        kill_diff = (sum(v for p, v in kl.items() if _team(p) == 0)
                     - sum(v for p, v in kl.items() if _team(p) == 1))
        snap2 = snap_at(t + delta_s)
        gd2 = (sum((s.get("NetWorth") or 0.0) for p, s in snap2.items() if _team(p) == 0)
               - sum((s.get("NetWorth") or 0.0) for p, s in snap2.items() if _team(p) == 1))
        kd2 = (sum((s.get("Kills") or 0.0) for p, s in snap2.items() if _team(p) == 0)
               - sum((s.get("Kills") or 0.0) for p, s in snap2.items() if _team(p) == 1))

        for pid, s in snap.items():
            x_raw, y_raw = s.get("x"), s.get("y")
            if x_raw is None or y_raw is None:
                continue
            x, y = x_raw - COORD_OFFSET, y_raw - COORD_OFFSET
            hp, mhp = s.get("hp"), s.get("maxhp")
            mp, mmp = s.get("mana"), s.get("maxmana")
            t2 = t + ACTION_WINDOW_S
            s_next = _last_at(down.get(pid, []), t2) if pid in down else None
            died = any(t <= et <= t2 for et in events[pid]["death"])
            tp = any(t <= et <= t2 for et in events[pid]["tp"])
            cast = any(t <= et <= t2 for et in events[pid]["cast"])
            alive = bool(hp and hp > 0)
            if died or not alive:
                act = "dead"
            elif tp:
                act = "recall"
            elif cast:
                act = "cast"
            elif not s_next or s_next.get("x") is None:
                act = "hold"
            else:
                mx = (s_next["x"] - COORD_OFFSET) - x
                my = (s_next["y"] - COORD_OFFSET) - y
                dist = (mx * mx + my * my) ** 0.5
                if dist < MOVE_EPS:
                    act = "hold"
                else:
                    proj = (mx * sign[_team(pid)] + my * sign[_team(pid)]) / (dist * 1.41421)
                    act = "advance" if proj > DIR_EPS else ("retreat" if proj < -DIR_EPS else "roam")
            unseen = 0
            for ep in range(10):
                if _team(ep) == _team(pid):
                    continue
                ls = (snap.get(ep) or {}).get("_seen")
                if ls is None or (t - ls) > UNSEEN_S:
                    unseen += 1
            s2 = snap2.get(pid) or {}
            rows.append({
                "match_id": mid,
                "t": round(t, 1),
                "pid": pid,
                "hero_id": pid_hero.get(pid),
                "team": _team(pid),
                "state": {
                    "x": round(x, 1), "y": round(y, 1),
                    "x_norm": round(x / MAP_HALF, 4), "y_norm": round(y / MAP_HALF, 4),
                    "hp_ratio": round(hp / mhp, 3) if hp is not None and mhp else None,
                    "mana_ratio": round(mp / mmp, 3) if mp is not None and mmp else None,
                    "level": s.get("Level"),
                    "networth": s.get("NetWorth"),
                    "alive": bool(hp and hp > 0),
                    "gold_diff": round(gold_diff, 1),
                    "kill_diff": round(kill_diff, 1),
                    "n_enemy_unseen": unseen,
                },
                "action": act,
                "delta": {
                    "d_gold_diff": round(gd2 - gold_diff, 1) if snap2 else None,
                    "d_kill_diff": round(kd2 - kill_diff, 1) if snap2 else None,
                    "d_hp_ratio": (round(((s2.get("hp") or 0) / (s2.get("maxhp") or 1))
                                         - ((hp or 0) / (mhp or 1)), 3) if s2 else None),
                    "died": died,
                },
            })
    info = index.get(mid) or {}
    win = info.get("radiant_win")
    for r in rows:
        r["win"] = (None if win is None else (win if r["team"] == 0 else (not win)))
    return rows, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Dota2 dump -> (state, semantic_action, delta, win)")
    ap.add_argument("--dumps", required=True, help="目录或通配，含 *.jsonl.zst")
    ap.add_argument("--matches-csv", required=True)
    ap.add_argument("--heroes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="最多处理几局（0=全部）")
    args = ap.parse_args()

    files = (sorted(glob.glob(os.path.join(args.dumps, "**", "*.jsonl.zst"), recursive=True))
             if os.path.isdir(args.dumps) else sorted(glob.glob(args.dumps)))
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit("没有匹配的 dump")

    hero_map = json.load(open(args.heroes, encoding="utf-8"))
    index = load_match_index(args.matches_csv)

    n_rows, metas, acts = 0, [], Counter()
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for f in files:
            rows, meta = analyse(f, hero_map, index)
            metas.append(meta)
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                acts[r["action"]] += 1
            n_rows += len(rows)
            print(f"{meta['match_id']:>12}  usable={meta['usable']!s:<5} "
                  f"xy={meta.get('xy_ratio')}  abil={meta.get('abilities')}  rows={len(rows)}")

    usable = sum(1 for m in metas if m["usable"])
    print(f"\n{usable}/{len(metas)} 局通过闸门，共 {n_rows} 行 -> {args.out}")
    if n_rows:
        print("动作分布:", dict(acts.most_common()))
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"matches": metas, "rows": n_rows, "actions": dict(acts),
                   "delta_s": DELTA_S, "action_window_s": ACTION_WINDOW_S}, fh,
                  ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
