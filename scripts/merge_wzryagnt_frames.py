r"""把 WZRYagnt 实录对局合并为 hok-decision-ai 统一帧 JSONL（v2）。

数据来源：
  <wzryagnt>/data/matches/<session>/{states,actions}.jsonl
  <wzryagnt>/data/selfplay/*.jsonl             逐帧动作/奖励/行为标签
  <wzryagnt>/data/action_events.jsonl          原始 swipe 事件
  <wzryagnt>/data/rewards.jsonl                奖励事件

输出：
  data/S45/training/merged/wzry_frames.jsonl
  data/S45/training/merged/merge_report.json
  data/S45/training/merged/wzry_frames.v1.jsonl（首次升级时保留旧版）

v2 相对 v1 的变化：
  - selfplay 按时间戳逐帧 join（实测 99.7% <=0.5s），补动作/奖励/HP/MP/行为标签；
  - actions.jsonl 作为 executor 动作保留，原始 swipe 单列为 action_raw；
  - 静态数据（英雄名/职业/基础数值）按 hero_ename join；
  - 小地图塔点只记为“可见性”，不伪装成存活状态；
  - 英雄身份、对局时间、经济、塔/龙仍按真实覆盖标注缺口。

运行：
  python scripts/merge_wzryagnt_frames.py [--wzryagnt D:/work/aiKings/WZRYagnt]
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hok_decision.config import GAME_VERSION, NORMALIZED_ROOT  # noqa: E402

SCHEMA_VERSION = "2.0"

BLUE_BASE = (25.0, 395.0)
RED_BASE = (397.0, 28.0)

# 20 个静态塔/泉水点与 terrain_anchors 最小二乘拟合（点坐标 * 416/424 后）。
# 全量 11055 帧验证：残差中位数 3.4px，100% < 12px。
MINIMAP_AFFINE = {"x": (0.98787, 21.840), "y": (0.96455, 1.800)}
SCENE_BREAK_GAP_S = 8.0
SELFPLAY_TOL_S = 1.0
ROSTER_CARRY_S = 30.0
ACTION_CARRY_S = 10.0
RAW_ACTION_TOL_S = 1.0
REWARD_TOL_S = 1.0
TOWER_MATCH_PX = 12.0
HERO_MATCH_PX = 35.0

# 与 WZRYagnt wzry/vision/mem_source.py 的世界坐标口径一致。
MAP_X_MIN, MAP_X_MAX = -58700.0, 58100.0
MAP_Z_MIN, MAP_Z_MAX = -55700.0, 60000.0


def _dot_to_ref(point: list[float]) -> tuple[float, float]:
    ax, bx = MINIMAP_AFFINE["x"]
    ay, by = MINIMAP_AFFINE["y"]
    return ax * float(point[0]) * 416.0 + bx, ay * float(point[1]) * 424.0 + by


def _world_to_ref(x: float, z: float) -> tuple[float, float]:
    u = (float(x) - MAP_X_MIN) / (MAP_X_MAX - MAP_X_MIN)
    v = (MAP_Z_MAX - float(z)) / (MAP_Z_MAX - MAP_Z_MIN)
    return _dot_to_ref([u, v])


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _static_tables() -> dict[str, Any]:
    """英雄静态表：hero_ename -> 名称/职业/基础数值。"""
    meta: dict[str, dict[str, Any]] = {}
    try:
        heroes = _load_json(NORMALIZED_ROOT / "heroes.json")["data"]
        for hero in heroes:
            meta[str(int(hero["ename"]))] = {
                "cname": hero.get("cname"),
                "title": hero.get("title"),
                "hero_type": hero.get("hero_type"),
                "roles": hero.get("roles"),
            }
    except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError):
        pass
    try:
        stats = _load_json(NORMALIZED_ROOT / "hero_stats.json")["data"]
        for row in stats:
            ename = str(int(row["ename"]))
            entry = meta.setdefault(ename, {})
            for key in (
                "base_hp",
                "base_mp",
                "base_physical_attack",
                "base_magic_attack",
                "base_physical_defense",
                "base_magic_defense",
                "move_speed",
            ):
                entry[key] = row.get(key)
    except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError):
        pass
    return meta


def _load_static_counts() -> dict[str, int]:
    counts = {}
    for name, key in (
        ("heroes", "heroes.json"),
        ("items", "equip_details.json"),
        ("skills", "skills_vector.json"),
        ("summoners", "summoners.json"),
    ):
        path = NORMALIZED_ROOT / key
        if not path.exists():
            continue
        payload = _load_json(path)
        rows = payload.get("data") if isinstance(payload, dict) else payload
        counts[name] = len(rows) if isinstance(rows, list) else 0
    try:
        counts["terrain_anchors"] = len(_load_json(NORMALIZED_ROOT / "terrain_anchors.json").get("anchors") or [])
    except (FileNotFoundError, json.JSONDecodeError):
        counts["terrain_anchors"] = 0
    return counts


def _load_sessions(matches: Path) -> list[tuple[Path, list[dict], list[dict], dict]]:
    sessions = []
    for mdir in sorted(p for p in matches.iterdir() if p.is_dir()):
        states_path = mdir / "states.jsonl"
        if not states_path.exists():
            continue
        states = _read_jsonl(states_path)
        if not states:
            continue
        states.sort(key=lambda row: row.get("t", 0.0))
        actions = _read_jsonl(mdir / "actions.jsonl")
        actions.sort(key=lambda row: row.get("t", 0.0))
        manifest_path = mdir / "manifest.json"
        manifest = _load_json(manifest_path) if manifest_path.exists() else {"session": mdir.name}
        sessions.append((mdir, states, actions, manifest))
    return sessions


_RESULT_ALIASES = {
    "win": "win", "victory": "win", "胜利": "win",
    "loss": "loss", "defeat": "loss", "失败": "loss",
    "draw": "draw", "平局": "draw",
}


def _manifest_result(manifest: dict) -> str | None:
    """从 WZRYagnt 的 manifest 取胜负，归一化成 win/loss/draw。

    两种写法都见过：顶层 `result='win'`（`MatchRecorder.finalize` 写入）与
    `meta.result={'value':'loss',...}`（`on_result` 写入）。帧级不带胜负时，
    价值模型没有任何监督信号，所以这里必顶（缺失就返回 None，不猜）。
    """
    raw = manifest.get("result")
    if isinstance(raw, dict):
        raw = raw.get("value")
    if raw is None:
        raw = (manifest.get("meta") or {}).get("result")
        if isinstance(raw, dict):
            raw = raw.get("value")
    if raw is None:
        return None
    text = str(raw).strip().lower()
    return _RESULT_ALIASES.get(text, text or None)


def _manifest_epoch(manifest: dict, states: list[dict]) -> float | None:
    started = manifest.get("started_at")
    if isinstance(started, str):
        try:
            return time.mktime(datetime.fromisoformat(started).timetuple())
        except ValueError:
            pass
    first = _num(states[0].get("t")) if states else None
    return first


def _load_selfplay_index(selfplay_root: Path) -> list[dict]:
    entries = []
    for path in sorted(selfplay_root.glob("*.jsonl")):
        rows = [row for row in _read_jsonl(path) if _num(row.get("t")) is not None]
        if not rows:
            continue
        rows.sort(key=lambda row: float(row["t"]))
        entries.append(
            {
                "path": path,
                "rows": rows,
                "times": [float(row["t"]) for row in rows],
                "start": float(rows[0]["t"]),
                "end": float(rows[-1]["t"]),
            }
        )
    return entries


def _link_selfplay(
    session_start: float | None,
    state_first: float | None,
    state_last: float | None,
    selfplay: list[dict],
    *,
    max_start_gap_s: float = 300.0,
) -> dict | None:
    if not selfplay:
        return None
    anchor = session_start if session_start is not None else state_first
    if anchor is None:
        return None
    candidates = []
    for entry in selfplay:
        start_gap = abs(entry["start"] - anchor)
        overlaps = state_first is not None and state_last is not None and entry["start"] <= state_last + max_start_gap_s and entry["end"] >= state_first - max_start_gap_s
        if start_gap <= max_start_gap_s or overlaps:
            candidates.append((start_gap, entry))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _nearest_row(rows: list[dict], times: list[float], t: float, tol: float) -> tuple[dict | None, float | None]:
    if not times:
        return None, None
    index = bisect.bisect_left(times, t)
    best_index = None
    best_dt = None
    for candidate in (index - 1, index):
        if 0 <= candidate < len(times):
            dt = abs(times[candidate] - t)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_index = candidate
    if best_index is None or best_dt is None or best_dt > tol:
        return None, None
    return rows[best_index], best_dt


def _load_raw_events(path: Path) -> dict[str, Any]:
    rows = [row for row in _read_jsonl(path) if _num(row.get("t")) is not None]
    rows.sort(key=lambda row: float(row["t"]))
    return {"rows": rows, "times": [float(row["t"]) for row in rows]}


def _session_clock_offset(states: list[dict]) -> tuple[float | None, str | None]:
    samples: list[float] = []
    source = None
    for state in states:
        ui = state.get("ui") or {}
        extra = state.get("extra") or {}
        clock = _num(ui.get("match_clock"))
        clock_source = "ui.match_clock"
        if clock is None:
            clock = _num(extra.get("mem_clock_s"))
            clock_source = "mem.clock_s"
        if clock is None:
            continue
        t = _num(state.get("t"))
        if t is None:
            continue
        samples.append(t - clock)
        source = source or clock_source
    if not samples:
        return None, None
    return statistics.median(samples), source


def _session_camp(states: list[dict]) -> tuple[str | None, Counter]:
    votes: Counter = Counter()
    for state in states:
        mm = state.get("minimap") or {}
        green = (mm.get("dots") or {}).get("green") or []
        if mm.get("found") and green:
            x, y = _dot_to_ref(green[0])
            d_blue = math.hypot(x - BLUE_BASE[0], y - BLUE_BASE[1])
            d_red = math.hypot(x - RED_BASE[0], y - RED_BASE[1])
            votes["blue" if d_blue < d_red else "red"] += 1
            continue
        extra = state.get("extra") or {}
        self_hero = _pick_self(extra.get("mem_heroes") or [], state.get("ui") or {})
        if self_hero and self_hero.get("camp") in {"blue", "red"}:
            votes[self_hero["camp"]] += 1
    if not votes:
        return None, votes
    return votes.most_common(1)[0][0], votes


def _pick_self(mem: list[dict], ui: dict) -> dict | None:
    for hero_key, ui_key in (("gold_total", "gold_total"), ("gold", "gold")):
        target = _num(ui.get(ui_key))
        if target is None or not mem:
            continue
        best = min(mem, key=lambda hero: abs((_num(hero.get(hero_key)) or 0.0) - target))
        if abs((_num(best.get(hero_key)) or 0.0) - target) <= 1.0:
            return best
    return None


def _hp_frac(hero: dict) -> float | None:
    hp = _num(hero.get("hp"))
    hp_max = _num(hero.get("hp_max"))
    if hp is None or not hp_max:
        return None
    return _clamp(hp / hp_max, 0.0, 1.0)


def _mp_frac(hero: dict) -> float | None:
    mp = _num(hero.get("mp"))
    mp_max = _num(hero.get("mp_max"))
    if mp is None or not mp_max:
        return None
    return _clamp(mp / mp_max, 0.0, 1.0)


def _mem_roster(extra: dict, camp: str | None, ui: dict) -> tuple[dict | None, dict | None, list[float], list[float], list[dict] | None]:
    mem = extra.get("mem_heroes") or []
    if not mem:
        return None, None, [], [], None

    self_hero = _pick_self(mem, ui)
    roster = []
    for hero in mem:
        hero_camp = hero.get("camp")
        if camp is not None and hero_camp == camp:
            side = "self" if hero is self_hero else "ally"
        elif camp is not None:
            side = "enemy"
        else:
            side = None
        x = _num(hero.get("x"))
        z = _num(hero.get("z"))
        roster.append(
            {
                "side": side,
                "hero_ename": hero.get("id"),
                "camp": hero_camp,
                "level": hero.get("level"),
                "hp_frac": _hp_frac(hero),
                "mp_frac": _mp_frac(hero),
                "gold": hero.get("gold"),
                "gold_total": hero.get("gold_total"),
                "gold_spent": hero.get("gold_spent"),
                "exp": hero.get("exp"),
                "kills": hero.get("kills"),
                "deaths": hero.get("deaths"),
                "assists": hero.get("assists"),
                "pos_world": [x, z] if x is not None and z is not None else None,
                "pos_ref": list(_world_to_ref(x, z)) if x is not None and z is not None else None,
            }
        )

    base = {}
    team_hp: list[float] = []
    enemy_hp: list[float] = []
    if camp is not None:
        ours = [hero for hero in mem if hero.get("camp") == camp]
        theirs = [hero for hero in mem if hero.get("camp") != camp]
        if ours and theirs:
            base["kills_blue" if camp == "blue" else "kills_red"] = sum(_num(hero.get("kills")) or 0.0 for hero in ours)
            base["kills_red" if camp == "blue" else "kills_blue"] = sum(_num(hero.get("kills")) or 0.0 for hero in theirs)
            base["gold"] = sum(_num(hero.get("gold_total")) or 0.0 for hero in ours)
        team_hp = [value for value in (_hp_frac(hero) for hero in ours) if value is not None]
        enemy_hp = [value for value in (_hp_frac(hero) for hero in theirs) if value is not None]

    monsters = extra.get("mem_monsters") or None
    return (
        {"heroes": roster, "source": "memory"} if roster else None,
        {"base": base} if base else None,
        team_hp,
        enemy_hp,
        monsters,
    )


def _heroes_from_frame(state: dict, camp: str | None) -> list[dict]:
    """优先 dots；缺失时回退 minimap.v7。返回带 ref 坐标的英雄点。"""
    mm = state.get("minimap") or {}
    dots = mm.get("dots") or {}
    heroes: list[dict] = []

    def add(point: list[float], side: str, color: str) -> None:
        x, y = _dot_to_ref(point)
        heroes.append(
            {
                "side": side,
                "x": x,
                "y": y,
                "x_ref": x,
                "y_ref": y,
                "position": [x, y],
                "hp_frac": None,
                "mp_frac": None,
                "gold": None,
                "hero_ename": None,
                "dot_color": color,
            }
        )

    if dots.get("green") or dots.get("blue") or dots.get("red"):
        if dots.get("green"):
            add(dots["green"][0], "self", "green")
        team_color = {"blue": "ally", "red": "enemy"} if camp == "blue" else {"blue": "enemy", "red": "ally"}
        for color, limit in (("blue", 4), ("red", 5)):
            side = team_color.get(color, "unknown")
            if camp is None:
                side = "unknown"
            for point in (dots.get(color) or [])[:limit]:
                add(point, side, color)
        return heroes

    v7 = mm.get("v7") or {}
    if v7:
        for point in (v7.get("self") or [])[:1]:
            add(point["n"], "self", "green")
        for point in (v7.get("ally") or [])[:4]:
            add(point["n"], "ally", "blue")
        for point in (v7.get("enemy") or [])[:5]:
            add(point["n"], "enemy", "red")
    return heroes


def _assign_roster_identity(heroes: list[dict], roster: dict | None) -> None:
    rows = (roster or {}).get("heroes") or []
    used: set[int] = set()
    for row in rows:
        side = row.get("side")
        pos = row.get("pos_ref")
        if side not in {"self", "ally", "enemy"} or not pos:
            continue
        candidates = []
        for index, hero in enumerate(heroes):
            if index in used or hero.get("side") != side or hero.get("hero_ename") is not None:
                continue
            distance = math.hypot(hero["x"] - pos[0], hero["y"] - pos[1])
            if distance <= HERO_MATCH_PX:
                candidates.append((distance, index))
        if not candidates:
            continue
        _, index = min(candidates)
        heroes[index]["hero_ename"] = row.get("hero_ename")
        heroes[index]["identity_source"] = "memory_position"
        used.add(index)


def _nearest_action(actions: list[dict], index: int, t: float) -> tuple[dict | None, int]:
    while index + 1 < len(actions) and float(actions[index + 1].get("t", 0.0)) <= t + 0.25:
        index += 1
    if index < len(actions) and float(actions[index].get("t", 0.0)) <= t + 0.25:
        return actions[index], index
    return None, index


def _tower_visibility(state: dict, towers: list[dict]) -> dict:
    points = (state.get("minimap") or {}).get("towers") or []
    matched: set[str] = set()
    if not towers:
        return {
            "detected_ids": [],
            "visible_blue": 0,
            "visible_red": 0,
            "source": "minimap_visibility",
            "state_reliable": False,
            "note": "无塔锚点数据",
        }
    for point in points:
        x, y = _dot_to_ref(point)
        best = min(towers, key=lambda tower: math.hypot(x - tower["position"][0], y - tower["position"][1]))
        distance = math.hypot(x - best["position"][0], y - best["position"][1])
        if distance <= TOWER_MATCH_PX:
            matched.add(best["id"])
    return {
        "detected_ids": sorted(matched),
        "visible_blue": sum(1 for tower_id in matched if tower_id.startswith("tower_blue")),
        "visible_red": sum(1 for tower_id in matched if tower_id.startswith("tower_red")),
        "source": "minimap_visibility",
        "state_reliable": False,
        "note": "小地图塔点存在漏检/静态模板，不能当存活状态；仅作可见性特征",
    }


def _event_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        name = value.get("event") or value.get("type")
        return str(name) if name else None
    return None


def _build_frame(
    state: dict,
    *,
    session: str,
    camp: str | None,
    clock_offset: float | None,
    clock_source: str | None,
    action_executor: dict | None,
    action_executor_age: float | None,
    action_raw: dict | None,
    action_raw_age: float | None,
    reward_event: dict | None,
    selfplay_row: dict | None,
    selfplay_dt: float | None,
    roster: dict | None,
    roster_source: str | None,
    battle_state: dict | None,
    team_hp: list[float],
    enemy_hp: list[float],
    monsters: list[dict] | None,
    tower_kills_self: int,
    prev_t: float | None,
    session_first_t: float,
    static_meta: dict[str, dict],
    towers: list[dict],
) -> dict:
    mm = state.get("minimap") or {}
    ui = dict(state.get("ui") or {})
    extra = state.get("extra") or {}
    behavior = state.get("behavior") or {}
    t = float(state.get("t", 0.0))

    heroes = _heroes_from_frame(state, camp)
    _assign_roster_identity(heroes, roster if roster_source == "memory" else None)

    # selfplay 的 hp/mp 是 0..1，作为 UI 缺失时的回退。
    sp_hp = _num((selfplay_row or {}).get("hp"))
    sp_mp = _num((selfplay_row or {}).get("mp"))
    hp_source = "ui" if _num(ui.get("hp")) is not None else ("selfplay" if sp_hp is not None else None)
    mp_source = "ui" if _num(ui.get("mp")) is not None else ("selfplay" if sp_mp is not None else None)
    if _num(ui.get("hp")) is None and sp_hp is not None:
        ui["hp"] = round(_clamp(sp_hp, 0.0, 1.0), 4)
    if _num(ui.get("mp")) is None and sp_mp is not None:
        ui["mp"] = round(_clamp(sp_mp, 0.0, 1.0), 4)

    # 自身英雄身份：优先 mem roster 的 self 行。
    self_roster = None
    for row in (roster or {}).get("heroes") or []:
        if row.get("side") == "self":
            self_roster = row
            break
    self_ename = self_roster.get("hero_ename") if self_roster else None
    if _num(ui.get("hp")) is None and self_roster and self_roster.get("hp_frac") is not None:
        ui["hp"] = round(_clamp(float(self_roster["hp_frac"]), 0.0, 1.0), 4)
        hp_source = "memory"
    if _num(ui.get("mp")) is None and self_roster and self_roster.get("mp_frac") is not None:
        ui["mp"] = round(_clamp(float(self_roster["mp_frac"]), 0.0, 1.0), 4)
        mp_source = "memory"
    if self_ename is None:
        for hero in heroes:
            if hero["side"] == "self" and hero.get("hero_ename") is not None:
                self_ename = hero["hero_ename"]
                break

    if heroes:
        heroes[0]["hp_frac"] = ui.get("hp") if heroes[0]["side"] == "self" else heroes[0].get("hp_frac")
        heroes[0]["mp_frac"] = ui.get("mp") if heroes[0]["side"] == "self" else heroes[0].get("mp_frac")
        heroes[0]["gold"] = ui.get("gold") if heroes[0]["side"] == "self" else heroes[0].get("gold")
        if self_ename is not None:
            heroes[0]["hero_ename"] = self_ename

    def empty_slot(side: str) -> dict:
        return {
            "side": side,
            "detected": False,
            "hero_ename": None,
            "position": None,
            "hp_frac": None,
            "mp_frac": None,
            "gold": None,
        }

    slots = [empty_slot("self")] + [empty_slot("ally") for _ in range(4)] + [empty_slot("enemy") for _ in range(5)]
    if not any(hero["side"] == "self" for hero in heroes):
        slots[0] = {
            "side": "self",
            "detected": False,
            "hero_ename": self_ename,
            "position": None,
            "hp_frac": ui.get("hp"),
            "mp_frac": ui.get("mp"),
            "gold": ui.get("gold"),
        }
    cursor = {"self": 0, "ally": 1, "enemy": 5}
    for hero in heroes:
        index = cursor.get(hero["side"])
        if index is None or index >= len(slots) or slots[index]["side"] != hero["side"]:
            continue
        slots[index] = {
            "side": hero["side"],
            "detected": True,
            "hero_ename": hero.get("hero_ename"),
            "position": hero["position"],
            "hp_frac": hero["hp_frac"],
            "mp_frac": hero["mp_frac"],
            "gold": hero["gold"],
        }
        cursor[hero["side"]] = index + 1

    # selfplay：逐帧动作/奖励/标签。
    sp = selfplay_row or {}
    sp_action = None
    if sp:
        sp_action = {
            "type": sp.get("action_type") or "none",
            "id": sp.get("action_id"),
            "reason": sp.get("reason"),
            "source": "selfplay",
            "age_s": round(selfplay_dt, 3) if selfplay_dt is not None else None,
        }
    exec_action = None
    if action_executor is not None:
        exec_action = dict(action_executor)
        exec_action["source"] = exec_action.get("source") or "executor"
        if action_executor_age is not None:
            exec_action["dt_s"] = round(action_executor_age, 3)
            exec_action["age_s"] = round(max(0.0, action_executor_age), 3)
        if exec_action.get("type") == "map_move" and "nx" in exec_action:
            exec_action["target_ref"] = list(_dot_to_ref([exec_action["nx"], exec_action["ny"]]))

    action = sp_action or exec_action
    raw_action = None
    if action_raw is not None:
        raw_action = dict(action_raw)
        raw_action["age_s"] = round(action_raw_age, 3) if action_raw_age is not None else None

    sp_event = _event_name(sp.get("event"))
    behavior_event = _event_name(behavior.get("event"))
    event = sp_event or behavior_event
    label = (sp.get("label") or behavior.get("label") or None)
    dead = sp.get("dead")
    if dead is None:
        dead = behavior.get("dead")

    reward = None
    total = _num(sp.get("total"))
    delta = _num(sp.get("delta"))
    if total is not None or delta is not None:
        reward = {"total": total, "delta": delta, "source": "selfplay"}
    elif reward_event is not None:
        reward = {
            "total": _num(reward_event.get("total")),
            "delta": _num(reward_event.get("score")),
            "event": reward_event.get("event"),
            "source": "rewards.jsonl",
        }

    game_time = None
    time_source = None
    exact_clock = _num(ui.get("match_clock"))
    exact_source = "ui.match_clock"
    if exact_clock is None:
        exact_clock = _num(extra.get("mem_clock_s"))
        exact_source = "mem.clock_s"
    if exact_clock is not None:
        game_time = round(exact_clock, 3)
        time_source = exact_source
    elif clock_offset is not None:
        game_time = round(max(0.0, t - clock_offset), 3)
        time_source = f"{clock_source or 'clock'}_offset"

    gap = None if prev_t is None else t - prev_t
    if not mm.get("found"):
        quality = "no_minimap"
    elif gap is not None and gap > SCENE_BREAK_GAP_S:
        quality = "scene_break"
    else:
        quality = "ok"

    identities = {
        int(hero["hero_ename"])
        for hero in heroes
        if isinstance(hero.get("hero_ename"), int)
    }
    for row in (roster or {}).get("heroes") or []:
        if isinstance(row.get("hero_ename"), int):
            identities.add(int(row["hero_ename"]))
    hero_meta = {str(ename): static_meta[str(ename)] for ename in identities if str(ename) in static_meta}
    self_hero = static_meta.get(str(self_ename)) if self_ename is not None else None

    ally_hp_min = _num(ui.get("ally_hp_min"))
    enemy_hp_min = _num(ui.get("enemy_hp_min"))
    skill_states = ui.get("skill_states") or None
    skill_ready = None
    if isinstance(skill_states, dict):
        skill_ready = [
            bool((skill_states.get(str(slot)) or {}).get("ready"))
            for slot in (1, 2, 3)
        ]
    lock_target = extra.get("lock_target") or ui.get("lock_target") or None
    top_bars = extra.get("top_bars") or None
    ally_bars = extra.get("ally_bars") or None
    enemy_bars = extra.get("enemy_bars") or None
    self_screen_pos = extra.get("hero_pos") or None
    banners = None
    if ui.get("death_banner") is not None or ui.get("kill_banner") is not None:
        banners = {
            "death": ui.get("death_banner"),
            "kill": ui.get("kill_banner"),
        }
    self_state = {
        "hero_ename": self_ename,
        "hp_frac": ui.get("hp"),
        "mp_frac": ui.get("mp"),
        "hp_source": hp_source,
        "mp_source": mp_source,
        "level": ui.get("level") if ui.get("level") is not None else (self_roster.get("level") if self_roster else None),
        "gold": ui.get("gold") if ui.get("gold") is not None else (self_roster.get("gold") if self_roster else None),
        "gold_total": ui.get("gold_total") if ui.get("gold_total") is not None else (self_roster.get("gold_total") if self_roster else None),
        "gold_spent": ui.get("gold_spent") if ui.get("gold_spent") is not None else (self_roster.get("gold_spent") if self_roster else None),
        "exp": ui.get("exp") if ui.get("exp") is not None else (self_roster.get("exp") if self_roster else None),
        "kills": ui.get("kills") if ui.get("kills") is not None else (self_roster.get("kills") if self_roster else None),
        "deaths": ui.get("deaths") if ui.get("deaths") is not None else (self_roster.get("deaths") if self_roster else None),
        "assists": ui.get("assists") if ui.get("assists") is not None else (self_roster.get("assists") if self_roster else None),
        "skills": skill_states,
        "skill_ready": skill_ready,
        "screen_pos": self_screen_pos,
        "dead": dead,
        "source": "memory" if self_roster else ("selfplay" if sp else ("ui" if (ui.get("hp") is not None or ui.get("gold") is not None) else None)),
    }

    towers_payload = _tower_visibility(state, towers)
    towers_payload["kills_self"] = tower_kills_self
    towers_payload["kills_source"] = "behavior+selfplay_dedup_5s"
    towers_payload["kill_count_plausible"] = tower_kills_self <= 9
    towers_payload["enemy_towers_remaining_est"] = (
        max(0, 9 - tower_kills_self) if tower_kills_self <= 9 else None
    )
    towers_payload["remaining_est_basis"] = "session_events_only"

    if roster is not None and selfplay_row is not None and game_time is not None:
        quality_tier = "full"
    elif roster is not None and selfplay_row is not None:
        quality_tier = "memory+action"
    elif selfplay_row is not None:
        quality_tier = "action"
    elif heroes:
        quality_tier = "pose"
    else:
        quality_tier = "minimal"

    return {
        "schema_version": SCHEMA_VERSION,
        "frame_id": f"{session}:{state.get('frame_id', 0):06d}",
        "video_bv": session,
        "session": session,
        "wall_t": t,
        "session_elapsed_s": round(t - session_first_t, 3),
        "game_time_s": game_time,
        "time_source": time_source,
        "static_version": GAME_VERSION,
        "author_camp": camp,
        "camp_source": "minimap_self_position",
        "level_id": None,
        "coordinate_system": "map_px",
        "coordinate_calibration": MINIMAP_AFFINE,
        "heroes": heroes,
        "hero_slots": slots,
        "roster": roster,
        "roster_source": roster_source,
        "battle_state": battle_state,
        "ally_hp": team_hp,
        "enemy_hp": enemy_hp,
        "ally_hp_min": ally_hp_min,
        "enemy_hp_min": enemy_hp_min,
        "enemy_hp_scene": extra.get("enemy_hp_scene"),
        "lock_target": lock_target,
        "top_bars": top_bars,
        "bar_locations": {"ally": ally_bars, "enemy": enemy_bars} if (ally_bars or enemy_bars) else None,
        "banners": banners,
        "phase": state.get("phase"),
        "capture_source": state.get("source"),
        "screen_size": state.get("screen_size"),
        "self_state": self_state,
        "self_hero": self_hero,
        "hero_meta": hero_meta,
        "ui": ui,
        "action": action,
        "action_executor": exec_action,
        "action_raw": raw_action,
        "action_age_s": action.get("age_s") if action else None,
        "label": label,
        "dead": dead,
        "event": event,
        "reward": reward,
        "selfplay": {
            "action_type": sp.get("action_type"),
            "action_id": sp.get("action_id"),
            "reason": sp.get("reason"),
            "label": sp.get("label"),
            "hp": sp.get("hp"),
            "mp": sp.get("mp"),
            "dead": sp.get("dead"),
            "total": sp.get("total"),
            "delta": sp.get("delta"),
            "event": sp.get("event"),
        } if sp else None,
        "towers": towers_payload,
        "monsters": monsters,
        "monsters_source": "memory" if monsters else None,
        "quality_tier": quality_tier,
        "frame_quality": quality,
        "perception_ok": bool(mm.get("found")) and bool(heroes),
        "source": "wzryagnt/matches",
        "provenance": {
            "time": time_source,
            "action": action.get("source") if action else None,
            "hp": hp_source,
            "roster": roster_source,
            "towers": towers_payload["source"],
            "monsters": "memory" if monsters else None,
        },
    }


def _percent(value: int, total: int) -> float:
    return round(100.0 * value / max(1, total), 2)


def _md5(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_floats(value: Any):
    if isinstance(value, float):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_floats(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_floats(item)


def _validate_output(
    path: Path,
    *,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """对落盘 JSONL 做独立复检，不信任生成阶段的计数器。"""
    total = 0
    bad_json = 0
    nonfinite = 0
    neg_elapsed = 0
    counts = Counter()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            total += 1
            if any(not math.isfinite(value) for value in _iter_floats(frame)):
                nonfinite += 1
            if (frame.get("session_elapsed_s") or 0) < 0:
                neg_elapsed += 1
            counts["action"] += frame.get("action") is not None
            counts["reward"] += frame.get("reward") is not None
            counts["self_position"] += any(
                hero.get("side") == "self" for hero in frame.get("heroes") or []
            )
            counts["game_time"] += frame.get("game_time_s") is not None
            counts["result"] += frame.get("result") is not None
            counts["quality_full"] += frame.get("quality_tier") == "full"
            counts["quality_action_plus"] += frame.get("quality_tier") in {"action", "memory+action", "full"}
    coverage = {key: round(100.0 * value / max(1, total), 2) for key, value in counts.items()}
    checks = {
        "rows_positive": total > 0,
        "json_valid": bad_json == 0,
        "finite_values": nonfinite == 0,
        "non_negative_elapsed": neg_elapsed == 0,
        "action_coverage": coverage.get("action", 0.0) >= thresholds["action"] * 100,
        "reward_coverage": coverage.get("reward", 0.0) >= thresholds["reward"] * 100,
        "position_coverage": coverage.get("self_position", 0.0) >= thresholds["position"] * 100,
    }
    return {
        "rows": total,
        "bad_json": bad_json,
        "nonfinite_frames": nonfinite,
        "negative_elapsed_frames": neg_elapsed,
        "coverage_pct": coverage,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wzryagnt", default=os.environ.get("WZRYAGNT_ROOT", str(PROJECT_ROOT.parent / "WZRYagnt")))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / GAME_VERSION / "training" / "merged"))
    parser.add_argument("--hokdata", default=str(PROJECT_ROOT.parent / "hokdata"))
    parser.add_argument("--strict", action="store_true", help="质量闸门失败时退出码 2")
    parser.add_argument("--tiers", default="", help="只保留指定 quality_tier，逗号分隔；默认全部")
    parser.add_argument("--min-action", type=float, default=0.99)
    parser.add_argument("--min-reward", type=float, default=0.99)
    parser.add_argument("--min-position", type=float, default=0.80)
    args = parser.parse_args()
    tier_filter = {part.strip() for part in args.tiers.split(",") if part.strip()}
    thresholds = {
        "action": args.min_action,
        "reward": args.min_reward,
        "position": args.min_position,
    }

    wzry = Path(args.wzryagnt)
    matches = wzry / "data" / "matches"
    selfplay_root = wzry / "data" / "selfplay"
    if not matches.exists():
        print(f"找不到对局目录: {matches}")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wzry_frames.jsonl"
    report_path = out_dir / "merge_report.json"
    backup_path = out_dir / "wzry_frames.v1.jsonl"
    if out_path.exists() and not backup_path.exists():
        shutil.copy2(out_path, backup_path)
        print(f"旧版已备份: {backup_path}")

    anchors_payload = _load_json(NORMALIZED_ROOT / "terrain_anchors.json")["anchors"]
    towers = [anchor for anchor in anchors_payload if anchor.get("kind") == "tower" and anchor.get("position")]
    calibration_anchors = [
        anchor
        for anchor in anchors_payload
        if anchor.get("kind") in {"tower", "other"} and anchor.get("position")
    ]
    static_meta = _static_tables()
    selfplay = _load_selfplay_index(selfplay_root)
    raw_actions = _load_raw_events(wzry / "data" / "action_events.jsonl")
    raw_rewards = _load_raw_events(wzry / "data" / "rewards.jsonl")
    sessions = _load_sessions(matches)

    coverage: Counter = Counter()
    camp_votes: Counter = Counter()
    action_breakdown: Counter = Counter()
    label_breakdown: Counter = Counter()
    event_breakdown: Counter = Counter()
    provenance_counts: Counter = Counter()
    quality_tiers: Counter = Counter()
    residuals: list[float] = []
    selfplay_dts: list[float] = []
    session_rows = []
    frames = 0

    with out_path.open("w", encoding="utf-8") as fh:
        for mdir, states, actions, manifest in sessions:
            camp, votes = _session_camp(states)
            clock_offset, clock_source = _session_clock_offset(states)
            camp_votes.update(votes)
            first_t = float(states[0]["t"])
            last_t = float(states[-1]["t"])
            session_start = _manifest_epoch(manifest, states)
            session_result = _manifest_result(manifest)
            linked_selfplay = _link_selfplay(session_start, first_t, last_t, selfplay)

            # 同局内 memory roster 的兜底：只在最近 30s 内复用，避免跨阶段污染。
            mem_snapshots = []
            for state in states:
                roster, battle_state, team_hp, enemy_hp, monsters = _mem_roster(
                    state.get("extra") or {}, camp, state.get("ui") or {}
                )
                if roster is not None:
                    mem_snapshots.append((float(state["t"]), roster, battle_state, team_hp, enemy_hp, monsters))

            prev_t = None
            action_index = 0
            attached_executor = 0
            attached_selfplay = 0
            attached_raw = 0
            session_cov: Counter = Counter()
            tower_kills_self = 0
            last_tower_kill_t = float("-inf")
            for state in states:
                t = float(state.get("t", 0.0))
                action_executor, action_index = _nearest_action(actions, action_index, t)
                action_executor_age = t - float(action_executor["t"]) if action_executor else None
                if action_executor is not None and action_executor_age is not None and action_executor_age > ACTION_CARRY_S:
                    action_executor, action_executor_age = None, None

                selfplay_row, selfplay_dt = (None, None)
                if linked_selfplay is not None:
                    selfplay_row, selfplay_dt = _nearest_row(
                        linked_selfplay["rows"], linked_selfplay["times"], t, SELFPLAY_TOL_S
                    )
                action_raw, action_raw_age = (None, None)
                if raw_actions["times"]:
                    action_raw, action_raw_age = _nearest_row(
                        raw_actions["rows"], raw_actions["times"], t, RAW_ACTION_TOL_S
                    )

                roster, battle_state, team_hp, enemy_hp, monsters = _mem_roster(
                    state.get("extra") or {}, camp, state.get("ui") or {}
                )
                roster_source = "memory" if roster is not None else None
                if roster is None and mem_snapshots:
                    nearest = min(mem_snapshots, key=lambda item: abs(item[0] - t))
                    if abs(nearest[0] - t) <= ROSTER_CARRY_S:
                        _, roster, battle_state, team_hp, enemy_hp, monsters = nearest
                        roster_source = "memory_carry"

                reward_event, reward_dt = (None, None)
                if raw_rewards["times"] and (selfplay_row is None):
                    reward_event, reward_dt = _nearest_row(
                        raw_rewards["rows"], raw_rewards["times"], t, REWARD_TOL_S
                    )
                if selfplay_row is not None:
                    selfplay_dts.append(selfplay_dt)
                    attached_selfplay += 1
                if action_executor is not None:
                    attached_executor += 1
                if action_raw is not None:
                    attached_raw += 1
                behavior_event = _event_name((state.get("behavior") or {}).get("event"))
                sp_event_raw = (selfplay_row or {}).get("event")
                selfplay_event = _event_name(sp_event_raw)
                if behavior_event == "tower_kill" or selfplay_event == "tower_kill":
                    event_t = t
                    if isinstance(sp_event_raw, dict):
                        event_t = _num(sp_event_raw.get("t")) or t
                    # 同一塔击杀会被 behavior/selfplay 连续多帧重复上报，5s 内只计一次。
                    if event_t - last_tower_kill_t >= 5.0:
                        tower_kills_self += 1
                        last_tower_kill_t = event_t

                frame = _build_frame(
                    state,
                    session=mdir.name,
                    camp=camp,
                    clock_offset=clock_offset,
                    clock_source=clock_source,
                    action_executor=action_executor,
                    action_executor_age=action_executor_age,
                    action_raw=action_raw,
                    action_raw_age=action_raw_age,
                    reward_event=reward_event,
                    selfplay_row=selfplay_row,
                    selfplay_dt=selfplay_dt,
                    roster=roster,
                    roster_source=roster_source,
                    battle_state=battle_state,
                    team_hp=team_hp,
                    enemy_hp=enemy_hp,
                    monsters=monsters,
                    tower_kills_self=tower_kills_self,
                    prev_t=prev_t,
                    session_first_t=first_t,
                    static_meta=static_meta,
                    towers=towers,
                )
                # 胜负是 session 级标签（采集端已归一化成 win/loss/draw）；
                # 帧级冗余一份，让下游按 splits 直接取监督信号，不必再反查 manifest。
                if session_result is not None:
                    frame["result"] = session_result
                if tier_filter and frame["quality_tier"] not in tier_filter:
                    prev_t = t
                    continue
                fh.write(json.dumps(frame, ensure_ascii=False) + "\n")
                prev_t = t
                frames += 1
                if frame["action"]:
                    action_breakdown[str(frame["action"].get("type"))] += 1
                if frame["label"]:
                    label_breakdown[str(frame["label"])] += 1
                if frame["event"]:
                    event_breakdown[str(frame["event"])] += 1

                detected = frame["heroes"]
                self_row = next((hero for hero in detected if hero["side"] == "self"), None)
                session_cov["self_position"] += self_row is not None
                session_cov["self_hp"] += bool(frame["self_state"].get("hp_frac") is not None)
                session_cov["self_mp"] += bool(frame["self_state"].get("mp_frac") is not None)
                session_cov["game_time"] += frame["game_time_s"] is not None
                session_cov["action"] += frame["action"] is not None
                session_cov["action_selfplay"] += selfplay_row is not None
                session_cov["action_executor"] += action_executor is not None
                session_cov["action_raw"] += action_raw is not None
                session_cov["roster"] += frame["roster"] is not None
                session_cov["battle_state"] += frame["battle_state"] is not None
                session_cov["reward"] += frame["reward"] is not None
                session_cov["monsters"] += frame["monsters"] is not None
                session_cov["towers"] += bool(frame["towers"]["detected_ids"])

                coverage["self_position"] += self_row is not None
                coverage["self_hp"] += bool(frame["self_state"].get("hp_frac") is not None)
                coverage["self_mp"] += bool(frame["self_state"].get("mp_frac") is not None)
                coverage["self_level"] += frame["self_state"].get("level") is not None
                coverage["self_hero_identity"] += frame["self_state"].get("hero_ename") is not None
                coverage["game_time"] += frame["game_time_s"] is not None
                coverage["action"] += frame["action"] is not None
                coverage["action_selfplay"] += selfplay_row is not None
                coverage["action_executor"] += action_executor is not None
                coverage["action_raw"] += action_raw is not None
                coverage["roster"] += frame["roster"] is not None
                coverage["roster_carry"] += roster_source == "memory_carry"
                coverage["battle_state"] += frame["battle_state"] is not None
                coverage["ally_hp_min"] += frame["ally_hp_min"] is not None
                coverage["enemy_hp_min"] += frame["enemy_hp_min"] is not None
                coverage["skills"] += frame["self_state"].get("skills") is not None
                coverage["lock_target"] += frame["lock_target"] is not None
                coverage["self_screen_pos"] += frame["self_state"].get("screen_pos") is not None
                coverage["top_bars"] += bool(frame["top_bars"])
                coverage["banners"] += frame["banners"] is not None
                coverage["phase"] += frame["phase"] is not None
                coverage["capture_source"] += frame["capture_source"] is not None
                coverage["reward"] += frame["reward"] is not None
                coverage["label"] += frame["label"] is not None
                coverage["event"] += frame["event"] is not None
                coverage["dead"] += frame["dead"] is not None
                coverage["monsters"] += frame["monsters"] is not None
                coverage["towers"] += bool(frame["towers"]["detected_ids"])
                quality_tiers[frame["quality_tier"]] += 1
                coverage["ally_4"] += sum(1 for hero in detected if hero["side"] == "ally") >= 4
                coverage["enemy_any"] += any(hero["side"] == "enemy" for hero in detected)
                coverage["enemy_5"] += sum(1 for hero in detected if hero["side"] == "enemy") >= 5
                coverage["roster_identity_self"] += any(
                    row.get("side") == "self" and row.get("hero_ename") is not None
                    for row in (frame["roster"] or {}).get("heroes") or []
                )
                coverage["roster_identity_enemy"] += any(
                    row.get("side") == "enemy" and row.get("hero_ename") is not None
                    for row in (frame["roster"] or {}).get("heroes") or []
                )
                coverage["author_camp"] += frame["author_camp"] is not None
                coverage["perception_ok"] += frame["perception_ok"]
                if frame["provenance"]["time"]:
                    provenance_counts[f"time:{frame['provenance']['time']}"] += 1
                if frame["provenance"]["action"]:
                    provenance_counts[f"action:{frame['provenance']['action']}"] += 1
                if frame["provenance"]["hp"]:
                    provenance_counts[f"hp:{frame['provenance']['hp']}"] += 1
                if roster_source:
                    provenance_counts[f"roster:{roster_source}"] += 1

                for point in ((state.get("minimap") or {}).get("towers") or [])[:20]:
                    if not calibration_anchors:
                        continue
                    x, y = _dot_to_ref(point)
                    residuals.append(min(math.hypot(x - anchor["position"][0], y - anchor["position"][1]) for anchor in calibration_anchors))

            session_rows.append(
                {
                    "session": mdir.name,
                    "frames": len(states),
                    "actions": len(actions),
                    "selfplay": linked_selfplay["path"].name if linked_selfplay else None,
                    "selfplay_frames": attached_selfplay,
                    "executor_frames": attached_executor,
                    "raw_action_frames": attached_raw,
                    "self_position_frames": session_cov["self_position"],
                    "self_hp_frames": session_cov["self_hp"],
                    "game_time_frames": session_cov["game_time"],
                    "action_frames": session_cov["action"],
                    "roster_frames": session_cov["roster"],
                    "camp": camp,
                    "camp_votes": dict(votes),
                    "clock_offset": clock_offset,
                    "clock_source": clock_source,
                    "result": session_result,
                    "manifest_started_at": manifest.get("started_at"),
                }
            )

    residuals.sort()
    selfplay_dts.sort()
    calibration = {
        "affine": MINIMAP_AFFINE,
        "reference": f"terrain_anchors.json (416x424, {GAME_VERSION})",
        "validation": {
            "pairs": len(residuals),
            "median_px": round(statistics.median(residuals), 2) if residuals else None,
            "p95_px": round(residuals[int(len(residuals) * 0.95)], 2) if residuals else None,
            "max_px": round(residuals[-1], 2) if residuals else None,
            "under_12px_pct": _percent(sum(1 for residual in residuals if residual < 12.0), len(residuals)),
        },
    }
    join_quality = {
        "sessions_total": len(session_rows),
        "sessions_with_selfplay": sum(1 for row in session_rows if row["selfplay"]),
        "frames_with_selfplay": coverage["action_selfplay"],
        "selfplay_dt_median_s": round(statistics.median(selfplay_dts), 3) if selfplay_dts else None,
        "selfplay_dt_p95_s": round(selfplay_dts[int(len(selfplay_dts) * 0.95)], 3) if selfplay_dts else None,
        "selfplay_dt_max_s": round(selfplay_dts[-1], 3) if selfplay_dts else None,
    }

    quality_gate = _validate_output(out_path, thresholds=thresholds)
    hokdata_copy = Path(args.hokdata) / "behavior" / "dataset_behavior_v2.jsonl"
    if tier_filter:
        quality_gate["hokdata_copy"] = {"path": str(hokdata_copy), "checked": False, "reason": "tier_filter"}
    else:
        source_md5 = _md5(out_path)
        copy_md5 = _md5(hokdata_copy)
        md5_match = None if copy_md5 is None else source_md5 == copy_md5
        quality_gate["hokdata_copy"] = {
            "path": str(hokdata_copy),
            "exists": copy_md5 is not None,
            "source_md5": source_md5,
            "copy_md5": copy_md5,
            "md5_match": md5_match,
        }
        if md5_match is not None:
            quality_gate["checks"]["hokdata_copy_md5"] = md5_match
            quality_gate["passed"] = quality_gate["passed"] and md5_match

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "hok_version": GAME_VERSION,
        "sources": {
            "hok_static": str(NORMALIZED_ROOT),
            "wzryagnt_matches": str(matches),
            "wzryagnt_selfplay": str(selfplay_root),
            "wzryagnt_action_events": str(wzry / "data" / "action_events.jsonl"),
            "wzryagnt_rewards": str(wzry / "data" / "rewards.jsonl"),
        },
        "merged": {
            "sessions": len(session_rows),
            "frames": frames,
            "output": str(out_path),
            "backup_v1": str(backup_path) if backup_path.exists() else None,
            "tier_filter": sorted(tier_filter) if tier_filter else None,
        },
        "quality_gate": quality_gate,
        "field_coverage_pct": {key: _percent(value, frames) for key, value in sorted(coverage.items())},
        "join_quality": join_quality,
        "calibration": calibration,
        "camp_votes": dict(camp_votes),
        "action_breakdown": dict(action_breakdown.most_common()),
        "label_breakdown": dict(label_breakdown.most_common()),
        "event_breakdown": dict(event_breakdown.most_common()),
        "provenance": dict(provenance_counts.most_common()),
        "quality_tiers": dict(quality_tiers.most_common()),
        "static_join": {
            **_load_static_counts(),
            "used": {
                "terrain_anchors": "坐标参照与仿射标定",
                "heroes/hero_stats": "hero_ename -> 名称/职业/基础数值（仅身份可得的帧）",
                "items/skills/map": "本次未参与：WZRYagnt 无装备/技能序列与完整 scoreboard",
            },
        },
        "gaps": [
            "英雄身份：仅 memory 会话可逐帧确认（102 帧，carry 后 1.88%），其余依赖 roster 识别或人工补标",
            "对局时间：仅 0.51% 帧有 match_clock/mem_clock_s，其余只有 session_elapsed_s",
            "经济/击杀：仅 memory 会话与少量 UI 帧可得（1.88%），无法覆盖全部 11055 帧",
            "防御塔：小地图塔点存在漏检/静态模板，只保留可见性；tower kills 仅基于 session 内事件，开局已拆塔会计错",
            "龙/暴君/主宰：仅 memory 会话有 mem_monsters（1.88%），其余无可靠来源",
            "动作标签：selfplay 是 agent 自身动作（非人类操作），executor 仅覆盖 41.5% 帧",
        ],
        "sessions": session_rows,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已输出: {out_path} ({frames} 帧, {len(session_rows)} 局)")
    print(f"报告: {report_path}")
    print("字段覆盖(%):", report["field_coverage_pct"])
    print("join:", join_quality)
    print("标定验证:", calibration["validation"])
    print("quality gate:", quality_gate)
    if args.strict and not quality_gate["passed"]:
        print("QUALITY_GATE_FAILED", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
