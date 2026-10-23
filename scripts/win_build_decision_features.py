r"""L1-L4 决策点特征提取（Win 运行，输出 E:\hokdata\decision_features.npz）。

L1 英雄/技能共性：技能强度、攻击强度（roster 未知时用掩码+0）
L2 对手/己方个体习惯：60s 滑动窗口（移动速度/方差/进攻性/换线/回城/野区距离）
L3 局面特征：位置/推进/路线/资源距离/scoreboard
L4 观测掩码：每个价值维度 observed 0/1 + roster_known

运行:
    E:\py312\python.exe E:\project\hok-decision-ai\scripts\win_build_decision_features.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

HOKDATA_ROOT = Path(os.environ.get("HOKDATA_ROOT", str(Path(__file__).resolve().parents[1].parent / "hokdata")))
DATASET = HOKDATA_ROOT / "dataset"
DATA = PROJECT_ROOT / "data" / "S45" / "normalized"
OUT_NPZ = HOKDATA_ROOT / "decision_features.npz"
OUT_META = HOKDATA_ROOT / "decision_features.json"

HUD_AFFINE = ((3.9619274696916493, 1.3453769972204605e-05), (-1.9928688568260355e-05, 4.038131527190943), (-196.00459141224727, -105.2570867363273))
BLUE_BASE = (25.0, 395.0)
RED_BASE = (397.0, 28.0)
MAP_MAX_DIST = (416.0**2 + 424.0**2) ** 0.5
WINDOW_S = 60.0
LEVEL_MAP = {"1500+": 2, "1800+": 3, "2100+": 4, "2300+": 5, "2500+": 6, "赛事": 7}

FEATURE_NAMES = [
    # 基础
    "time_norm", "level_norm", "camp_is_red",
    # L3 位置
    "self_x_norm", "self_y_norm", "enemy_x_norm", "enemy_y_norm",
    "centroid_dist_norm", "self_spread", "enemy_spread",
    "self_dist_enemy_base_norm", "enemy_dist_self_base_norm",
    "self_in_enemy_half", "self_lane_0", "self_lane_1", "self_lane_2",
    "enemy_lane_0", "enemy_lane_1", "enemy_lane_2",
    "jungle_proximity_self", "jungle_proximity_enemy",
    # L2 60s 习惯窗口
    "self_speed_mean", "enemy_speed_mean",
    "self_move_var", "enemy_move_var",
    "self_aggression", "enemy_aggression",
    "self_lane_switch", "enemy_lane_switch",
    "self_recall_count", "enemy_recall_count",
    # L1 英雄/技能强度（roster 未知时=0）
    "roster_known", "self_skill_power", "enemy_skill_power",
    "self_attack_power", "enemy_attack_power",
    # scoreboard（作者视角）
    "kill_diff_norm", "economy_norm", "tower_norm", "dragon_norm", "hp_diff",
    # L4 观测掩码
    "mask_kill", "mask_economy", "mask_tower", "mask_dragon", "mask_hp",
]


def _to_ref(hero: dict, cs: str) -> tuple[float, float]:
    if cs == "hud_roi":
        a, b, c = HUD_AFFINE
        x = a[0] * hero["x"] + a[1] * hero["y"] + c[0]
        y = b[0] * hero["x"] + b[1] * hero["y"] + c[1]
        return x, y
    return hero.get("x_ref", hero["x"]), hero.get("y_ref", hero["y"])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_resource_points() -> list[tuple[float, float]]:
    map_data = json.loads((DATA / "map.json").read_text(encoding="utf-8"))
    points = []
    for rp in map_data["resource_points"]:
        if rp.get("position") and rp.get("kind") in {"buff", "camp", "neutral"}:
            points.append(tuple(rp["position"]))
    return points


def _load_skill_power() -> dict[int, float]:
    payload = json.loads((DATA / "skills_vector.json").read_text(encoding="utf-8"))
    power = {}
    for row in payload["data"]:
        ename = int(row["ename"])
        dmg = row["damage"].get("max") or 0.0
        eff = sum(row["effect_vector"])
        power[ename] = 0.5 + dmg / 1200.0 + min(1.0, eff / 10.0)
    return power


def _load_attack_power() -> dict[int, float]:
    payload = json.loads((DATA / "hero_stats.json").read_text(encoding="utf-8"))
    return {
        int(row["ename"]): (float(row.get("base_physical_attack") or 0) + float(row.get("base_magic_attack") or 0)) / 400.0
        for row in payload["data"]
    }


def _hero_stats(frame: dict, side: str, skill_power: dict, attack_power: dict) -> tuple[float, float, float]:
    """返回 (skill, attack, known)。roster 未知→0。"""
    roster = frame.get("roster") or {}
    heroes = roster.get("heroes") or roster.get("list") or []
    if isinstance(heroes, dict):
        heroes = list(heroes.values())
    enames = [
        int(h.get("hero_ename") or h.get("ename") or 0)
        for h in heroes
        if str(h.get("side", "")) == side and (h.get("hero_ename") or h.get("ename"))
    ]
    if not enames:
        return 0.0, 0.0, 0.0
    return _mean([skill_power.get(e, 0.0) for e in enames]), _mean([attack_power.get(e, 0.0) for e in enames]), 1.0


def _centroid(heroes: list[dict], side: str, cs: str) -> tuple[float, float] | None:
    pts = [_to_ref(h, cs) for h in heroes if h.get("side") == side]
    if not pts:
        return None
    return _mean([p[0] for p in pts]), _mean([p[1] for p in pts])


def _lane(y: float) -> int:
    if y < 140:
        return 0
    if y < 280:
        return 1
    return 2


def _features_for_frame(
    frame: dict,
    window: list[dict],
    resources: list[tuple[float, float]],
    skill_power: dict[int, float],
    attack_power: dict[int, float],
) -> list[float]:
    cs = frame.get("coordinate_system", "hud_roi")
    heroes = frame.get("heroes", [])
    self_c = _centroid(heroes, "self", cs) or BLUE_BASE
    enemy_c = _centroid(heroes, "enemy", cs) or RED_BASE
    t = float(frame.get("game_time_s") or 0.0)
    camp_red = 1.0 if str(frame.get("author_camp", "")) == "red" else 0.0

    f: list[float] = []
    f += [min(1.0, t / 1500.0), LEVEL_MAP.get(str(frame.get("level_id", "")), 3) / 7.0, camp_red]
    sx, sy = self_c
    ex, ey = enemy_c
    f += [sx / 416.0, sy / 424.0, ex / 416.0, ey / 424.0]
    f += [((sx - ex) ** 2 + (sy - ey) ** 2) ** 0.5 / MAP_MAX_DIST]

    self_pts = [_to_ref(h, cs) for h in heroes if h.get("side") == "self"]
    enemy_pts = [_to_ref(h, cs) for h in heroes if h.get("side") == "enemy"]
    self_spread = _mean([((px - sx) ** 2 + (py - sy) ** 2) ** 0.5 for px, py in self_pts]) / MAP_MAX_DIST if self_pts else 0.0
    enemy_spread = _mean([((px - ex) ** 2 + (py - ey) ** 2) ** 0.5 for px, py in enemy_pts]) / MAP_MAX_DIST if enemy_pts else 0.0
    f += [self_spread, enemy_spread]
    f += [((sx - RED_BASE[0]) ** 2 + (sy - RED_BASE[1]) ** 2) ** 0.5 / MAP_MAX_DIST,
          ((ex - BLUE_BASE[0]) ** 2 + (ey - BLUE_BASE[1]) ** 2) ** 0.5 / MAP_MAX_DIST]
    f += [1.0 if sx > 208.0 else 0.0]
    self_lanes = [0, 0, 0]
    enemy_lanes = [0, 0, 0]
    for px, py in self_pts:
        self_lanes[_lane(py)] += 1
    for px, py in enemy_pts:
        enemy_lanes[_lane(py)] += 1
    f += [v / max(1, len(self_pts)) for v in self_lanes]
    f += [v / max(1, len(enemy_pts)) for v in enemy_lanes]
    if resources:
        f += [min((((sx - px) ** 2 + (sy - py) ** 2) ** 0.5 for px, py in resources)) / MAP_MAX_DIST,
              min((((ex - px) ** 2 + (ey - py) ** 2) ** 0.5 for px, py in resources)) / MAP_MAX_DIST]
    else:
        f += [0.5, 0.5]

    # L2 60s 习惯窗口
    w_self = [(_centroid(fr.get("heroes", []), "self", fr.get("coordinate_system", "hud_roi")) or self_c) for fr in window]
    w_enemy = [(_centroid(fr.get("heroes", []), "enemy", fr.get("coordinate_system", "hud_roi")) or enemy_c) for fr in window]
    self_speed = _mean([((w_self[i + 1][0] - w_self[i][0]) ** 2 + (w_self[i + 1][1] - w_self[i][1]) ** 2) ** 0.5 for i in range(len(w_self) - 1)]) / 50.0 if len(w_self) > 1 else 0.0
    enemy_speed = _mean([((w_enemy[i + 1][0] - w_enemy[i][0]) ** 2 + (w_enemy[i + 1][1] - w_enemy[i][1]) ** 2) ** 0.5 for i in range(len(w_enemy) - 1)]) / 50.0 if len(w_enemy) > 1 else 0.0
    self_var = np.var([p[0] for p in w_self]) / 10000.0 if len(w_self) > 1 else 0.0
    enemy_var = np.var([p[0] for p in w_enemy]) / 10000.0 if len(w_enemy) > 1 else 0.0
    self_aggr = _mean([1.0 if p[0] > 208.0 else 0.0 for p in w_self])
    enemy_aggr = _mean([1.0 if p[0] < 208.0 else 0.0 for p in w_enemy])
    self_switch = sum(1 for i in range(len(w_self) - 1) if _lane(w_self[i][1]) != _lane(w_self[i + 1][1]))
    enemy_switch = sum(1 for i in range(len(w_enemy) - 1) if _lane(w_enemy[i][1]) != _lane(w_enemy[i + 1][1]))
    self_recall = sum(1 for p in w_self if ((p[0] - BLUE_BASE[0]) ** 2 + (p[1] - BLUE_BASE[1]) ** 2) ** 0.5 < 80)
    enemy_recall = sum(1 for p in w_enemy if ((p[0] - RED_BASE[0]) ** 2 + (p[1] - RED_BASE[1]) ** 2) ** 0.5 < 80)
    f += [self_speed, enemy_speed, self_var, enemy_var, self_aggr, enemy_aggr,
          self_switch / 12.0, enemy_switch / 12.0, self_recall / 12.0, enemy_recall / 12.0]

    # L1 英雄/技能
    sk, at, known = _hero_stats(frame, "self", skill_power, attack_power)
    ek, ea, eknown = _hero_stats(frame, "enemy", skill_power, attack_power)
    f += [float(min(known, eknown)), sk, ek, at, ea]

    # scoreboard + L4 掩码
    camp = str(frame.get("author_camp", "blue"))
    bs = (frame.get("battle_state") or {}).get("base") or {}
    kb = bs.get("kills_blue")
    kr = bs.get("kills_red")
    if camp == "red":
        kb, kr = kr, kb
    kill = ((kb - kr) / 10.0) if isinstance(kb, (int, float)) and isinstance(kr, (int, float)) else 0.0
    gold = (bs.get("gold") / 10000.0) if isinstance(bs.get("gold"), (int, float)) else 0.0
    tower = (bs.get("towers") / 10.0) if isinstance(bs.get("towers"), (int, float)) else 0.0
    dragon = (bs.get("dragons") / 5.0) if isinstance(bs.get("dragons"), (int, float)) else 0.0
    ally_hp = [v for v in (frame.get("ally_hp") or []) if isinstance(v, (int, float))]
    enemy_hp = [v for v in (frame.get("enemy_hp") or []) if isinstance(v, (int, float))]
    hp = (_mean(ally_hp) - _mean(enemy_hp)) if ally_hp and enemy_hp else 0.0
    f += [kill, gold, tower, dragon, hp]
    f += [1.0 if isinstance(kb, (int, float)) and isinstance(kr, (int, float)) else 0.0,
          1.0 if isinstance(bs.get("gold"), (int, float)) else 0.0,
          1.0 if isinstance(bs.get("towers"), (int, float)) else 0.0,
          1.0 if isinstance(bs.get("dragons"), (int, float)) else 0.0,
          1.0 if (ally_hp and enemy_hp) else 0.0]
    return f


def main() -> int:
    print("loading catalogs...")
    resources = _load_resource_points()
    skill_power = _load_skill_power()
    attack_power = _load_attack_power()

    frames_by_video: dict[str, list[dict]] = defaultdict(list)
    for line in (DATASET / "dataset_final_v4.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            continue
        frames_by_video[str(frame.get("video_bv"))].append(frame)

    rows: list[list[float]] = []
    meta_rows: list[dict] = []
    for frames in frames_by_video.values():
        frames.sort(key=lambda fr: float(fr.get("game_time_s") or 0.0))
        for i, frame in enumerate(frames):
            t = float(frame.get("game_time_s") or 0.0)
            window = [fr for fr in frames[max(0, i - 12) : i] if t - float(fr.get("game_time_s") or 0.0) <= WINDOW_S]
            vec = _features_for_frame(frame, window, resources, skill_power, attack_power)
            rows.append(vec)
            meta_rows.append({
                "video_bv": frame.get("video_bv"),
                "frame_id": frame.get("frame_id"),
                "game_time_s": t,
                "level_id": frame.get("level_id"),
                "camp": frame.get("author_camp"),
            })

    matrix = np.asarray(rows, dtype=np.float32)
    np.savez_compressed(OUT_NPZ, features=matrix)
    OUT_META.write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "count": len(rows),
                "frames": meta_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {matrix.shape} -> {OUT_NPZ}")
    print("feature_names:", FEATURE_NAMES)
    print("BUILD_DECISION_FEATURES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
