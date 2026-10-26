r"""把腾讯 AI Lab Real-Time HoK Dataset（100 局样本）合并为统一帧 JSONL。

来源：https://github.com/yangzelong14/Real-Time-HoK-Dataset
  - 5v5 真实对局，30 秒采样；每局含 wincp/losecp 双方队伍级 scoreboard + 5 英雄数据；
  - 无坐标、无血量、无等级；2019 年旧版本；最后一行为全量重置行（已剔除）。

阵营映射（避免用胜负泄漏特征）：
  - 每局按 game_id md5 决定 wincp 映射到 blue 还是 red；
  - author_camp 另取一位哈希，独立于胜负，保证正负样本均衡。

输出：data/S45/training/merged/tencent_rt_frames.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hok_decision.config import GAME_VERSION, NORMALIZED_ROOT  # noqa: E402

RESET_TOWER_COUNT = 9
RESET_MID_COUNT = 3


def _bits(game_id: str) -> tuple[bool, bool]:
    digest = hashlib.md5(game_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16)
    return bool(value & 1), bool(value & 2)


def _load_games(source: Path) -> list[tuple[str, str, dict]]:
    games = []
    seen: Counter = Counter()
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        game_id, body = line.split(None, 1)
        seen[game_id] += 1
        unique_id = game_id if seen[game_id] == 1 else f"{game_id}#{seen[game_id]}"
        games.append((unique_id, game_id, json.loads(body)))
    return games


def _is_reset_row(row: dict) -> bool:
    return all(
        row[side].get("tower_cnt") == RESET_TOWER_COUNT
        and row[side].get("middleTower_cnt") == RESET_MID_COUNT
        for side in ("wincp", "losecp")
    )


def _objective_series(games: list[tuple[str, str, dict]]) -> dict[tuple[str, str], dict[str, int]]:
    series: dict[tuple[str, str], dict[str, int]] = {}
    for game_id, _, game in games:
        times = sorted(game, key=float)
        if times and _is_reset_row(game[times[-1]]):
            times = times[:-1]
        for cp in ("wincp", "losecp"):
            counts: dict[str, int] = {}
            total = 0
            prev = {"baron": "0", "darkTyrant": "0"}
            for t in times:
                row = game[t][cp]
                for key in ("baron", "darkTyrant"):
                    value = str(row.get(key) or "0")
                    if value == "1" and prev[key] != "1":
                        total += 1
                    prev[key] = value
                counts[t] = total
            series[(game_id, cp)] = counts
    return series


def _team_rows(team: dict, base_side: str) -> list[dict]:
    heroes = team.get("hero") or {}
    rows = []
    for index, (hero_id, stats) in enumerate(heroes.items()):
        if base_side == "self":
            side = "self" if index == 0 else "ally"
        else:
            side = "enemy"
        rows.append(
            {
                "side": side,
                "hero_ename": int(hero_id),
                "profession": stats.get("profession") or [],
                "gold": stats.get("gold"),
                "kills": stats.get("killCnt"),
                "deaths": stats.get("deadCnt"),
                "assists": stats.get("assistCnt"),
            }
        )
    return rows


def _slots(roster: list[dict]) -> list[dict]:
    def empty(side: str) -> dict:
        return {"side": side, "detected": False, "hero_ename": None, "position": None,
                "hp_frac": None, "mp_frac": None, "gold": None}

    slots = [empty("self")] + [empty("ally") for _ in range(4)] + [empty("enemy") for _ in range(5)]
    cursor = {"self": 0, "ally": 1, "enemy": 5}
    for row in roster:
        idx = cursor.get(row["side"])
        if idx is None or idx >= len(slots) or slots[idx]["side"] != row["side"]:
            continue
        slots[idx] = {
            "side": row["side"], "detected": True, "hero_ename": row["hero_ename"],
            "position": None, "hp_frac": None, "mp_frac": None, "gold": row["gold"],
        }
        cursor[row["side"]] = idx + 1
    return slots


def _build_frame(
    game_id: str,
    source_game_id: str,
    time_key: str,
    game: dict,
    *,
    winner_side: str,
    author_camp: str,
    objective_series: dict[tuple[str, str], dict[str, int]],
) -> dict:
    loser_side = "red" if winner_side == "blue" else "blue"
    side_of_cp = {"wincp": winner_side, "losecp": loser_side}
    author_cp = "wincp" if author_camp == winner_side else "losecp"
    enemy_cp = "losecp" if author_cp == "wincp" else "wincp"

    row = game[time_key]
    own = row[author_cp]
    opp = row[enemy_cp]

    own_rows = _team_rows(own, "self")
    opp_rows = _team_rows(opp, "enemy")
    roster = own_rows + opp_rows
    slots = _slots(roster)

    own_kills = own.get("kill") or 0
    opp_kills = opp.get("kill") or 0
    kills = {author_camp: own_kills, ("red" if author_camp == "blue" else "blue"): opp_kills}
    own_towers = own.get("tower_cnt")
    opp_towers = opp.get("tower_cnt")
    towers_diff = (own_towers - opp_towers) if isinstance(own_towers, int) and isinstance(opp_towers, int) else None
    own_gold = own.get("gold")
    opp_gold = opp.get("gold")
    gold_diff = (own_gold - opp_gold) if isinstance(own_gold, (int, float)) and isinstance(opp_gold, (int, float)) else None
    own_dragons = objective_series.get((game_id, author_cp), {}).get(time_key, 0)
    opp_dragons = objective_series.get((game_id, enemy_cp), {}).get(time_key, 0)

    base = {
        "kills_blue": kills.get("blue"),
        "kills_red": kills.get("red"),
        "gold": gold_diff,
        "towers": towers_diff,
        "dragons": own_dragons - opp_dragons,
        "towers_remaining_self": own_towers,
        "towers_remaining_enemy": opp_towers,
        "middle_towers_self": own.get("middleTower_cnt"),
        "middle_towers_enemy": opp.get("middleTower_cnt"),
        "crystal_towers_self": own.get("crystalTower_cnt"),
        "crystal_towers_enemy": opp.get("crystalTower_cnt"),
        "baron_active_self": str(own.get("baron") or "0") == "1",
        "baron_active_enemy": str(opp.get("baron") or "0") == "1",
        "dark_tyrant_active_self": str(own.get("darkTyrant") or "0") == "1",
        "dark_tyrant_active_enemy": str(opp.get("darkTyrant") or "0") == "1",
        "soldier_distance_self": own.get("soldier.distance"),
        "soldier_distance_enemy": opp.get("soldier.distance"),
        "gold_self": own_gold,
        "gold_enemy": opp_gold,
        "dragons_self": own_dragons,
        "dragons_enemy": opp_dragons,
        "participation_self": own.get("participation"),
        "gold_blue": own_gold if author_camp == "blue" else opp_gold,
        "gold_red": opp_gold if author_camp == "blue" else own_gold,
    }

    return {
        "frame_id": f"{game_id}:{float(time_key):.1f}",
        "video_bv": game_id,
        "source_game_id": source_game_id,
        "game_time_s": round(float(time_key) * 60.0, 1),
        "time_source": "tencent_rt_snapshot",
        "author_camp": author_camp,
        "winner_side": winner_side,
        "author_cp": author_cp,
        "cp_side_map": side_of_cp,
        "camp_source": "md5_balanced",
        "level_id": None,
        "coordinate_system": "none",
        "heroes": [],
        "hero_slots": slots,
        "roster": {"heroes": roster, "source": "tencent_rt"},
        "battle_state": {"base": base},
        "ally_hp": [],
        "enemy_hp": [],
        "action": None,
        "frame_quality": "ok",
        "perception_ok": True,
        "phase_hint": row.get("sum"),
        "source": "tencent_rt_hok_sample100",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=str(PROJECT_ROOT / "data" / GAME_VERSION / "raw" / "external" / "tencent_rt_hok_sample_100.json"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "data" / GAME_VERSION / "training" / "merged"),
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"找不到数据文件: {source}")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tencent_rt_frames.jsonl"
    report_path = out_dir / "tencent_rt_report.json"

    games = _load_games(source)
    objectives = _objective_series(games)
    heroes_catalog = {
        int(row["ename"])
        for row in json.loads((NORMALIZED_ROOT / "heroes.json").read_text(encoding="utf-8"))["data"]
    }

    dropped_reset = 0
    frames = 0
    winners = Counter()
    authors = Counter()
    hero_ids = set()
    time_min = 0.0
    time_max = 0.0

    with out_path.open("w", encoding="utf-8") as fh:
        for game_id, source_game_id, game in games:
            times = sorted(game, key=float)
            if times and _is_reset_row(game[times[-1]]):
                times = times[:-1]
                dropped_reset += 1
            blue_is_winner, author_is_winner = _bits(game_id)
            winner_side = "blue" if blue_is_winner else "red"
            author_camp = winner_side if author_is_winner else ("red" if winner_side == "blue" else "blue")
            winners[winner_side] += 1
            authors[author_camp] += 1
            for t in times:
                frame = _build_frame(
                    game_id, source_game_id, t, game,
                    winner_side=winner_side,
                    author_camp=author_camp,
                    objective_series=objectives,
                )
                fh.write(json.dumps(frame, ensure_ascii=False) + "\n")
                frames += 1
                time_max = max(time_max, frame["game_time_s"])
                for row in frame["roster"]["heroes"]:
                    hero_ids.add(row["hero_ename"])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hok_version": GAME_VERSION,
        "source": {
            "file": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "url": "https://github.com/yangzelong14/Real-Time-HoK-Dataset",
            "note": "Tencent AI Lab 100-game sample; no explicit license, research use only",
        },
        "merged": {"games": len(games), "frames": frames, "output": str(out_path)},
        "dropped_reset_rows": dropped_reset,
        "winner_side_balance": dict(winners),
        "author_camp_balance": dict(authors),
        "time_range_s": [time_min, time_max],
        "hero_identity": {
            "unique": len(hero_ids),
            "not_in_s45_catalog": sorted(hero_ids - heroes_catalog),
        },
        "field_coverage_pct": {
            "hero_identity": 100.0,
            "scoreboard_gold": 100.0,
            "scoreboard_kills": 100.0,
            "scoreboard_towers": 100.0,
            "objectives": 100.0,
            "game_time": 100.0,
            "positions": 0.0,
            "hp": 0.0,
            "human_action": 0.0,
        },
        "gaps": [
            "无坐标：heroes/hero_slots 只有身份与经济，position=null",
            "无血量/蓝量/等级：ally_hp/enemy_hp 为空",
            "30 秒采样：只能做 Δ=30s 的收益标签，细粒度动作不可得",
            "2019 旧版本：英雄/装备数值与 S45 不一致，仅适合宏观胜负/经济建模",
            "towers/gold/dragons 均为作者视角差值（正=己方优势）；towers_remaining_* 保留原始剩余塔数",
            "baron/darkTyrant 为增益状态位，dragons=0→1 跳变累计值，非官方龙计数",
            "winner/author 阵营由 md5 平衡分配，不代表真实蓝红方",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已输出: {out_path} ({frames} 帧, {len(games)} 局)")
    print(f"报告: {report_path}")
    print("胜负平衡:", dict(winners), "视角平衡:", dict(authors))
    print("剔除重置行局数:", dropped_reset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
