r"""解析 HiderX/ali 的 KPL SQL 数据，归一化为 5v5 人类对局 JSONL。

来源：https://github.com/HiderX/ali （data/SQLinsert）
覆盖：22 个联赛、1,494 场、6,044 局、60,440 条选手记录。
字段：camp/hero/summoner/KDA/金币/伤害/参团/MVP/胜负；无位置、无动作、无 BP ban。

输出：
  <hokdata>/external/kpl_sql/battles.jsonl
  <hokdata>/external/kpl_sql/players.jsonl
  <hokdata>/external/kpl_sql/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _convert(value: str):
    v = value.strip()
    if v.upper() == "NULL":
        return None
    if v.upper() == "TRUE":
        return True
    if v.upper() == "FALSE":
        return False
    if len(v) >= 2 and v[0] == v[-1] == "'":
        return v[1:-1]
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def parse_inserts(path: Path, table: str) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        rf"INSERT\s+INTO\s+{table}\s*\((.*?)\)\s*VALUES\s*\((.*?)\);",
        re.S | re.I,
    )
    rows = []
    for match in pattern.finditer(text):
        cols = [c.strip() for c in match.group(1).split(",")]
        raw_values = match.group(2).replace("\r", " ").replace("\n", " ")
        try:
            values = next(csv.reader([raw_values], skipinitialspace=True))
        except (csv.Error, StopIteration):
            continue
        if len(values) != len(cols):
            continue
        rows.append({col: _convert(val) for col, val in zip(cols, values)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(Path(__file__).resolve().parents[2] / "hokdata" / "external" / "kpl_sql" / "raw"))
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2] / "hokdata"))
    args = parser.parse_args()

    src = Path(args.source)
    base = Path(args.root) / "external" / "kpl_sql"
    base.mkdir(parents=True, exist_ok=True)

    battles = parse_inserts(src / "battle.sql", "Battle")
    belongs = {r["battle_id"]: r["match_id"] for r in parse_inserts(src / "battle_belongs.sql", "battle_belongs")}
    matches = {r["match_id"]: r for r in parse_inserts(src / "matches.sql", "matches")}
    match_league = {r["match_id"]: r["league_id"] for r in parse_inserts(src / "match_belongs.sql", "match_belongs")}
    leagues = {r["league_id"]: r for r in parse_inserts(src / "league.sql", "League")}
    players = parse_inserts(src / "a_player_plays_a_game.sql", "a_player_plays_a_game")

    players_by_battle: dict[str, list[dict]] = defaultdict(list)
    for row in players:
        players_by_battle[str(row.get("battle_id"))].append(row)

    out_battles = []
    out_players = []
    player_counts = Counter()
    for battle in battles:
        battle_id = str(battle.get("battle_id"))
        match_id = belongs.get(battle_id)
        league_id = match_league.get(match_id)
        league = leagues.get(league_id, {})
        match = matches.get(match_id, {})
        rows = players_by_battle.get(battle_id, [])
        player_counts[len(rows)] += 1
        blue = [p for p in rows if p.get("camp") == 1]
        red = [p for p in rows if p.get("camp") == 2]

        def row_out(p: dict) -> dict:
            return {
                "team_id": p.get("team_id"),
                "player_name": p.get("player_name"),
                "camp": p.get("camp"),
                "hero_id": p.get("hero_id"),
                "summoner_id": p.get("summoner_id"),
                "is_win": bool(p.get("is_win")),
                "is_mvp": bool(p.get("is_mvp")),
                "is_lose_mvp": bool(p.get("is_lose_mvp")),
                "kills": p.get("kill_num"),
                "deaths": p.get("death_num"),
                "assists": p.get("assist_num"),
                "kda": p.get("kda"),
                "gold": p.get("gold"),
                "damage_total": p.get("hurt_total"),
                "damage_taken": p.get("be_hurt_total"),
                "damage_to_hero": p.get("hurt_to_hero_total"),
                "damage_taken_by_hero": p.get("be_hurt_by_hero_total"),
                "participation_rate": p.get("participation_rate"),
                "mvp_score": p.get("mvp_score"),
            }

        blue_out = [row_out(p) for p in blue]
        red_out = [row_out(p) for p in red]
        out_battles.append(
            {
                "battle_id": battle_id,
                "battle_seq": battle.get("battle_seq"),
                "match_id": match_id,
                "league_id": league_id,
                "league_name": league.get("league_name"),
                "year": league.get("year"),
                "start_time": match.get("start_time"),
                "win_camp": battle.get("win_camp"),
                "blue_win": battle.get("win_camp") == 1,
                "blue_picks": [p["hero_id"] for p in blue_out],
                "red_picks": [p["hero_id"] for p in red_out],
                "blue_players": blue_out,
                "red_players": red_out,
                "source": "hiderx_ali_kpl_sql",
            }
        )
        for p in rows:
            out_players.append(
                {
                    "battle_id": battle_id,
                    "match_id": match_id,
                    "league_id": league_id,
                    "year": league.get("year"),
                    **row_out(p),
                }
            )

    with (base / "battles.jsonl").open("w", encoding="utf-8") as fh:
        for row in out_battles:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (base / "players.jsonl").open("w", encoding="utf-8") as fh:
        for row in out_players:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    hero_ids = {p["hero_id"] for p in out_players if p.get("hero_id") is not None}
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "https://github.com/HiderX/ali (data/SQLinsert)",
        "battles": len(out_battles),
        "players": len(out_players),
        "matches": len({b["match_id"] for b in out_battles}),
        "leagues": len({b["league_id"] for b in out_battles}),
        "years": dict(sorted(Counter(b["year"] for b in out_battles).items())),
        "win_balance": dict(Counter(b["win_camp"] for b in out_battles)),
        "players_per_battle": dict(player_counts),
        "heroes": len(hero_ids),
        "teams": len({p["team_id"] for p in out_players if p.get("team_id") is not None}),
        "note": "post-game 5v5 human KPL data; picks + player stats + winner; no positions/actions/BP bans",
    }
    (base / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"battles={len(out_battles)} players={len(out_players)} matches={summary['matches']} years={summary['years']}")
    print("players_per_battle", dict(player_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
