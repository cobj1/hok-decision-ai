r"""抓取 KPL 官方公开比赛数据（5v5 人类对局，post-game）。

接口（pvp.qq.com 同源）：
  /leaguesite/matches/open?league_id=...
  /leaguesite/match/battles/open?match_id=...
  /leaguesite/battle/open?battle_id=...

输出：
  <hokdata>/external/kpl/raw/league_<id>/{match,battle}_*.json
  <hokdata>/external/kpl/battles.jsonl
  <hokdata>/external/kpl/summary.json

用法：
  python scripts/fetch_kpl_battles.py --league-ids 20260001 --max-matches 20
  python scripts/fetch_kpl_battles.py --league-ids 20260001
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://prod.comp.smoba.qq.com/leaguesite"
HEADERS = {
    "accept": "application/json, text/plain, */*",
    "origin": "https://pvp.qq.com",
    "referer": "https://pvp.qq.com/",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
    ),
}
SLEEP_S = 0.25


def _get(path: str, params: dict) -> dict | None:
    try:
        resp = requests.get(f"{BASE}/{path}", params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("code") == 200:
            return payload
    except (requests.RequestException, ValueError):
        pass
    return None


def _team(row: dict) -> dict:
    return {
        "team_id": row.get("team_id"),
        "team_name": row.get("team_name"),
        "team_abbreviation": row.get("team_abbreviation"),
        "win": bool(row.get("is_win")),
        "kills": row.get("kill_num"),
        "deaths": row.get("death_num"),
        "assists": row.get("assist_num"),
        "gold": row.get("gold"),
        "towers": row.get("push_tower_num"),
        "dragons": {
            "big": row.get("kill_big_dragon_num"),
            "tyrant": row.get("kill_tyrant_num"),
            "dark_tyrant": row.get("kill_dark_tyrant_num"),
            "shadow": row.get("kill_shadow_dragon_num"),
            "prophet": row.get("kill_prophet_dragon_num"),
            "storm_king": row.get("kill_storm_dragon_king_num"),
        },
    }


def _player(row: dict) -> dict:
    summoner = row.get("SummonerAbilityInfo") or {}
    equips = [e.get("equip_id") for e in (row.get("BriefEquipList") or []) if e.get("equip_id")]
    return {
        "camp": row.get("camp"),
        "team_id": row.get("team_id"),
        "player_name": row.get("actual_player_name") or row.get("player_name"),
        "position": row.get("position"),
        "position_desc": row.get("position_desc"),
        "hero_id": row.get("hero_id"),
        "hero_name": row.get("hero_name"),
        "kills": row.get("kill_num"),
        "deaths": row.get("death_num"),
        "assists": row.get("assist_num"),
        "gold": row.get("gold"),
        "damage_to_hero": row.get("hurt_to_hero_total"),
        "damage_taken": row.get("be_hurt_by_hero_total"),
        "participation_rate": row.get("participation_rate"),
        "mvp": bool(row.get("is_mvp")),
        "mvp_score": row.get("mvp_score"),
        "summoner_id": summoner.get("summoner_ability_id"),
        "equips": equips,
    }


def _normalize(league_id: str, match_id: str, data: dict) -> dict:
    bp = []
    for row in data.get("bp_list") or []:
        bp.append(
            {
                "camp": row.get("camp"),
                "action": "pick" if row.get("is_ban_or_pick") == 1 else "ban",
                "hero_id": row.get("hero_id"),
                "hero_name": row.get("hero_name"),
                "position": row.get("position"),
            }
        )
    videos = [v.get("video_url") for v in (data.get("video_list") or []) if v.get("video_url")]
    return {
        "battle_id": data.get("battle_id"),
        "league_id": league_id,
        "match_id": match_id,
        "battle_seq": data.get("battle_seq"),
        "duration_s": (data.get("game_duration") or 0) / 1000.0,
        "win_camp": data.get("win_camp"),
        "teams": {"1": _team(data.get("camp1") or {}), "2": _team(data.get("camp2") or {})},
        "players": [_player(p) for p in (data.get("battle_player_list") or [])],
        "bp": bp,
        "videos": videos,
        "source": "kpl_official_api",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-ids", nargs="+", default=["20260001"])
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2] / "hokdata"))
    parser.add_argument("--max-matches", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=SLEEP_S)
    args = parser.parse_args()

    base = Path(args.root) / "external" / "kpl"
    raw = base / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    rows = []
    stats = Counter()
    for league_id in args.league_ids:
        league = _get("matches/open", {"league_id": league_id})
        if not league:
            print(f"league {league_id}: fetch failed")
            continue
        matches = league.get("results") or []
        if args.max_matches:
            matches = matches[: args.max_matches]
        print(f"league {league_id}: {len(matches)} matches")
        out_league = raw / f"league_{league_id}"
        out_league.mkdir(parents=True, exist_ok=True)

        for match in matches:
            match_id = str(match.get("match_id"))
            match_path = out_league / f"match_{match_id}.json"
            payload = _get("match/battles/open", {"match_id": match_id})
            if not payload:
                stats["match_failed"] += 1
                continue
            match_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            battles = payload.get("results") or []
            for battle in battles:
                battle_id = str(battle.get("battle_id"))
                battle_path = out_league / f"battle_{battle_id}.json"
                if battle_path.exists() and battle_path.stat().st_size > 0:
                    data = json.loads(battle_path.read_text(encoding="utf-8")).get("data")
                else:
                    time.sleep(args.sleep)
                    payload = _get("battle/open", {"battle_id": battle_id})
                    if not payload:
                        stats["battle_failed"] += 1
                        continue
                    battle_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                    data = payload.get("data")
                if not data:
                    stats["battle_empty"] += 1
                    continue
                rows.append(_normalize(league_id, match_id, data))
                stats["battles"] += 1
            time.sleep(args.sleep)

    if rows:
        # 去重（同一 battle 可能跨 league 重复）
        unique = {r["battle_id"]: r for r in rows}
        rows = list(unique.values())
        out = base / "battles.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        hero_ids = {p["hero_id"] for r in rows for p in r["players"] if p.get("hero_id")}
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "https://pvp.qq.com/matchdata/",
            "leagues": args.league_ids,
            "battles": len(rows),
            "matches": len({r["match_id"] for r in rows}),
            "teams": len({r["teams"][c]["team_id"] for r in rows for c in ("1", "2")}),
            "heroes": len(hero_ids),
            "win_balance": dict(Counter(r["win_camp"] for r in rows)),
            "duration_s": {
                "min": min(r["duration_s"] for r in rows),
                "max": max(r["duration_s"] for r in rows),
            },
            "players_per_battle": dict(Counter(len(r["players"]) for r in rows)),
            "bp_per_battle": dict(Counter(len(r["bp"]) for r in rows)),
            "videos": sum(1 for r in rows if r["videos"]),
            "stats": dict(stats),
            "note": "post-game 5v5 human data: BP/player stats/builds/winner/VOD; no positions or action stream",
        }
        (base / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"battles={len(rows)} matches={summary['matches']} heroes={len(hero_ids)} -> {out}")
        print("stats:", dict(stats))
    else:
        print("no battles fetched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
