r"""合并 KPL 两套公开数据为统一 composition 数据集（职业档 5v5 人类对局）。

输入：
  <hokdata>/external/kpl_sql/battles.jsonl   （2019-2024，pick+选手数据+胜负，无 ban）
  <hokdata>/external/kpl/battles.jsonl       （2026，BP+选手数据+胜负+VOD）

输出：
  <hokdata>/external/kpl/composition.jsonl
  <hokdata>/external/kpl/composition_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _sql_row(row: dict) -> dict:
    blue = row.get("blue_players") or []
    red = row.get("red_players") or []
    return {
        "battle_id": row.get("battle_id"),
        "source": "kpl_sql_2019_2024",
        "year": row.get("year"),
        "league_id": row.get("league_id"),
        "match_id": row.get("match_id"),
        "duration_s": None,
        "blue_win": bool(row.get("win_camp") == 1),
        "blue_picks": row.get("blue_picks") or [],
        "red_picks": row.get("red_picks") or [],
        "blue_bans": [],
        "red_bans": [],
        "blue_gold": sum(p.get("gold") or 0 for p in blue),
        "red_gold": sum(p.get("gold") or 0 for p in red),
        "blue_kills": sum(p.get("kills") or 0 for p in blue),
        "red_kills": sum(p.get("kills") or 0 for p in red),
        "level_id": 7,
    }


def _api_row(row: dict) -> dict:
    bp = row.get("bp") or []
    def side(camp: int, action: str) -> list[int]:
        return [x["hero_id"] for x in bp if x.get("camp") == camp and x.get("action") == action and x.get("hero_id")]
    teams = row.get("teams") or {}
    blue = teams.get("1") or {}
    red = teams.get("2") or {}
    return {
        "battle_id": row.get("battle_id"),
        "source": "kpl_api_2025_2026",
        "year": int(str(row.get("league_id", ""))[:4]) if row.get("league_id") else None,
        "league_id": row.get("league_id"),
        "match_id": row.get("match_id"),
        "duration_s": row.get("duration_s"),
        "blue_win": bool(row.get("win_camp") == 1),
        "blue_picks": side(1, "pick"),
        "red_picks": side(2, "pick"),
        "blue_bans": side(1, "ban"),
        "red_bans": side(2, "ban"),
        "blue_gold": blue.get("gold"),
        "red_gold": red.get("gold"),
        "blue_kills": blue.get("kills"),
        "red_kills": red.get("kills"),
        "level_id": 7,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2] / "hokdata"))
    args = parser.parse_args()
    base = Path(args.root) / "external" / "kpl"
    sql_path = Path(args.root) / "external" / "kpl_sql" / "battles.jsonl"
    api_path = base / "battles.jsonl"

    rows = []
    if sql_path.exists():
        rows.extend(_sql_row(json.loads(line)) for line in sql_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if api_path.exists():
        rows.extend(_api_row(json.loads(line)) for line in api_path.read_text(encoding="utf-8").splitlines() if line.strip())

    out = base / "composition.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "battles": len(rows),
        "sources": dict(Counter(r["source"] for r in rows)),
        "years": dict(sorted(Counter(r["year"] for r in rows).items())),
        "blue_win_rate": round(sum(1 for r in rows if r["blue_win"]) / max(1, len(rows)), 4),
        "with_bans": sum(1 for r in rows if r["blue_bans"] or r["red_bans"]),
        "with_duration": sum(1 for r in rows if r["duration_s"]),
        "note": "KPL professional 5v5 human post-game: picks/bans/winner/stats; no positions or action stream",
    }
    (base / "composition_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
