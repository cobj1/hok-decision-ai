r"""把 Kaggle LoL 逐分钟 timeline 抽成价值模型快照（5/10/15 分钟）。

来源：https://www.kaggle.com/datasets/nathansmallcalder/league-of-legends-ranked-post-match-and-timeline
  MatchTimelineTbl.csv: 单个玩家视角的逐分钟位置/经济/目标/胜负
  TeamMatchTbl.csv: 双方阵容与团队目标
  MatchTbl.csv: patch/时长/段位

输出：
  <hokdata>/external/lol_kaggle/matches.jsonl
  <hokdata>/external/lol_kaggle/snapshot_meta.jsonl
  <hokdata>/external/lol_kaggle/snapshots.npz
  <hokdata>/external/lol_kaggle/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FEATURES = [
    "minute", "position_x", "position_y", "total_gold", "current_gold", "cs", "jungle_cs", "xp", "level",
    "damage_to_champs", "damage_taken", "kills", "deaths", "assists",
    "team_kills", "team_towers", "team_inhibitors", "team_dragons", "team_barons",
    "enemy_kills", "enemy_towers", "enemy_inhibitors", "enemy_dragons", "enemy_barons",
    "gold_diff", "xp_diff", "team_gold_diff",
]
MINUTES = (5, 10, 15)


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except ValueError:
        return 0.0


def _auc(scores: list[float], labels: list[int]) -> float | None:
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        rank_sum += ((i + j) / 2 + 1) * sum(y for _, y in pairs[i : j + 1])
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2] / "hokdata"))
    args = parser.parse_args()
    base = Path(args.root) / "external" / "lol_kaggle"
    if not (base / "MatchTimelineTbl.csv").exists():
        print(f"missing CSVs in {base}")
        return 1

    matches = {}
    with (base / "MatchTbl.csv").open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            matches[row["MatchId"]] = {
                "match_id": row["MatchId"],
                "patch": row.get("Patch"),
                "queue_type": row.get("QueueType"),
                "rank_fk": row.get("RankFk"),
                "duration_s": _f(row, "GameDuration"),
            }

    teams = {}
    with (base / "TeamMatchTbl.csv").open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            match_id = row["MatchFk"]
            entry = teams.setdefault(match_id, {})
            entry["blue_champs"] = [int(row[f"B{i}Champ"]) for i in range(1, 6) if row.get(f"B{i}Champ")]
            entry["red_champs"] = [int(row[f"R{i}Champ"]) for i in range(1, 6) if row.get(f"R{i}Champ")]
            entry["blue_win"] = str(row.get("BlueWin", "")).strip() == "1"
            entry["red_win"] = str(row.get("RedWin", "")).strip() == "1"
            entry["blue_kills"] = _f(row, "BlueKills")
            entry["red_kills"] = _f(row, "RedKills")
            entry["blue_towers"] = _f(row, "BlueTowerKills")
            entry["red_towers"] = _f(row, "RedTowerKills")

    summoners = {}
    with (base / "SummonerMatchTbl.csv").open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            summoners[row["SummonerMatchId"]] = {
                "match_id": row["MatchFk"],
                "champion_fk": int(row["ChampionFk"]) if row.get("ChampionFk") else None,
            }

    with (base / "matches.jsonl").open("w", encoding="utf-8") as fh:
        for match_id, meta in matches.items():
            team = teams.get(match_id, {})
            fh.write(json.dumps({**meta, **team}, ensure_ascii=False) + "\n")

    feature_rows = []
    labels = []
    minutes = []
    meta_out = []
    counts = Counter()
    per_minute_labels: dict[int, list[int]] = {m: [] for m in MINUTES}
    per_minute_gold: dict[int, list[float]] = {m: [] for m in MINUTES}
    per_minute_teamgold: dict[int, list[float]] = {m: [] for m in MINUTES}

    with (base / "MatchTimelineTbl.csv").open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            minute = int(_f(row, "Minute"))
            if minute not in MINUTES:
                continue
            sid = row["SummonerMatchFk"]
            sm = summoners.get(sid)
            if not sm:
                continue
            match_id = sm["match_id"]
            team = teams.get(match_id, {})
            label = int(_f(row, "Win"))
            feature = [
                minute, _f(row, "PositionX"), _f(row, "PositionY"), _f(row, "TotalGold"), _f(row, "CurrentGold"),
                _f(row, "MinionsKilled"), _f(row, "JungleMinionsKilled"), _f(row, "Xp"), _f(row, "Level"),
                _f(row, "DmgDealtToChampions"), _f(row, "DmgTaken"), _f(row, "Kills"), _f(row, "Deaths"),
                _f(row, "Assists"), _f(row, "TeamKills"), _f(row, "TeamTowers"), _f(row, "TeamInhibitors"),
                _f(row, "TeamDragons"), _f(row, "TeamBarons"), _f(row, "EnemyKills"), _f(row, "EnemyTowers"),
                _f(row, "EnemyInhibitors"), _f(row, "EnemyDragons"), _f(row, "EnemyBarons"),
                _f(row, "GoldDiff"), _f(row, "XpDiff"), _f(row, "TeamGoldDiff"),
            ]
            feature_rows.append(feature)
            labels.append(label)
            minutes.append(minute)
            counts[minute] += 1
            per_minute_labels[minute].append(label)
            per_minute_gold[minute].append(feature[24])
            per_minute_teamgold[minute].append(feature[26])
            meta_out.append(
                {
                    "match_id": match_id,
                    "summoner_match_id": sid,
                    "champion_fk": sm["champion_fk"],
                    "minute": minute,
                    "blue_champs": team.get("blue_champs"),
                    "red_champs": team.get("red_champs"),
                    "blue_win": team.get("blue_win"),
                    "patch": matches.get(match_id, {}).get("patch"),
                    "rank_fk": matches.get(match_id, {}).get("rank_fk"),
                    "duration_s": matches.get(match_id, {}).get("duration_s"),
                }
            )

    x = np.asarray(feature_rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8)
    m = np.asarray(minutes, dtype=np.int16)
    np.savez_compressed(base / "snapshots.npz", features=x, labels=y, minutes=m, feature_names=np.asarray(FEATURES))
    with (base / "snapshot_meta.jsonl").open("w", encoding="utf-8") as fh:
        for row in meta_out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    auc = {}
    for minute in MINUTES:
        auc[str(minute)] = {
            "rows": counts[minute],
            "win_rate": round(sum(per_minute_labels[minute]) / max(1, len(per_minute_labels[minute])), 4),
            "gold_diff_auc": round(_auc(per_minute_gold[minute], per_minute_labels[minute]), 4),
            "team_gold_diff_auc": round(_auc(per_minute_teamgold[minute], per_minute_labels[minute]), 4),
        }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "https://www.kaggle.com/datasets/nathansmallcalder/league-of-legends-ranked-post-match-and-timeline",
        "license": "MIT",
        "matches": len(matches),
        "snapshots": int(x.shape[0]),
        "players_tracked": len({r["summoner_match_id"] for r in meta_out}),
        "features": len(FEATURES),
        "minutes": dict(counts),
        "auc": auc,
        "note": "single-player perspective per-minute trajectory (position/gold/xp/items/objectives) + win; not full 10-player positions",
    }
    (base / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
