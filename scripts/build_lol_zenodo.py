r"""把 Zenodo LoL 对局 CSV 归一化为 JSONL，并计算 scoreboard->胜负的 AUC。

来源：https://zenodo.org/records/8303397 （CC BY 4.0，64,556 局，20%/100% 两个快照）
输出：<hokdata>/external/lol_zenodo/matches.jsonl + summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


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
        avg_rank = (i + j) / 2 + 1
        rank_sum += avg_rank * sum(label for _, label in pairs[i : j + 1])
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _row(rec: dict) -> dict:
    blue = {
        "kills": _num(rec["blueChampionKill"]),
        "gold": _num(rec["blueTotalGold"]),
        "towers": _num(rec["blueTowerKill"]),
        "inhibitors": _num(rec["blueInhibitorKill"]),
        "dragons": _num(rec["blueDragonKill"]),
        "barons": _num(rec["blueBaronKill"]),
        "cs": _num(rec["blueMinionsKilled"]) + _num(rec["blueJungleMinionsKilled"]),
        "level": _num(rec["blueAvgPlayerLevel"]),
    }
    red = {
        "kills": _num(rec["redChampionKill"]),
        "gold": _num(rec["redTotalGold"]),
        "towers": _num(rec["redTowerKill"]),
        "inhibitors": _num(rec["redInhibitorKill"]),
        "dragons": _num(rec["redDragonKill"]),
        "barons": _num(rec["redBaronKill"]),
        "cs": _num(rec["redMinionsKilled"]) + _num(rec["redJungleMinionsKilled"]),
        "level": _num(rec["redAvgPlayerLevel"]),
    }
    return {
        "match_id": rec["matchID"],
        "time_percent": int(_num(rec["timePercent"])),
        "full_time_ms": int(_num(rec["fullTimeMS"])),
        "blue": blue,
        "red": red,
        "blue_win": str(rec["blueWin"]).strip().lower() == "true",
        "source": "zenodo_lol_8303397",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2] / "hokdata"))
    args = parser.parse_args()
    base = Path(args.root) / "external" / "lol_zenodo"
    files = sorted(base.glob("full_data_*.csv"))
    if not files:
        print(f"no csv in {base}")
        return 1

    rows = []
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                if rec.get("matchID"):
                    rows.append(_row(rec))

    out = base / "matches.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    features = ("gold", "kills", "towers", "inhibitors", "dragons", "barons", "cs", "level")
    by_file = defaultdict(list)
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                if rec.get("matchID"):
                    by_file[path.stem].append(_row(rec))

    auc_report = {}
    for name, subset in by_file.items():
        labels = [1 if r["blue_win"] else 0 for r in subset]
        auc_report[name] = {
            "rows": len(subset),
            "matches": len({r["match_id"] for r in subset}),
            "blue_win_rate": round(sum(labels) / len(labels), 4),
            "auc": {
                feature: round(_auc([r["blue"][feature] - r["red"][feature] for r in subset], labels), 4)
                for feature in features
            },
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "https://zenodo.org/records/8303397",
        "license": "CC BY 4.0",
        "rows": len(rows),
        "matches": len({r["match_id"] for r in rows}),
        "files": {p.stem: sum(1 for _ in p.open(encoding='utf-8')) - 1 for p in files},
        "time_percent_hist": dict(sorted(Counter(r["time_percent"] for r in rows).items())),
        "auc_by_snapshot": auc_report,
        "note": "团队级 scoreboard；无逐玩家位置/动作；用于胜负/价值模型预训练或跨游戏验证",
    }
    (base / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rows={len(rows)} matches={summary['matches']} -> {out}")
    print("AUC:", json.dumps({k: v["auc"] for k, v in auc_report.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
