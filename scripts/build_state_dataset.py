"""把人工/录屏识别的状态帧转成预训练 token 语料。

输入 JSON 格式（逐帧）：
{
  "frames": [
    {
      "frame_id": "20260829_001",
      "game_time_s": 600.0,
      "heroes": [
        {"hero_ename": 105, "side": "self",  "hp_frac": 0.8, "mp_frac": 0.6,
         "gold": 3200.0, "position": [52.0, 350.0]},
        ... 共 10 名：self×1, ally×4, enemy×5
      ]
    }
  ]
}

输出：
  data/S45/training/state_tokens.json
  每帧: {"frame_id": ..., "game_time_s": ..., "tokens": [...]}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from hok_decision.config import GAME_VERSION, NORMALIZED_ROOT
from hok_decision.model.state_tokenizer import tokenize_state

TRAINING_DIR = NORMALIZED_ROOT.parent / "training"


def build_state_tokens(frames: list[dict]) -> list[dict]:
    rows = []
    for frame in frames:
        heroes = frame["heroes"]
        sides = [str(h["side"]) for h in heroes]
        assert sides.count("self") == 1, f"self 必须 1 名: {frame.get('frame_id')}"
        assert sides.count("ally") == 4, f"ally 必须 4 名: {frame.get('frame_id')}"
        assert sides.count("enemy") == 5, f"enemy 必须 5 名: {frame.get('frame_id')}"
        tokens = tokenize_state(
            game_time_s=float(frame["game_time_s"]), heroes=heroes
        )
        rows.append(
            {
                "frame_id": str(frame.get("frame_id", "")),
                "game_time_s": float(frame["game_time_s"]),
                "tokens": tokens,
            }
        )
    return rows


def build_from_file(source: Path, output: Path | None = None) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    frames = payload["frames"]
    rows = build_state_tokens(frames)
    # 按八档水平分流：每档独立训练数据
    by_level: dict[str, list[dict]] = {}
    for frame, row in zip(frames, rows):
        level_id = str(frame.get("level_id", "all"))
        by_level.setdefault(level_id, []).append(row)

    output = output or TRAINING_DIR / "state_tokens.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "version": GAME_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": str(source),
                "count": len(rows),
                "frames": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for level_id, level_rows in by_level.items():
        level_path = output.parent / f"state_tokens_level_{level_id}.json"
        level_path.write_text(
            json.dumps(
                {
                    "version": GAME_VERSION,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source": str(source),
                    "level_id": level_id,
                    "count": len(level_rows),
                    "frames": level_rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return output


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("用法: python build_state_dataset.py <标注帧.json>")
        raise SystemExit(1)
    path = build_from_file(Path(sys.argv[1]))
    print(f"已生成: {path}")
