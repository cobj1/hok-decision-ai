r"""ASR 意图 vs search 输出校对（Win 运行）。

口径（用户确认）：
  先 search，后对比；模型只用来给注意力二阶权重 w(s)=attention(局势;θ)，
  动作序列/DecisionOutput 全部来自 beam_search。

流程：
  1. 读 asr_calibration_final_v2.json 的 ASR 动作意图（bv, frame_t, action）；
  2. 每个唯一决策点构造 SearchState（小地图 self/enemy + sb2b scoreboard + towers_state）；
  3. AttentionWeightedValue 加载 attention_value_td.pt，按帧上下文输出动态权重；
  4. beam_search(with_decision_output=True) 得到最优动作序列 + 目标收益向量；
  5. 对比：ASR action -> 期望 MacroAction.kind，与搜索最优首动作是否一致。

运行（分片）:
    E:\py312\python.exe E:\project\hok-decision-ai\scripts\win_asr_vs_search.py --offset 0 --limit 300 --out E:\hokdata\asr_vs_search_part_00.json
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

from hok_decision.schema.models import MapData  # noqa: E402
from hok_decision.search.attention_weights import AttentionWeightedValue  # noqa: E402
from hok_decision.search.planner import SearchState, beam_search  # noqa: E402
from hok_decision.simulator.units import Hero  # noqa: E402

HOKDATA_ROOT = Path(os.environ.get("HOKDATA_ROOT", str(Path(__file__).resolve().parents[1].parent / "hokdata")))
DATASET = HOKDATA_ROOT / "dataset"
ASR_REPORT = HOKDATA_ROOT / "asr_calibration_final_v2.json"
SPLIT_FILE = HOKDATA_ROOT / "asr_train_split.json"
MODELS_DIR = PROJECT_ROOT / "data" / "S45" / "models"
MAP_JSON = PROJECT_ROOT / "data" / "S45" / "normalized" / "map.json"

HUD_AFFINE = ((3.9619274696916493, 1.3453769972204605e-05), (-1.9928688568260355e-05, 4.038131527190943), (-196.00459141224727, -105.2570867363273))
BLUE_BASE = (25.0, 395.0)
RED_BASE = (397.0, 28.0)
LEVEL_MAP = {"1500+": 2, "1800+": 3, "2100+": 4, "2300+": 5, "2500+": 6, "赛事": 7}
CONTEXT_NAMES = [
    "time_norm", "econ_sat", "enemy_threat", "own_hp_frac", "tower_adv",
    "dragon_avail", "team_aa_power", "team_caster_power",
    "build_attack_power", "build_defense_power", "build_utility_power",
    "skill_power", "match_level",
]
GOAL_DIMS = ["kill", "economy", "tower", "dragon", "survival", "advance"]

# ASR 动作 -> 期望搜索首动作
EXPECT_KIND = {
    "kill": "attack_enemy",
    "fight": "attack_enemy",
    "retreat": "retreat",
    "tower": "push_tower",
    "jungle": "capture_resource",
    "dragon": "capture_resource",
    "lane": "capture_resource",
}
DRAGON_IDS = {"tyrant", "overlord", "storm_dragon"}


def _to_ref(hero: dict, cs: str) -> tuple[float, float]:
    if cs == "hud_roi":
        a, b, c = HUD_AFFINE
        x = a[0] * hero["x"] + a[1] * hero["y"] + c[0]
        y = b[0] * hero["x"] + b[1] * hero["y"] + c[1]
        return x, y
    return hero.get("x_ref", hero["x"]), hero.get("y_ref", hero["y"])


def _mean(values) -> float:
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def _clamp_gold(value) -> float:
    return value if 0.0 <= value <= 15000.0 else 0.0


def _load_sb2b() -> dict[tuple[str, str], dict]:
    table: dict[tuple[str, str], dict] = {}
    for p in DATASET.glob("*.sb2b.json"):
        payload = json.loads(p.read_text(encoding="utf-8"))
        bv = payload.get("bv", p.stem.split(".sb2b")[0])
        for entry in payload.get("scoreboard", []):
            if entry.get("frame"):
                table[(str(bv), str(entry["frame"]))] = entry.get("state") or {}
    return table


SB2B = _load_sb2b()


def _raw_author(frame: dict) -> dict:
    camp = str(frame.get("author_camp", "blue"))
    state = SB2B.get((str(frame.get("video_bv", "")), str(frame.get("frame_id", ""))))
    if state:
        blue = state.get("blue") or {}
        red = state.get("red") or {}

        def num(v):
            return float(v) if isinstance(v, (int, float)) else 0.0

        if camp == "blue":
            self_k, enemy_k = num(blue.get("kills")), num(red.get("kills"))
            self_t, enemy_t = num(blue.get("towers")), num(red.get("towers"))
            self_d, enemy_d = num(blue.get("dragons")), num(red.get("dragons"))
            self_g = _clamp_gold(num(blue.get("gold")))
        else:
            self_k, enemy_k = num(red.get("kills")), num(blue.get("kills"))
            self_t, enemy_t = num(red.get("towers")), num(blue.get("towers"))
            self_d, enemy_d = num(red.get("dragons")), num(blue.get("dragons"))
            self_g = _clamp_gold(num(red.get("gold")))
        return {
            "kills_self": self_k, "kills_enemy": enemy_k,
            "towers": min(9.0, max(-9.0, self_t - enemy_t)),
            "dragons": min(3.0, max(-3.0, self_d - enemy_d)),
            "gold": self_g,
        }
    bs = (frame.get("battle_state") or {}).get("base") or {}
    kb = bs.get("kills_blue")
    kr = bs.get("kills_red")
    if camp == "red":
        kb, kr = kr, kb

    def num2(v):
        return float(v) if isinstance(v, (int, float)) else 0.0

    return {
        "kills_self": num2(kb), "kills_enemy": num2(kr),
        "towers": min(9.0, max(-9.0, num2(bs.get("towers")))),
        "dragons": min(3.0, max(-3.0, num2(bs.get("dragons")))),
        "gold": _clamp_gold(num2(bs.get("gold"))),
    }


def _context_vector(frame: dict) -> list[float]:
    raw = _raw_author(frame)
    t = float(frame.get("game_time_s", 0))
    level_id = LEVEL_MAP.get(str(frame.get("level_id", "")), 3)
    ally_hp = _mean(frame.get("ally_hp") or [])
    ctx = [0.0] * len(CONTEXT_NAMES)
    ctx[CONTEXT_NAMES.index("time_norm")] = t / 1500.0
    ctx[CONTEXT_NAMES.index("econ_sat")] = min(1.0, t / 1500.0)
    ctx[CONTEXT_NAMES.index("enemy_threat")] = 0.5
    ctx[CONTEXT_NAMES.index("own_hp_frac")] = ally_hp if ally_hp else 0.7
    ctx[CONTEXT_NAMES.index("tower_adv")] = raw["towers"] / 10.0
    ctx[CONTEXT_NAMES.index("dragon_avail")] = 1.0 if raw["dragons"] > 0 or t >= 120.0 else 0.0
    ctx[CONTEXT_NAMES.index("team_aa_power")] = 0.8
    ctx[CONTEXT_NAMES.index("team_caster_power")] = 0.8
    ctx[CONTEXT_NAMES.index("build_attack_power")] = 0.8
    ctx[CONTEXT_NAMES.index("build_defense_power")] = 0.8
    ctx[CONTEXT_NAMES.index("build_utility_power")] = 0.8
    ctx[CONTEXT_NAMES.index("skill_power")] = 0.8
    ctx[CONTEXT_NAMES.index("match_level")] = level_id / 7.0
    return ctx


def _build_state(frame: dict) -> SearchState:
    heroes = frame.get("heroes") or []
    cs = str(frame.get("coordinate_system", "hud_roi"))
    self_h = [h for h in heroes if str(h.get("side")) == "self"]
    enemy_h = [h for h in heroes if str(h.get("side")) == "enemy"]
    if not self_h or not enemy_h:
        raise ValueError("minimap 缺少 self/enemy 英雄")

    ally_hp = [v for v in (frame.get("ally_hp") or []) if isinstance(v, (int, float))]
    enemy_hp = [v for v in (frame.get("enemy_hp") or []) if isinstance(v, (int, float))]

    state = SearchState(game_time=float(frame.get("game_time_s", 0)))
    for i, h in enumerate(self_h):
        hp = ally_hp[i] if i < len(ally_hp) and 0.0 <= ally_hp[i] <= 1.0 else 0.7
        state.heroes[f"blue_{i}"] = Hero(
            unit_id=f"blue_{i}", name="self", side="blue",
            position=_to_ref(h, cs), hp=hp, max_hp=1.0,
        )
    for i, h in enumerate(enemy_h):
        hp = enemy_hp[i] if i < len(enemy_hp) and 0.0 <= enemy_hp[i] <= 1.0 else 0.7
        state.heroes[f"red_{i}"] = Hero(
            unit_id=f"red_{i}", name="enemy", side="red",
            position=_to_ref(h, cs), hp=hp, max_hp=1.0,
        )

    # 经济：sb2b 总经济均分给各英雄（search 内经济差特征使用）
    raw = _raw_author(frame)
    n_self = len(self_h)
    n_enemy = len(enemy_h)
    for h in state.heroes.values():
        if h.side == "blue":
            state.gold[h.unit_id] = raw["gold"] / n_self if n_self else 0.0
        else:
            state.gold[h.unit_id] = 0.0

    # 塔血量：towers_state 的 true/false 只表示可见性（蓝塔恒 true），
    # 不是存活状态；逐塔血量未知，统一按满血初始化，由 search 内推塔/守塔产生变化。
    for tower_id in (frame.get("towers_state") or {}):
        state.tower_hp[tower_id] = 1.0

    # 龙：scoreboard 数量 -> 已捕获资源（用于初值；搜索过程中会继续捕获）
    for rid, count in (("tyrant", max(0, min(2, int(raw["dragons"])))),):
        for _ in range(count):
            if rid not in state.captured_resources:
                state.captured_resources.add(rid)
                break
    return state


def _expect_match(action_type: str, first_action) -> bool:
    kind = first_action.kind
    target = first_action.target_id or ""
    if action_type in {"kill", "fight"}:
        return kind == "attack_enemy"
    if action_type == "retreat":
        return kind == "retreat"
    if action_type == "tower":
        return kind == "push_tower"
    if action_type == "dragon":
        return kind == "capture_resource" and any(d in target for d in DRAGON_IDS)
    if action_type == "jungle":
        return kind == "capture_resource" and not any(d in target for d in DRAGON_IDS)
    if action_type == "lane":
        # 带线/清线最接近的宏观动作：打资源（非龙）或推进
        return (kind == "capture_resource" and not any(d in target for d in DRAGON_IDS)) or kind == "push_tower"
    return False


def main() -> int:
    args = sys.argv[1:]
    try:
        offset = int(args[args.index("--offset") + 1]) if "--offset" in args else 0
    except (ValueError, IndexError):
        offset = 0
    try:
        limit = int(args[args.index("--limit") + 1]) if "--limit" in args else 300
    except (ValueError, IndexError):
        limit = 300
    try:
        out_path = Path(args[args.index("--out") + 1]) if "--out" in args else Path(r"E:\hokdata\asr_vs_search.json")
    except IndexError:
        out_path = Path(r"E:\hokdata\asr_vs_search.json")
    try:
        model_name = args[args.index("--model") + 1] if "--model" in args else "attention_value_td.pt"
    except IndexError:
        model_name = "attention_value_td.pt"
    try:
        subset = args[args.index("--subset") + 1] if "--subset" in args else "all"
    except IndexError:
        subset = "all"

    map_data = MapData.model_validate(json.loads(MAP_JSON.read_text(encoding="utf-8")))
    anchors = {
        "fountain_blue": {"position": BLUE_BASE},
        "fountain_red": {"position": RED_BASE},
    }

    # v4 帧索引：(bv, round(t,1)) -> frame
    frames_index: dict[tuple[str, float], dict] = {}
    for line in (DATASET / "dataset_final_v4.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (str(frame.get("video_bv")), round(float(frame.get("game_time_s", 0)), 1))
        frames_index[key] = frame

    asr_report = json.loads(ASR_REPORT.read_text(encoding="utf-8"))
    rows = asr_report.get("llm_rows", [])
    by_point: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in rows:
        by_point[(str(r["bv"]), round(float(r["frame_t"]), 1))].append(r)
    if subset in {"train", "holdout"} and SPLIT_FILE.exists():
        split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
        allowed = set(split.get(subset, []))
        by_point = {k: v for k, v in by_point.items() if k[0] in allowed}
        print(f"subset={subset} videos={len(allowed)}")
    points = sorted(by_point.items())[offset: offset + limit]
    print(f"total_points={len(by_point)} chunk_points={len(points)}")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch={torch.__version__} device={device}")
    model_path = MODELS_DIR / model_name
    evaluator = AttentionWeightedValue(context=[0.0] * 13, model_path=model_path, device=device)
    print("model:", model_name, "feature_dim:", evaluator.feature_dim)
    print("model_loaded:", evaluator.model is not None, "fallback_weights:", evaluator.base_weights[:3])

    out_rows = []
    missing_frames = 0
    state_errors = 0
    for (bv, t), asr_rows in points:
        frame = frames_index.get((bv, t))
        if frame is None:
            missing_frames += 1
            continue
        try:
            state = _build_state(frame)
        except ValueError:
            state_errors += 1
            continue
        evaluator.context = _context_vector(frame)
        best = beam_search(
            state,
            map_data=map_data,
            anchors=anchors,
            horizon_s=12.0,
            beam_width=2,
            action_duration_s=4.0,
            max_candidates=8,
            value_context=evaluator,
            with_decision_output=True,
        )
        if best is None or best.decision_output is None or not best.actions:
            continue
        first = best.actions[0]
        goal = best.decision_output.goal_vector
        for r in asr_rows:
            action_type = str(r["action"])
            out_rows.append(
                {
                    "bv": bv,
                    "frame_t": r["frame_t"],
                    "action": action_type,
                    "first_action": first.label,
                    "first_kind": first.kind,
                    "target_id": first.target_id,
                    "goal_vector": goal,
                    "hit": bool(_expect_match(action_type, first)),
                    "value": best.value,
                    "n_actions": len(best.actions),
                }
            )
        if (len(out_rows) // 200) > 0 and len(out_rows) % 200 < len(asr_rows):
            print(f"processed points~{len(out_rows)} rows", flush=True)

    report = {
        "rows": out_rows,
        "total": len(out_rows),
        "hits": sum(1 for r in out_rows if r["hit"]),
        "hit_rate": round(sum(1 for r in out_rows if r["hit"]) / max(1, len(out_rows)), 4),
        "missing_frames": missing_frames,
        "state_errors": state_errors,
        "points_in_chunk": len(points),
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"total={report['total']} hits={report['hits']} hit_rate={report['hit_rate']} missing={missing_frames} state_err={state_errors}")
    print(f"saved -> {out_path}")
    print("ASR_VS_SEARCH_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
