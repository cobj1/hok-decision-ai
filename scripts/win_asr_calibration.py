r"""ASR 解说意图 vs 收益向量 校对 v2（Win 运行）。

- 用 sb2b 的 state.time（视频时钟）标定每个视频的 ASR→对局时间偏移；
- 用本地 Ollama LLM 从解说文本提取下一步动作意图（不用正则）；
- 与决策点收益向量比对方向，输出校对报告。

运行:
    E:\py312\python.exe E:\project\hok-decision-ai\scripts\win_asr_calibration.py
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

HOKDATA_ROOT = Path(os.environ.get("HOKDATA_ROOT", str(Path(__file__).resolve().parents[1].parent / "hokdata")))
DATASET = HOKDATA_ROOT / "dataset"
TRANSCRIPTS = HOKDATA_ROOT / "transcripts"
POOL_TXT = TRANSCRIPTS / "pool"
OUT = HOKDATA_ROOT / "asr_calibration_report_v2.json"
DEEPSEEK_URL = "https://api.deepseek.com/anthropic/v1/messages"
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_KEY_FILE = HOKDATA_ROOT / "deepseek_key.txt"


def _deepseek_key() -> str:
    import os

    if os.environ.get("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"]
    if DEEPSEEK_KEY_FILE.exists():
        return DEEPSEEK_KEY_FILE.read_text(encoding="utf-8").strip()
    return ""

ACTION_DIM = {
    "dragon": ("dragon", True),
    "tower": ("tower", True),
    "lane": ("advance", True),
    "jungle": ("economy", True),
    "kill": ("kill", True),
    "fight": ("kill", True),
    "retreat": ("survival", True),
}
DIMS = ["kill", "economy", "tower", "dragon", "survival", "advance"]


def _load_segments(bv: str) -> list[dict]:
    # 优先 faster-whisper 逐句时间戳（.fw.json），回退普通 json / pool txt
    for p in [POOL_TXT / f"{bv}.fw.json", TRANSCRIPTS / f"{bv}.json", POOL_TXT / f"{bv}.txt"]:
        if not p.exists():
            continue
        if p.suffix == ".json":
            payload = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [
                    {"start": float(s.get("start_ms") or 0) / 1000.0, "text": s.get("text", "")}
                    for s in payload
                ]
            return []
        text = p.read_text(encoding="utf-8", errors="ignore")
        segs = []
        for m in re.finditer(r"\[(\d+):(\d+):(\d+)\.(\d+)\s*-->\s*[^\]]+\]\s*([^\[]+)", text):
            h, mi, s, ms = (int(x) for x in m.groups()[:4])
            segs.append({"start": h * 3600 + mi * 60 + s + ms / 1000.0, "text": m.group(5)})
        return segs
    return []


def _parse_video_clock(value) -> float | None:
    """'mm:ss' → 秒。"""
    if not value or not isinstance(value, str):
        return None
    parts = value.strip().split(":")
    if len(parts) == 2:
        try:
            return int(parts[0]) * 60 + int(parts[1])
        except ValueError:
            return None
    return None


def _video_offsets() -> dict[str, float]:
    """用 sb2b 的 state.time（视频时钟）与 v4 game_time_s 计算每视频偏移（中位数）。"""
    clock_by_frame: dict[tuple[str, str], float] = {}
    for p in DATASET.glob("*.sb2b.json"):
        payload = json.loads(p.read_text(encoding="utf-8"))
        bv = payload.get("bv", p.stem.split(".sb2b")[0])
        for entry in payload.get("scoreboard", []):
            clock = _parse_video_clock((entry.get("state") or {}).get("time"))
            if clock is not None and entry.get("frame"):
                clock_by_frame[(str(bv), str(entry["frame"]))] = clock
    deltas: dict[str, list[float]] = defaultdict(list)
    for line in (DATASET / "dataset_final_v4.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (str(d.get("video_bv", "")), str(d.get("frame_id", "")))
        if key in clock_by_frame:
            deltas[str(d.get("video_bv"))].append(clock_by_frame[key] - float(d.get("game_time_s", 0)))
    return {bv: statistics.median(v) for bv, v in deltas.items() if v}


def _v4_frames(bv: str) -> list[dict]:
    frames = []
    for line in (DATASET / "dataset_final_v4.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("video_bv") == bv:
            frames.append(d)
    frames.sort(key=lambda f: float(f.get("game_time_s", 0)))
    return frames


def _llm_extract(items: list[dict]) -> dict[str, list[dict]]:
    """让 DeepSeek 从若干解说段落提取动作意图，返回 {id: [actions]}。"""
    key = _deepseek_key()
    if not key:
        print("deepseek key missing")
        return {}
    prompt = (
        "你是王者荣耀对局解说分析器。下面是若干段解说文本，每段带 id。"
        "请判断每段文本中提到的『下一步要做的动作』，动作类型只能是："
        "dragon(打龙/暴君/主宰)、tower(推塔/拆塔)、lane(带线/清线)、"
        "jungle(刷野/反野)、kill(抓人/击杀)、retreat(撤退/回城)、fight(打团/开团)。"
        "只输出JSON数组，不要解释：\n"
        '[{"id":"0","actions":[{"type":"dragon","confidence":0.9}]}]\n\n'
    )
    for item in items:
        prompt += f"id:{item['id']} 文本:{item['text'][:200]}\n"
    body = json.dumps(
        {
            "model": DEEPSEEK_MODEL,
            "max_tokens": 4000,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    req = urllib.request.Request(
        DEEPSEEK_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    )
    payload = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as exc:
            print(f"deepseek err attempt{attempt}:", exc, flush=True)
    if payload is None:
        return {}
    content = payload.get("content") or []
    raw = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    m = re.search(r"\[.*\]", raw, flags=re.S)
    if not m:
        return {}
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    result: dict[str, list[dict]] = {}
    for row in parsed:
        if not isinstance(row, dict):
            continue
        key_id = str(row.get("id", ""))
        result[key_id] = row.get("actions", []) or []
    return result


def main() -> int:
    print("START", flush=True)
    args = sys.argv[1:]
    try:
        offset = int(args[args.index("--offset") + 1]) if "--offset" in args else 0
    except (ValueError, IndexError):
        offset = 0
    try:
        limit = int(args[args.index("--limit") + 1]) if "--limit" in args else 5
    except (ValueError, IndexError):
        limit = 5
    try:
        out_path = Path(args[args.index("--out") + 1]) if "--out" in args else OUT
    except IndexError:
        out_path = OUT
    data = np.load(r"E:\hokdata\decision_benefit_labels.npz", allow_pickle=True)
    points = json.loads(str(data["points_json"]))
    by_bv: dict[str, list[dict]] = defaultdict(list)
    for p in points:
        by_bv[p["bv"]].append(p)

    report = {"llm_rows": [], "hit_rate": 0.0, "total": 0, "hits": 0,
              "evaluable_total": 0, "evaluable_hits": 0, "evaluable_rate": 0.0,
              "debug_unknown": [], "chosen_bvs": []}
    all_rows = []
    bv_segs: dict[str, list[dict]] = {bv: _load_segments(bv) for bv in by_bv}
    usable = [bv for bv, ss in sorted(bv_segs.items(), key=lambda kv: -len(kv[1])) if len(ss) >= 20]
    chosen_bvs = usable[offset: offset + limit]
    report["chosen_bvs"] = chosen_bvs
    print("chosen:", chosen_bvs, flush=True)
    for bv in chosen_bvs:
        segs_list = bv_segs[bv]
        pts = by_bv[bv]  # 全视频决策点（约 5s 间隔），覆盖中后期事件密集段
        # 为每个决策点取 [t-20s, t+10s] 的解说窗口作为 LLM 上下文
        windows = []
        for p in pts:
            t = float(p["t"])
            context = " ".join(
                s["text"] for s in segs_list if t - 20.0 <= s["start"] <= t + 10.0
            )[:500]
            if len(context) >= 15:
                windows.append({"id": str(len(windows)), "frame_t": t, "text": context, "p": p})
        if not windows:
            continue
        # LLM 批量提取（每批 15 个窗口）
        for batch_start in range(0, len(windows), 15):
            batch = windows[batch_start : batch_start + 15]
            llm = _llm_extract([{"id": m["id"], "text": m["text"]} for m in batch])
            n_actions = sum(len(v) for v in llm.values())
            print(f"{bv} batch {batch_start // 15 + 1} rows={len(llm)} actions={n_actions}", flush=True)
            for m in batch:
                actions = llm.get(m["id"], [])
                if not actions:
                    continue
                ben = m["p"]["benefits"]
                key = next((k for k in ("20", "15", "10", "5") if k in ben), None)
                if key is None:
                    continue
                vec = np.asarray(ben[key], dtype=float)
                for action in actions:
                    if not isinstance(action, dict):
                        if len(report["debug_unknown"]) < 30:
                            report["debug_unknown"].append({"bv": bv, "t": m["frame_t"], "non_dict": repr(action)})
                        continue
                    atype = action.get("type")
                    if atype not in ACTION_DIM:
                        if len(report["debug_unknown"]) < 30:
                            report["debug_unknown"].append({"bv": bv, "t": m["frame_t"], "action": action, "ctx": m["text"][:80]})
                        continue
                    dim, positive = ACTION_DIM[atype]
                    idx = DIMS.index(dim)
                    hit = bool(vec[idx] > 0) if positive else bool(vec[idx] < 0)
                    all_rows.append(
                        {
                            "bv": bv,
                            "frame_t": m["frame_t"],
                            "action": atype,
                            "dim": dim,
                            "target_value": float(vec[idx]),
                            "delta": key,
                            "confidence": action.get("confidence"),
                            "vec20": vec.tolist(),
                            "hit": hit,
                            "text": m["text"][:100],
                        }
                    )
    report["llm_rows"] = all_rows
    report["total"] = len(all_rows)
    report["hits"] = sum(1 for r in all_rows if r["hit"])
    report["hit_rate"] = round(report["hits"] / max(1, report["total"]), 4)
    ev = [r for r in all_rows if r["target_value"] != 0.0]
    report["evaluable_total"] = len(ev)
    report["evaluable_hits"] = sum(1 for r in ev if r["hit"])
    report["evaluable_rate"] = round(report["evaluable_hits"] / max(1, len(ev)), 4)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out_path}")
    print(f"total={report['total']} hits={report['hits']} hit_rate={report['hit_rate']} evaluable={report['evaluable_total']}/{report['evaluable_hits']}={report['evaluable_rate']}")
    print("ASR_CALIBRATION_V2_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
