"""splits.json 的唯一读取口径：只认按基础局切分的 ``tencent_rt_base``。

历史读法有三套：有的只读 ``tencent_rt``（含 ``#2`` 视角），有的合并两键，
缺键时还会回退到 80/20 随机切分——同一批帧在不同脚本里可能落到不同 split。
这里统一为：基础局映射 + 查表时剥掉 ``#`` 后缀；缺文件/缺命名空间直接报错。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

NAMESPACE = "tencent_rt_base"
VALID_SPLITS = frozenset({"train", "val", "test"})


def load_splits(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"splits.json 缺失：{p}；禁止回退到随机/哈希切分，请先运行 build_hokdata.py"
        )
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("splits.json 顶层必须是对象")
    base = payload.get(NAMESPACE)
    if not isinstance(base, dict) or not base:
        raise ValueError(f"splits.json 缺少非空命名空间 {NAMESPACE}")
    bad_keys = sorted(k for k in base if "#" in str(k))
    if bad_keys:
        raise ValueError(f"{NAMESPACE} 必须按基础局切分，发现带 # 的键：{bad_keys[:5]}")
    invalid = sorted({str(v) for v in base.values()} - VALID_SPLITS)
    if invalid:
        raise ValueError(f"{NAMESPACE} 含非法 split 值：{invalid}")
    return {str(k): str(v) for k, v in base.items()}


def split_of(splits: Mapping[str, str], key: str) -> str | None:
    """先按原键查，再按基础局（剥 ``#`` 后缀）查。"""
    raw = str(key)
    return splits.get(raw) or splits.get(raw.split("#", 1)[0])
