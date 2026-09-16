"""爬虫基础：带重试的 HTTP 抓取与原始数据落盘。"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from hok_decision.config import USER_AGENT

logger = logging.getLogger(__name__)


class CrawlerError(RuntimeError):
    """爬取失败（网络、超时或数据校验失败）。"""


def fetch(
    url: str,
    *,
    timeout: float = 30.0,
    retries: int = 3,
    backoff_s: float = 1.0,
    params: dict[str, Any] | None = None,
) -> requests.Response:
    """GET 一个 URL，带有限重试与退避。"""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:  # noqa: PERF203
            last_error = exc
            if attempt < retries - 1:
                time.sleep(backoff_s * (attempt + 1))
    raise CrawlerError(f"GET {url} 失败: {last_error}") from last_error


def save_raw(
    content: bytes,
    *,
    root: Path,
    filename: str,
) -> Path:
    """把原始响应写入 raw 目录，返回路径。"""
    root.mkdir(parents=True, exist_ok=True)
    target = root / filename
    target.write_bytes(content)
    return target


def source_ref(url: str, name: str, is_primary: bool = False) -> dict[str, Any]:
    """生成统一的数据来源记录。"""
    return {
        "url": url,
        "name": name,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "is_primary": is_primary,
    }
