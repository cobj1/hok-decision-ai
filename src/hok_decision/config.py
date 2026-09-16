"""全局配置：版本标签、数据目录与数据源 URL。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# 版本标签：锁定当前赛季，数据目录与样本均带此标签。
# 更新方法：官方赛季公告确认后修改此处并重新运行爬虫。
GAME_VERSION: str = "S45"

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_ROOT: Path = PROJECT_ROOT / "data"
RAW_ROOT: Path = DATA_ROOT / GAME_VERSION / "raw"
NORMALIZED_ROOT: Path = DATA_ROOT / GAME_VERSION / "normalized"

USER_AGENT: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

# 官方资料库 JSON 端点（一手来源）
OFFICIAL_HERO_LIST_URL: str = "https://pvp.qq.com/web201605/js/herolist.json"
OFFICIAL_ITEM_LIST_URL: str = "https://pvp.qq.com/web201605/js/item.json"
OFFICIAL_SUMMONER_URL: str = "https://pvp.qq.com/web201605/js/summoner.json"

# 官方地图介绍页（王者峡谷）
OFFICIAL_MAP_PAGE_URL: str = "https://pvp.qq.com/coming/system/wtc-map.html"

# 地图资源机制文章（次级来源，数值需版本验证）
MAP_MECHANICS_URL: str = "https://news.17173.com/z/pvp/content/07302025/102109377.shtml"

MAP_ASSET_ROOT: Path = RAW_ROOT / "map_assets"


@dataclass(frozen=True)
class SourceRef:
    """数据来源记录，全部结构化数据必须携带。"""

    url: str
    name: str
    retrieved_at: str
    is_primary: bool
