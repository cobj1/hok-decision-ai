"""爬虫归一化逻辑测试（离线数据，不访问网络）。"""

from __future__ import annotations

import pytest

from hok_decision.crawlers.base import CrawlerError
from hok_decision.crawlers.official_pvp import _normalize_roles
from hok_decision.schema.models import HeroInfo, ItemInfo


def test_normalize_roles() -> None:
    assert _normalize_roles("1|5") == [1, 5]
    assert _normalize_roles(3) == [3]
    assert _normalize_roles("") == []
    with pytest.raises(CrawlerError):
        _normalize_roles("1|x")


def test_hero_info_from_raw_payload() -> None:
    raw = {
        "ename": 105,
        "cname": "廉颇",
        "id_name": "lianpo",
        "title": "正义爆轰",
        "hero_type": 3,
        "roles": "1|5",
    }
    hero = HeroInfo.model_validate(raw)
    assert hero.ename == 105
    assert hero.roles == [1, 5]


def test_item_info_from_raw_payload() -> None:
    raw = {
        "item_id": 1111,
        "item_name": "铁剑",
        "item_type": 1,
        "price": 165,
        "total_price": 275,
        "des1": "<p>+20物理攻击</p>",
    }
    item = ItemInfo.model_validate(raw)
    assert item.total_price == 275
