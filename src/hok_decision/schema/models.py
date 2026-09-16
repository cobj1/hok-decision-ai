"""结构化数据模型（Pydantic v2）。

原则：模型字段与游戏数值解耦——数值全部来自数据层 JSON，
模型只定义结构；每个数据文件带 version 与 source 溯源。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HeroInfo(BaseModel):
    """官方英雄列表条目（不含战斗数值）。"""

    model_config = ConfigDict(extra="allow")

    ename: int
    cname: str
    id_name: str
    title: str
    hero_type: int
    roles: list[int] = Field(default_factory=list)

    @field_validator("roles", mode="before")
    @classmethod
    def _parse_roles(cls, value: object) -> object:
        """官方字段为 '1|5' 字符串或单 int，统一为 int 列表。"""
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            return [int(part) for part in text.split("|") if part.strip()]
        if isinstance(value, list):
            return [int(item) for item in value]
        return value


class HeroStats(BaseModel):
    """英雄战斗数值（可含 1 级基础值、成长、满级值；未拿到一律 null）。"""

    ename: int
    cname: str
    level: int = 1
    base_hp: float | None = None
    base_mp: float | None = None
    base_physical_attack: float | None = None
    base_magic_attack: float | None = None
    base_physical_defense: float | None = None
    base_magic_defense: float | None = None
    base_attack_speed: float | None = None
    attack_range: float | None = None
    attack_range_label: str | None = None
    crit_rate: float | None = None
    crit_effect: float | None = None
    move_speed: float | None = None
    attack_interval: float | None = None
    hp_regen_5s: float | None = None
    mp_regen_5s: float | None = None
    max_hp: float | None = None
    max_mp: float | None = None
    max_physical_attack: float | None = None
    max_physical_defense: float | None = None
    max_magic_defense: float | None = None
    max_attack_speed: float | None = None
    growth: dict[str, float] = Field(default_factory=dict)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    source: dict[str, Any] = Field(default_factory=dict)
    verified: bool = False


class SkillInfo(BaseModel):
    """英雄技能数据（来源以官方 herodetail 页面为主）。"""

    model_config = ConfigDict(extra="allow")

    ename: int
    cname: str
    skill_id: str | None = None
    slot: int
    name: str
    type: str | None = None
    cooldown: list[float] = Field(default_factory=list)
    cooldown_raw: str = ""
    mana_cost: list[float] = Field(default_factory=list)
    mana_cost_raw: str = ""
    description: str = ""
    control_effect: str | None = None
    data_tables: dict[str, str] = Field(default_factory=dict)
    labels: list[str] = Field(default_factory=list)
    source: dict[str, Any] = Field(default_factory=dict)
    verified: bool = False


class ItemInfo(BaseModel):
    """官方装备列表条目。"""

    model_config = ConfigDict(extra="allow")

    item_id: int
    item_name: str
    item_type: int
    price: float
    total_price: float
    des1: str = ""


class SummonerSpell(BaseModel):
    """官方召唤师技能条目。"""

    model_config = ConfigDict(extra="allow")

    summoner_id: int
    summoner_name: str
    summoner_rank: str
    summoner_description: str


class ResourcePoint(BaseModel):
    """地图资源点：野怪/中立资源/塔/兵线节点。

    position 为空表示该数值尚未校准；后续由录屏地形定位管线回填。
    spawn_rule 说明该资源的生成/刷新逻辑：
      - static         一直存在（塔、基地）
      - fixed          到时间生成，击杀后按 respawn_s 刷新
      - once           只在 initial_spawn_s 生成一次
      - timed_window   只在一个时间窗口内生成/刷新
      - dynamic        生成位置不固定（如风暴龙王双龙坑随机），
                       实际位置需通过 minimap 观测回填
    """

    id: str
    name: str
    kind: Literal["buff", "camp", "neutral", "tower", "lane", "other"]
    side: Literal["blue", "red", "neutral"] = "neutral"
    position: tuple[float, float] | None = None
    initial_spawn_s: float | None = None
    respawn_s: float | None = None
    buff_duration_s: float | None = None
    spawn_rule: Literal["static", "fixed", "once", "timed_window", "dynamic"] = "fixed"
    possible_locations: list[tuple[float, float]] | None = None
    active_until_s: float | None = None
    notes: str = ""
    source: dict[str, Any] = Field(default_factory=dict)
    verified: bool = False


class Lane(BaseModel):
    """一条兵线：发育路/对抗路/中路。"""

    id: str
    name: str
    waypoints: list[tuple[float, float]] = Field(default_factory=list)
    notes: str = ""


class MapMechanics(BaseModel):
    """地图机制规则（当前版本整理，字段可扩展）。"""

    minion_rules: dict[str, Any] = Field(default_factory=dict)
    tower_rules: dict[str, Any] = Field(default_factory=dict)
    dragon_bond_s: float | None = None
    notes: list[str] = Field(default_factory=list)
    source: dict[str, Any] = Field(default_factory=dict)


class MapData(BaseModel):
    """5v5 王者峡谷地图数据。"""

    version: str
    name: str = "王者峡谷"
    width: float | None = None
    height: float | None = None
    unit_note: str = "地图单位待标定：从录屏地形定位回填"
    lanes: list[Lane] = Field(default_factory=list)
    resource_points: list[ResourcePoint] = Field(default_factory=list)
    mechanics: MapMechanics | None = None
    fetched_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class DamageConfig(BaseModel):
    """伤害结算口径（保守/期望）。"""

    defense_constant: float = 602.0
    friendly_crit: bool = False
    enemy_damage_mode: Literal["none", "expected", "max"] = "expected"
    crit_default_multiplier: float = 2.0


class DataManifest(BaseModel):
    """一次爬取产物的清单与溯源。"""

    version: str
    fetched_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sources: list[dict[str, Any]] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
