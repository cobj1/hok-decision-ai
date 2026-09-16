"""资源点刷新调度器。

根据 map.json 中每个资源点的 spawn_rule / initial_spawn_s / respawn_s /
active_until_s / possible_locations，在模拟推进过程中维护其状态。

风暴龙王（dynamic）特殊处理：到时间后进入 available，但 current_position 为
None，必须由实时小地图观测（observe）回填实际所在龙坑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ResourceStatus = Literal["pending", "available", "waiting", "expired"]


@dataclass
class ResourceState:
    """单个资源点的运行期状态。"""

    resource_id: str
    name: str
    kind: str
    side: str
    spawn_rule: str
    base_position: tuple[float, float] | None
    possible_locations: list[tuple[float, float]] = field(default_factory=list)
    initial_spawn_s: float | None = None
    respawn_s: float | None = None
    active_until_s: float | None = None
    status: ResourceStatus = "pending"
    current_position: tuple[float, float] | None = None
    next_spawn_at: float | None = None
    spawned_count: int = 0

    def __post_init__(self) -> None:
        self.next_spawn_at = self.initial_spawn_s
        self.status = "pending" if self.initial_spawn_s is not None else "available"
        if self.initial_spawn_s is None and self.spawn_rule == "static":
            self.status = "available"
            self.current_position = self.base_position

    def mark_available(self, t: float) -> None:
        self.status = "available"
        self.spawned_count += 1
        self.current_position = None if self.spawn_rule == "dynamic" else self.base_position
        self.next_spawn_at = None

    def mark_expired(self) -> None:
        self.status = "expired"
        self.current_position = None
        self.next_spawn_at = None


class ResourceScheduler:
    """按游戏时钟推进资源点状态。"""

    def __init__(self, resources: list[dict[str, Any]] | None = None) -> None:
        self.states: dict[str, ResourceState] = {}
        if resources:
            for rp in resources:
                self.add_resource(rp)

    @classmethod
    def from_map_data(cls, map_data: Any) -> "ResourceScheduler":
        """从 MapData 对象构建。"""
        return cls([rp.model_dump() for rp in map_data.resource_points])

    def add_resource(self, rp: dict[str, Any]) -> None:
        state = ResourceState(
            resource_id=str(rp["id"]),
            name=str(rp.get("name", rp["id"])),
            kind=str(rp.get("kind", "other")),
            side=str(rp.get("side", "neutral")),
            spawn_rule=str(rp.get("spawn_rule", "fixed")),
            base_position=(
                tuple(rp["position"]) if rp.get("position") is not None else None
            ),
            possible_locations=[
                tuple(pos) for pos in (rp.get("possible_locations") or [])
            ],
            initial_spawn_s=rp.get("initial_spawn_s"),
            respawn_s=rp.get("respawn_s"),
            active_until_s=rp.get("active_until_s"),
        )
        self.states[state.resource_id] = state

    def tick(self, t: float) -> list[str]:
        """推进到 t 秒，返回状态发生变化的资源 id。"""
        changed: list[str] = []
        for state in self.states.values():
            old = state.status
            if state.status == "pending":
                if state.next_spawn_at is not None and t >= state.next_spawn_at:
                    state.mark_available(t)
            elif state.status == "available":
                if state.active_until_s is not None and t >= state.active_until_s:
                    state.mark_expired()
            elif state.status == "waiting":
                if state.next_spawn_at is not None and t >= state.next_spawn_at:
                    if state.active_until_s is not None and t >= state.active_until_s:
                        state.mark_expired()
                    else:
                        state.mark_available(t)
            if state.status != old:
                changed.append(state.resource_id)
        return changed

    def kill(self, resource_id: str, t: float) -> bool:
        """标记资源被击杀。返回是否成功。"""
        state = self.states.get(resource_id)
        if state is None or state.status != "available":
            return False
        if state.spawn_rule == "once" or state.respawn_s is None:
            state.mark_expired()
        else:
            state.status = "waiting"
            state.next_spawn_at = t + state.respawn_s
            state.current_position = None
        return True

    def observe(self, resource_id: str, position: tuple[float, float]) -> bool:
        """观测到资源实际位置（dynamic 资源必须调用）。"""
        state = self.states.get(resource_id)
        if state is None or state.status != "available":
            return False
        if state.spawn_rule != "dynamic":
            state.current_position = position
            return True
        # 动态资源：接受任意位置观测，若在候选点内则记录
        state.current_position = position
        return True

    def available(self) -> list[ResourceState]:
        return [s for s in self.states.values() if s.status == "available"]

    def get(self, resource_id: str) -> ResourceState | None:
        return self.states.get(resource_id)
