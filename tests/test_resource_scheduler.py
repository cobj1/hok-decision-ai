"""资源点刷新调度器测试。"""

from __future__ import annotations

import pytest

from hok_decision.simulator.engine import SimEngine
from hok_decision.simulator.resources import ResourceScheduler


def _rp(
    rid: str,
    *,
    spawn_rule: str = "fixed",
    initial: float | None = 30.0,
    respawn: float | None = 90.0,
    active_until: float | None = None,
    possible: list[tuple[float, float]] | None = None,
) -> dict:
    return {
        "id": rid,
        "name": rid,
        "kind": "neutral",
        "side": "neutral",
        "position": [100.0, 200.0],
        "initial_spawn_s": initial,
        "respawn_s": respawn,
        "spawn_rule": spawn_rule,
        "possible_locations": possible,
        "active_until_s": active_until,
        "notes": "",
        "source": {},
        "verified": True,
    }


def test_buff_initial_spawn_and_respawn() -> None:
    sched = ResourceScheduler([_rp("buff", initial=30.0, respawn=90.0)])
    sched.tick(20.0)
    assert sched.get("buff").status == "pending"  # type: ignore[union-attr]
    sched.tick(30.0)
    buff = sched.get("buff")
    assert buff is not None and buff.status == "available"
    assert buff.current_position == (100.0, 200.0)
    assert sched.kill("buff", t=40.0)
    assert sched.get("buff").status == "waiting"  # type: ignore[union-attr]
    sched.tick(129.0)
    assert sched.get("buff").status == "waiting"  # type: ignore[union-attr]
    sched.tick(130.0)
    assert sched.get("buff").status == "available"  # type: ignore[union-attr]


def test_once_resource_no_respawn() -> None:
    sched = ResourceScheduler([_rp("red_falcon", spawn_rule="once", initial=30.0, respawn=None)])
    sched.tick(30.0)
    assert sched.get("red_falcon").status == "available"  # type: ignore[union-attr]
    assert sched.kill("red_falcon", t=40.0)
    assert sched.get("red_falcon").status == "expired"  # type: ignore[union-attr]
    sched.tick(500.0)
    assert sched.get("red_falcon").status == "expired"  # type: ignore[union-attr]


def test_timed_window_resource_expires() -> None:
    sched = ResourceScheduler(
        [_rp("space_spirit", spawn_rule="timed_window", initial=60.0, respawn=60.0, active_until=240.0)]
    )
    sched.tick(60.0)
    assert sched.get("space_spirit").status == "available"  # type: ignore[union-attr]
    sched.tick(245.0)
    assert sched.get("space_spirit").status == "expired"  # type: ignore[union-attr]
    # 击杀后等待也受窗口限制
    sched2 = ResourceScheduler(
        [_rp("space_spirit2", spawn_rule="timed_window", initial=60.0, respawn=60.0, active_until=240.0)]
    )
    sched2.tick(100.0)
    assert sched2.kill("space_spirit2", t=100.0)
    assert sched2.get("space_spirit2").status == "waiting"  # type: ignore[union-attr]
    sched2.tick(160.0)
    assert sched2.get("space_spirit2").status == "available"  # type: ignore[union-attr]
    sched2.tick(300.0)
    assert sched2.get("space_spirit2").status == "expired"  # type: ignore[union-attr]


def test_dynamic_storm_dragon_requires_observation() -> None:
    sched = ResourceScheduler(
        [
            _rp(
                "storm_dragon",
                spawn_rule="dynamic",
                initial=1200.0,
                respawn=240.0,
                possible=[(150.0, 112.0), (284.0, 276.0)],
            )
        ]
    )
    sched.tick(1199.0)
    assert sched.get("storm_dragon").status == "pending"  # type: ignore[union-attr]
    sched.tick(1200.0)
    st = sched.get("storm_dragon")
    assert st is not None and st.status == "available"
    assert st.current_position is None  # 未观测前不知道在哪个龙坑
    assert sched.observe("storm_dragon", (284.0, 276.0))
    assert st.current_position == (284.0, 276.0)
    assert sched.kill("storm_dragon", t=1250.0)
    assert st.status == "waiting"
    assert st.current_position is None
    sched.tick(1490.0)
    assert st.status == "available"
    assert st.current_position is None  # 重生后仍需重新观测


def test_static_tower_always_available() -> None:
    sched = ResourceScheduler([_rp("tower", spawn_rule="static", initial=None, respawn=None)])
    sched.tick(0.0)
    assert sched.get("tower").status == "available"  # type: ignore[union-attr]


def test_engine_integrates_resource_scheduler() -> None:
    scheduler = ResourceScheduler([_rp("buff", initial=30.0, respawn=90.0)])
    engine = SimEngine()
    engine.attach_resources(scheduler)
    engine.run_until(max_time=31.0, dt=1.0)
    assert scheduler.get("buff").status == "available"  # type: ignore[union-attr]
