"""refnav.planner gap picking and the stall landing, against the fake PX4 and ray-cast camera in test_refnav_core."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_refnav_core import _start_leg, fly  # noqa: E402

from refnav import core as C  # noqa: E402
from refnav import planner as P  # noqa: E402


def test_a_frozen_carrot_lands_instead_of_hovering_forever() -> None:
    m, _, _ = fly(C.Params(route=[(8.0, 0.0)], min_corridor_coverage=1.01, stall_s=5.0), seconds=60.0)
    assert m.result.startswith("landed: no progress"), m.result
    assert m.counts["stalls"] == 1


def test_avoid_flies_around_an_obstacle_on_the_route() -> None:
    m, px4, _ = fly(C.Params(route=[(8.0, 0.0)], avoid=True), walls=[(0, 4.5, -1.0, 1.0)], seconds=90.0)
    assert m.result == "complete", m.result
    assert m.counts["obstacle_stops"] == 0
    assert abs(px4.pos[0] - 8.0) < 0.4


def test_avoid_gives_up_safely_when_nothing_is_clear() -> None:
    m, px4, _ = fly(C.Params(route=[(8.0, 0.0)], avoid=True), walls=[(0, 4.5, -20.0, 20.0)], seconds=120.0)
    assert m.result.startswith("landed"), m.result
    assert px4.pos[0] < 4.5 - 0.25


def test_memory_keeps_an_obstacle_after_it_leaves_the_view() -> None:
    mem = P.PointMemory(keep_s=3.0)
    seen = np.array([[1.0, 0.0, 0.0]])
    mem.add(0.0, seen, (0.0, 0.0), 0.0)
    now_abeam = mem.merged(np.empty((0, 3)), (1.0, -0.5), 0.0, np.radians(34.0))
    assert np.allclose(now_abeam, [[0.0, 0.5, 0.0]])


def test_memory_drops_what_the_camera_can_see_again() -> None:
    mem = P.PointMemory(keep_s=3.0)
    mem.add(0.0, np.array([[2.0, 0.0, 0.0]]), (0.0, 0.0), 0.0)
    assert len(mem.merged(np.empty((0, 3)), (0.0, 0.0), 0.0, np.radians(34.0))) == 0


def test_memory_forgets_after_keep_s() -> None:
    mem = P.PointMemory(keep_s=1.0)
    mem.add(0.0, np.array([[1.0, 0.0, 0.0]]), (0.0, 0.0), 0.0)
    mem.add(2.0, np.array([[1.0, 0.0, 0.0]]), (0.0, 0.0), 0.0)
    assert len(mem.frames) == 1


def _turn_to_check(log) -> float:
    t_turn = next(t for t, ph, *_ in log if ph == "turn")
    return next(t for t, ph, *_ in log if t > t_turn and ph == "check") - t_turn


def test_settle_gate_holds_the_turn_until_heading_has_held() -> None:
    _, _, quick = fly(C.Params(route=[(0.0, 3.0)]), seconds=12.0)     # only the first turn is needed
    m, _, slow = fly(C.Params(route=[(0.0, 3.0)], settle_s=1.0), seconds=40.0)
    assert m.result == "complete", m.result
    assert _turn_to_check(slow) >= _turn_to_check(quick) + 0.9


def test_settle_gate_waits_while_the_aircraft_is_off_the_leg_start() -> None:
    m = _start_leg(C.Params(route=[(3.0, 0.0)], settle_s=0.5, settle_m=0.15))
    off = C.Vehicle(pos=(0.3, 0.0, -1.0), heading=0.0)
    on = C.Vehicle(pos=(0.05, 0.0, -1.0), heading=0.0)
    askew = C.Vehicle(pos=(0.05, 0.0, -1.0), heading=0.5)
    assert not m._turn_settled(10.0, 0.0, off)
    assert not m._turn_settled(10.0, 0.0, on)
    assert not m._turn_settled(10.4, 0.0, on)
    assert m._turn_settled(10.6, 0.0, on)
    assert not m._turn_settled(10.7, 0.0, off)
    assert not m._turn_settled(10.8, 0.0, on)          # leaving restarts the hold
    assert not m._turn_settled(11.2, 0.0, on)
    assert m._turn_settled(11.35, 0.0, on)
    assert not m._turn_settled(11.4, 0.0, askew)       # heading outside face_tol_deg resets it too
