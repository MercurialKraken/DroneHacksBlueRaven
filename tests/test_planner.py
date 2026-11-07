"""refnav.planner and the stall landing, against the fake PX4 and ray-cast camera in test_refnav_core."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_refnav_core import fly  # noqa: E402

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
