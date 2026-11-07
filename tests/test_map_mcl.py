"""nav.map_mcl.ParticleCorrector on the synthetic maps of test_map_align."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nav.map_align import SE2  # noqa: E402
from nav.map_mcl import MclParams, ParticleCorrector  # noqa: E402
from tests.test_map_align import ROOM_WALLS, corridor, room, wall_points  # noqa: E402


def odom_for(truth, true_T):
    c, s = math.cos(-true_T.yaw), math.sin(-true_T.yaw)
    ox, oy = truth[0] - true_T.x, truth[1] - true_T.y
    return (c * ox - s * oy, s * ox + c * oy, truth[2], truth[3] - true_T.yaw)


def test_converges_to_the_true_correction_in_a_room():
    cm = room()
    truth = (1.4, 1.2, 0.8, math.radians(20))
    true_T = SE2(0.12, -0.08, math.radians(3))
    odom = odom_for(truth, true_T)
    pc = ParticleCorrector(cm, SE2(), MclParams(init_xy_m=0.15, init_yaw_deg=4.0, max_rate_xy=1.0, max_rate_yaw_deg=20.0))
    for k in range(40):
        pc.update(k * 0.2, odom, wall_points(ROOM_WALLS, truth, rng=np.random.default_rng(k)))
    est = pc.out.pose(odom)
    assert math.hypot(est[0] - truth[0], est[1] - truth[1]) < 0.04
    assert abs(math.degrees(est[3] - truth[3])) < 1.0
    assert pc.spread[0] < 0.05 and pc.spread[1] < 0.05


def test_a_straight_corridor_leaves_the_along_axis_wide():
    cm = corridor()
    truth = (0.0, 0.5, 0.8, 0.0)
    pc = ParticleCorrector(cm, SE2(0.0, 0.0, 0.0), MclParams(init_xy_m=0.1))
    for k in range(40):
        pc.update(k * 0.2, truth, wall_points([(-6, 0, 6, 0), (-6, 1, 6, 1)], truth, rng=np.random.default_rng(k)))
    sx, sy, _ = pc.spread
    assert sx > 3 * sy                                  # along the corridor is honestly uncertain
    assert abs(pc.out.pose(truth)[1] - 0.5) < 0.03


def test_frames_that_match_nothing_only_widen_the_cloud():
    cm = room()
    truth = (1.4, 1.2, 0.8, 0.0)
    pc = ParticleCorrector(cm, SE2())
    rng = np.random.default_rng(5)
    junk = np.column_stack([rng.uniform(0.5, 2.5, 400), rng.uniform(-1, 1, 400), rng.uniform(-0.3, 0.3, 400)])
    s0 = None
    for k in range(10):
        pc.update(k * 0.2, truth, junk)
        s0 = s0 or pc.spread[0]
    assert pc.n["weighted"] == 0 and pc.n["skipped"] == 10
    assert pc.spread[0] > s0                            # random walk only: wider, never confidently wrong


def test_output_is_rate_limited():
    cm = room()
    truth = (1.4, 1.2, 0.8, 0.0)
    true_T = SE2(0.5, 0.0, 0.0)
    odom = odom_for(truth, true_T)
    pc = ParticleCorrector(cm, SE2(), MclParams(init_xy_m=0.6, max_rate_xy=0.15))
    prev = pc.out.pose(odom)
    for k in range(1, 20):
        cur = pc.update(k * 0.2, odom, wall_points(ROOM_WALLS, truth, rng=np.random.default_rng(k))).pose(odom)
        assert math.hypot(cur[0] - prev[0], cur[1] - prev[1]) <= 0.15 * 0.2 + 1e-9
        prev = cur
