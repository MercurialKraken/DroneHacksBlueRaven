"""nav.map_align on synthetic maps with exact distance fields."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nav.course_map import CourseMap  # noqa: E402
from nav.map_align import SE2, AlignParams, CorrectorParams, DriftCorrector, align, level_to_map  # noqa: E402

CELL = 0.05


def grid_map(dist_fn, lo, hi):
    xs, ys, zs = (np.arange(lo[i], hi[i] + 1e-9, CELL) for i in range(3))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    f = dist_fn(X, Y, Z).astype(np.float32)
    return CourseMap({"localization": f, "planning": f}, (xs[0], ys[0], zs[0]), CELL)


def room():
    # 4 x 3 m room plus a 1 m stub wall from the north wall at x = 2 (gives yaw and x/y everywhere)
    def d(X, Y, Z):
        box = np.minimum.reduce([X, 4.0 - X, Y, 3.0 - Y])
        stub = np.where(Y > 2.0, np.abs(X - 2.0), np.hypot(X - 2.0, Y - 2.0))     # a thin wall
        return np.minimum(box, stub)
    return grid_map(d, (-0.5, -0.5, 0.0), (4.5, 3.5, 2.0))


def corridor():
    return grid_map(lambda X, Y, Z: np.minimum(Y, 1.0 - Y), (-6.0, -0.5, 0.0), (6.0, 1.5, 2.0))


def wall_points(walls, pose, n=1500, rng=None):
    """Map points on wall segments [(x0,y0,x1,y1)], within 0.4-3.5 m and +-0.3 m of the body height,
    returned in the LEVEL body frame for `pose`."""
    rng = rng or np.random.default_rng(1)
    x, y, z, yaw = pose
    seg = [walls[i] for i in rng.integers(0, len(walls), n * 3)]
    t = rng.random(len(seg))
    P = np.array([[a + (c - a) * u, b + (d - b) * u, z + rng.uniform(-0.3, 0.3)] for (a, b, c, d), u in zip(seg, t)])
    r = np.hypot(P[:, 0] - x, P[:, 1] - y)
    P = P[(r > 0.4) & (r < 3.5)][:n]
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = P[:, 0] - x, P[:, 1] - y
    return np.stack([c * dx + s * dy, s * dx - c * dy, z - P[:, 2]], axis=1)


ROOM_WALLS = [(0, 0, 4, 0), (4, 0, 4, 3), (4, 3, 0, 3), (0, 3, 0, 0), (2.0, 3.0, 2.0, 2.0)]


def test_level_to_map_inverts_the_generator():
    pose = (1.0, 2.0, 0.8, math.radians(30))
    pts = wall_points(ROOM_WALLS, pose)
    P = level_to_map(pts, pose)
    assert np.abs(room().distance(P, "localization")).max() < 0.03


def test_room_offset_is_recovered():
    cm = room()
    truth = (1.4, 1.2, 0.8, math.radians(20))
    pts = wall_points(ROOM_WALLS, truth)
    prior = (truth[0] - 0.15, truth[1] + 0.10, 0.8, truth[3] - math.radians(3))
    r = align(cm, pts, prior)
    assert r.ok, r.why
    assert r.observable.all()
    assert abs(prior[0] + r.delta[0] - truth[0]) < 0.02
    assert abs(prior[1] + r.delta[1] - truth[1]) < 0.02
    assert abs(math.degrees(prior[3] + r.delta[2] - truth[3])) < 0.5
    assert r.rms_after < r.rms_before


def test_position_along_a_straight_corridor_is_not_corrected():
    cm = corridor()
    truth = (0.0, 0.5, 0.8, 0.0)
    pts = wall_points([(-6, 0, 6, 0), (-6, 1, 6, 1)], truth)
    prior = (0.3, 0.42, 0.8, math.radians(2))
    r = align(cm, pts, prior)
    assert r.ok, r.why
    assert not r.observable[0] and r.delta[0] == 0.0           # along the corridor: unknown, left alone
    assert r.observable[1] and abs(prior[1] + r.delta[1] - 0.5) < 0.02
    assert abs(math.degrees(prior[3] + r.delta[2])) < 0.5      # two parallel walls do fix the heading


def test_corrector_converges_smoothly_and_ignores_a_jump():
    cm = room()
    truth = (1.4, 1.2, 0.8, math.radians(20))
    pts = wall_points(ROOM_WALLS, truth)
    # odom = map shifted: the true map <- odom is (0.2, -0.1, 4 deg); the corrector starts at identity
    true_T = SE2(0.2, -0.1, math.radians(4))
    c, s = math.cos(-true_T.yaw), math.sin(-true_T.yaw)
    ox, oy = truth[0] - true_T.x, truth[1] - true_T.y
    odom = (c * ox - s * oy, s * ox + c * oy, 0.8, truth[3] - true_T.yaw)
    dc = DriftCorrector(cm, SE2(), CorrectorParams(tau_s=1.0, max_rate_xy=0.2, max_rate_yaw=math.radians(5)))
    steps = []
    for k in range(60):                                         # 12 s at 5 Hz
        before = dc.T.pose(odom)
        after = dc.update(k * 0.2, odom, pts).pose(odom)
        steps.append(math.hypot(after[0] - before[0], after[1] - before[1]))   # how far the corrected drone moves
    est = dc.T.pose(odom)
    assert math.hypot(est[0] - truth[0], est[1] - truth[1]) < 0.03
    assert abs(math.degrees(est[3] - truth[3])) < 0.5
    assert max(steps) <= 0.2 * 0.2 + 1e-9                       # never faster than max_rate_xy
    # an odom pose 0.9 m off (outside the inlier gate) must not drag the correction after it
    jumped = (odom[0] + 0.9, odom[1], odom[2], odom[3])
    for k in range(10):
        dc.update(12.2 + 0.2 * k, jumped, pts)
    moved = dc.T.pose(odom)
    assert math.hypot(moved[0] - est[0], moved[1] - est[1]) < 0.1


def test_correction_ned_maps_raw_px4_onto_the_corrected_pose():
    from nav.map_align import correction_ned
    start = SE2(-2.5146, 1.1144, 0.0)                    # launch_sim: odom axes = map axes
    T = SE2(start.x + 0.12, start.y - 0.30, math.radians(4.0))   # a corrector state
    xC, yC, yawC = correction_ned(start, T)
    for odom in ((1.6, 0.2, 0.8, math.radians(80)), (-0.5, 3.0, 0.8, math.radians(-30))):   # odom ENU poses
        mx, my, _, myaw = T.pose(odom)                     # where the corrector puts the drone in the map
        want_n, want_e = my - start.y, mx - start.x        # that pose in PX4 NED from the launch point
        n, e = odom[1], odom[0]                            # the raw PX4 position (NED)
        c, s = math.cos(yawC), math.sin(yawC)
        got_n, got_e = c * n - s * e + xC, s * n + c * e + yC
        assert abs(got_n - want_n) < 1e-9 and abs(got_e - want_e) < 1e-9
        heading_px4 = math.pi / 2 - odom[3]                # NED heading from ENU yaw
        want_heading = math.pi / 2 - myaw
        assert abs(math.remainder(heading_px4 + yawC - want_heading, 2 * math.pi)) < 1e-9


def test_points_that_match_no_wall_are_rejected():
    cm = room()
    rng = np.random.default_rng(3)
    junk = np.column_stack([rng.uniform(0.5, 2.5, 600), rng.uniform(-1, 1, 600), rng.uniform(-0.3, 0.3, 600)])
    r = align(cm, junk, (1.4, 1.2, 0.8, 0.0), AlignParams())
    assert not r.ok
