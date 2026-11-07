"""nav.wall_centring on a synthetic corridor: the map says the walls are at y = +-0.5; the 'real' walls the
simulated depth camera sees may be moved, missing or cluttered, and the drone's believed pose may be off."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nav.course_map import CourseMap  # noqa: E402
from nav.path_geometry import build_path  # noqa: E402
from nav.wall_centring import CentringParams, Measurement, WallCentring, points_from_depth  # noqa: E402

CELL = 0.05
HALF_T = 0.02                                   # walls 4 cm thick: faces at +-0.48 for centres at +-0.5
Z = 0.77
FOV = math.radians(38.6)                         # half of skybeta-09's 77 deg (fx 642 at 1024 px)


def box_sdf(X, Y, cx, cy, hx, hy):
    qx, qy = np.abs(X - cx) - hx, np.abs(Y - cy) - hy
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)


def corridor_map(walls=((0.5, -1.0, 6.0), (-0.5, -1.0, 6.0))) -> CourseMap:
    xs, ys, zs = np.arange(-2, 8.001, CELL), np.arange(-2, 2.001, CELL), np.arange(0, 2.001, CELL)
    X, Y, _ = np.meshgrid(xs, ys, zs, indexing="ij")
    d = np.full(X.shape, 5.0)
    for yc, x0, x1 in walls:
        d = np.minimum(d, box_sdf(X, Y, (x0 + x1) / 2, yc, (x1 - x0) / 2, HALF_T))
    d = d.astype(np.float32)
    return CourseMap({"localization": d, "planning": d}, (xs[0], ys[0], zs[0]), CELL)


def sense(true_pose, believed_pose, faces, rays=96, levels=(-0.1, 0.0, 0.1), noise=0.01, extra=None, seed=0):
    """Wall points the camera sees from the TRUE pose, expressed in the map with the BELIEVED pose.
    faces: [(y_face, x0, x1)]; extra: (M, 2) true-world points to add (people, clutter)."""
    rng = np.random.default_rng(seed)
    (tx, ty, tyaw), (bx, by, byaw) = true_pose, believed_pose
    hits = []
    for th in np.linspace(-FOV, FOV, rays):
        a = tyaw + th
        dx, dy = math.cos(a), math.sin(a)
        best = None
        for yf, x0, x1 in faces:
            if abs(dy) < 1e-9:
                continue
            t = (yf - ty) / dy
            if 0.2 < t < 3.0 and x0 <= tx + t * dx <= x1 and (best is None or t < best):
                best = t
        if best is not None:
            hits.append((tx + best * dx, ty + best * dy))
    P = np.asarray(hits + ([] if extra is None else [tuple(q) for q in extra]), dtype=float).reshape(-1, 2)
    P = P + rng.normal(0, noise, P.shape)
    c, s = math.cos(byaw - tyaw), math.sin(byaw - tyaw)
    rel = P - [tx, ty]
    Pb = np.c_[c * rel[:, 0] - s * rel[:, 1] + bx, s * rel[:, 0] + c * rel[:, 1] + by]
    return np.concatenate([np.c_[Pb, np.full(len(Pb), Z + h)] for h in levels])


def faces_of(left_y, right_y, x0=-1.0, x1=6.0):
    """The wall FACES the corridor sees, for walls centred at left_y / right_y."""
    out = []
    if left_y is not None:
        out.append((left_y - HALF_T, x0, x1))
    if right_y is not None:
        out.append((right_y + HALF_T, x0, x1))
    return out


CM = corridor_map()
PATH = build_path([(-0.5, 0.0), (5.5, 0.0)], None, z=Z)


def measure(true_pose, believed_pose, faces, **kw):
    wc = WallCentring(CM)
    s = PATH.project(believed_pose[:2])
    return wc.measure(sense(true_pose, believed_pose, faces, **kw), (believed_pose[0], believed_pose[1], Z), PATH, s)


def test_finds_the_mapped_walls():
    wc = WallCentring(CM)
    eL, eR = wc.mapped_walls(np.array([1.0, 0.0]), np.array([1.0, 0.0]), Z, np.array([0.6, 1.0, 1.5]))
    assert np.allclose(eL, 0.48, atol=0.03) and np.allclose(eR, -0.48, atol=0.03)


def test_matching_world_gives_no_offset():
    m = measure((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), faces_of(0.5, -0.5))
    assert m.ok and m.why == "both walls" and abs(m.offset) < 0.02


def test_a_shifted_corridor_moves_the_centre():
    m = measure((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), faces_of(0.62, -0.38))        # built 12 cm to the left
    assert m.ok and m.offset == pytest.approx(0.12, abs=0.02)
    assert m.resid_left == pytest.approx(0.12, abs=0.02) and m.resid_right == pytest.approx(0.12, abs=0.02)


def test_pose_error_cancels_so_the_true_drone_is_centred():
    # real walls as mapped; the drone is truly at y = -0.10 but believes y = 0
    m = measure((1.0, -0.10, 0.0), (1.0, 0.0, 0.0), faces_of(0.5, -0.5))
    assert m.ok and m.offset == pytest.approx(0.10, abs=0.02)
    # steering the believed y to 0 + offset moves the true drone by +offset: back to y = 0, the real centre.
    # A heading error: the nose is truly 3 deg left of what the drone believes, so the real corridor ~1 m
    # ahead lies ~tan(3 deg) = 5 cm to the believed right. Steering there leads it into the real corridor.
    m2 = measure((1.0, 0.0, math.radians(3.0)), (1.0, 0.0, 0.0), faces_of(0.5, -0.5))
    assert m2.ok and m2.offset == pytest.approx(-math.tan(math.radians(3.0)) * 1.0, abs=0.02)


def test_a_wider_corridor_keeps_the_middle():
    m = measure((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), faces_of(0.6, -0.6))
    assert m.ok and abs(m.offset) < 0.02 and m.resid_left > 0.08 and m.resid_right < -0.08


def test_one_wall_seen_keeps_the_mapped_distance_from_it():
    m = measure((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), faces_of(0.58, None))
    assert m.ok and m.why == "one wall" and m.offset == pytest.approx(0.08, abs=0.02)


def test_people_behind_the_wall_line_and_clutter_in_the_corridor_are_ignored():
    people = [(x, 1.2) for x in np.linspace(1.5, 2.5, 40)] + [(x, -0.05) for x in np.linspace(1.8, 2.0, 20)]
    m = measure((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), faces_of(0.5, -0.5), extra=people)
    assert m.ok and abs(m.offset) < 0.02


# ---------------------------------------------------------------------------------------- nets
def net_side(y_face, x0=0.8, x1=3.0, behind=(0.3, 1.5), frac_behind=0.6, n=120, seed=1):
    """Returns from a net on the side at y_face: a few strand hits on it, most from things BEHIND it."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(x0, x1, n)
    sign = 1.0 if y_face > 0 else -1.0
    through = rng.random(n) < frac_behind
    y = np.where(through, y_face + sign * rng.uniform(*behind, n), y_face + rng.normal(0, 0.01, n))
    return np.c_[xs, y]


def test_a_see_through_side_is_ignored_and_the_solid_side_is_used():
    # the map says boards on both sides; the real right side is mesh: most returns come from behind it
    faces = faces_of(0.55, None)                                           # real left wall 5 cm further out
    m = measure((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), faces, extra=net_side(-0.48))
    assert m.ok and m.why == "one wall" and "see-through" in m.right
    assert m.offset == pytest.approx(0.05, abs=0.02)                       # from the solid left wall only


def test_returns_scattered_just_behind_a_net_are_not_a_wall():
    # strands plus clutter within match_m behind the net: few land far behind, but they scatter
    rng = np.random.default_rng(3)
    xs = rng.uniform(0.8, 3.0, 150)
    clutter = np.c_[xs, -0.48 - rng.uniform(0.0, 0.22, 150)]
    m = measure((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), faces_of(0.5, None), extra=clutter)
    assert m.ok and "scattered" in m.right and abs(m.offset) < 0.02


def test_a_side_the_map_marks_as_a_net_is_never_used():
    # map: a board on the left, a NET on the right (in the planning field, not in the walls-only field)
    xs, ys, zs = np.arange(-2, 8.001, CELL), np.arange(-2, 2.001, CELL), np.arange(0, 2.001, CELL)
    X, Y, _ = np.meshgrid(xs, ys, zs, indexing="ij")
    wall = box_sdf(X, Y, 2.5, 0.5, 3.5, HALF_T)
    net = box_sdf(X, Y, 2.5, -0.5, 3.5, 0.005)
    cm = CourseMap({"localization": wall.astype(np.float32), "planning": np.minimum(wall, net).astype(np.float32)},
                   (xs[0], ys[0], zs[0]), CELL)
    wc = WallCentring(cm)
    # the camera sees the net's strands exactly where mapped, but with things moving behind it
    pts = sense((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), faces_of(0.5, None), extra=net_side(-0.495, frac_behind=0.2))
    m = wc.measure(pts, (1.0, 0.0, Z), PATH, PATH.project((1.0, 0.0)))
    assert m.ok and m.why == "one wall" and "net" in m.right and abs(m.offset) < 0.02


def test_no_measurement_outside_corridors_or_while_turning():
    open_map = corridor_map(walls=((0.5, -1.0, 0.5),))                         # walls end behind the drone
    wc = WallCentring(open_map)
    pts = sense((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), faces_of(0.5, -0.5))
    assert not wc.measure(pts, (1.0, 0.0, Z), PATH, PATH.project((1.0, 0.0))).ok
    wc = WallCentring(CM)
    m = wc.measure(pts, (1.0, 0.0, Z), PATH, PATH.project((1.0, 0.0)), yaw_rate=math.radians(40))
    assert not m.ok and m.why == "turning"
    m = wc.measure(np.zeros((0, 3)), (1.0, 0.0, Z), PATH, PATH.project((1.0, 0.0)))
    assert not m.ok and "too few" in m.left and "too few" in m.right


def test_filter_converges_caps_holds_and_decays():
    p = CentringParams(tau_s=0.5, cap_m=0.15, hold_s=1.0, decay_s=1.0)
    wc = WallCentring(CM, p)
    wc.update(0.0, Measurement(True, "both walls", 0.10))
    assert wc.offset == pytest.approx(0.10)                              # first measurement taken as is
    t = 0.0
    for _ in range(50):
        t += 0.2
        wc.update(t, Measurement(True, "both walls", 0.40))
    assert wc.offset == pytest.approx(0.15)                              # capped
    t_last = t
    for _ in range(4):
        t += 0.2
        wc.update(t, None)
    assert wc.offset == pytest.approx(0.15)                              # held for hold_s
    while t < t_last + 6.0:
        t += 0.2
        wc.update(t, Measurement(False, "turning"))
    assert abs(wc.offset) < 0.01                                         # then faded out


def test_points_from_depth_back_projects_through_the_intrinsics():
    K = np.array([[348.4, 0, 256.0], [0, 348.4, 160.0], [0, 0, 1]])
    D = np.full((320, 512), 2000, dtype=np.uint16)                       # a wall 2 m ahead, 16UC1 millimetres
    D[:10] = 0                                                           # invalid pixels
    P = points_from_depth(D, K, np.eye(4), stride=8, scale=0.001)
    assert np.allclose(P[:, 2], 2.0) and len(P) == (320 // 8 - 1) * (512 // 8)
    centre = P[np.argmin(np.hypot(P[:, 0], P[:, 1]))]
    assert abs(centre[0]) < 0.03 and abs(centre[1]) < 0.03
    # optical (x right, y down, z forward) -> body FLU (x forward, y left, z up)
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    Pb = points_from_depth(D, K, T, stride=8, scale=0.001)
    assert np.allclose(Pb[:, 0], 2.0)
