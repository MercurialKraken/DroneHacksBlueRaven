"""nav.live_map on a synthetic room: a static map with one wall, a depth camera that ray-casts a
scene, and obstacles the static map doesn't know about. numpy only, no course numbers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nav.course_map import CourseMap  # noqa: E402
from nav.live_map import (CombinedMap, LiveObstacleLayer, LiveParams, ObstacleCloud,  # noqa: E402
                          PoseGate, min_filter)

CELL = 0.05
ORIGIN = np.array([-0.5, -1.5, -0.5])
SHAPE = (81, 61, 41)                       # x -0.5..3.5, y -1.5..1.5, z -0.5..1.5
WALL_X = 3.0                               # the mapped wall: solid for x >= 3.0

W, H, F = 160, 100, 100.0                  # small pinhole camera
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
CAM = np.array([0.0, 0.0, 0.5])            # camera position in map, looking along +x
R_MAP_OPT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)   # optical -> map (cols)


def course() -> CourseMap:
    axes = [ORIGIN[a] + CELL * np.arange(SHAPE[a]) for a in range(3)]
    X, _, _ = np.meshgrid(*axes, indexing="ij")
    return CourseMap({"planning": WALL_X - X}, ORIGIN, CELL)


def T_cam(pos=CAM) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_MAP_OPT
    T[:3, 3] = pos
    return T


def render(boxes=(), wall=True) -> np.ndarray:
    """Depth (m along the optical axis) of the wall at x = WALL_X plus axis-aligned boxes
    [(xmin, xmax, ymin, ymax, zmin, zmax)]; 0 where nothing is hit."""
    v, u = np.mgrid[0:H, 0:W]
    d_opt = np.stack([(u - K[0, 2]) / F, (v - K[1, 2]) / F, np.ones_like(u, dtype=float)], axis=-1)
    d = d_opt @ R_MAP_OPT.T                                # ray directions in map; forward component 1
    best = np.full((H, W), np.inf)
    if wall:
        best = np.where(d[..., 0] > 0, (WALL_X - CAM[0]) / d[..., 0], np.inf)
    for (x0, x1, y0, y1, z0, z1) in boxes:
        with np.errstate(divide="ignore", invalid="ignore"):
            lo = np.stack([(x0 - CAM[0]) / d[..., 0], (y0 - CAM[1]) / d[..., 1], (z0 - CAM[2]) / d[..., 2]], -1)
            hi = np.stack([(x1 - CAM[0]) / d[..., 0], (y1 - CAM[1]) / d[..., 1], (z1 - CAM[2]) / d[..., 2]], -1)
        tmin = np.nanmax(np.minimum(lo, hi), axis=-1)
        tmax = np.nanmin(np.maximum(lo, hi), axis=-1)
        hit = (tmax >= tmin) & (tmax > 0)
        best = np.where(hit & (tmin < best), tmin, best)
    depth = best                                           # forward component of d is 1: t is depth
    return np.where(np.isfinite(depth), depth, 0.0)


BOX = (1.4, 1.6, -0.2, 0.2, 0.2, 0.8)                     # an unmapped obstacle 1.4 m ahead


def fresh(**kw) -> LiveObstacleLayer:
    kw = {"hit_stride_px": 2, "free_stride_px": 2, **kw}  # this camera is coarse: keep rays < 1 voxel apart
    return LiveObstacleLayer(course(), params=LiveParams(**kw))


def test_mapped_wall_adds_no_obstacles():
    live = fresh()
    for _ in range(3):
        st = live.integrate(render(), K, T_cam())
    assert st.explained > 0 and st.unexplained == 0 and st.explained + st.off_map == st.hits
    assert live.trusted
    assert len(live.obstacle_voxels()) == 0
    cm = CombinedMap(course(), live)
    p = np.array([[1.0, 0.0, 0.5]])
    assert cm.distance(p)[0] == pytest.approx(2.0, abs=1e-6)            # static only


def test_unmapped_box_appears_after_min_hits_and_lowers_clearance():
    live = fresh()
    live.integrate(render([BOX]), K, T_cam())
    assert len(live.obstacle_voxels()) == 0                               # one frame is not enough
    st = live.integrate(render([BOX]), K, T_cam())
    assert st.unexplained > 0 and st.obstacles > 0
    pts = live.obstacle_points()
    assert np.all((pts[:, 0] > 1.3) & (pts[:, 0] < 1.7))                  # on the box's front face
    cm = CombinedMap(course(), live)
    p = np.array([[1.0, 0.0, 0.5]])                                        # 0.4 m in front of the box
    d, g = cm.query(p)
    assert d[0] == pytest.approx(0.4, abs=0.06)
    assert g[0, 0] < -0.9                                                  # points away from the box
    assert not cm.clearance_ok(p, need=0.5)[0] and cm.clearance_ok(p, need=0.3)[0]


def test_single_frame_phantom_is_ignored_and_cleared():
    live = fresh()
    live.integrate(render([BOX]), K, T_cam())                             # phantom for one frame
    for _ in range(3):
        live.integrate(render(), K, T_cam())                              # then rays see through it
    assert len(live.obstacle_voxels()) == 0


def test_obstacle_that_leaves_is_cleared():
    live = fresh()
    for _ in range(2):
        live.integrate(render([BOX]), K, T_cam())
    assert len(live.obstacle_voxels()) > 0
    for _ in range(6):
        live.integrate(render(), K, T_cam())
    assert len(live.obstacle_voxels()) == 0


def test_no_return_is_unknown_not_free():
    live = fresh()
    D = render()
    D[:, : W // 2] = 0.0                                                   # left half returns nothing
    live.integrate(D, K, T_cam())
    left = np.array([[2.0, 1.0, 0.5]])                                     # the camera's left = map +y
    right = np.array([[2.0, -1.0, 0.5]])
    assert not live.observed(left)[0]
    assert live.observed(right)[0]


def test_min_filter_joins_sparse_returns_and_keeps_empty_space_empty():
    D = np.full((20, 20), np.inf, dtype=np.float32)
    D[:, ::4] = 1.5                                                        # net-like: every 4th column
    out = min_filter(D, 5)
    assert np.isfinite(out[:, 2:18]).all()                                 # the gaps are bridged
    E = np.full((20, 20), np.inf, dtype=np.float32)
    assert np.isinf(min_filter(E, 5)).all()                                # no returns stay no returns


def test_off_map_returns_are_ignored():
    live = fresh(max_range_m=10.0)
    st = live.integrate(render([(4.0, 4.2, -0.5, 0.5, 0.0, 1.0)], wall=False), K, T_cam())
    assert st.off_map == st.hits and st.unexplained == 0


def test_misaligned_map_is_refused_not_trusted():
    live = fresh(max_obstacle_voxels=5)
    for _ in range(2):
        st = live.integrate(render([BOX]), K, T_cam())
    assert st.notes and not live.trusted and live.as_course_map() is None
    cm = CombinedMap(course(), live)
    assert cm.distance(np.array([[1.0, 0.0, 0.5]]))[0] == pytest.approx(2.0, abs=1e-6)


def test_frame_with_a_wrong_pose_is_refused():
    live = fresh()
    D = render([BOX])                                                      # rendered from the TRUE pose
    for _ in range(3):
        st = live.integrate(D, K, T_cam(CAM + [0.3, 0.0, 0.0]))            # pose estimate 0.3 m off
    assert st.rejected and st.misfit is not None and st.misfit > 0.9
    assert len(live.obstacle_voxels()) == 0 and not live.observed(np.array([[1.0, 0.0, 0.5]]))[0]
    st = live.integrate(D, K, T_cam())                                     # the right pose is accepted
    assert not st.rejected and st.misfit is not None and st.misfit < 0.1


def test_obstacle_cloud_rebuilds_the_same_combined_map():
    live = fresh()
    for _ in range(2):
        live.integrate(render([BOX]), K, T_cam())
    c = course()
    p = np.random.default_rng(1).uniform([0.5, -1.0, 0.0], [2.5, 1.0, 1.0], size=(200, 3))
    direct = CombinedMap(c, live).distance(p)
    via_cloud = CombinedMap(c, ObstacleCloud(c, live.obstacle_points())).distance(p)
    assert np.allclose(direct, via_cloud)
    assert np.allclose(CombinedMap(c, ObstacleCloud(c, live.obstacle_points(), trusted=False)).distance(p),
                       c.distance(p))


def test_pose_gate():
    g = PoseGate(recover_s=3.0, stale_s=1.0, max_speed_mps=2.5)
    assert g.check(0.0, [0, 0, 1])[0] is False                             # no health heard yet
    g.on_health(0.0, True)
    assert g.check(0.1, [0, 0, 1]) == (True, "ok")
    g.on_health(1.0, False)
    assert g.check(1.1, [0.01, 0, 1])[0] is False                          # unhealthy
    g.on_health(1.5, True)
    assert g.check(2.0, [0.02, 0, 1])[0] is False                          # recovering until 4.0
    g.on_health(4.1, True)
    assert g.check(4.2, [0.03, 0, 1])[0] is True
    g.on_health(4.4, True)
    ok, why = g.check(4.4, [0.63, 0, 1])                                   # 0.6 m in 0.2 s: a VIO jump
    assert not ok and "jumped" in why
    g.on_health(5.0, True)
    assert g.check(5.0, [0.63, 0, 1])[0] is False                          # still recovering
    g.on_health(7.5, True)
    assert g.check(7.5, [0.64, 0, 1])[0] is True
    assert g.check(9.0, [0.64, 0, 1])[0] is False                          # health went stale


def test_combined_without_live_layer_is_the_static_map():
    c = course()
    cm = CombinedMap(c)
    p = np.random.default_rng(0).uniform([-0.4, -1.4, -0.4], [3.4, 1.4, 1.4], size=(50, 3))
    assert np.allclose(cm.distance(p), c.distance(p))
    assert np.isnan(cm.distance(np.array([[9.0, 0.0, 0.0]])))[0]
    assert not cm.clearance_ok(np.array([[9.0, 0.0, 0.0]]), need=0.1)[0]
