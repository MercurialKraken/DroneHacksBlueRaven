"""nav/depth_points.py: depth image -> odom points with the pose at the capture stamp."""
from __future__ import annotations

import math

import numpy as np
import pytest

from nav.depth_points import R_FLU_OPTICAL, depth_to_odom, points_from_depth
from nav.pose_buffer import PoseBuffer

K = np.array([[200.0, 0.0, 160.0], [0.0, 200.0, 100.0], [0.0, 0.0, 1.0]])     # a 320x200 depth image


def wall_image(dist_m: float, shape=(200, 320)) -> np.ndarray:
    """A wall square-on to the camera at dist_m, 16UC1 millimetres, with every third pixel missing (0)."""
    D = np.full(shape, int(round(dist_m * 1000)), dtype=np.uint16)
    D.ravel()[::3] = 0
    return D


def T_base_cam(xyz=(0.104, 0.038, 0.030)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_FLU_OPTICAL
    T[:3, 3] = xyz
    return T


def yaw_q(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])     # wxyz


def test_optical_axes_map_to_flu():
    np.testing.assert_allclose(R_FLU_OPTICAL @ [0, 0, 1], [1, 0, 0])      # optical forward -> base forward
    np.testing.assert_allclose(R_FLU_OPTICAL @ [1, 0, 0], [0, -1, 0])     # optical right -> base right (-left)
    np.testing.assert_allclose(R_FLU_OPTICAL @ [0, 1, 0], [0, 0, -1])     # optical down -> base down


def test_a_wall_ahead_lands_ahead_in_odom_using_the_pose_at_the_capture_stamp():
    buf = PoseBuffer(horizon_s=2.0, max_gap_s=0.2)
    for k in range(11):                                                    # drone flying north (+y) at 0.5 m/s, facing north
        t = 10.0 + 0.1 * k
        buf.add_position(t, [1.0, 0.5 * (t - 10.0), 0.7])
        buf.add_attitude(t, yaw_q(math.pi / 2))
    out = depth_to_odom(wall_image(2.0), K, T_base_cam(), buf, stamp=10.4)   # captured at 10.4 s (y = 0.2)
    assert out is not None
    pts, cam = out
    np.testing.assert_allclose(cam, [1.0 - 0.038, 0.2 + 0.104, 0.73], atol=1e-9)
    np.testing.assert_allclose(pts[:, 1], 0.2 + 0.104 + 2.0, atol=1e-6)   # the wall: 2 m north of the camera
    assert pts[:, 0].min() < 0.0 < 2.0 < pts[:, 0].max()                   # spans left-right (x) ...
    assert pts[:, 2].min() < 0.73 < pts[:, 2].max()                        # ... and up-down
    assert len(pts) < (200 // 2) * (320 // 2)                              # zeros (no data) dropped


def test_no_pose_at_the_stamp_drops_the_frame():
    buf = PoseBuffer(horizon_s=1.0, max_gap_s=0.1)
    buf.add_position(10.0, [0, 0, 0.7])
    buf.add_attitude(10.0, yaw_q(0.0))
    buf.add_position(10.1, [0, 0, 0.7])
    buf.add_attitude(10.1, yaw_q(0.0))
    assert depth_to_odom(wall_image(2.0), K, T_base_cam(), buf, stamp=10.5) is None     # newer than the newest pose
    assert depth_to_odom(wall_image(2.0), K, T_base_cam(), buf, stamp=10.05) is not None


def test_range_limits_and_scale():
    D = np.zeros((200, 320), dtype=np.uint16)
    D[:, :100] = 150                                                       # 0.15 m: too close
    D[:, 100:200] = 2000
    D[:, 200:] = 6000                                                      # 6 m: too far
    P = points_from_depth(D, K, np.eye(4), stride=2, min_m=0.25, max_m=4.5, scale=0.001)
    np.testing.assert_allclose(P[:, 2], 2.0)
    assert len(P) == (200 // 2) * (100 // 2)


def test_feeds_the_grid_end_to_end():
    from nav.local_grid import LocalGrid
    buf = PoseBuffer(horizon_s=2.0, max_gap_s=0.2)
    for k in range(5):
        buf.add_position(10.0 + 0.1 * k, [0.0, 0.0, 0.7])
        buf.add_attitude(10.0 + 0.1 * k, yaw_q(0.0))                       # facing +x
    g = LocalGrid()
    for s in (10.1, 10.2):
        pts, cam = depth_to_odom(wall_image(1.5), K, T_base_cam(), buf, stamp=s)
        g.integrate(pts, cam, 0.7, s)
    i, j, ok = g.cells(np.array([[0.104 + 1.5, 0.038]]))
    assert ok[0] and g.occupied()[i[0], j[0]]
    i, j, ok = g.cells(np.array([[0.8, 0.038]]))
    assert not g.occupied()[i[0], j[0]]
