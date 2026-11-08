"""nav/mppi_ros.py pure helpers (no ROS needed): route -> odom, status JSON, occupancy layout, sample colours."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from nav.local_grid import LocalGrid
from nav.local_nav import LocalNav
from nav.mppi_ros import occupancy, route_to_odom, sample_lines, status_dict


def test_route_forward_left_maps_into_odom_with_the_takeoff_pose():
    # take-off at odom (1, 2) facing +y (north): forward = +y, left = -x
    W = route_to_odom([[1.0, 0.0], [1.0, 0.5]], (1.0, 2.0), math.pi / 2)
    np.testing.assert_allclose(W, [[1.0, 3.0], [0.5, 3.0]], atol=1e-12)


def test_status_is_json_with_the_numbers_we_watch():
    nav = LocalNav([(3.0, 0.0)], (0.0, 0.0), 0.0)
    pts = np.c_[np.full(50, 2.0), np.linspace(-1, 1, 50), np.full(50, 0.7)]
    for k in range(6):
        nav.add_depth(pts, (0.0, 0.0, 0.7), 0.7, 0.1 * k)
    nav.mppi.debug_samples = 40
    cmd = nav.step((0.0, 0.0), (0.0, 0.0), 0.6)
    d = json.loads(json.dumps(status_dict(cmd)))
    for key in ("state", "vx", "vy", "yaw", "s", "e_lat", "ess", "crash_frac", "blocked", "ms", "depth_age"):
        assert key in d
    lines = sample_lines(cmd.mppi, 10)
    assert 0 < len(lines) <= 10
    assert all(len(p) == nav.mppi.p.horizon and len(c) == 4 and all(0 <= v <= 1 for v in c) for p, c in lines)


def test_occupancy_is_row_major_with_the_corner_origin():
    g = LocalGrid()
    cam = np.array([0.0, 0.0, 0.7])
    blob = np.array([[2.0, 1.0, 0.7], [2.05, 1.0, 0.7], [2.1, 1.0, 0.7]])     # a 3-cell obstacle (lone cells are dropped)
    for _ in range(2):
        g.integrate(blob, cam, 0.7)
    data, origin, cell, w, h = occupancy(g)
    assert len(data) == w * h and data.dtype == np.int8
    i, j, _ = g.cells(np.array([[2.0, 1.0]]))
    assert data[j[0] * w + i[0]] == 100                                    # ROS: index = row (y) * width + column (x)
    assert origin[0] == pytest.approx(g.origin[0] - cell / 2)
    assert (data == -1).any() and (data == 0).any()
