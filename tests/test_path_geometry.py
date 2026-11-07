"""nav.path_geometry on synthetic corridors (analytic distance fields): fillets, clearance limits,
sharp corners, windowed projection, heights and checkpoint crossings. No course numbers."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nav.path_geometry import build_path, checkpoint_crossings, smooth_path  # noqa: E402


class Boxes:
    """Free space = a union of axis-aligned rectangles (x0, x1, y0, y1); distance to the walls is
    approximated (conservatively) by the best rectangle's inner distance. Height ignored."""

    def __init__(self, *rects):
        self.rects = rects

    def distance(self, p, field="planning"):
        p = np.atleast_2d(p)
        d = np.full(len(p), -np.inf)
        for x0, x1, y0, y1 in self.rects:
            inner = np.minimum.reduce([p[:, 0] - x0, x1 - p[:, 0], p[:, 1] - y0, y1 - p[:, 1]])
            d = np.maximum(d, inner)
        return d


OPEN = Boxes((-50, 50, -50, 50))
L_CORRIDOR = Boxes((-1.0, 2.0, 0.0, 1.0), (1.0, 2.0, 0.0, 4.0))      # 1 m wide, turning north at x 1-2
ROUTE_L = [(-0.5, 0.5), (1.5, 0.5), (1.5, 3.5)]


def test_straight_line():
    p = build_path([(0, 0), (3, 0)])
    assert p.length == pytest.approx(3.0)
    assert np.allclose(p.psi, 0.0) and np.all(p.k == 0.0)
    assert np.max(np.diff(p.s)) <= 0.02 + 1e-9 and p.radii == []


def test_open_space_corner_is_a_fillet_at_max_radius():
    p = build_path([(0, 0), (3, 0), (3, 3)], OPEN, max_radius=1.0)
    c = p.corners[0]
    assert c.radius == pytest.approx(1.0) and c.limit == "max" and c.turn_rad == pytest.approx(math.pi / 2)
    assert p.length == pytest.approx(2 + math.pi / 2 + 2, abs=0.01)          # 2 m + quarter circle + 2 m
    on_arc = p.k > 0
    assert np.allclose(p.k[on_arc], 1.0)                                     # left turn, radius 1
    dpsi = np.abs(np.diff(np.unwrap(p.psi)))
    assert dpsi.max() < 0.03                                                 # heading is continuous
    assert p.sample(p.length)["psi"] == pytest.approx(math.pi / 2)


def test_short_legs_limit_the_radius():
    p = build_path([(0, 0), (0.6, 0), (0.6, 0.6), (1.2, 0.6)], OPEN, max_radius=5.0)
    for c in p.corners:                                                      # tangent = r for 90 deg
        assert c.limit == "leg" and c.radius == pytest.approx(0.3, abs=1e-6)


def test_clearance_limits_the_fillet_and_every_sample_keeps_it():
    p = build_path(ROUTE_L, L_CORRIDOR, drone_radius=0.28, margin=0.10, max_radius=2.0)
    c = p.corners[0]
    assert c.limit == "clearance" and 0.0 < c.radius < 1.0
    assert np.all(p.clearance >= 0.38 - 1e-9)
    bigger = build_path(ROUTE_L, None, max_radius=c.radius + 0.05)           # a slightly bigger arc...
    assert np.min(L_CORRIDOR.distance(bigger.xyz)) < 0.38                    # ...would cut the corner


def test_a_little_room_gives_a_small_radius():
    snug = Boxes((-1.0, 2.0, 0.0, 0.8), (1.2, 2.0, 0.0, 4.0))                # 0.8 m: 2 cm to spare
    p = build_path([(-0.5, 0.4), (1.6, 0.4), (1.6, 3.5)], snug, drone_radius=0.28, margin=0.10)
    assert 0.0 < p.corners[0].radius < 0.15 and np.all(p.clearance >= 0.38 - 1e-9)


def test_no_room_leaves_a_sharp_corner_to_rotate_in_place():
    tight = Boxes((-1.0, 2.0, 0.0, 0.76), (1.22, 1.98, 0.0, 4.0))            # 0.76 m: exactly radius + margin
    p = build_path([(-0.5, 0.38), (1.6, 0.38), (1.6, 3.5)], tight, drone_radius=0.28, margin=0.10)
    c = p.corners[0]
    assert c.radius == 0.0 and c.limit == "clearance"
    i = int(np.argmin(np.hypot(p.x - 1.6, p.y - 0.38)))
    assert math.isinf(p.k[i])                                                # stop and rotate here
    assert p.project((1.6, 0.38)) == pytest.approx(p.corners[0].s, abs=0.03)


def test_projection_window_does_not_jump_corridors():
    p = build_path([(0, 0), (4, 0), (4, 1), (0, 1)], OPEN, max_radius=0.3)  # a U-turn, legs 1 m apart
    q = (1.0, 0.9)                                                           # closer to the return leg
    s_free = p.project(q)
    s_win = p.project(q, s_lo=0.5, s_hi=1.6)                                 # but we are on the outbound leg
    assert p.sample(s_free)["y"] == pytest.approx(1.0, abs=0.01)
    assert p.sample(s_win)["y"] == pytest.approx(0.0, abs=0.01) and s_win == pytest.approx(1.0, abs=0.01)


def test_heights_ramp_on_legs_and_hold_on_arcs():
    p = build_path([(0, 0, 1.0), (3, 0, 0.8), (3, 3, 0.8)], OPEN, max_radius=0.5)
    assert p.z[0] == pytest.approx(1.0) and p.z[-1] == pytest.approx(0.8)
    assert np.allclose(p.z[p.k > 0], 0.8)
    assert np.all(np.diff(p.z) <= 1e-12)                                     # never climbs back


def test_checkpoint_crossings():
    p = build_path([(0, 0), (2.0, 0)])
    cps = [{"id": "CPa", "line": [[1.0, -1.0], [1.0, 1.0]], "direction": [1, 0]},
           {"id": "CPb", "line": [[1.9, -1.0], [1.9, 1.0]], "direction": [1, 0]},
           {"id": "CPc", "options": [{"name": "far", "line": [[5, -1], [5, 1]], "direction": [1, 0]}]},
           {"id": "PAD", "centre": [2.0, 0.0], "outer_radius": 0.5}]
    r = {x["checkpoint"]: x for x in checkpoint_crossings(p, cps, drone_radius=0.28)}
    assert r["CPa"]["whole_drone_across"] and r["CPa"]["past_m"] == pytest.approx(1.0)
    assert not r["CPb"]["whole_drone_across"] and r["CPb"]["past_m"] == pytest.approx(0.1)
    assert not r["CPc far"]["whole_drone_across"]
    assert r["PAD"]["whole_drone_inside"]


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        build_path([(0, 0)])
    with pytest.raises(ValueError):
        build_path([(0, 0), (0, 0)])
    with pytest.raises(ValueError):
        build_path(np.zeros((3, 4)))
