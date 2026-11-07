"""The route nudged into clear space on the real course map (refnav.map_route)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from nav.course_map import CourseMap
from nav.frames import load_launch
from refnav import follow as F
from refnav import map_route as M

ROOT = Path(__file__).resolve().parents[1]
CM = CourseMap.load(ROOT / "maps" / "course_sdf.npz")
LAUNCH = load_launch(ROOT / "maps" / "course.yaml", "launch_sim")
Z = 0.8 + LAUNCH["z"]


def _route_ned():
    pts = json.loads((ROOT / "route.json").read_text())["route"]
    return [(0.0, 0.0)] + [(f, -l) for f, l in pts]              # forward/left from a north-facing launch


def _clearance(path: F.Path) -> np.ndarray:
    x, y = M.MapFrame(LAUNCH).to_map(path.x, path.y)
    return CM.distance(np.c_[x, y, np.full(len(x), Z)])


def test_the_map_frame_round_trips_and_puts_the_launch_at_the_spawn() -> None:
    fr = M.MapFrame(LAUNCH)
    x, y = fr.to_map(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    assert (x[0], y[0]) == pytest.approx((LAUNCH["x"], LAUNCH["y"]))
    assert (x[1] - x[0], y[1] - y[0]) == pytest.approx((0.0, 1.0))          # forward = course +y here
    n, e = fr.to_ned(x, y)
    assert np.allclose(n, [0, 1]) and np.allclose(e, [0, 0])


def test_route_v5_keeps_its_clearance_after_nudging() -> None:
    build = M.builder(CM, LAUNCH, Z, 0.28, 0.15, 1.0, 0.3)
    plain = F.fillet_path(_route_ned(), 0.3)
    nudged = build(_route_ned())
    before, after = _clearance(plain), _clearance(nudged)
    assert np.nanmin(after) >= np.nanmin(before)
    assert np.mean(after < 0.42) < np.mean(before < 0.42) / 2, (np.mean(before < 0.42), np.mean(after < 0.42))
    assert np.nanmin(after) >= 0.28 + 0.08, np.nanmin(after)              # never closer than radius + 8 cm
    assert abs(nudged.length - plain.length) < 1.0                         # the same route, not a new one


def test_corners_move_only_where_clearance_is_short_and_within_the_shift() -> None:
    fr = M.MapFrame(LAUNCH)
    C = np.asarray(_route_ned())
    xy = np.stack(fr.to_map(C[:, 0], C[:, 1]), 1)
    out = M.clear_corners(xy, CM, Z, 0.43, 0.3)
    d0 = CM.distance(np.c_[xy, np.full(len(xy), Z)])
    for p, q, d in zip(xy[1:-1], out[1:-1], d0[1:-1]):
        moved = float(np.hypot(*(q - p))) if len(out) == len(xy) else None
        if moved is not None:
            assert moved <= 0.3 + 1e-6
            if d >= 0.43:
                assert moved < 1e-9
    assert np.allclose(out[0], xy[0]) and np.allclose(out[-1], xy[-1])


def test_the_built_path_is_in_ned_with_consistent_headings() -> None:
    path = M.builder(CM, LAUNCH, Z, 0.28, 0.15, 1.0, 0.3)(_route_ned())
    i = len(path.s) // 3
    j = i + 5
    step = math.atan2(path.y[j] - path.y[i], path.x[j] - path.x[i])
    assert abs(float(F.wrap(step - path.psi[i]))) < math.radians(10)
    assert path.x[0] == pytest.approx(0.0, abs=0.02) and path.y[0] == pytest.approx(0.0, abs=0.02)


def test_the_follower_flies_the_whole_map_path() -> None:
    """End to end on the real map's path: strictly increasing s, and a lagging vehicle reaches the end."""
    import test_follow as TF
    path = M.builder(CM, LAUNCH, Z, 0.28, 0.15, 1.0, 0.3)(_route_ned())
    assert (np.diff(path.s) > 0).all()
    c, xt, s, t = TF._fly_lagging(path, TF._follower(path), seconds=200.0)
    assert c.done, (path.length, s[-1])
    assert xt.max() < 0.2


def test_speed_drops_where_the_mapped_path_is_tight() -> None:
    path = M.builder(CM, LAUNCH, Z, 0.28, 0.15, 1.0, 0.3)(_route_ned())
    v = F.speed_profile(path, 0.4, math.radians(45), 0.25, 0.5, 0.5, (0.40, 0.60, 0.15))
    tight = path.clearance < 0.42
    assert tight.any() and (v[tight] <= 0.15 + 0.4 * 0.1 + 1e-6).all()
    assert v.max() == pytest.approx(0.4)
