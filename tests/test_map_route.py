"""The route as an elastic band on the real course map (refnav.map_route)."""
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
MAP_MARGIN_M = 0.20            # refnav.node's map_margin_m


def _route_ned():
    pts = json.loads((ROOT / "routes" / "route_v5.json").read_text())["route"]   # a fixed route: the band tests pin its spots
    return [(0.0, 0.0)] + [(f, -l) for f, l in pts]              # forward/left from a north-facing launch


NARROW = (0.53, -0.95)          # the one stretch of route_v5's corridor whose centre is only ~0.40 m clear


def _clearance(path: F.Path) -> np.ndarray:
    x, y = M.MapFrame(LAUNCH).to_map(path.x, path.y)
    return CM.distance(np.c_[x, y, np.full(len(x), Z)])


def _build() -> F.Path:
    return M.builder(CM, LAUNCH, Z, M.DRONE_RADIUS_M + MAP_MARGIN_M, 0.3)(_route_ned())


def test_the_map_frame_round_trips_and_puts_the_launch_at_the_spawn() -> None:
    fr = M.MapFrame(LAUNCH)
    x, y = fr.to_map(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    assert (x[0], y[0]) == pytest.approx((LAUNCH["x"], LAUNCH["y"]))
    assert (x[1] - x[0], y[1] - y[0]) == pytest.approx((0.0, 1.0))          # forward = course +y here
    n, e = fr.to_ned(x, y)
    assert np.allclose(n, [0, 1]) and np.allclose(e, [0, 0])


def test_the_band_keeps_the_corridor_centre_where_the_route_ran_close() -> None:
    path, plain = _build(), F.fillet_path(_route_ned(), 0.3)
    d = path.clearance
    assert np.allclose(d, _clearance(path), equal_nan=True)
    assert np.nanmin(d) >= 0.40 and np.nanmin(_clearance(plain)) < 0.40
    x, y = M.MapFrame(LAUNCH).to_map(path.x, path.y)
    dip = (d[1:-1] <= d[:-2]) & (d[1:-1] <= d[2:]) & (d[1:-1] < 0.47)       # every tight spot along the path
    away = np.hypot(x[1:-1] - NARROW[0], y[1:-1] - NARROW[1]) > 0.3
    assert dip.any() and (d[1:-1][dip & away] >= 0.44).all(), d[1:-1][dip & away].min()


def test_the_band_stays_within_max_shift_of_the_route_and_passes_its_points() -> None:
    path = _build()
    fr = M.MapFrame(LAUNCH)
    C = np.asarray(_route_ned())
    route = np.stack(fr.to_map(C[:, 0], C[:, 1]), 1)
    xy = np.stack(fr.to_map(path.x, path.y), 1)
    a, b = route[:-1], route[1:]
    t = np.clip(np.einsum("nmk,mk->nm", xy[:, None] - a, b - a) / ((b - a) ** 2).sum(1), 0.0, 1.0)
    off = np.linalg.norm(xy[:, None] - (a + t[..., None] * (b - a)), axis=2).min(1)
    assert off.max() <= 0.3 + 0.005
    assert max(np.hypot(*(xy - q).T).min() for q in route) <= 0.3 + 0.005
    assert np.allclose(xy[[0, -1]], route[[0, -1]], atol=1e-6)


def test_the_band_is_smooth_enough_to_fly_without_stopping() -> None:
    path = _build()
    assert np.abs(path.k).max() < 4.0                               # tightest radius 0.25 m
    assert np.degrees(np.abs(F.wrap(np.diff(path.psi)))).max() < 6.0
    v = F.speed_profile(path, 0.4, math.radians(45), 0.25, 0.5, (0.40, 0.60, 0.15))
    assert v[1:-1].min() > 0.1


def test_the_built_path_is_in_ned_with_consistent_headings() -> None:
    path = _build()
    i = len(path.s) // 3
    j = i + 5
    step = math.atan2(path.y[j] - path.y[i], path.x[j] - path.x[i])
    assert abs(float(F.wrap(step - path.psi[i]))) < math.radians(10)
    assert path.x[0] == pytest.approx(0.0, abs=0.02) and path.y[0] == pytest.approx(0.0, abs=0.02)


def test_the_follower_flies_the_whole_map_path() -> None:
    """End to end on the real map's path: strictly increasing s, and a lagging vehicle reaches the end."""
    import test_follow as TF
    path = _build()
    assert (np.diff(path.s) > 0).all()
    c, xt, s, t = TF._fly_lagging(path, TF._follower(path), seconds=200.0)
    assert c.done, (path.length, s[-1])
    assert xt.max() < 0.2


def test_speed_drops_where_the_mapped_path_is_tight() -> None:
    path = _build()
    v = F.speed_profile(path, 0.4, math.radians(45), 0.25, 0.5, (0.40, 0.60, 0.15))
    tight = path.clearance < 0.42
    assert tight.any() and (v[tight] <= 0.15 + 0.4 * 0.1 + 1e-6).all()
    assert v.max() == pytest.approx(0.4)


def test_the_flown_route_clears_every_surface_at_its_height() -> None:
    """route.json at main.py's height (the window route at the window's centre height) keeps 0.40 m everywhere."""
    import re
    height = float(re.search(r"^HEIGHT_M = ([0-9.]+)", (ROOT / "main.py").read_text(), re.M).group(1))
    pts = json.loads((ROOT / "route.json").read_text())["route"]
    path = M.builder(CM, LAUNCH, height + LAUNCH["z"], M.DRONE_RADIUS_M + MAP_MARGIN_M, 0.3)([(0.0, 0.0)] + [(f, -l) for f, l in pts])
    assert np.nanmin(path.clearance) >= 0.40
