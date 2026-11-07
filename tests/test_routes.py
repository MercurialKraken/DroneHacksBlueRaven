"""make_routes produces the route files we flew / plan to fly, and they keep the planned clearance."""
import json
import math

import pytest

from course_planner import load_map, rasterise
from course_planner.make_routes import FLYX_SPAWN_XY, HERE, build

pytest.importorskip("scipy", reason="route generation requires scipy; see make_routes.build")


def _xy(L, q):
    return (L[0] - q[1], L[1] + q[0])     # (forward, left) with start heading north -> (east, north)


def test_routes_shape_and_params():
    r = build()
    assert len(r["route_v1.json"][0]) == 29
    for name in ("route_fast.json", "route_v2.json", "route_v3.json"):
        assert len(r[name][0]) == 19
    v3 = r["route_v3.json"][1]
    assert v3["min_corridor_coverage"] == 0.4
    json.dumps(v3)


def test_committed_route_files_match_the_generator():
    root = HERE.parent
    for name, (route, params) in build().items():
        doc = json.loads((root / "routes" / name).read_text())
        assert doc["params"] == params, name
        assert doc["route"] == [[float("%.3f" % a), float("%.3f" % b)] for a, b in route], name
    flown, v5 = (json.loads((root / name).read_text())["route"] for name in ("route.json", "routes/route_v5.json"))
    assert flown[:14] == v5[:14]           # the flown route is v5 to the tunnel exit, then through the window


def test_v5_is_v4_flown_from_the_spawn():
    m = load_map(HERE / "course_map.json")
    r = build()
    v4 = [_xy(m["launch_mark_xy"], q) for q in r["route_v4.json"][0]]
    v5 = [_xy(FLYX_SPAWN_XY, q) for q in r["route_v5.json"][0]]
    assert len(v4) == len(v5)
    assert max(math.dist(a, b) for a, b in zip(v4, v5)) < 0.05     # same course points, within pull-back rounding


def test_fast_route_keeps_clearance_and_ends_on_pad():
    m = load_map(HERE / "course_map.json")
    g = rasterise(m, 0.025)
    L = m["launch_mark_xy"]
    route, _ = build()["route_v3.json"]
    pts = [tuple(L)] + [_xy(L, q) for q in route]
    for a, b in zip(pts, pts[1:]):
        n = max(2, int(math.dist(a, b) / 0.01))
        worst = min(g.clearance_at(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
                    for t in (i / (n - 1) for i in range(n)))
        assert worst >= 0.34 - 0.025
    assert math.dist(pts[-1], m["landing_pad"]["centre_xy"]) < 0.05
