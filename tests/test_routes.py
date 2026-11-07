"""make_routes produces the route files we flew / plan to fly, and they keep the planned clearance."""
import json
import math

from course_planner import load_map, rasterise
from course_planner.make_routes import HERE, build


def _xy(L, q):
    return (L[0] - q[1], L[1] + q[0])     # (forward, left) with start heading north -> (east, north)


def test_routes_shape_and_params():
    r = build()
    assert len(r["route_v1.json"][0]) == 29
    for name in ("route_fast.json", "route_v2.json", "route_v3.json"):
        assert len(r[name][0]) == 19
    v3 = r["route_v3.json"][1]
    assert v3["min_corridor_coverage"] == 0.4
    assert len(v3["leg_yaw_off_deg"]) == 19
    assert v3["leg_yaw_off_deg"][6] == 15.0 and sum(abs(x) for x in v3["leg_yaw_off_deg"]) == 15.0
    assert all(isinstance(x, float) for x in v3["leg_yaw_off_deg"])   # ROS param lists must be all floats
    json.dumps(v3)


def test_v4_leg_ends_survive_the_overshoot():
    m = load_map(HERE / "course_map.json")
    g = rasterise(m, 0.025)
    L = m["launch_mark_xy"]
    route, params = build()["route_v4.json"]
    assert params["corridor_half_m"] >= 0.29 and params["stop_m"] > 0.4
    pts = [tuple(L)] + [_xy(L, q) for q in route]
    assert len(pts) == 20
    for a, b in zip(pts[:-2], pts[1:-1]):                 # every leg but the last, which lands on the pad
        d = math.dist(a, b)
        past = (b[0] + (b[0] - a[0]) / d * 0.13, b[1] + (b[1] - a[1]) / d * 0.13)
        assert g.clearance_at(*past) >= 0.29
    assert math.dist(pts[-1], m["landing_pad"]["centre_xy"]) < 0.05


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
