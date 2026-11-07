"""Planner checks. Run from the repo root: python3 -m pytest tests/test_planner.py -q"""
import copy
import math
from pathlib import Path

import numpy as np
import pytest

from course_planner import Params, load_map, plan, rasterise, to_px4_ned

MAP = Path(__file__).resolve().parents[1] / "course_planner" / "course_map.json"


@pytest.fixture(scope="module")
def m():
    return load_map(MAP)


@pytest.fixture(scope="module")
def plans(m):
    return {r: plan(m, r) for r in ("gate_b", "window")}


def _xy_path(pl):
    return [(w.x, w.y) for w in pl.waypoints if w.phase != "return"]


def _seg_intersect(p1, p2, q1, q2):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1, d2 = cross(q1, q2, p1), cross(q1, q2, p2)
    d3, d4 = cross(p1, p2, q1), cross(p1, p2, q2)
    return d1 * d2 < 0 and d3 * d4 < 0


def _first_crossing(path, line):
    for i, (a, b) in enumerate(zip(path, path[1:])):
        if _seg_intersect(a, b, line[0], line[1]):
            return i
    return None


def _min_clearance(grid, path):
    worst = math.inf
    for a, b in zip(path, path[1:]):
        for t in np.linspace(0, 1, max(2, int(math.dist(a, b) / 0.01))):
            worst = min(worst, grid.clearance_at(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return worst


@pytest.mark.parametrize("route", ["gate_b", "window"])
def test_path_keeps_clearance(m, plans, route):
    pl = plans[route]
    grid = rasterise(m, pl.params.grid_res_m, open_openings=("window",) if route == "window" else ())
    assert _min_clearance(grid, _xy_path(pl)) >= pl.params.safe_r - pl.params.grid_res_m


@pytest.mark.parametrize("route", ["gate_b", "window"])
def test_checkpoints_crossed_in_order(m, plans, route):
    path = _xy_path(plans[route])
    lines = []
    for cp in m["checkpoints"]:
        spec = cp["options"][route] if cp["kind"] == "choice" else cp
        if spec["kind"] == "line":
            lines.append(spec["line"])
        elif spec["kind"] == "opening":
            op = m["openings"][0]
            c, u, h = np.asarray(op["centre_xy"]), np.asarray(op["along_wall_unit"]), op["width_m"] / 2
            lines.append([tuple(c - u * h), tuple(c + u * h)])
    idx = [_first_crossing(path, ln) for ln in lines]
    assert None not in idx, idx
    assert idx == sorted(idx)


def test_window_height(m, plans):
    op = m["openings"][0]
    for w in plans["window"].waypoints:
        if "window" in w.label:
            assert op["bottom_m"] + 0.2 <= w.z <= op["top_m"] - 0.2


def test_tunnel_is_flown_straight_at_mid_height(m, plans):
    t = m["tunnel"]
    inside = [w for w in plans["gate_b"].waypoints if w.phase == "tunnel"]
    assert inside, "no waypoints inside the tunnel"
    for w in inside:
        assert abs(w.x - t["entry_xy"][0]) < 0.02
        assert abs(w.z - (t["floor_m"] + t["inner_height_m"] / 2)) < 0.01


def test_never_over_a_boulder(m, plans):
    for pl in plans.values():
        for a, b in zip(_xy_path(pl), _xy_path(pl)[1:]):
            for bo in m["boulders"]:
                for t in np.linspace(0, 1, 50):
                    q = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
                    assert math.dist(q, bo["centre_xy"]) > bo["radius_m"] + pl.params.drone_radius_m


def test_return_clears_walls_and_stays_under_net(m, plans):
    ret = [w for w in plans["gate_b"].waypoints if w.phase == "return"]
    cruise = max(w.z for w in ret)
    assert cruise >= max(w["height_m"] for w in m["walls"]) + 0.3
    assert cruise <= m["bounds"]["height_m"] - 0.5
    assert math.dist((ret[-1].x, ret[-1].y), m["launch_mark_xy"]) < 0.05


def test_fits_in_time_limit(plans):
    for pl in plans.values():
        assert pl.time_s(include_return=True) < 170


def test_plans_on_a_changed_course(m):
    """Generality: nudge walls, boulders and the pad; the planner should adapt, not break."""
    m2 = copy.deepcopy(m)
    for bo in m2["boulders"]:
        bo["centre_xy"][0] += 0.15
    for w in m2["walls"]:
        if w["name"].startswith("M2"):
            w["footprint_xy"] = [[x + 0.10, y] for x, y in w["footprint_xy"]]
    m2["landing_pad"]["centre_xy"][1] -= 0.25
    for route in ("gate_b", "window"):
        pl = plan(m2, route)
        grid = rasterise(m2, pl.params.grid_res_m, open_openings=("window",) if route == "window" else ())
        assert _min_clearance(grid, _xy_path(pl)) >= pl.params.safe_r - pl.params.grid_res_m
        pad = [w for w in pl.waypoints if w.label == "touchdown + disarm"][0]
        assert math.dist((pad.x, pad.y), m2["landing_pad"]["centre_xy"]) < 0.05


def test_ned_conversion(m, plans):
    ned = to_px4_ned(plans["gate_b"], m["launch_mark_xy"])
    assert ned[0]["x"] == 0 and ned[0]["y"] == 0 and ned[0]["z"] == 0
    assert ned[1]["z"] == pytest.approx(-0.9)
    first_move = next(w for w in ned if abs(w["x"]) > 0.5)
    assert first_move["x"] > 0 and abs(first_move["y"]) < 0.1     # first corridor runs north
    # a waypoint's yaw faces the next leg: from the Gate A approach the drone flies east
    gate_a = next(w for w in ned if w["label"] == "CP2 Gate A (approach)")
    assert gate_a["yaw"] == pytest.approx(math.pi / 2, abs=0.05)


def test_too_wide_drone_is_reported(m):
    with pytest.raises(ValueError):
        plan(m, "gate_b", Params(drone_radius_m=0.45))
