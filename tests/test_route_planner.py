"""nav.route_planner: the route planned at startup from the course map."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from nav.course_map import CourseMap
from nav.route_planner import PlannerParams, gate_problems, plan_from_files, plan_route
from refnav import core as C

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "tests" / "data" / "course_v3_published"      # the published course, frozen (see its README)
SDF, YAML = V3 / "course_sdf.npz", V3 / "course.yaml"
HEIGHT = 0.7                                   # main.py HEIGHT_M: the window's centre height above the launch


def _cross(p, q, a, b) -> bool:
    """Do segments pq and ab intersect (proper crossing)?"""
    def side(u, v, w):
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
    return side(a, b, p) * side(a, b, q) < 0 and side(p, q, a) * side(p, q, b) < 0


def _polyline(plan):
    return [plan.launch_xy] + list(plan.map_xy)


# ------------------------------------------------------------------------------------- the real course
def test_same_map_same_route_bit_for_bit() -> None:
    a = plan_from_files(SDF, YAML, "launch_sim", HEIGHT, "window")
    b = plan_from_files(SDF, YAML, "launch_sim", HEIGHT, "window")
    assert a.route == b.route and a.digest == b.digest
    cm = CourseMap.load(SDF)                   # a fresh copy of the fields changes nothing
    import yaml
    cfg = yaml.safe_load(YAML.read_text(encoding="utf-8"))
    launch = {"x": -2.5146, "y": 1.1144, "yaw_deg": 90.0, "z": 0.07}
    lim = cfg["limits"]
    c = plan_route(CourseMap({k: v.copy() for k, v in cm.fields.items()}, cm.origin, cm.cell), cfg["checkpoints"],
                   launch, HEIGHT + 0.07, PlannerParams(lim["drone_radius"], lim["drone_radius"]
                                                        + lim["clearance_margin"] + 0.01), "window")
    assert c.route == a.route


def test_window_route_crosses_every_checkpoint_and_ends_on_the_pad() -> None:
    plan = plan_from_files(SDF, YAML, "launch_sim", HEIGHT, "window")
    assert gate_problems(plan) == []
    got = {c["checkpoint"]: c for c in plan.crossings}
    assert got["CP1"]["past_m"] > 0.0                       # CP1 as drawn: only partly (see course.yaml)
    for cp in ("CP2", "CP3", "CP4 window"):
        assert got[cp]["whole_drone_across"], got[cp]
    assert got["CP5"]["end_from_centre_m"] < 0.05
    assert [w for w in plan.warnings if not w.startswith("CP1")] == []


def test_the_route_keeps_its_clearance_on_the_map() -> None:
    plan = plan_from_files(SDF, YAML, "launch_sim", HEIGHT, "window")
    cm = CourseMap.load(SDF)
    P = np.asarray(_polyline(plan))
    for a, b in zip(P, P[1:]):
        n = max(2, int(math.dist(a, b) / 0.01) + 1)
        pts = a + (b - a) * np.linspace(0.0, 1.0, n)[:, None]
        d = cm.distance(np.c_[pts, np.full(n, plan.z)])
        assert np.nanmin(d) >= 0.39 - 0.025, (a, b, np.nanmin(d))   # nearest-cell planning vs trilinear lookup


def test_gate_b_route_never_flies_through_the_window() -> None:
    import yaml
    cfg = yaml.safe_load(YAML.read_text(encoding="utf-8"))
    window = next(o for o in cfg["checkpoints"][3]["options"] if o["name"] == "window")["line"]
    plan = plan_from_files(SDF, YAML, "launch_sim", HEIGHT, "Gate B")
    assert gate_problems(plan) == []
    assert {c["checkpoint"]: c for c in plan.crossings}["CP4 Gate B"]["whole_drone_across"]
    P = _polyline(plan)
    assert not any(_cross(p, q, *window) for p, q in zip(P, P[1:]))


def test_the_window_needs_the_windows_height() -> None:
    with pytest.raises(ValueError, match="opening's height"):
        plan_from_files(SDF, YAML, "launch_sim", 1.3, "window")      # z 1.37 m: above the window's top edge


def test_the_planned_route_is_a_valid_refnav_route() -> None:
    plan = plan_from_files(SDF, YAML, "launch", HEIGHT, "window")    # from the real launch mark
    prm, _ = C.params_from_doc({"route": plan.route, "params": {}})
    assert prm.route == plan.route
    x0, y0 = plan.launch_xy
    for (f, left), (x, y) in zip(plan.route, plan.map_xy):            # launch yaw 90: forward = +y, left = -x
        assert (f, left) == pytest.approx((y - y0, -(x - x0)), abs=1e-3)


# ---------------------------------------------------------------- a toy course: the route follows the map
def _box_distance(P: np.ndarray, lo, hi) -> np.ndarray:
    q = np.maximum(np.maximum(np.asarray(lo) - P, P - np.asarray(hi)), 0.0)
    return np.linalg.norm(q, axis=-1)


def _two_rooms(door_y: float) -> CourseMap:
    """A 4 x 4 m room split by a wall at x = 0 with a 1 m door starting at door_y. Unsigned distance is enough here."""
    c = 0.05
    xs, ys, zs = np.arange(-2.2, 2.2 + 1e-9, c), np.arange(-2.2, 2.2 + 1e-9, c), np.arange(0.0, 2.0 + 1e-9, c)
    P = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    boxes = [((-2.2, -2.2, 0), (-2.0, 2.2, 2)), ((2.0, -2.2, 0), (2.2, 2.2, 2)),        # outer walls
             ((-2.2, -2.2, 0), (2.2, -2.0, 2)), ((-2.2, 2.0, 0), (2.2, 2.2, 2)),
             ((-0.05, -2.2, 0), (0.05, door_y, 2)), ((-0.05, door_y + 1.0, 0), (0.05, 2.2, 2))]  # divider + door
    d = np.minimum.reduce([_box_distance(P, lo, hi) for lo, hi in boxes] + [P[..., 2]])
    return CourseMap({"planning": d}, (xs[0], ys[0], zs[0]), c)


TOY_CPS = [{"id": "CP1", "line": None, "direction": [1.0, 0.0]},
           {"id": "CP2", "centre": [1.4, 0.0], "outer_radius": 0.7}]


@pytest.mark.parametrize("door_y", [-1.5, 0.4])
def test_the_route_follows_the_door(door_y: float) -> None:
    cps = [dict(TOY_CPS[0], line=[[0.0, door_y], [0.0, door_y + 1.0]]), TOY_CPS[1]]
    plan = plan_route(_two_rooms(door_y), cps, {"x": -1.4, "y": 0.0, "yaw_deg": 0.0, "z": 0.0}, 1.0)
    assert gate_problems(plan) == []
    P = _polyline(plan)
    hits = [(p, q) for p, q in zip(P, P[1:]) if (p[0] < 0.0) != (q[0] < 0.0)]
    assert len(hits) == 1                                   # crosses the divider exactly once...
    (p, q), = hits
    y_at = p[1] + (q[1] - p[1]) * (0.0 - p[0]) / (q[0] - p[0])
    assert door_y + 0.39 - 0.05 <= y_at <= door_y + 1.0 - 0.39 + 0.05     # ...through the door, clear of its posts
    assert math.dist(P[-1], (1.4, 0.0)) < 0.05
    f, left = plan.route[-1]                                # yaw 0: forward = +x, left = +y
    assert (f, left) == pytest.approx((1.4 - plan.launch_xy[0], 0.0 - plan.launch_xy[1]), abs=0.06)
