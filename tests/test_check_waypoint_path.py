"""tools/check_waypoint_path.py measures the path refnav's waypoint mode flies, placed at the launch pose, on the map."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "courses" / "start-2026-11-07"
sys.path.insert(0, str(ROOT / "tools"))
import check_waypoint_path as T  # noqa: E402

from nav.course_map import CourseMap  # noqa: E402
from nav.frames import load_launch  # noqa: E402


def test_the_clearance_is_measured_on_the_flown_path_at_flight_height():
    cm = CourseMap.load(START / "course_sdf.npz")
    launch = {"x": -2.515, "y": 0.914, "yaw_deg": 90.0, "z": 0.07}
    r = T.check({"route": [[1.0, 0.0]]}, cm, launch)                     # 1 m north up the first corridor
    xs = np.linspace(0.0, 1.0, len(r["clearance"]))
    want = cm.distance(np.c_[np.full_like(xs, -2.515), 0.914 + xs, np.full_like(xs, 0.77)])
    assert r["z"] == 0.77 and np.allclose(r["clearance"], want, atol=0.02)
    assert r["min"] == np.nanmin(r["clearance"])


def test_a_route_into_a_wall_fails():
    stage1 = str(START / "routes" / "route_v6_start2_stage1.json")
    assert T.main([stage1, "--course", "start-2026-11-07", "--pose", "-2.515,0.914,0"]) == 1               # launched facing east: into the net


@pytest.mark.parametrize("name, key", [("route_v6_start2", "launch"), ("route_v6_start2_sim", "launch_sim"),
                                       ("route_v6_start2_stage1", "launch"), ("route_v6_start2_stage1_sim", "launch_sim")])
def test_the_arcs_keep_chelseas_cp2_line_through_the_gap(name, key):
    """Her approach, widest point and exit lie on the gap's normal: the arcs must not cut that line."""
    cm = CourseMap.load(START / "course_sdf.npz")
    r = T.check(json.loads((START / "routes" / f"{name}.json").read_text()), cm, load_launch(START / "course.yaml", key))
    assert r["min"] >= 0.34 and np.hypot(r["at_map"][0] + 1.155, r["at_map"][1] - 2.300) < 0.1, r
