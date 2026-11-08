"""tools/check_route.py measures the path refnav's waypoint mode flies, placed at the launch pose, on the map."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_route as T  # noqa: E402

from nav.course_map import CourseMap  # noqa: E402


def test_the_clearance_is_measured_on_the_flown_path_at_flight_height():
    cm = CourseMap.load(ROOT / "maps" / "course_sdf.npz")
    launch = {"x": -2.515, "y": 0.914, "yaw_deg": 90.0, "z": 0.07}
    r = T.check({"route": [[1.0, 0.0]]}, cm, launch)                     # 1 m north up the first corridor
    xs = np.linspace(0.0, 1.0, len(r["clearance"]))
    want = cm.distance(np.c_[np.full_like(xs, -2.515), 0.914 + xs, np.full_like(xs, 0.77)])
    assert r["z"] == 0.77 and np.allclose(r["clearance"], want, atol=0.02)
    assert r["min"] == np.nanmin(r["clearance"])


def test_a_route_into_a_wall_fails():
    stage1 = str(ROOT / "routes" / "route_v6_start2_stage1.json")
    assert T.main([stage1, "--pose", "-2.515,0.914,0"]) == 1               # launched facing east: into the net
