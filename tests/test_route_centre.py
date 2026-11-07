"""nav.route_planner's keep_centre_m option on the real course files: the route stays as centred as the
course allows (a physical course of boards and tarp is never exactly the published one). The default
(None) must not change: the sims' route digest depends on it."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytest.importorskip("yaml")
from nav.course_map import CourseMap  # noqa: E402
from nav.route_planner import PlannerParams, gate_problems, plan_from_files  # noqa: E402

SDF, CFG = ROOT / "maps" / "course_sdf.npz", ROOT / "maps" / "course.yaml"


def clearance_along(plan, cm, step=0.02):
    P = np.asarray([plan.launch_xy] + plan.map_xy)
    dense = [P[0]]
    for a, b in zip(P, P[1:]):
        n = max(2, int(math.dist(a, b) / step) + 1)
        dense.extend(a + (b - a) * t for t in np.linspace(0.0, 1.0, n)[1:])
    D = np.asarray(dense)
    return cm.distance(np.c_[D, np.full(len(D), plan.z)], "planning")


def test_default_is_the_tight_planner():
    assert PlannerParams().keep_centre_m is None
    a = plan_from_files(SDF, CFG, "launch_sim", 0.70, "window")
    b = plan_from_files(SDF, CFG, "launch_sim", 0.70, "window", keep_centre_m=None)
    assert a.digest == b.digest


@pytest.mark.parametrize("cp4", ["gate_b", "window"])
def test_keep_centre_never_tightens_and_still_passes_every_gate(cp4):
    cm = CourseMap.load(SDF)
    tight = plan_from_files(SDF, CFG, "launch", 0.70, cp4)
    centred = plan_from_files(SDF, CFG, "launch", 0.70, cp4, keep_centre_m=0.02)
    assert gate_problems(centred) == []
    dt, dc = clearance_along(tight, cm), clearance_along(centred, cm)
    assert np.nanmin(dc) >= np.nanmin(dt) - 0.005                    # the bottleneck is no worse
    assert np.nanpercentile(dc, 10) >= np.nanpercentile(dt, 10) - 1e-9   # and the route is no tighter overall
    assert np.mean(dc < 0.45) <= np.mean(dt < 0.45) + 1e-9
    assert centred.length_m < tight.length_m * 1.05                  # at little extra length
    again = plan_from_files(SDF, CFG, "launch", 0.70, cp4, keep_centre_m=0.02)
    assert again.digest == centred.digest                             # still deterministic
