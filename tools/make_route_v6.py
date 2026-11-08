#!/usr/bin/env python3
"""OFFLINE: the take-off-frame route files for the course with the start rebuilt on 2026-11-07.

    python tools/make_route_v6.py          # writes routes/route_v6_start2{,_stage1}{,_sim}.json

The CP2 gap (~0.75 m between Wall A and Wall B, Darion's drawing) is narrower than the planner's
clearance rule allows, so it is placed BY HAND, as Arnav asked: a point before the gap on its normal,
the gap's widest point (searched on the map along the gap line), a point after it on the normal, all at
the fixed flight height (over the 0.30 m cone). A lead-in straight ahead from the pad lets the drone set
off without turning on the spot. The start-up planner (nav.route_planner) routes the rest (CP3, Gate B,
the pad) from the exit point on the same map. Waypoints only: the map is never read in flight.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nav.course_map import CourseMap  # noqa: E402
from nav.frames import load_launch  # noqa: E402
from nav.route_planner import PlannerParams, gate_problems, plan_route  # noqa: E402

HEIGHT_M, CP4 = 0.7, "Gate B"
LEAD_IN_M = 0.5                 # straight ahead from the pad before the first turn
GAP_STANDOFF_M = 0.45           # before/after the gap, along its normal
STAGE1_EXTRA = 0                # stage 1 lands this many planner corners past the gap's exit point. 0: it lands at
                                # the exit (whole drone across CP2), staying within ~2.5 m of the start, where an
                                # error in the start heading moves it least (the next corner is 4.5 m further east)


def main() -> int:
    cfg_path = ROOT / "maps" / "course.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cm = CourseMap.load(ROOT / "maps" / "course_sdf.npz")
    cp2 = next(c for c in cfg["checkpoints"] if c["id"] == "CP2")
    a, b = np.asarray(cp2["line"], dtype=float)
    n = np.asarray(cp2["direction"], dtype=float)
    n /= np.linalg.norm(n)
    lim = cfg["limits"]
    params = PlannerParams(drone_radius_m=float(lim["drone_radius"]),
                           clearance_m=float(lim["drone_radius"]) + float(lim["clearance_margin"]) + 0.01,
                           keep_centre_m=0.02)
    old = json.loads((ROOT / "route.json").read_text(encoding="utf-8"))["params"]
    fly = {**old, "height_m": HEIGHT_M, "speed_mps": 0.3}          # Arnav: fixed 0.7 m, <= 0.3 m/s at CP2
    for key, suffix in (("launch", ""), ("launch_sim", "_sim")):
        L = load_launch(cfg_path, key)
        z = L["z"] + HEIGHT_M
        s = np.linspace(0.05, 0.95, 91)                              # the gap's widest point, on its line
        line = a + np.outer(s, b - a)
        d = cm.distance(np.c_[line, np.full(len(s), z)], "planning")
        centre = line[int(np.nanargmax(d))]
        pre, post = centre - GAP_STANDOFF_M * n, centre + GAP_STANDOFF_M * n
        yaw = math.radians(L["yaw_deg"])
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        lead = np.array([L["x"], L["y"]]) + LEAD_IN_M * fwd
        rest_launch = {"x": float(post[0]), "y": float(post[1]), "yaw_deg": math.degrees(math.atan2(n[1], n[0])),
                       "z": L["z"]}
        rest_cps = [c for c in cfg["checkpoints"] if c["id"] not in ("CP1", "CP2")]
        plan = plan_route(cm, rest_cps, rest_launch, z, params, CP4)
        if gate_problems(plan):
            print(f"{key}: the planner refused the rest of the course: {gate_problems(plan)}")
            return 1
        corners_map = [lead, pre, centre, post] + [np.asarray(q) for q in plan.map_xy]
        labels = ["lead-in", "CP2 approach", "CP2 gap centre", "CP2 exit"] + list(plan.labels)
        c, sn = math.cos(yaw), math.sin(yaw)
        route = [((q[0] - L["x"]) * c + (q[1] - L["y"]) * sn, -(q[0] - L["x"]) * sn + (q[1] - L["y"]) * c)
                 for q in corners_map]
        src = (f"tools/make_route_v6.py on maps/course.yaml {cfg['course']['name']} (start rebuilt 2026-11-07, "
               f"Darion's corrected drawing, approximate), launch_key {key}, {HEIGHT_M} m, CP4 {CP4}. CP2 by hand: "
               f"gap widest point ({centre[0]:.3f}, {centre[1]:.3f}) map, clearance {np.nanmax(d):.3f} m (drone "
               f"radius {lim['drone_radius']}); approach/exit {GAP_STANDOFF_M} m along the normal. Rest: the "
               f"planner from the exit point, digest {plan.digest}.")
        for name, upto in ((f"route_v6_start2{suffix}.json", len(route)),
                           (f"route_v6_start2_stage1{suffix}.json", 4 + STAGE1_EXTRA)):
            pts = ",\n".join(f"  [{f:.3f}, {l:.3f}]" for f, l in route[:upto])
            stage = "" if upto == len(route) else "STAGE 1 (take off, through the CP2 gap, land past it). "
            text = ('{\n "from": ' + json.dumps(stage + src) + ',\n "frame": "metres from the take-off pose: '
                    '[forward, left]; launch_key ' + key + '",\n "labels": ' + json.dumps(labels[:upto]) +
                    ',\n "route": [\n' + pts + '\n ],\n "params": ' + json.dumps(fly) + '\n}\n')
            json.loads(text)
            (ROOT / "routes" / name).write_text(text, encoding="utf-8")
            print(f"wrote routes/{name}: {upto} corners; gap centre {np.nanmax(d):.3f} m clear")
    return 0


if __name__ == "__main__":
    sys.exit(main())
