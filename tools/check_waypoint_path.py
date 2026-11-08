#!/usr/bin/env python3
"""OFFLINE (laptop): the clearance of the path the drone flies for a route file (waypoint mode), on a course map.

    python tools/check_waypoint_path.py routes/route_v6_start2.json                      # the real mark, maps/course_sdf.npz
    python tools/check_waypoint_path.py routes/route_v6_start2_sim.json --launch launch_sim --map other.npz
    python tools/check_waypoint_path.py routes/x.json --pose -2.515,0.914,90             # map x, y and yaw (deg) of take-off

The path is built exactly as refnav.launch.py's mission builds it (the route's corners as arcs within corner_cut_m),
placed at the launch pose and flight height, and measured against the map's planning field. The flight reads none
of this: it is how a route is checked against the map before it flies. (tools/check_route.py on chelsea/course-map
measures the straight legs and her shell's flight; this one measures refnav's arcs.)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nav.course_map import CourseMap  # noqa: E402
from nav.frames import load_launch  # noqa: E402
from refnav import core as C  # noqa: E402
from refnav import follow as F  # noqa: E402
from refnav import map_route as M  # noqa: E402

DRONE_PARAMS = {"path_source": "waypoints", "corner_cut_m": 0.05, "height_m": 0.7}   # refnav.launch.py's


def check(doc: dict[str, Any], cm: CourseMap, launch: dict[str, float]) -> dict[str, Any]:
    """-> the flown path's clearance (centre to the nearest surface) at each sample, its minimum and where."""
    prm, _ = C.params_from_doc(doc, DRONE_PARAMS)
    corners = [(0.0, 0.0)] + [(f, -l) for f, l in prm.route]
    path = F.fillet_path(corners, prm.fillet_m, min(prm.arrive_m, prm.corner_cut_m))
    frame = M.MapFrame(launch)
    x, y = frame.to_map(path.x, path.y)
    z = prm.height_m + launch["z"]
    d = cm.distance(np.c_[x, y, np.full(len(x), z)])
    i = int(np.nanargmin(d)) if np.isfinite(d).any() else 0
    cx, cy = frame.to_map(*np.array(corners).T)
    near = int(np.argmin(np.hypot(cx - x[i], cy - y[i])))
    return {"clearance": d, "min": float(d[i]), "at_map": (float(x[i]), float(y[i])), "at_s": float(path.s[i]),
            "length": path.length, "z": z, "near_corner": near, "outside_map": int(np.isnan(d).sum())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("route", help="route file: {route: [[forward, left], ...], params: {...}}")
    ap.add_argument("--map", default=str(ROOT / "maps" / "course_sdf.npz"))
    ap.add_argument("--course", default=str(ROOT / "maps" / "course.yaml"), help="for --launch and base_link_z")
    ap.add_argument("--launch", default="launch", help="course.yaml entry: launch (the real mark) or launch_sim")
    ap.add_argument("--pose", help="x,y,yaw_deg of the take-off pose in the map frame, instead of --launch")
    a = ap.parse_args(argv)
    launch = load_launch(a.course, a.launch)
    if a.pose:
        launch["x"], launch["y"], launch["yaw_deg"] = (float(v) for v in a.pose.split(","))
    doc = json.loads(Path(a.route).read_text(encoding="utf-8"))
    r = check(doc, CourseMap.load(a.map), launch)
    labels = doc.get("labels", []) if isinstance(doc, dict) else []
    k = r["near_corner"]
    name = "take-off" if k == 0 else f"corner {k}" + (f" ({labels[k - 1]})" if k <= len(labels) and labels[k - 1] else "")
    print(f"{a.route}: {r['length']:.2f} m at z {r['z']:.2f} m, take-off ({launch['x']:.3f}, {launch['y']:.3f}) "
          f"yaw {launch['yaw_deg']:.1f} deg, map {a.map}")
    print(f"minimum clearance {r['min']:.3f} m (drone radius {M.DRONE_RADIUS_M} m: {r['min'] - M.DRONE_RADIUS_M:+.3f} m "
          f"to spare) at map ({r['at_map'][0]:.3f}, {r['at_map'][1]:.3f}), {r['at_s']:.2f} m along, near {name}")
    if r["outside_map"]:
        print(f"WARNING: {r['outside_map']} samples are outside the map")
    return 0 if r["min"] > M.DRONE_RADIUS_M and not r["outside_map"] else 1


if __name__ == "__main__":
    sys.exit(main())
