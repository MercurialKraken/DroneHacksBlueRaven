"""Regenerate the route.json files the starter's refnav mission flies.

    python3 -m course_planner.make_routes            # writes routes/ next to this package
    python3 -m course_planner.make_routes --out DIR  # somewhere else

Outputs (all for the Gate B route, all from course_map.json, nothing hand-typed):
  route_v1.json    29 legs: every vertex the planner produced. Flew legs 1-11 in the sim (run 1).
  route_fast.json  19 legs: v1 with corner vertices merged where the straight line keeps full clearance.
  route_v2.json    route_fast's points.
  route_v3.json    route_fast + min_corridor_coverage 0.4. Predicted to hit the net.
  route_v4.json    v3 with leg ends pulled back from walls, plus wider stop checks. Assumes take-off at the 0.914 mark.
  route_v5.json    v4 from the sim's real spawn (-2.5146, 1.1144). Current candidate; route.json.

refnav route format: [[forward_m, left_m], ...] from the take-off point in the drone's take-off heading.
The drone starts facing north up the first corridor (confirmed by run 1), so forward = north, left = west.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .geometry import load_map, rasterise
from .planner import plan, to_refnav_route

HERE = Path(__file__).resolve().parent

BASE_PARAMS = {"height_m": 0.8, "speed_mps": 0.4, "arrive_m": 0.15, "corridor_half_m": 0.25, "stop_m": 0.3}
# stop 0.65 = braking 0.34 + radius 0.29 + a little; corridor 0.30 = radius + 1 cm: a 0.9 m corridor's centre line
# leaves 0.45 m to each wall, so the checked band must stay well inside that for 10 cm of pose error
V4_PARAMS = dict(BASE_PARAMS, stop_m=0.65, corridor_half_m=0.30, min_corridor_coverage=0.4)
FLYX_SPAWN_XY = (-2.5146, 1.1144)   # FlyX environment.json spawn, measured on the T4 replica; NOT the 0.914 mark
OVERSHOOT_M = 0.13          # past the end of a leg, as a PX4 model fitted to run_0613 predicts
SAFE_END_M = 0.32           # drone radius 0.29 + 0.03


def _merge_corners(route, m, params, keep_idx):
    """Drop vertices whose neighbours can be joined by a straight line that still keeps the drone
    centre >= safe radius from every wall. keep_idx: indices (into [origin] + route) never dropped."""
    grid = rasterise(m, params.grid_res_m)
    L = m["launch_mark_xy"]
    pts = [(0.0, 0.0)] + [tuple(map(float, q)) for q in route]
    orig = list(pts)

    def xy(q):                      # (forward, left) -> course (east, north); start heading = north
        return (L[0] - q[1], L[1] + q[0])

    def min_clear(a, b):
        a, b = xy(a), xy(b)
        n = max(2, int(math.dist(a, b) / 0.01))
        return min(grid.clearance_at(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
                   for t in (i / (n - 1) for i in range(n)))

    changed = True
    while changed:
        changed = False
        for i in range(1, len(pts) - 1):
            if orig.index(pts[i]) in keep_idx:
                continue
            if min_clear(pts[i - 1], pts[i + 1]) >= params.safe_r:
                del pts[i]
                changed = True
                break
    return pts[1:]


def _pull_back(route, m, res, overshoot=OVERSHOOT_M, need=SAFE_END_M):
    """Move each leg's end back along the leg until the drone keeps `need` from every wall even after
    overshooting it by `overshoot`. The last point (the pad) stays: the drone lands there."""
    grid = rasterise(m, res)
    L = m["launch_mark_xy"]
    out = []
    prev = (0.0, 0.0)
    for i, q in enumerate(route):
        f, l = float(q[0]), float(q[1])
        if i < len(route) - 1:
            d = math.hypot(f - prev[0], l - prev[1])
            uf, ul = (f - prev[0]) / d, (l - prev[1]) / d
            while d > 0.1 and grid.clearance_at(L[0] - (l + ul * overshoot), L[1] + f + uf * overshoot) < need:
                f, l, d = f - uf * 0.01, l - ul * 0.01, d - 0.01
        out.append((f, l))
        prev = (f, l)
    return out


def _dump(path, route, params):
    s = ('{"route": [\n' + ",\n".join("  [%.3f, %.3f]" % tuple(q) for q in route)
         + '\n ],\n "params": ' + json.dumps(params) + "\n}\n")
    json.loads(s)
    Path(path).write_text(s, newline="\n")       # LF everywhere, so files match whichever OS generated them


def build(map_path=HERE / "course_map.json"):
    # geometry.py's chamfer fallback is about 2 % off, enough to move almost every waypoint and change route_v1
    # from 29 legs to 28. Route files must not depend on what happens to be installed.
    try:
        import scipy.ndimage  # noqa: F401
    except ImportError as e:
        raise RuntimeError("generating routes needs scipy: without it they differ from routes/ (pip install scipy)") from e
    m = load_map(map_path)
    pl = plan(m, "gate_b", with_return=False)
    v1 = [tuple(map(float, q)) for q in to_refnav_route(pl, m["launch_mark_xy"], math.pi / 2)]
    # Keep: CP1 end (idx 2), the point just past Gate A (12), the tunnel line-up and straight run (13-17), the pad (last).
    keep = {2, 12, len(v1)} | set(range(13, 18))
    fast = _merge_corners(v1, m, pl.params, keep)
    v4 = _pull_back(fast, m, pl.params.grid_res_m)
    # v5: same course path, flown from where the sim actually spawns the drone. Routes are relative to the
    # take-off point, so a spawn 0.2 m north of the mark moves every waypoint 0.2 m north (T4: leg 1 ended
    # 0.25 m from the net). Same vertex layout as v1, so the keep indices still apply.
    m5 = dict(m, launch_mark_xy=list(FLYX_SPAWN_XY))
    pl5 = plan(m5, "gate_b", with_return=False)
    v1_5 = [tuple(map(float, q)) for q in to_refnav_route(pl5, m5["launch_mark_xy"], math.pi / 2)]
    if len(v1_5) != len(v1):
        raise RuntimeError(f"plan from the spawn has {len(v1_5)} vertices, not {len(v1)}: keep indices would be wrong")
    v5 = _pull_back(_merge_corners(v1_5, m5, pl5.params, keep), m5, pl5.params.grid_res_m)
    return {
        "route_v5.json": (v5, V4_PARAMS),
        "route_v4.json": (v4, V4_PARAMS),
        "route_v1.json": (v1, BASE_PARAMS),
        "route_fast.json": (fast, BASE_PARAMS),
        "route_v2.json": (fast, BASE_PARAMS),
        "route_v3.json": (fast, dict(BASE_PARAMS, min_corridor_coverage=0.4)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(HERE / "course_map.json"))
    ap.add_argument("--out", default=str(HERE.parent / "routes"))
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, (route, params) in build(a.map).items():
        _dump(out / name, route, params)
        print(f"{name}: {len(route)} legs")


if __name__ == "__main__":
    main()
