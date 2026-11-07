"""Regenerate the route.json files the starter's refnav mission flies.

    python3 -m course_planner.make_routes            # writes routes/ next to this package
    python3 -m course_planner.make_routes --out DIR  # somewhere else

Outputs (all for the Gate B route, all from course_map.json, nothing hand-typed):
  route_v1.json    29 legs: every vertex the planner produced. Flew legs 1-11 in the sim (run 1).
  route_fast.json  19 legs: v1 with corner vertices merged where the straight line keeps full clearance.
  route_v2.json    route_fast + 30 deg camera offsets on legs 1 and 7. FROZE on leg 1 (run 2): don't use.
  route_v3.json    route_fast + 15 deg offset on leg 7 only + min_corridor_coverage 0.4. Predicted to hit the net.
  route_v4.json    v3 with leg ends pulled back from walls, plus braking and wider stop checks. Current candidate.

refnav route format: [[forward_m, left_m], ...] from the take-off point in the drone's take-off heading.
The drone starts facing north up the first corridor (confirmed by run 1), so forward = north, left = west.
leg_yaw_off_deg needs the 3-edit patch to refnav/core.py described in HANDOFF.md.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .geometry import load_map, rasterise
from .planner import plan, to_refnav_route

HERE = Path(__file__).resolve().parent

BASE_PARAMS = {"height_m": 0.8, "speed_mps": 0.4, "lead_max_m": 0.5, "arrive_m": 0.15,
               "corridor_half_m": 0.25, "stop_m": 0.3, "hold_s": 0.5}
FAST_PARAMS = dict(BASE_PARAMS, hold_s=0.2, yaw_rate_dps=90.0)
# v4, from run_0613: 0.65 = braking 0.34 + radius 0.29 + a little; 0.35 = radius + margin. lead and brake come
# from a PX4 model fitted to that bag, which predicts about 0.13 m past a leg end instead of 0.30 m.
V4_PARAMS = dict(FAST_PARAMS, lead_max_m=0.3, brake_mps2=0.1, stop_m=0.65, end_clear_m=0.35, corridor_half_m=0.35,
                 min_corridor_coverage=0.4)
OVERSHOOT_M = 0.13          # past the end of a leg, flying V4_PARAMS
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
    Path(path).write_text(s)


def build(map_path=HERE / "course_map.json"):
    m = load_map(map_path)
    pl = plan(m, "gate_b", with_return=False)
    v1 = [tuple(map(float, q)) for q in to_refnav_route(pl, m["launch_mark_xy"], math.pi / 2)]
    # Keep: CP1 end (idx 2), the point just past Gate A (12), the tunnel line-up and straight run (13-17), the pad (last).
    keep = {2, 12, len(v1)} | set(range(13, 18))
    fast = _merge_corners(v1, m, pl.params, keep)
    off_v2 = [0.0] * len(fast); off_v2[0] = 30.0; off_v2[6] = 30.0
    off_v3 = [0.0] * len(fast); off_v3[6] = 15.0
    v4 = _pull_back(fast, m, pl.params.grid_res_m)
    return {
        "route_v4.json": (v4, dict(V4_PARAMS, leg_yaw_off_deg=off_v3)),
        "route_v1.json": (v1, BASE_PARAMS),
        "route_fast.json": (fast, FAST_PARAMS),
        "route_v2.json": (fast, dict(FAST_PARAMS, leg_yaw_off_deg=off_v2)),
        "route_v3.json": (fast, dict(FAST_PARAMS, min_corridor_coverage=0.4, leg_yaw_off_deg=off_v3)),
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
