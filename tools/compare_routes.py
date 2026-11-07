#!/usr/bin/env python3
"""Compare routes offline exactly as refnav would fly them: the same elastic band on the course map, the same
speed profile and the same parameters (route.json's params, then flight.yaml, then main.py's height). No ROS.

    python3 tools/compare_routes.py                                # route.json's points vs the startup planner
    python3 tools/compare_routes.py --cp4 "Gate B" --launch-key launch_sim

For each route: path length, profile flight time, minimum and tight clearance along the flown path, and how far
the path's centre gets past each checkpoint line (nav.path_geometry.checkpoint_crossings).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nav.course_map import CourseMap  # noqa: E402
from nav.frames import load_launch  # noqa: E402
from nav.path_geometry import checkpoint_crossings  # noqa: E402
from nav.route_planner import plan_from_files  # noqa: E402
from refnav import core as C  # noqa: E402
from refnav import follow as F  # noqa: E402
from refnav import map_route as M  # noqa: E402
from refnav.launch import flight_params  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence


def main_py_height() -> float:
    """HEIGHT_M from main.py, read as text (importing main.py would start the flight)."""
    for line in (ROOT / "main.py").read_text(encoding="utf-8").splitlines():
        if line.startswith("HEIGHT_M"):
            return float(line.split("=", 1)[1].split("#", 1)[0])
    raise ValueError("main.py has no HEIGHT_M")


def evaluate(name: str, route: list[tuple[float, float]], doc: dict[str, Any], overrides: dict[str, Any],
             cm: CourseMap, launch: dict[str, float], checkpoints: list[dict[str, Any]], cp4: str,
             margin: float, shift: float) -> dict[str, Any]:
    prm, _ = C.params_from_doc(dict(doc, route=route), overrides)
    z = prm.height_m + launch["z"]
    pts_ned = [(0.0, 0.0)] + [(f, -l) for f, l in prm.route]     # facing north at arming: forward = N, left = -E
    path = M.builder(cm, launch, z, M.DRONE_RADIUS_M + margin, shift)(pts_ned)
    v = F.speed_profile(path, prm.speed_mps, math.radians(prm.yaw_rate_dps), prm.a_lat_mps2, prm.accel_mps2,
                        (prm.tight_m, prm.roomy_m, prm.crawl_mps))
    ds = np.diff(path.s)
    vm = np.maximum((v[1:] + v[:-1]) / 2.0, F.MIN_SPEED)
    x, y = M.MapFrame(launch).to_map(path.x, path.y)
    flown = [dict(cp, options=[o for o in cp["options"] if o["name"].lower() == cp4.lower()])
             if "options" in cp else cp for cp in checkpoints]
    cross = checkpoint_crossings(SimpleNamespace(xyz=np.c_[x, y, np.full(len(x), z)]), flown, M.DRONE_RADIUS_M)
    clear = path.clearance
    return {"name": name, "points": len(route), "length_m": round(float(path.length), 2),
            "time_s": round(float(np.sum(ds / vm)), 1), "min_clear_m": round(float(np.nanmin(clear)), 3),
            "share_under_0.40": round(float(np.mean(clear < 0.40)), 3), "crossings": cross}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route-file", default=str(ROOT / "route.json"))
    ap.add_argument("--sdf", default=str(ROOT / "maps" / "course_sdf.npz"))
    ap.add_argument("--yaml", default=str(ROOT / "maps" / "course.yaml"))
    ap.add_argument("--launch-key", default="launch_sim")
    ap.add_argument("--cp4", default="window")
    ap.add_argument("--margin", type=float, default=0.20, help="refnav.node map_margin_m")
    ap.add_argument("--shift", type=float, default=0.3, help="refnav.node map_max_shift_m")
    a = ap.parse_args(argv)

    import yaml
    doc = json.loads(Path(a.route_file).read_text(encoding="utf-8"))
    overrides = dict(flight_params(ROOT), height_m=main_py_height())
    cfg = yaml.safe_load(Path(a.yaml).read_text(encoding="utf-8"))
    cm, launch = CourseMap.load(a.sdf), load_launch(a.yaml, a.launch_key)
    prm, _ = C.params_from_doc(doc, overrides)
    plan = plan_from_files(a.sdf, a.yaml, a.launch_key, prm.height_m, a.cp4)
    rows = [evaluate("route.json", [tuple(q) for q in doc["route"]], doc, overrides, cm, launch,
                     cfg["checkpoints"], a.cp4, a.margin, a.shift),
            evaluate(f"planner ({plan.digest})", plan.route, doc, overrides, cm, launch,
                     cfg["checkpoints"], a.cp4, a.margin, a.shift)]
    print(f"height {prm.height_m} m above launch ({a.launch_key}), CP4 {a.cp4}, speed {prm.speed_mps} m/s")
    for r in rows:
        cross = ", ".join(f"{c['checkpoint']} {c.get('past_m', c.get('end_from_centre_m'))}" for c in r["crossings"])
        print(f"{r['name']:24s} {r['points']:3d} pts  {r['length_m']:5.2f} m  {r['time_s']:5.1f} s  "
              f"min clear {r['min_clear_m']:.3f} m  <0.40 m: {100 * r['share_under_0.40']:.1f}%  | {cross}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
