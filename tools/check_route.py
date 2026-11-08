#!/usr/bin/env python3
"""OFFLINE (laptop) check of a take-off-frame route file against the course map: straight segments and a
toy-drone flight through the skybeta shell (the kinematic drone of tests/test_skybeta_mission.py).

    python tools/check_route.py routes/route_v6_start2.json launch [corner_radius_m]

Reports each straight leg's minimum clearance at flight height (0.7 m) on maps/course_sdf.npz, then flies
the route through mission_core and reports the flown clearance and the checkpoint crossings. The map is
only as good as the drawing it came from: treat the numbers as a prior, not as the course.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "skybeta" / "blueraven_skybeta"), str(REPO / "tests")]
from blueraven_skybeta.mission_core import MissionParams, SkybetaMission, load_route  # noqa: E402
from nav.course_map import CourseMap  # noqa: E402
from nav.frames import load_launch, map_from_startup_odom  # noqa: E402
from nav.path_geometry import checkpoint_crossings  # noqa: E402
from test_skybeta_mission import Sim, run, switch_to_offboard, to_ready  # noqa: E402

route_path, key = sys.argv[1], sys.argv[2]
radius = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
cfg = yaml.safe_load((REPO / "maps" / "course.yaml").read_text(encoding="utf-8"))
cm = CourseMap.load(REPO / "maps" / "course_sdf.npz")
launch = load_launch(REPO / "maps" / "course.yaml", key)
T = map_from_startup_odom(launch, (0.0, 0.0, 0.0, 0.0))
corners = load_route(route_path)
z = launch["z"] + 0.7
P = np.array([(T @ [f, l, 0, 1])[:2] for f, l in corners])
labels = json.loads(Path(route_path).read_text(encoding="utf-8")).get("labels", [])
print(f"{Path(route_path).name} from {key}: {len(corners) - 1} corners")
worst = []
for i in range(1, len(P)):
    a, b = P[i - 1], P[i]
    n = max(2, int(np.linalg.norm(b - a) / 0.02))
    pts = np.column_stack([a + np.outer(np.linspace(0, 1, n), b - a), np.full(n, z)])
    d = cm.distance(pts, "planning")
    lab = labels[i - 1] if i - 1 < len(labels) else ""
    worst.append(float(np.nanmin(d)))
    print(f"  leg {i:2d} -> map ({b[0]:+.2f}, {b[1]:+.2f}) {lab:22s} min clearance {np.nanmin(d):.3f} m"
          f"  corner {cm.distance(np.array([[b[0], b[1], z]]), 'planning')[0]:.3f}")
print(f"  straight segments: min clearance {min(worst):.3f} m (drone radius {cfg['limits']['drone_radius']})")

m = SkybetaMission(MissionParams(mission="route", height_m=0.7, route_max_radius_m=radius, cruise_mps=0.3), corners)
sim = Sim()
t = to_ready(m, sim)
switch_to_offboard(t, sim)
track = []
t_end = t + 400
while m.phase not in ("DONE", "ABORTED") and t < t_end:
    t, _ = run(m, sim, t, t + 1.0, hook=lambda tt, s: track.append((m.phase, s.x, s.y, s.z)))
xyz = np.array([(T @ [x, y, zz, 1])[:3] for _, x, y, zz in track])
fl = np.array([ph in ("HOVER", "FOLLOW") for ph, *_ in track])
d = cm.distance(xyz[fl], "planning")
print(f"  toy flight (corner radius {radius}): {m.phase} in {t:.0f} s, min clearance {np.nanmin(d):.3f} m, "
      f"landed at map ({xyz[-1, 0]:+.2f}, {xyz[-1, 1]:+.2f})")
for c in checkpoint_crossings(SimpleNamespace(xyz=xyz), cfg["checkpoints"], float(cfg["limits"]["drone_radius"])):
    print("   ", c)
