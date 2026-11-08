"""SCRATCH: planned-path clearance of a route at a chosen height (tools/check_waypoint_path.check with its height
override changed), per labelled section; and clear_xy() for flown trails. The CAD window (z 0.305-1.219) is measured
at its centre height 0.762 inside WIN_R of the window centre, i.e. horizontal clearance to the jambs only.
usage (from hfr): py -3 ../route_clear.py ROUTE.json HEIGHT"""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "tools")
import check_waypoint_path as CW
from nav.course_map import CourseMap, course_dir
from nav.frames import load_launch
from refnav import map_route as M

COURSE = course_dir("start-2026-11-07-tape")
CM = CourseMap.load(COURSE / "course_sdf.npz")
LAUNCH = load_launch(COURSE / "course.yaml", "launch")
LAUNCH["x"], LAUNCH["y"], LAUNCH["yaw_deg"] = -2.515, 1.466, 90.0
FRAME = M.MapFrame(LAUNCH)
WIN = (-0.471, -2.272)
WIN_R = 0.6


def clear_xy(n, e, height):
    """pad (forward, right) samples -> (clearance at flight z, in-window-zone mask, horizontal clearance at z 0.762)."""
    x, y = FRAME.to_map(np.asarray(n, float), np.asarray(e, float))
    z = height + LAUNCH["z"]
    d = CM.distance(np.c_[x, y, np.full(len(x), z)])
    win = np.hypot(x - WIN[0], y - WIN[1]) < WIN_R
    dw = CM.distance(np.c_[x, y, np.full(len(x), 0.762)])
    return d, win, dw, x, y


if __name__ == "__main__":
    route, h = sys.argv[1], float(sys.argv[2])
    doc = json.loads(Path(route).read_text())
    CW.DRONE_PARAMS["height_m"] = h
    r = CW.check(doc, CM, LAUNCH)
    from refnav import core as C, follow as F
    prm, _ = C.params_from_doc(doc, CW.DRONE_PARAMS)
    corners = [(0.0, 0.0)] + [(f, -l) for f, l in prm.route]
    path = F.fillet_path(corners, prm.fillet_m, min(prm.arrive_m, prm.corner_cut_m))
    d, win, dw, x, y = clear_xy(path.x, path.y, h)
    assert np.allclose(d, r["clearance"], equal_nan=True)
    print(f"{route}: z {r['z']:.2f} m, tool min {r['min']:.3f} at map {tuple(round(c, 3) for c in r['at_map'])}, "
          f"{r['at_s']:.2f} m along; outside map {r['outside_map']}")
    out = ~win
    i = int(np.nanargmin(np.where(out, d, np.inf)))
    print(f"outside the window zone (> {WIN_R} m from its centre): min {d[i]:.3f} m at map ({x[i]:.3f}, {y[i]:.3f}), s {path.s[i]:.2f}")
    j = int(np.nanargmin(np.where(win, dw, np.inf)))
    print(f"window zone, horizontal (jambs, z 0.762): min {dw[j]:.3f} m at map ({x[j]:.3f}, {y[j]:.3f}); at flight z the CAD map gives {np.nanmin(d[win]):.3f}")
    # per section between route points
    fol = F.Follower(path, np.ones(len(path.s)), None, [(f, -l) for f, l in prm.route])
    labels = doc.get("labels", [])
    ss = [0.0] + fol.points_s
    for k in range(len(ss) - 1):
        m = (path.s >= ss[k]) & (path.s <= ss[k + 1])
        dd = np.where(win[m], dw[m], d[m])
        print(f"  {k:2d}->{k + 1:2d} {labels[k] if k < len(labels) else '':38s} min {np.nanmin(dd):.3f}{'  (window: jambs)' if win[m].any() else ''}")
