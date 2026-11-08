"""SIM: refnav (blind) on route_full_hold at a given speed and height through the skybeta adapter + lag Stack, with a
steady body push, and how far the flown path strays from the planned (filleted) path. usage: refnav_speed.py SPEED [push]"""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "tests")
import test_refnav_skybeta as TR
from refnav import core as C

speed = float(sys.argv[1]); pushed = len(sys.argv) > 2
doc = json.loads(Path("courses/start-2026-11-07-tape/routes/route_full_hold.json").read_text())
doc["params"]["speed_mps"] = speed
trail = []
m, sk, sent = TR.flight(doc=doc, prm_over={"height_m": 1.1}, tau=0.9, lag_s=0.9 + 0.6, delay=0.6, trail=trail,
                        push_body=(-0.056, 0.100) if pushed else (0.0, 0.0), limit_s=300.0)
fol = np.array([p[:2] for t, ph, p in trail if ph == "follow"])
path = m.follower.path if m.follower else None
pts = None
for attr in ("pts", "points", "xy", "P"):
    if path is not None and hasattr(path, attr):
        pts = np.asarray(getattr(path, attr))[:, :2]
        break
if pts is None and path is not None:
    s = np.linspace(0, path.length, 2000)
    pts = np.array([path.at(x)[:2] if hasattr(path, "at") else path.point(x)[:2] for x in s])
def dist_to_poly(p, P):
    a, b = P[:-1], P[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    return float(np.min(np.hypot(*(a + ab * t[:, None] - p).T)))
xt = np.array([dist_to_poly(p, pts) for p in fol]) if pts is not None and len(fol) else np.array([np.nan])
fs = [t for t, ph, _ in trail if ph == "follow"]
print(json.dumps({"speed": speed, "push": pushed, "result": m.result, "why": m.why,
                  "follow_s": round(fs[-1] - fs[0], 1) if fs else None,
                  "crosstrack_p50_m": round(float(np.median(xt)), 3), "crosstrack_p95_m": round(float(np.percentile(xt, 95)), 3),
                  "crosstrack_max_m": round(float(xt.max()), 3), "points_passed": m.follower.points_passed if m.follower else None}))
