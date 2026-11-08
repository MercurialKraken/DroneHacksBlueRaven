"""PX4 velocity setpoint vs achieved velocity in the follow phase, real flight vs T4, in PX4's own NED frame.
Real: the setpoint is the interface's enu_to_ned(twist) (flight_interface.py; offset not applied to twists).
T4: /fmu/in/trajectory_setpoint as recorded.  Deficit = achieved - setpoint, also in the body frame (PX4 heading).

    python compare_tracking.py <name>=<flight.csv>[:lag_s] ...
"""
import json
import sys

import numpy as np


def load(path):
    rows = {"lpos": [], "odom": [], "twist": [], "status": [], "tsp": [], "truth": []}
    for line in open(path, encoding="utf-8"):
        k, rest = line.rstrip("\n").split(",", 1)
        if k == "status":
            t, js = rest.split(",", 1)
            rows[k].append((float(t), json.loads(js)))
        elif k in rows:
            rows[k].append([float(x) for x in rest.split(",")])
    return rows


def ang(a, b):
    return np.degrees(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1)))


for arg in sys.argv[1:]:
    name, rest = arg.split("=", 1)
    path, lag = (rest.rsplit(":", 1) + ["0.55"])[:2] if ":" in rest[2:] else (rest, "0.55")
    lag = float(lag)
    R = load(path)
    L = np.array(R["lpos"])
    ph = {}
    for t, s in R["status"]:
        ph.setdefault(s["phase"], t)
    tf, tl = ph["follow"], ph["land"]
    if R["tsp"]:
        T = np.array(R["tsp"])
        sp_t, sp = T[:, 0], T[:, 1:3]                                   # NED (n, e)
        src = "trajectory_setpoint"
    else:
        W = np.array(R["twist"])
        sp_t, sp = W[:, 0], np.c_[W[:, 2], W[:, 1]]                     # enu_to_ned: (n, e) = (y, x)
        src = "enu_to_ned(twist)"
    m = (sp_t > tf) & (sp_t < tl) & np.isfinite(sp).all(1) & (np.hypot(*sp.T) > 0.05)
    t, c = sp_t[m], sp[m]
    a = np.c_[np.interp(t + lag, L[:, 0], L[:, 4]), np.interp(t + lag, L[:, 0], L[:, 5])]   # achieved (vn, ve)
    hd = np.interp(t, L[:, 0], np.unwrap(L[:, 7]))                                       # NED heading
    d = a - c
    fwd = d[:, 0] * np.cos(hd) + d[:, 1] * np.sin(hd)
    right = -d[:, 0] * np.sin(hd) + d[:, 1] * np.cos(hd)
    along = (a * c).sum(1) / np.hypot(*c.T)
    xt = np.array([s.get("xtrack", np.nan) for tt, s in R["status"] if s["phase"] == "follow"])
    e = ang(c, a)          # NED: + = clockwise = to the RIGHT (NED is left-handed in the horizontal)
    print(f"== {name} ({src}, achieved {lag:.2f} s later): {m.sum()} samples")
    print(f"   |setpoint| median {np.median(np.hypot(*c.T)):.3f} m/s; achieved along it {np.median(along):.3f} m/s "
          f"({np.median(along / np.hypot(*c.T)) * 100:.0f}%); |achieved| {np.median(np.hypot(*a.T)):.3f}")
    print(f"   achieved direction vs setpoint: median {np.median(e):+.1f} deg (NED: + = right, - = left), "
          f"IQR {np.percentile(e, 75) - np.percentile(e, 25):.0f}")
    print(f"   deficit (achieved - setpoint) in BODY frame: forward {np.median(fwd):+.3f}, right {np.median(right):+.3f} m/s "
          f"(p25..p75 right {np.percentile(right, 25):+.3f}..{np.percentile(right, 75):+.3f})")
    print(f"   refnav cross-track in follow: median {np.nanmedian(xt):.3f} m, p95 {np.nanpercentile(xt, 95):.3f}")
