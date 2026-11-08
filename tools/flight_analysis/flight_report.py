"""route1-chelsea flight report from extract_flight_sim.py's CSV (real or T4): take-off drift, the HOLD (does it get back
to the take-off point?), offset from the path per leg, the CP1 crossing, and PX4 setpoint vs achieved velocity.

    python flight_report.py <flight.csv> [--real]      # --real: also the CP1 crossing on the tape map (launch mark)

The take-off frame is PX4's own: its position and heading on the ground before the climb (forward, right).
"""
import json
import math
import sys

import numpy as np

path_csv = sys.argv[1]
REAL = "--real" in sys.argv
R = {"lpos": [], "status": [], "twist": [], "tsp": [], "vstatus": [], "truth": [], "att": [], "attsp": [], "lpsp": []}
for line in open(path_csv, encoding="utf-8"):
    k, rest = line.rstrip("\n").split(",", 1)
    if k == "status":
        t, js = rest.split(",", 1)
        R[k].append((float(t), json.loads(js)))
    elif k in R:
        R[k].append([float(x) for x in rest.split(",")])
L = np.array(R["lpos"])                          # t, x(n), y(e), z(d), vx, vy, vz, heading, xy_valid, eph
ST = R["status"]
ph = {}
for t, s in ST:
    ph.setdefault(s["phase"], t)
t_climb = ph["climb"]
t_hold = ph.get("hover")
t_follow, t_land = ph["follow"], ph.get("land", L[-1, 0])
gnd = (L[:, 0] > t_climb - 15) & (L[:, 0] < t_climb - 0.5)
n0, e0, z0 = (float(np.median(L[gnd, i])) for i in (1, 2, 3))
h0 = float(np.angle(np.mean(np.exp(1j * L[gnd, 7]))))
fwd_u, right_u = np.array([math.cos(h0), math.sin(h0)]), np.array([-math.sin(h0), math.cos(h0)])
D = np.c_[L[:, 1] - n0, L[:, 2] - e0]
F, Rt = D @ fwd_u, D @ right_u                   # take-off frame (forward, right)
H = -(L[:, 3] - z0)
T = L[:, 0] - t_climb


def at(t_abs):
    i = min(int(np.searchsorted(L[:, 0], t_abs)), len(L) - 1)
    return F[i], Rt[i], H[i]


def fmt(f, r):
    return f"{math.hypot(f, r):.3f} m (fwd {f:+.3f}, right {r:+.3f})"


print(f"== {path_csv}")
lift = T[np.argmax((H > 0.03) & (T > 0))]
end_climb = t_hold or t_follow
f, r, h = at(end_climb)
print(f"CLIMB: lift-off t+{lift:.2f} s; at its end (t+{end_climb - t_climb:.2f} s, height {h:.2f} m) the drone is {fmt(f, r)} from the take-off point")
for hh in (0.2, 0.3, 0.4, 0.5):
    m = (T > lift) & (H >= hh)
    if m.any():
        j = int(np.argmax(m))
        print(f"   drift when first at {hh:.1f} m: {fmt(F[j], Rt[j])}")
if t_hold:
    f1, r1, _ = at(t_hold)
    f2, r2, h2 = at(t_follow)
    m = (L[:, 0] >= t_hold) & (L[:, 0] < t_follow)
    dist = np.hypot(F[m], Rt[m])
    tail = (L[:, 0] >= t_follow - 3) & (L[:, 0] < t_follow)
    slope = np.polyfit(L[tail, 0], np.hypot(F[tail], Rt[tail]), 1)[0] if tail.sum() > 5 else float("nan")
    print(f"HOLD ({t_follow - t_hold:.1f} s): start {fmt(f1, r1)}; END {fmt(f2, r2)} at height {h2:.2f} m; "
          f"min {dist.min():.3f} m; last 3 s trend {slope * 100:+.1f} cm/s")
    verdict = ("BACK ON THE TAKE-OFF POINT (<= 0.05 m): take-off drift was flown out" if math.hypot(f2, r2) <= 0.05 else
               "STILL CONVERGING at the end (trend < -1 cm/s): the hold was too short" if slope < -0.01 else
               f"SETTLED {math.hypot(f2, r2):.2f} m OFF the take-off point: a steady push, not just take-off drift")
    print(f"   => {verdict}")
else:
    print("HOLD: none in this flight (no 'hover' phase)")
path = np.array(next(s["path"] for t, s in ST if s.get("path")))          # pad (n, e) = (forward, right)
m = (L[:, 0] >= t_follow) & (L[:, 0] <= t_land)
P = np.c_[F[m], Rt[m]]
near = np.array([path[np.argmin(np.linalg.norm(path - p, axis=1))] for p in P])
tang = np.gradient(path, axis=0)
tang /= np.linalg.norm(tang, axis=1)[:, None]
idx = np.array([np.argmin(np.linalg.norm(path - p, axis=1)) for p in P])
off = P - near
lat = tang[idx, 0] * off[:, 1] - tang[idx, 1] * off[:, 0]                  # + = right of travel, - = left
print(f"FOLLOW ({t_land - t_follow:.1f} s): distance from the path median {np.median(np.abs(lat)):.3f} m, p95 "
      f"{np.percentile(np.abs(lat), 95):.3f}; side: {np.mean(lat < 0) * 100:.0f}% LEFT of travel")
for name, sel in (("leg 1 (forward)", P[:, 0] < 0.45), ("arc", (P[:, 0] >= 0.45) & (P[:, 1] < 0.8)), ("leg 2 (rightward, CP1)", P[:, 1] >= 0.8)):
    if sel.any():
        print(f"   {name:24s}: lateral median {np.median(lat[sel]):+.3f} m (- = left), max left {(-lat[sel]).max():.3f} m")
if REAL:   # CP1 line on the tape map -> take-off frame, assuming the drone sat on the launch mark facing course north
    LO = np.array([-2.515, 0.914])
    to_pad = lambda xy: np.array([xy[1] - LO[1], xy[0] - LO[0]])         # map (x, y) -> (forward, right)
    a, b = to_pad([-1.339, 2.591]), to_pad([-0.857, 1.715])
    n = np.array([b[1] - a[1], -(b[0] - a[0])]); n /= np.linalg.norm(n)
    s = (P - a) @ n
    s *= np.sign(s[-1]) if s[-1] != 0 else 1
    if (s > 0).any() and (s < 0).any():
        c = P[int(np.argmax(s > 0))]
        u = (b - a) / np.linalg.norm(b - a)
        along = float((c - (a + b) / 2) @ u)
        print(f"CP1 (tape map, if placed on the mark): crossed {along:+.3f} m from the gap centre (+ = towards Wall B's end); "
              f"{0.5 - abs(along) - 0.29:+.3f} m to spare on the near side")
fL = np.array([s.get("xtrack", np.nan) for t, s in ST if s["phase"] == "follow"])
sc = np.array([s.get("scale", np.nan) for t, s in ST if s["phase"] == "follow"])
print(f"refnav: cross-track median {np.nanmedian(fL):.3f} m; reference waiting (scale 0) {np.nanmean(sc == 0) * 100:.0f}% of follow")
# PX4 setpoint vs achieved velocity, body frame, per phase
if R["tsp"]:
    S_ = np.array(R["tsp"]); spt, sp = S_[:, 0], S_[:, 1:3]; src = "trajectory_setpoint"
else:
    W = np.array(R["twist"]); spt, sp = W[:, 0], np.c_[W[:, 2], W[:, 1]]; src = "enu_to_ned(twist)"
for name, a_, b_ in (("hold", t_hold, t_follow), ("follow", t_follow, t_land)):
    if a_ is None:
        continue
    k = (spt > a_ + 1) & (spt < b_) & np.isfinite(sp).all(1)
    if k.sum() < 10:
        continue
    lag = 0.55
    ach = np.c_[np.interp(spt[k] + lag, L[:, 0], L[:, 4]), np.interp(spt[k] + lag, L[:, 0], L[:, 5])]
    hd = np.interp(spt[k], L[:, 0], np.unwrap(L[:, 7]))
    d = ach - sp[k]
    bf = d[:, 0] * np.cos(hd) + d[:, 1] * np.sin(hd)
    br = -d[:, 0] * np.sin(hd) + d[:, 1] * np.cos(hd)
    print(f"PX4 {name}: setpoint ({src}) median |{np.median(np.hypot(*sp[k].T)):.3f}| m/s; achieved-minus-setpoint in the BODY "
          f"frame: forward {np.median(bf):+.3f}, right {np.median(br):+.3f} m/s (p25..p75 right {np.percentile(br, 25):+.3f}..{np.percentile(br, 75):+.3f})")
land = np.c_[F, Rt][L[:, 0] <= (ph.get("done", L[-1, 0]))][-1]
print(f"LANDED at fwd {land[0]:+.3f}, right {land[1]:+.3f} (route end {path[-1][0]:+.3f}, {path[-1][1]:+.3f}): "
      f"{np.linalg.norm(land - path[-1]):.3f} m off")

# ---- the response delay and the push, from PX4's own setpoint (Chelsea: is it a push? what is the real lag?) ----
if R["tsp"]:
    S_ = np.array(R["tsp"]); spt, sp = S_[:, 0], S_[:, 1:3]
else:
    W = np.array(R["twist"]); spt, sp = W[:, 0], np.c_[W[:, 2], W[:, 1]]
k = (spt > t_follow + 1) & (spt < t_land) & np.isfinite(sp).all(1)
best = None
for lag in np.arange(0.2, 2.61, 0.1):
    ach = np.c_[np.interp(spt[k] + lag, L[:, 0], L[:, 4]), np.interp(spt[k] + lag, L[:, 0], L[:, 5])]
    hd = np.interp(spt[k], L[:, 0], np.unwrap(L[:, 7]))
    c, s = np.cos(hd), np.sin(hd)
    cb = np.c_[c * sp[k, 0] + s * sp[k, 1], -s * sp[k, 0] + c * sp[k, 1]]
    vb = np.c_[c * ach[:, 0] + s * ach[:, 1], -s * ach[:, 0] + c * ach[:, 1]]
    b = (vb - cb).mean(0)
    r2 = 1 - float(((vb - cb - b) ** 2).sum()) / float(((vb - vb.mean(0)) ** 2).sum())
    if best is None or r2 > best[1]:
        best = (lag, r2, b)
if best:
    print(f"RESPONSE (follow, PX4 setpoint -> achieved): best-fit delay {best[0]:.1f} s (refnav assumes 0.55 on the drone); "
          f"push at it: forward {best[2][0]:+.3f}, right {best[2][1]:+.3f} m/s (body), fit R^2 {best[1]:.2f}")
if R["att"] and R["attsp"]:
    A_, AS = np.array(R["att"]), np.array(R["attsp"])
    for name, a_, b_ in (("ground", t_climb - 15, t_climb - 1), ("hold", t_hold, t_follow), ("follow", t_follow, t_land)):
        if a_ is None:
            continue
        ma = (A_[:, 0] >= a_) & (A_[:, 0] < b_); ms = (AS[:, 0] >= a_) & (AS[:, 0] < b_)
        if ma.sum() and ms.sum():
            print(f"TILT {name:6s}: actual roll {np.median(A_[ma, 1]):+.2f} / pitch {np.median(A_[ma, 2]):+.2f} deg; "
                  f"PX4's setpoint roll {np.median(AS[ms, 1]):+.2f} / pitch {np.median(AS[ms, 2]):+.2f} deg "
                  f"(roll + = right down, pitch + = nose up)")
