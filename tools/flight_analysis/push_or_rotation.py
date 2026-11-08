"""Is the drone's 'push' a constant push or a rotated response? Fits PX4's achieved velocity (vehicle_local_position)
against the velocity setpoint it was given, a lag later, in the BODY frame (PX4 heading):
  push model:      v = cmd + b                 (b constant, body frame)
  rotation model:  v = s * R(theta) * cmd      (a rotated, scaled response)
  general:         v = A cmd + b               (A 2x2, b constant)
Real flight: the setpoint is enu_to_ned(twist) (flight_interface.py); T4: /fmu/in/trajectory_setpoint.

    python push_or_rotation.py <name>=<flight.csv>:<lag_s>:<phase[,phase]> ...
"""
import json
import math
import sys

import numpy as np


def load(path):
    R = {"lpos": [], "twist": [], "tsp": [], "status": []}
    for line in open(path, encoding="utf-8"):
        k, rest = line.rstrip("\n").split(",", 1)
        if k == "status":
            t, js = rest.split(",", 1)
            R[k].append((float(t), json.loads(js)["phase"]))
        elif k in R:
            R[k].append([float(x) for x in rest.split(",")])
    return R


def fit(name, path, lag, phases):
    R = load(path)
    L = np.array(R["lpos"])
    spans, cur = [], None
    for t, p in R["status"]:
        if p in phases and cur is None:
            cur = t
        elif p not in phases and cur is not None:
            spans.append((cur, t)); cur = None
    if cur is not None:
        spans.append((cur, R["status"][-1][0]))
    if R["tsp"]:
        S = np.array(R["tsp"]); t_c, c = S[:, 0], S[:, 1:3]
    else:
        W = np.array(R["twist"]); t_c, c = W[:, 0], np.c_[W[:, 2], W[:, 1]]
    m = np.zeros(len(t_c), bool)
    for a, b in spans:
        m |= (t_c >= a + 1.0) & (t_c <= b)
    m &= np.isfinite(c).all(1)
    t, c = t_c[m], c[m]
    v = np.c_[np.interp(t + lag, L[:, 0], L[:, 4]), np.interp(t + lag, L[:, 0], L[:, 5])]
    h = np.interp(t, L[:, 0], np.unwrap(L[:, 7]))
    cs, sn = np.cos(h), np.sin(h)
    body = lambda x: np.c_[cs * x[:, 0] + sn * x[:, 1], -sn * x[:, 0] + cs * x[:, 1]]   # (forward, right)
    cb, vb = body(c), body(v)
    sst = float(((vb - vb.mean(0)) ** 2).sum())
    r2 = lambda pred: 1 - float(((vb - pred) ** 2).sum()) / sst
    b = (vb - cb).mean(0)                                        # push model
    r_push = r2(cb + b)
    z_c, z_v = cb[:, 0] + 1j * cb[:, 1], vb[:, 0] + 1j * vb[:, 1]   # rotation model: complex gain
    g = (np.conj(z_c) @ z_v) / (np.conj(z_c) @ z_c)
    pr = g * z_c
    r_rot = r2(np.c_[pr.real, pr.imag])
    X = np.c_[cb, np.ones(len(cb))]                              # general A + b
    coef, *_ = np.linalg.lstsq(X, vb, rcond=None)
    r_gen = r2(X @ coef)
    th = -math.degrees(math.atan2(g.imag, g.real))               # body (fwd, right): + = rotated to the LEFT
    print(f"== {name}: {len(t)} samples ({', '.join(phases)}), lag {lag} s; |cmd| median {np.median(np.hypot(*cb.T)):.3f}, |v| {np.median(np.hypot(*vb.T)):.3f} m/s")
    print(f"   push model     v = cmd + b:      b = fwd {b[0]:+.3f}, right {b[1]:+.3f} m/s        R^2 {r_push:.2f}")
    print(f"   rotation model v = s R(th) cmd:  s = {abs(g):.2f}, th = {th:+.0f} deg (+ = left)    R^2 {r_rot:.2f}")
    A, bb = coef[:2].T, coef[2]
    print(f"   general        v = A cmd + b:    A = [[{A[0,0]:+.2f} {A[0,1]:+.2f}] [{A[1,0]:+.2f} {A[1,1]:+.2f}]], b = ({bb[0]:+.3f}, {bb[1]:+.3f})  R^2 {r_gen:.2f}")


for arg in sys.argv[1:]:
    name, rest = arg.split("=", 1)
    path, lag, ph = rest.rsplit(":", 2)
    fit(name, path, float(lag), ph.split(","))
