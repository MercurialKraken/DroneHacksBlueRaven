"""VIO / EV pipeline health in a skybeta bag: cuVSLAM status (track time, state), EV into PX4 (rate, age), EKF2 flags,
PX4 position vs cuVSLAM odometry. usage: py -3 bag_health.py <bag_dir>"""
import glob
import math
import sys

import numpy as np


def fields(m):
    return [k for k in (getattr(m, '__slots__', None) or vars(m).keys()) if not k.startswith('_')]
from mcap.records import Channel, Message, Schema
from mcap.stream_reader import StreamReader
from mcap_ros2.decoder import DecoderFactory

want = {"/visual_slam/status", "/fmu/in/vehicle_visual_odometry", "/fmu/out/estimator_status_flags",
        "/fmu/out/vehicle_local_position", "/visual_slam/tracking/odometry"}
dec, schemas, chans, decs, rows = DecoderFactory(), {}, {}, {}, {k: [] for k in want}
seen = set()
with open(sorted(glob.glob(sys.argv[1] + "/*.mcap"))[0], "rb") as fh:
    try:
        for r in StreamReader(fh, skip_magic=False).records:
            if isinstance(r, Schema):
                schemas[r.id] = r
            elif isinstance(r, Channel):
                chans[r.id] = r
                seen.add(r.topic)
            elif isinstance(r, Message):
                ch = chans.get(r.channel_id)
                if ch is None or ch.topic not in want:
                    continue
                if ch.id not in decs:
                    decs[ch.id] = dec.decoder_for(ch.message_encoding, schemas[ch.schema_id])
                try:
                    rows[ch.topic].append((r.log_time * 1e-9, decs[ch.id](r.data)))
                except Exception:
                    pass
    except Exception:
        pass
print("counts:", {k.split("/")[-1]: len(v) for k, v in rows.items()})
t0 = min(v[0][0] for v in rows.values() if v)
T1 = 31.0                                                # follow starts ~31 s into the 160418 bag

st = rows["/visual_slam/status"]
if st:
    m0 = st[0][1]
    flds = fields(m0)
    print("visual_slam/status fields:", flds)
    for f in flds:
        try:
            vals = np.array([float(getattr(m, f)) for _, m in st])
        except Exception:
            continue
        tt = np.array([t for t, _ in st]) - t0
        a = vals[tt >= T1]
        if len(a):
            print(f"  {f:34s} follow: min {a.min():.4g} p50 {np.median(a):.4g} p95 {np.percentile(a, 95):.4g} max {a.max():.4g}"
                  f" | changes {int(np.sum(np.diff(a) != 0))}")

ev = rows["/fmu/in/vehicle_visual_odometry"]
if ev:
    tt = np.array([t for t, _ in ev]) - t0
    age = np.array([(m.timestamp - m.timestamp_sample) * 1e-6 for _, m in ev])
    f = tt >= T1
    gaps = np.diff(tt[f])
    print(f"EV -> PX4: {f.sum()} msgs in follow, rate {f.sum() / max(tt[f][-1] - tt[f][0], 1e-9):.1f} Hz, gap p95 "
          f"{np.percentile(gaps, 95) * 1e3:.0f} ms max {gaps.max() * 1e3:.0f} ms; age (timestamp - timestamp_sample) "
          f"p50 {np.median(age[f]) * 1e3:.0f} p95 {np.percentile(age[f], 95) * 1e3:.0f} max {age[f].max() * 1e3:.0f} ms")
    pv = np.array([[m.position_variance[0], m.position_variance[1]] for _, m in ev])
    print(f"  EV position variance (x, y) p50 {np.median(pv[f], axis=0)} max {pv[f].max(axis=0)}")

es = rows["/fmu/out/estimator_status_flags"]
if es:
    m0 = es[0][1]
    flds = [k for k in fields(m0) if k.startswith(("cs_", "fs_", "reject_")) or "reject" in k]
    tt = np.array([t for t, _ in es]) - t0
    f = tt >= T1
    print("EKF2 flags during follow (fraction of samples true, only those not always equal):")
    for k in flds:
        v = np.array([bool(getattr(m, k)) for _, m in es])[f]
        if len(v) and 0 < v.mean() < 1 or (len(v) and k.startswith(("fs_", "reject")) and v.any()):
            print(f"  {k:32s} {v.mean():.2f} (transitions {int(np.sum(np.diff(v.astype(int)) != 0))})")
    for k in ("cs_ev_pos", "cs_ev_yaw", "cs_rng_hgt", "cs_baro_hgt"):
        if k in fields(m0):
            v = np.array([bool(getattr(m, k)) for _, m in es])[f]
            print(f"  {k:32s} fraction true {v.mean():.2f}")

lp, od = rows["/fmu/out/vehicle_local_position"], rows["/visual_slam/tracking/odometry"]
if lp and od:
    to = np.array([t for t, _ in od])
    po = np.array([[m.pose.pose.position.x, m.pose.pose.position.y] for _, m in od])
    tl = np.array([t for t, _ in lp])
    pl = np.array([[m.x, m.y] for _, m in lp])                     # NED
    # cuVSLAM ENU (x fwd at take-off = north, y left = west) vs PX4 NED (x north, y east): n = x_odom, e = -y_odom
    k = np.searchsorted(to, tl).clip(1, len(to) - 1)
    pon = np.c_[po[k, 0], -po[k, 1]]
    d = pl - pon
    d = d - d[(tl - t0) < 20].mean(axis=0) if ((tl - t0) < 20).any() else d          # remove the constant frame offset
    f = (tl - t0) >= T1
    n = np.hypot(d[f, 0], d[f, 1])
    print(f"PX4 local position vs cuVSLAM odometry in follow (frame offset removed): p50 {np.median(n):.3f} "
          f"p95 {np.percentile(n, 95):.3f} max {n.max():.3f} m")
print("bag topics:", sorted(seen))

# rigid 2D fit (rotation + translation) of cuVSLAM odometry onto PX4 local position, over the first 15 s of follow;
# then how far PX4's position wanders from the transformed odometry for the rest (a growing gap = the EV link/EKF lagging)
if lp and od:
    to = np.array([t for t, _ in od]); po = np.array([[m.pose.pose.position.x, m.pose.pose.position.y] for _, m in od])
    tl = np.array([t for t, _ in lp]); pl = np.array([[m.x, m.y] for _, m in lp])
    k = np.searchsorted(to, tl).clip(1, len(to) - 1)
    A, B = po[k], pl
    f = (tl - t0) >= T1
    fit = f & ((tl - t0) < T1 + 15)
    a, b = A[fit] - A[fit].mean(0), B[fit] - B[fit].mean(0)
    U, S, Vt = np.linalg.svd(a.T @ b)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:                                        # odom ENU -> PX4 NED is a reflection
        Vt[-1] *= -1
        R = (U @ Vt).T
    tr = B[fit].mean(0) - A[fit].mean(0) @ R.T
    res = np.hypot(*(A @ R.T + tr - B).T)
    print(f"rigid fit (first 15 s of follow): rotation {math.degrees(math.atan2(R[1, 0], R[0, 0])):.1f} deg, det {np.linalg.det(R):+.0f}")
    for lo, hi in ((T1, T1 + 15), (T1 + 15, T1 + 30), (T1 + 30, T1 + 45), (T1 + 45, 1e9)):
        g = ((tl - t0) >= lo) & ((tl - t0) < hi)
        if g.any():
            print(f"  PX4 vs transformed VIO, follow {lo - T1:3.0f}-{min(hi, (tl - t0).max()) - T1:3.0f} s: p50 {np.median(res[g]):.3f} max {res[g].max():.3f} m")

if lp and od:
    g = f
    for refl in (False, True):
        a0 = A[g].copy()
        if refl:
            a0[:, 1] *= -1                                          # ENU (x, y) -> (x, -y) first
        a, b = a0 - a0.mean(0), B[g] - B[g].mean(0)
        U, S, Vt = np.linalg.svd(a.T @ b)
        D = np.eye(2); D[1, 1] = np.sign(np.linalg.det(U @ Vt))
        R = (U @ D @ Vt).T
        tr = B[g].mean(0) - a0.mean(0) @ R.T
        res = np.hypot(*(a0 @ R.T + tr - B[g]).T)
        tt = tl[g] - t0 - T1
        print(f"WHOLE-follow fit, reflection={refl}: rot {math.degrees(math.atan2(R[1, 0], R[0, 0])):.1f} deg, residual p50 "
              f"{np.median(res):.3f} p95 {np.percentile(res, 95):.3f} max {res.max():.3f} m; by thirds: "
              + ", ".join(f"{np.median(res[(tt >= q0) & (tt < q1)]):.3f}" for q0, q1 in ((0, 14), (14, 28), (28, 99))))
