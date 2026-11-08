"""Every pose estimate in a skybeta bag on one picture: cuVSLAM odometry (= the VIO), the EV message the bridge sent PX4,
PX4's EKF2 local position (IMU + EV + rangefinder), an IMU-only dead-reckoned path (PX4 acceleration integrated twice
from MPPI follow start), and the rangefinder. Frames are aligned to cuVSLAM by a rigid fit (rotation/reflection +
translation) over the whole follow, then cuVSLAM is placed on the map from the take-off mark.
usage: plot_all_pose.py <bag_dir> <route.json> <course_sdf.npz> <out.png>"""
import glob
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mcap.records import Channel, Message, Schema  # noqa: E402
from mcap.stream_reader import StreamReader  # noqa: E402
from mcap_ros2.decoder import DecoderFactory  # noqa: E402

BAG, ROUTE, SDF, OUT = sys.argv[1:5]
START, H = np.array([-2.515, 1.466]), 3.048
want = {"/visual_slam/tracking/odometry", "/fmu/in/vehicle_visual_odometry", "/fmu/out/vehicle_local_position",
        "/blueraven/mppi/cmd", "/blueraven/mppi/start"}
dec, schemas, chans, decs, rows = DecoderFactory(), {}, {}, {}, {k: [] for k in want}
with open(sorted(glob.glob(BAG + "/*.mcap"))[0], "rb") as fh:
    try:
        for r in StreamReader(fh, skip_magic=False).records:
            if isinstance(r, Schema):
                schemas[r.id] = r
            elif isinstance(r, Channel):
                chans[r.id] = r
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

od = rows["/visual_slam/tracking/odometry"]
t0 = od[0][0]
tf = next((t for t, m in rows["/blueraven/mppi/cmd"] if json.loads(m.data).get("start") is not None), t0) - t0
pad = json.loads(rows["/blueraven/mppi/start"][0][1].data)["pad"] if rows["/blueraven/mppi/start"] else [0, 0, 0, 0]
to = np.array([t for t, _ in od]) - t0
O = np.array([[m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z] for _, m in od])
ev = rows["/fmu/in/vehicle_visual_odometry"]
te = np.array([t for t, _ in ev]) - t0
E = np.array([list(m.position) for _, m in ev])
lp = rows["/fmu/out/vehicle_local_position"]
tl = np.array([t for t, _ in lp]) - t0
L = np.array([[m.x, m.y, m.z] for _, m in lp])
Lv = np.array([[m.vx, m.vy] for _, m in lp])
La = np.array([[m.ax, m.ay] for _, m in lp])
DB = np.array([m.dist_bottom for _, m in lp])


def fit(src, t_src):
    """Rigid 2D map of src (N,2) onto cuVSLAM xy over the follow, allowing a reflection. -> (function, residual)."""
    k = np.searchsorted(to, t_src).clip(1, len(to) - 1)
    g = t_src >= tf
    best = None
    for refl in (False, True):
        a0 = src[g].copy()
        if refl:
            a0[:, 1] *= -1
        a, b = a0 - a0.mean(0), O[k[g], :2] - O[k[g], :2].mean(0)
        U, S, Vt = np.linalg.svd(a.T @ b)
        D = np.eye(2)
        D[1, 1] = np.sign(np.linalg.det(U @ Vt))
        R = (U @ D @ Vt).T
        tr = O[k[g], :2].mean(0) - a0.mean(0) @ R.T
        res = float(np.median(np.hypot(*(a0 @ R.T + tr - O[k[g], :2]).T)))
        if best is None or res < best[1]:
            best = (refl, res, R, tr)
    refl, res, R, tr = best

    def f(p):
        p = np.array(p, float)
        if refl:
            p[..., 1] = -p[..., 1]
        return p @ R.T + tr
    return f, res, (refl, R)


f_ev, r_ev, _ = fit(E[:, :2], te)
f_lp, r_lp, (refl_lp, R_lp) = fit(L[:, :2], tl)
# IMU-only: integrate PX4 NED acceleration twice from follow start, starting at PX4's position and velocity there
g = tl >= tf
ti, A, v, p = tl[g], La[g], Lv[g][0].copy(), L[g][0, :2].copy()
imu = [p.copy()]
for i in range(1, len(ti)):
    dt = ti[i] - ti[i - 1]
    v = v + A[i - 1] * dt
    p = p + v * dt
    imu.append(p.copy())
IMU = f_lp(np.array(imu))                                 # into cuVSLAM's frame with PX4's own frame fit


def to_map(p):
    return np.c_[START[0] - (p[:, 1] - pad[1]), START[1] + (p[:, 0] - pad[0])]


Om, Em, Lm, Im = to_map(O[:, :2]), to_map(f_ev(E[:, :2])), to_map(f_lp(L[:, :2])), to_map(IMU)
fo = to >= tf
fig = plt.figure(figsize=(17, 10))
gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1])
a = fig.add_subplot(gs[:, 0])
d = np.load(SDF)
o, c, sdf = d["origin"], float(d["cell"]), d["planning"]
k = int(round((1.0 - o[2]) / c))
X, Y = np.meshgrid(o[0] + c * np.arange(sdf.shape[0]), o[1] + c * np.arange(sdf.shape[1]), indexing="ij")
occ = sdf[:, :, k] < 0
a.scatter(X[occ], Y[occ], s=2, c="0.6", marker="s", linewidths=0, label="walls (map)")
a.plot([-H, H, H, -H, -H], [-H, -H, H, H, -H], "r-", lw=2.5, label="nets")
rt = json.load(open(ROUTE))["route"]
a.plot([START[0]] + [START[0] - q[1] for q in rt], [START[1]] + [START[1] + q[0] for q in rt], "--", color="#3060c0", lw=1, label="route")
a.plot(Im[:, 0], Im[:, 1], color="#999900", lw=1.5, label="IMU only (accel integrated twice from follow start)")
a.plot(Om[fo, 0], Om[fo, 1], color="#d02090", lw=3.5, alpha=0.6, label="cuVSLAM odometry (= the VIO)")
a.plot(Em[te >= tf, 0], Em[te >= tf, 1], color="#1070d0", lw=1.4, label=f"EV sent to PX4 (median {100 * r_ev:.1f} cm off cuVSLAM)")
a.plot(Lm[tl >= tf, 0], Lm[tl >= tf, 1], "k:", lw=1.6, label=f"PX4 EKF2 (IMU+EV+range; median {100 * r_lp:.1f} cm off)")
a.annotate("pilot: the real drone was\nAT the north net here", (Om[-1, 0], H), xytext=(-2.9, 3.4), color="red", fontsize=9,
           arrowprops=dict(arrowstyle="->", color="red"))
a.set_aspect("equal")
a.set_xlim(-3.4, 3.4)
a.set_ylim(-3.4, 3.9)
a.set_title("All position estimates, on the map", fontsize=11)
a.legend(fontsize=7.5, loc="lower left")
a.grid(alpha=0.2)

b = fig.add_subplot(gs[0, 1])
b.plot(to, Om[:, 0], color="#d02090", lw=3, alpha=0.5, label="cuVSLAM x (east)")
b.plot(tl, Lm[:, 0], "k:", label="PX4 x")
b.plot(to, Om[:, 1], color="#d02090", lw=3, alpha=0.5, ls="--", label="cuVSLAM y (north)")
b.plot(tl, Lm[:, 1], "k-.", lw=1, label="PX4 y")
b.plot(ti, Im[:, 0], color="#999900", label="IMU-only x")
b.plot(ti, Im[:, 1], color="#999900", ls="--", label="IMU-only y")
b.set_ylim(-3.5, 3.5)
b.axvline(tf, color="k", ls=":", lw=1)
b.set_title("Map position over time", fontsize=11)
b.legend(fontsize=7)
b.grid(alpha=0.2)

c2 = fig.add_subplot(gs[0, 2])
c2.plot(to, O[:, 2], color="#d02090", lw=2, label="cuVSLAM height")
c2.plot(te, -E[:, 2] if np.median(E[te > tf + 5, 2]) < 0 else E[:, 2], color="#1070d0", lw=1, label="EV to PX4 height")
c2.plot(tl, -L[:, 2], "k:", label="PX4 EKF2 height (rangefinder-based)")
c2.plot(tl, DB - DB[tl < 15].mean(), color="0.5", lw=1, label="rangefinder (minus its 0.10 m on the ground)")
c2.axvline(tf, color="k", ls=":", lw=1)
c2.set_ylim(-0.1, 1.4)
c2.set_title("Height estimates", fontsize=11)
c2.legend(fontsize=7)
c2.grid(alpha=0.2)

e = fig.add_subplot(gs[1, 1:])
ko = lambda tq: np.searchsorted(to, tq).clip(1, len(to) - 1)  # noqa: E731
e.plot(te, 100 * np.hypot(*(f_ev(E[:, :2]) - O[ko(te), :2]).T), color="#1070d0", label="EV to PX4 minus cuVSLAM")
e.plot(tl, 100 * np.hypot(*(f_lp(L[:, :2]) - O[ko(tl), :2]).T), "k", lw=1, label="PX4 EKF2 minus cuVSLAM")
e.plot(ti, 100 * np.hypot(*(IMU - O[ko(ti), :2]).T), color="#999900", label="IMU-only minus cuVSLAM")
e.plot(tl, 100 * np.abs((-L[:, 2]) - O[ko(tl), 2]), color="#d06000", lw=1, label="height: PX4 (rangefinder) minus cuVSLAM")
e.axvline(tf, color="k", ls=":", lw=1)
e.set_yscale("symlog", linthresh=5)
e.set_ylabel("cm (log above 5)")
e.set_xlabel("time in bag (s)")
e.set_title("How far each estimate is from cuVSLAM: the EV link and PX4 agree with it to ~1 cm, so they can't reveal its drift; "
            "only height has an independent sensor (rangefinder)", fontsize=10)
e.legend(fontsize=8, loc="upper left")
e.grid(alpha=0.2)
fig.suptitle("skybeta-10 scored MPPI run 2 (route_mppi_full_160418): every pose estimate in the bag", fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=100)
print("saved", OUT, f"| EV vs cuVSLAM median {100 * r_ev:.1f} cm, PX4 vs cuVSLAM {100 * r_lp:.1f} cm (reflection {refl_lp}); "
      f"IMU-only end error {np.hypot(*(IMU[-1] - O[ko(ti[-1:])[0], :2])):.2f} m after {ti[-1] - ti[0]:.0f} s")
