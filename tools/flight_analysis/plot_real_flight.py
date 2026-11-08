"""Plot a real route-1 flight from extract_flight.py's CSV: the flown path (PX4 EKF2 fused pose + cuVSLAM) against
refnav's planned path on the course map, and the fused pose over time.

    python plot_real_flight.py <flight.csv> <repo> <out prefix>
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

CSV, REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from make_start_obj import COURSE, start_geometry, wall_footprint  # noqa: E402
from make_route_cp2_simple import to_polygon  # noqa: E402
from nav.frames import load_launch  # noqa: E402

rows = {"lpos": [], "odom": [], "twist": [], "status": [], "vstatus": []}
for line in CSV.read_text(encoding="utf-8").splitlines():
    kind, rest = line.split(",", 1)
    if kind == "status":
        t, js = rest.split(",", 1)
        rows[kind].append((float(t), json.loads(js)))
    elif kind in rows:
        rows[kind].append([float(x) for x in rest.split(",")])
L = np.array(rows["lpos"]); O = np.array(rows["odom"]); TW = np.array(rows["twist"]); VS = np.array(rows["vstatus"])
ST = rows["status"]

# phases from refnav's status
phase_t = {}
for t, s in ST:
    phase_t.setdefault(s["phase"], t)
t_climb, t_follow, t_land = phase_t.get("climb"), phase_t.get("follow"), phase_t.get("land")
t_done = phase_t.get("done", ST[-1][0])
t0 = t_climb
armed = VS[VS[:, 2] == 2, 0]
t_arm, t_disarm = armed[0], VS[(VS[:, 0] > armed[0]) & (VS[:, 2] == 1), 0][0]

# the take-off (pad) frame refnav used: its 'take-off pose' = odom at wait->arm; forward = its heading
yaw0 = math.radians(-0.4)
p0 = np.array([0.007, 0.007])
f = np.array([math.cos(yaw0), math.sin(yaw0)]); r = np.array([math.sin(yaw0), -math.cos(yaw0)])


def pad(xy_enu):
    d = xy_enu - p0
    return np.c_[d @ f, d @ r]          # (forward, right)


# PX4 NED -> ENU (the interface's plain enu_to_ned, offset 0.0): x_enu = east, y_enu = north
px4_enu = np.c_[L[:, 2], L[:, 1]]
on_ground = (L[:, 0] > t_arm - 20) & (L[:, 0] < t_arm)
odom_at = np.c_[np.interp(L[:, 0], O[:, 0], O[:, 1]), np.interp(L[:, 0], O[:, 0], O[:, 2])]
shift = np.median((px4_enu - odom_at)[on_ground], axis=0)       # EKF origin vs cuVSLAM origin
px4_enu = px4_enu - shift
flying = (L[:, 0] >= t_climb) & (L[:, 0] <= t_disarm)
dis = np.linalg.norm((px4_enu - odom_at)[flying], axis=1)
P = pad(px4_enu); Q = pad(O[:, 1:3])
h_px4 = -(L[:, 3] - np.median(L[on_ground, 3]))
h_odom = O[:, 3] - np.median(O[(O[:, 0] > t_arm - 20) & (O[:, 0] < t_arm), 3])
hdg_px4 = np.degrees(np.unwrap(L[:, 7]))                       # NED heading: 90 deg = east = forward here
yaw_odom = np.degrees(np.unwrap(O[:, 4]))

path = np.array(next(s["path"] for t, s in ST if s.get("path")))     # pad (n, e) = (forward, right)
sp = np.array([(t, *s["setpoint"][:2], s["setpoint"][3]) for t, s in ST if s.get("setpoint")])
xs = np.array([(t, s.get("xtrack", np.nan), s.get("scale", np.nan), s.get("s_hat", np.nan), s.get("s_ref", np.nan))
               for t, s in ST if s["phase"] == "follow"])

# course map: launch mark, yaw 90 (forward = +y, right = +x)
cfg = yaml.safe_load(COURSE.read_text(encoding="utf-8"))
g = start_geometry(cfg)
walls = {w: np.asarray(wall_footprint(*g[k], g["t"], g["gap_mid"])) for w, k in (("Wall A", "A"), ("Wall B", "B"))}
Lc = load_launch(COURSE, "launch")
LO = np.array([Lc["x"], Lc["y"]]); ly = math.radians(Lc["yaw_deg"])
mf, mr = np.array([math.cos(ly), math.sin(ly)]), np.array([math.sin(ly), -math.cos(ly)])


def to_map(fr):
    return LO + np.outer(fr[:, 0], mf) + np.outer(fr[:, 1], mr)


cp2 = next(c for c in cfg["checkpoints"] if c["id"] == "CP2")
ca, cb = np.asarray(cp2["line"], float); cn = np.asarray(cp2["direction"], float); cn /= np.linalg.norm(cn)
R = float(cfg["limits"]["drone_radius"])
Pm = to_map(P[flying]); plan_m = to_map(path); Qm = to_map(Q[(O[:, 0] >= t_climb) & (O[:, 0] <= t_disarm)])
clr = {w: to_polygon(Pm, fp) for w, fp in walls.items()}
side = (Pm - ca) @ cn
ci = int(np.argmax(side > 0)) if (side > 0).any() else None
cross = Pm[ci] if ci is not None else None
mid = (ca + cb) / 2
along = float((cross - mid) @ ((cb - ca) / np.linalg.norm(cb - ca))) if cross is not None else float("nan")
# cross-track of the fused pose to the planned path, in the follow phase
fol = (L[:, 0] >= t_follow) & (L[:, 0] <= t_land)
dpath = np.array([np.min(np.linalg.norm(path - p, axis=1)) for p in P[fol]])
sgn = []
for p in P[fol]:
    j = int(np.argmin(np.linalg.norm(path - p, axis=1))); j2 = min(j + 1, len(path) - 1); j1 = max(j2 - 1, 0)
    tng = path[j2] - path[j1]; tng /= np.linalg.norm(tng) + 1e-12
    v = p - path[j]
    sgn.append(np.sign(tng[0] * v[1] - tng[1] * v[0]))       # + = right of the path, - = left (forward,right frame)
sgn = np.array(sgn)
end = P[(L[:, 0] >= t_land)][0]
land_pt = P[L[:, 0] <= t_disarm][-1]
climb_end = P[np.searchsorted(L[:, 0], t_follow)]

summary = {
    "follow_s": round(t_land - t_follow, 1), "flight_s": round(t_disarm - t_climb, 1),
    "climb_drift_m (fwd,right)": np.round(climb_end, 3).tolist(),
    "px4_vs_cuvslam_max_m": round(float(dis.max()), 3), "px4_vs_cuvslam_p95_m": round(float(np.percentile(dis, 95)), 3),
    "xtrack_follow_mean_m": round(float(dpath.mean()), 3), "xtrack_follow_p95_m": round(float(np.percentile(dpath, 95)), 3),
    "xtrack_side": f"{(sgn < 0).mean() * 100:.0f}% left of the path, {(sgn > 0).mean() * 100:.0f}% right",
    "speed_mean_follow_mps": round(float(xs[-1, 3] / (t_land - t_follow)), 3) if len(xs) else None,
    "scale0_frac_follow": round(float(np.mean(xs[:, 2] == 0)), 2) if len(xs) else None,
    "end_of_route_vs_last_point_m": round(float(np.linalg.norm(end - path[-1])), 3),
    "landed_at (fwd,right)": np.round(land_pt, 3).tolist(),
    "IF ON THE COURSE: min centre-to-wall m": {w: round(float(c.min()), 3) for w, c in clr.items()},
    "IF ON THE COURSE: CP2 crossed m from the gap centre (+ towards Wall B end)": round(along, 3),
    "IF ON THE COURSE: CP2 rotor margin each side m": round(0.5 - abs(along) - R, 3),
}
print(json.dumps(summary, indent=1))

# commanded twist (ENU) vs PX4's achieved velocity (NED -> ENU), follow phase
ve = np.c_[L[:, 5], L[:, 4]]
fw = (TW[:, 0] >= t_follow) & (TW[:, 0] <= t_land) & (np.hypot(TW[:, 1], TW[:, 2]) > 0.05)
cmd = TW[fw, 1:3]
lag = 0.55                                                    # compare with the velocity ~one lag later
ach = np.c_[np.interp(TW[fw, 0] + lag, L[:, 0], ve[:, 0]), np.interp(TW[fw, 0] + lag, L[:, 0], ve[:, 1])]
ang = np.degrees(np.arctan2(cmd[:, 0] * ach[:, 1] - cmd[:, 1] * ach[:, 0], (cmd * ach).sum(1)))   # + = achieved is CCW (left)
ratio = np.hypot(*ach.T) / np.hypot(*cmd.T)
along_ach = (cmd * ach).sum(1) / np.hypot(*cmd.T)
print(json.dumps({"cmd_vs_achieved (follow, achieved 0.55 s later)": {
    "angle_deg_median (+ = achieved left of command)": round(float(np.median(ang)), 1),
    "angle_deg_p10_p90": [round(float(np.percentile(ang, 10)), 1), round(float(np.percentile(ang, 90)), 1)],
    "speed_ratio_median": round(float(np.median(ratio)), 2),
    "achieved_along_command_median_mps": round(float(np.median(along_ach)), 3),
    "cmd_speed_median_mps": round(float(np.median(np.hypot(*cmd.T))), 3)}}, indent=1))

# ---------- figure 1: path ----------
fig, axs = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axs[0]
for w, fp in walls.items():
    ax.fill(*np.r_[fp, fp[:1]].T, color="#8a8f98", alpha=0.85, zorder=2)
    ax.text(*fp.mean(axis=0), w, ha="center", va="center", fontsize=9, color="white", zorder=3)
h = g["cone_size"] / 2
cb_ = np.array([g["cone"] + d for d in ([-h, -h], [h, -h], [h, h], [-h, h])])
ax.fill(*np.r_[cb_, cb_[:1]].T, color="#f28e2b", alpha=0.6, zorder=2); ax.text(*g["cone"], "cone", fontsize=8, ha="center")
ax.plot(*np.c_[ca, cb], color="#59a14f", lw=3, zorder=3, label="CP2 line (gap 1.00 m)")
ax.plot(*plan_m.T, color="black", lw=1.6, ls="--", zorder=4, label="refnav's planned path (arc)")
spm = to_map(sp[(sp[:, 0] >= t_follow) & (sp[:, 0] <= t_land), 1:3])
ax.plot(*spm.T, ".", color="#4e79a7", ms=3, zorder=4, label="refnav reference setpoint")
ax.plot(*Qm.T, color="#e15759", lw=1, alpha=0.7, zorder=5, label="cuVSLAM odometry")
tt = L[flying, 0] - t0
sc = ax.scatter(*Pm.T, c=tt, cmap="viridis", s=4, zorder=6, label="PX4 fused pose (EKF2)")
for k in range(0, len(Pm), max(1, len(Pm) // 18)):
    ax.add_patch(plt.Circle(Pm[k], R, fill=False, color="#76b7b2", lw=0.5, alpha=0.6, zorder=5))
ax.plot(*LO, "k^", ms=9, zorder=7, label="launch mark (take-off)")
ax.plot(*to_map(land_pt[None])[0], "rv", ms=9, zorder=7, label="landed (fused)")
if cross is not None:
    ax.plot(*cross, "o", mfc="none", mec="#59a14f", ms=12, mew=2, zorder=7, label=f"CP2 crossing ({along:+.2f} m from centre)")
fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01, label="s since take-off")
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_xlim(-3.0, -0.2); ax.set_ylim(0.6, 2.9)
ax.set_xlabel("course map x (m)"); ax.set_ylabel("course map y (m)")
ax.set_title("Route 1, real flight 160449: flown vs planned on the tape map\n"
             "(walls drawn where the course has them; assumes the drone sat on the launch mark facing north)", fontsize=10)
ax.legend(loc="lower right", fontsize=7.5)

ax = axs[1]
ax.plot(path[:, 1], path[:, 0], "k--", lw=1.6, label="planned")
ax.plot(sp[:, 2], sp[:, 1], ".", color="#4e79a7", ms=3, label="reference setpoint")
ax.plot(Q[:, 1], Q[:, 0], color="#e15759", lw=1, alpha=0.7, label="cuVSLAM")
sc2 = ax.scatter(P[flying, 1], P[flying, 0], c=tt, cmap="viridis", s=4, label="PX4 fused")
ax.plot(0, 0, "k^", ms=9); ax.plot(climb_end[1], climb_end[0], "s", color="#f28e2b", ms=8,
                                   label=f"end of climb ({climb_end[0]:+.2f} fwd, {climb_end[1]:+.2f} right)")
ax.plot(land_pt[1], land_pt[0], "rv", ms=9, label="landed")
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_xlabel("right of the take-off pose (m)"); ax.set_ylabel("forward of the take-off pose (m)")
ax.set_title("Same flight in refnav's own take-off frame", fontsize=10)
ax.legend(fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}_path.png", dpi=130)

# ---------- figure 2: fused pose over time ----------
fig, axs = plt.subplots(6, 1, figsize=(13, 15), sharex=True)
TL, TO = L[:, 0] - t0, O[:, 0] - t0


def shade(ax):
    for a, b, c, n in ((t_climb, t_follow, "#fce8b2", "climb"), (t_follow, t_land, "#d5e8d4", "follow"),
                       (t_land, t_disarm, "#f8cecc", "land")):
        ax.axvspan(a - t0, b - t0, color=c, alpha=0.6, lw=0)
        if ax is axs[0]:
            ax.text((a + b) / 2 - t0, 1.02, n, transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
    ax.axvline(t_arm - t0, color="k", lw=0.8, ls=":"); ax.axvline(t_disarm - t0, color="k", lw=0.8, ls=":")
    ax.grid(alpha=0.3)


ax = axs[0]; shade(ax)
ax.plot(TL, P[:, 0], color="#4e79a7", label="forward, PX4 fused"); ax.plot(TO, Q[:, 0], color="#4e79a7", lw=0.8, ls="--", label="forward, cuVSLAM")
ax.plot(TL, P[:, 1], color="#e15759", label="right, PX4 fused"); ax.plot(TO, Q[:, 1], color="#e15759", lw=0.8, ls="--", label="right, cuVSLAM")
ax.plot(sp[:, 0] - t0, sp[:, 1], ".", color="#4e79a7", ms=3, label="forward, reference")
ax.plot(sp[:, 0] - t0, sp[:, 2], ".", color="#e15759", ms=3, label="right, reference")
ax.set_ylabel("position (m)\ntake-off frame"); ax.legend(ncol=3, fontsize=7.5, loc="upper left")
ax = axs[1]; shade(ax)
ax.plot(TL, h_px4, color="#59a14f", label="height, PX4 fused"); ax.plot(TO, h_odom, color="#59a14f", lw=0.8, ls="--", label="height, cuVSLAM")
ax.axhline(0.7, color="k", lw=0.8, ls="--", label="target 0.7 m"); ax.set_ylabel("height (m)"); ax.legend(fontsize=7.5, loc="upper left")
ax = axs[2]; shade(ax)
ax.plot(TL, (90 - hdg_px4) - (-0.4), color="#b07aa1", label="heading vs take-off, PX4 fused (+ = left)")
ax.plot(TO, yaw_odom - (-0.4), color="#b07aa1", lw=0.8, ls="--", label="heading vs take-off, cuVSLAM (ENU, + = left)")
fsp = (sp[:, 0] >= t_follow) & (sp[:, 0] <= t_land)
ax.plot(sp[fsp, 0] - t0, -np.degrees(sp[fsp, 3]), ".", color="k", ms=3, label="refnav yaw setpoint (follow)")
ax.set_ylabel("heading (deg)"); ax.legend(fontsize=7.5, loc="lower left")
ax = axs[3]; shade(ax)
spd = np.hypot(L[:, 4], L[:, 5])
ax.plot(TL, spd, color="#76b7b2", label="ground speed, PX4 fused")
ax.plot(TW[:, 0] - t0, np.hypot(TW[:, 1], TW[:, 2]), color="k", lw=0.7, alpha=0.7, label="commanded twist speed")
ax.axhline(0.15, color="k", lw=0.8, ls="--", label="planned 0.15 m/s")
ax.set_ylabel("speed (m/s)"); ax.legend(fontsize=7.5, loc="upper left")
ax = axs[4]; shade(ax)
if len(xs):
    ax.plot(xs[:, 0] - t0, xs[:, 1], color="#e15759", label="refnav cross-track (m)")
    ax.plot(xs[:, 0] - t0, xs[:, 2], color="#4e79a7", label="refnav speed scale (0 = reference waits)")
ax.axhline(0.20, color="#e15759", lw=0.8, ls="--", label="xtrack_stop 0.20 m")
ax.set_ylabel("refnav"); ax.legend(fontsize=7.5, loc="upper left")
ax = axs[5]; shade(ax)
ax.plot(TW[fw, 0] - t0, ang, ".", color="#b07aa1", ms=3, label="achieved velocity direction minus commanded (deg, + = left)")
ax.plot(TW[fw, 0] - t0, ratio * 100, ".", color="#76b7b2", ms=3, label="achieved / commanded speed (%)")
ax.axhline(0, color="k", lw=0.8); ax.axhline(100, color="#76b7b2", lw=0.8, ls="--")
ax.set_ylim(-120, 200); ax.set_ylabel("command vs achieved"); ax.set_xlabel("seconds since take-off (climb start)")
ax.legend(fontsize=7.5, loc="upper left")
fig.suptitle("Route 1 real flight 160449: PX4 EKF2 fused pose over time (dashed: cuVSLAM)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}_pose.png", dpi=120)
