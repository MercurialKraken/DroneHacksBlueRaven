"""DEBUG/SIM ONLY: the full flight pipeline on a course map, with the real drone's errors, before the T4 sim.

    python tools/mppi_course_sim.py <course_dir> <route.json> [out.gif] [--errors=realistic] [--seed=N]

  course_dir: a course folder (course.yaml + course_sdf.npz), e.g. courses/start-2026-11-07-tape
  route.json: a routes/*.json file ([forward, left] from the take-off pose; *_sim.json -> launch_sim)
  --errors:   perfect | old_noise | realistic | t4plant | realpush | harsh   (see Errors)
  --nopush:   don't give MPPI the heading (no steady-push estimate; as before 2026-11-07 evening)
  --hoverpush=S  the push acts in HOVER too (S s long), held by a refnav-like position loop (0.5/s, <= 0.25 m/s)
  --hoverlearn=S as --hoverpush, and the hover's poses and commands feed MPPI's push estimate (LocalNav.observe)

Phases, each separate as on the drone (MPPI never takes off, hovers or lands):
  TAKEOFF  climb at the launch pose to the route's height_m at 0.35 m/s (xy held)
  HOVER    2 s still; depth fills the grid; MPPI not called
  FOLLOW   nav.local_nav (route -> smooth trajectory -> MPPI on the live depth grid) until ARRIVED
           (or BLOCKED / STUCK for 3 s -> land where it is)
  LAND     brake and descend in place at 0.35 m/s; MPPI not called
World: depth is sphere-traced through the course's 3D distance field ('localization': walls, barrier, tube, boulders;
plus the floor), so nets are invisible to it as on the real camera; clearance is judged on 'planning' (everything,
nets included) at the TRUE position. The controller only ever sees the BELIEVED pose (truth + estimate error, late,
with gaps) and depth placed with the believed pose at capture, exactly as on the drone.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nav.course_map import CourseMap                                                # noqa: E402
from nav.local_nav import LocalNav                                                  # noqa: E402
from nav.mppi_ros import route_to_odom                                              # noqa: E402

try:
    import yaml
except ImportError:                                                                 # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class Errors:
    """What the real drone gets wrong, between the true world and the controller.

    Sources (2026-11-07, skybeta-09 unless noted):
      odometry  /visual_slam/tracking/odometry 29.5 Hz. IN FLIGHT (hover_160526, Darion): delivery bunching, arrival gaps
                p99 306 ms, max 0.351 s, ~10 gaps >= 0.3 s in 43 s -> realistic: a 0.25-0.35 s gap every ~4 s. EV age at
                PX4 p50 58 ms, p90 259 ms. course1 walk: one 3.6 cm jump. Hover drift from take-off 0.12 m mean, 0.22 max.
      drift     not measurable from today's bags (no ground truth; course1 had the tilted-start frame). Calibrated on
                Hinson's T4 cuVSLAM runs (est vs truth p95 3-6 cm over the course): realistic ~2-3x that (1 % of
                distance + 0.03 m per radian of yaw, slip direction redrawn every 0.5 rad of turning), harsh ~10x (3 %
                + 0.10 m/rad). (A fixed slip direction, tried first, summed every turn into 0.7-0.9 m: too pessimistic.)
                T4 also showed a 0.40 m slip in one 90 deg in-place yaw at 29 deg/s, which the follower avoids by
                turning while moving.
      depth     on-camera, ~35 % valid pixels, capture->ROS ~49 ms; MEASURED error sigma ~6 cm at 2 m and 23 cm at 4 m
                (Hinson, 20:02) = 0.015 z^2 (realistic); harsh 0.025 z^2. Textureless surfaces return nothing (no IR
                projector): harsh hides 25 % of wall surface in 0.5 m patches (texture_holes).
      response  PX4 velocity loop modelled with tau 0.55 s (MPC_XY_VEL_P_ACC 1.8); the true one is unmeasured: 0.65 /
                0.8 s here, plus command latency (interface + DDS), MPC_ACC_HOR 3 m/s^2, and air disturbance."""
    odom_delay_s: float = 0.0
    gap_per_s: float = 0.0
    gap_s: tuple[float, float] = (0.15, 0.40)
    drift_frac: float = 0.0
    yaw_drift_dpm: float = 0.0
    yaw_slip_m_per_rad: float = 0.0
    jump_per_s: float = 0.0
    jump_m: float = 0.04
    depth_k: float = 0.0
    depth_dropout: float = 0.0
    depth_outliers: float = 0.0
    depth_delay_s: float = 0.0
    texture_holes: float = 0.0          # fraction of wall surface (0.5 m patches) that returns no depth
    tau_true: float = 0.55
    act_delay_s: float = 0.0
    acc_max: float = 1e9
    vel_noise: float = 0.0
    yaw_noise_deg: float = 0.0          # heading noise on each odometry sample (deg), on top of the yaw drift
    push_body: tuple[float, float] = (0.0, 0.0)   # steady velocity push in the BODY frame (forward, left), m/s: the real
                                                  # skybeta-09 flight 160449 (Chelsea): (-0.056, +0.100), never nulled by PX4


PRESETS = {
    "perfect": Errors(),
    "old_noise": Errors(depth_k=0.006, depth_dropout=0.65, depth_outliers=0.01),
    "realistic": Errors(odom_delay_s=0.08, gap_per_s=1 / 4, gap_s=(0.25, 0.35), drift_frac=0.01, yaw_drift_dpm=1.0,
                        yaw_slip_m_per_rad=0.03, jump_per_s=1 / 30, jump_m=0.04, depth_k=0.015, depth_dropout=0.65,
                        depth_outliers=0.02, depth_delay_s=0.06, tau_true=0.65, act_delay_s=0.05, acc_max=3.0,
                        vel_noise=0.02, yaw_noise_deg=0.3),
    # realistic, with the PX4 response Hinson fitted on the T4 through the real flight_interface: tau 0.9 s + 0.6 s dead time
    "t4plant": Errors(odom_delay_s=0.08, gap_per_s=1 / 4, gap_s=(0.25, 0.35), drift_frac=0.01, yaw_drift_dpm=1.0,
                      yaw_slip_m_per_rad=0.03, jump_per_s=1 / 30, jump_m=0.04, depth_k=0.015, depth_dropout=0.65,
                      depth_outliers=0.02, depth_delay_s=0.06, tau_true=0.9, act_delay_s=0.6, acc_max=3.0,
                      vel_noise=0.02, yaw_noise_deg=0.3),
    # t4plant + the real drone's steady body-frame back-left push from route 1 flight 160449
    "realpush": Errors(odom_delay_s=0.08, gap_per_s=1 / 4, gap_s=(0.25, 0.35), drift_frac=0.01, yaw_drift_dpm=1.0,
                       yaw_slip_m_per_rad=0.03, jump_per_s=1 / 30, jump_m=0.04, depth_k=0.015, depth_dropout=0.65,
                       depth_outliers=0.02, depth_delay_s=0.06, tau_true=0.9, act_delay_s=0.6, acc_max=3.0,
                       vel_noise=0.02, push_body=(-0.056, 0.100), yaw_noise_deg=0.3),
    "harsh": Errors(odom_delay_s=0.12, gap_per_s=1 / 10, drift_frac=0.03, yaw_drift_dpm=3.0, yaw_slip_m_per_rad=0.10,
                    jump_per_s=1 / 15, jump_m=0.06, depth_k=0.025, depth_dropout=0.75, depth_outliers=0.05,
                    depth_delay_s=0.10, texture_holes=0.25, tau_true=0.80, act_delay_s=0.10, acc_max=3.0, vel_noise=0.04,
                    yaw_noise_deg=0.6),
}


class CourseWorld:
    def __init__(self, cm: CourseMap) -> None:
        self.cm = cm

    def depth_points(self, cam: np.ndarray, yaw: float, rng: np.random.Generator, err: Errors,
                     hfov_deg: float = 72.0, vfov_deg: float = 55.0, n_h: int = 160, n_v: int = 11,
                     max_range: float = 5.0) -> np.ndarray:
        az = yaw + np.radians(np.linspace(-hfov_deg / 2, hfov_deg / 2, n_h))
        el = np.radians(np.linspace(-vfov_deg / 2, vfov_deg / 2, n_v))
        A, E = np.meshgrid(az, el, indexing="ij")
        d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
        t = np.full(len(d), 0.05)
        hit = np.zeros(len(d), bool)
        alive = np.ones(len(d), bool)
        for _ in range(48):                                                         # sphere tracing
            idx = np.nonzero(alive & ~hit)[0]
            if not len(idx):
                break
            P = cam[None, :] + t[idx, None] * d[idx]
            s = self.cm.distance(P, field="localization")
            out = ~np.isfinite(s)
            alive[idx[out]] = False
            s = np.where(out, 1.0, s)
            h = s < 0.01
            hit[idx[h]] = True
            t[idx[~h]] += np.maximum(0.9 * s[~h], 0.01)
            alive &= t < max_range
        tf = np.where(d[:, 2] < -1e-6, -cam[2] / np.minimum(d[:, 2], -1e-6), np.inf)   # the floor plane z = 0
        if err.texture_holes > 0:                                                   # blank wall patches: no return
            Q = cam[None, :] + t[:, None] * d
            cell = np.floor(Q / 0.5).astype(np.int64)
            hsh = ((cell[:, 0] * 73856093) ^ (cell[:, 1] * 19349663) ^ (cell[:, 2] * 83492791)) % 1000
            hit &= hsh >= int(1000 * err.texture_holes)
        r = np.minimum(np.where(hit, t, np.inf), tf)
        ok = np.isfinite(r) & (r <= max_range)
        r, d = r[ok], d[ok]
        r = r + rng.normal(0.0, 1.0, len(r)) * err.depth_k * r ** 2
        o = rng.random(len(r)) < err.depth_outliers
        r = np.where(o, rng.uniform(0.2, max_range, len(r)), r)
        keep = (rng.random(len(r)) >= err.depth_dropout) & (r > 0.05)
        r, d = r[keep], d[keep]
        return cam[None, :] + r[:, None] * d

    def clearance(self, xy: np.ndarray, z: float) -> np.ndarray:
        xy = np.atleast_2d(xy)
        return self.cm.distance(np.c_[xy, np.full(len(xy), z)], field="planning")


def load_launch(course_dir: Path, key: str) -> tuple[np.ndarray, float]:
    text = (course_dir / "course.yaml").read_text(encoding="utf-8")
    if yaml is not None:
        L = yaml.safe_load(text)[key]
        return np.array([float(L["x"]), float(L["y"])]), math.radians(float(L["yaw_deg"]))
    block = text.split(f"\n{key}:", 1)[1].split("\n\n", 1)[0]                     # minimal fallback parser
    val = {k: float(block.split(f"{k}:", 1)[1].split()[0]) for k in ("x", "y", "yaw_deg")}
    return np.array([val["x"], val["y"]]), math.radians(val["yaw_deg"])


def _rot(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


def fly(course_dir: str, route_file: str, err: Errors = PRESETS["old_noise"], seed: int = 3,
        keep_frames: bool = False, speed: float | None = None, matched: bool = False, use_yaw: bool = True,
        hover_s: float = 2.0, hover_push: bool = False, hover_learn: bool = False) -> dict:
    course_dir = Path(course_dir)
    cm = CourseMap.load(course_dir / "course_sdf.npz")
    world = CourseWorld(cm)
    route = json.loads(Path(route_file).read_text(encoding="utf-8"))
    key = "launch_sim" if str(route_file).endswith("_sim.json") else "launch"
    launch, yaw0 = load_launch(course_dir, key)
    height = float(route.get("params", {}).get("height_m", 0.7))
    wps = route_to_odom(route["route"], launch, yaw0)            # the odom frame starts at take-off = map here
    from nav.local_nav import LocalNavParams
    from nav.mppi import MppiParams
    mp = MppiParams(tau=err.tau_true, dead_time_s=err.act_delay_s) if matched else None
    nav = LocalNav(wps, launch, yaw0, nav=LocalNavParams(v_cruise=speed) if speed else None, mppi=mp)
    nav.mppi.debug_samples = 120 if keep_frames else 0
    p = nav.mppi.p
    dt = p.dt
    a_true = math.exp(-dt / err.tau_true)
    rng = np.random.default_rng(seed)
    x, v, z, yaw = launch.copy(), np.zeros(2), 0.07, yaw0        # truth
    e, eyaw = np.zeros(2), 0.0                                   # estimate error (believed = R(eyaw)(true-launch)+launch+e)
    drift_dir = rng.uniform(0, 2 * math.pi)
    yaw_sign = 1.0 if rng.random() < 0.5 else -1.0
    slip = rng.normal(size=2)
    slip /= max(float(np.hypot(*slip)), 1e-9)
    turned = 0.0
    hist: list[tuple[float, np.ndarray, np.ndarray, float]] = []  # (t, believed xy, believed v, believed yaw)
    depth_q: list[tuple[float, np.ndarray, np.ndarray, float]] = []
    act_q: list[tuple[float, np.ndarray]] = []
    u_applied = np.zeros(2)
    t_gap_end = -1.0
    phase, t, t_phase, t_bad = "TAKEOFF", 0.0, 0.0, None
    frames, track = [], []
    log: dict = {"phases": {}}
    max_err = 0.0

    def to_b(Q: np.ndarray) -> np.ndarray:                       # true xy -> believed xy
        return (Q - launch) @ _rot(eyaw).T + launch + e

    while t < 240.0:
        # estimate error grows with distance flown (wandering direction), steady yaw drift, rare jumps
        step_d = float(np.hypot(*v)) * dt
        drift_dir += rng.normal() * 0.3 * math.sqrt(dt)
        e = e + err.drift_frac * step_d * np.array([math.cos(drift_dir), math.sin(drift_dir)])
        eyaw += yaw_sign * math.radians(err.yaw_drift_dpm) / 60.0 * dt
        if rng.random() < err.jump_per_s * dt:
            ang = rng.uniform(0, 2 * math.pi)
            e = e + err.jump_m * np.array([math.cos(ang), math.sin(ang)])
        # the camera (truth) and its depth frame, placed with the believed pose at capture, arriving late
        cam = np.array([x[0] + 0.104 * math.cos(yaw), x[1] + 0.104 * math.sin(yaw), z + 0.03])
        if phase != "LAND":
            pts = world.depth_points(cam, yaw, rng, err)
            depth_q.append((t + err.depth_delay_s, np.c_[to_b(pts[:, :2]), pts[:, 2]],
                            np.r_[to_b(cam[None, :2])[0], cam[2]], t))
        while depth_q and depth_q[0][0] <= t + 1e-9:
            _, pb, cb, tc = depth_q.pop(0)
            nav.add_depth(pb, cb, z, tc)
        # odometry: gaps and latency
        if t >= t_gap_end and rng.random() < err.gap_per_s * dt:
            t_gap_end = t + rng.uniform(*err.gap_s)
        if t >= t_gap_end:
            yaw_meas = yaw + eyaw + math.radians(err.yaw_noise_deg) * rng.normal()     # the believed heading
            hist.append((t, to_b(x[None])[0], _rot(eyaw) @ v, yaw_meas))
        avail = [h for h in hist if h[0] <= t - err.odom_delay_s + 1e-9]
        pos_b, vel_b, stamp_b, yaw_b = ((avail[-1][1], avail[-1][2], avail[-1][0], avail[-1][3]) if avail
                                        else (launch.copy(), np.zeros(2), t, yaw0))
        max_err = max(max_err, float(np.hypot(*(to_b(x[None])[0] - x))))
        cmd = None
        if phase == "TAKEOFF":
            z = min(height, z + 0.35 * dt)
            if z >= height - 1e-6:
                phase, t_phase = "HOVER", t
        elif phase == "HOVER":
            if hover_push:                                           # refnav's hover: hold the take-off xy
                u_h = 0.5 * (to_b(launch[None])[0] - pos_b)
                n_h = float(np.hypot(*u_h))
                u_h = u_h * min(1.0, 0.25 / max(n_h, 1e-9))
                act_q.append((t + err.act_delay_s, _rot(-eyaw) @ u_h))
                if hover_learn:
                    nav.observe(pos_b, t, yaw_b, u_h, pose_stamp=stamp_b if matched else None)
            if t - t_phase >= hover_s:
                phase, t_phase = "FOLLOW", t
        elif phase == "FOLLOW":
            cmd = nav.step(pos_b, vel_b, t, pose_stamp=stamp_b if matched else None, yaw=yaw_b if use_yaw else None)
            act_q.append((t + err.act_delay_s, _rot(-eyaw) @ np.array([cmd.vx, cmd.vy])))   # flown in the true frame
            dy = (cmd.yaw - eyaw - yaw + math.pi) % (2 * math.pi) - math.pi
            dyaw = float(np.clip(dy, -math.radians(90) * dt, math.radians(90) * dt))
            yaw += dyaw
            e = e + err.yaw_slip_m_per_rad * abs(dyaw) * slip
            turned += abs(dyaw)
            if turned > 0.5:                                         # a new turn slips in a new direction
                turned = 0.0
                slip = rng.normal(size=2)
                slip /= max(float(np.hypot(*slip)), 1e-9)
            if cmd.state in ("BLOCKED", "STUCK"):
                t_bad = t if t_bad is None else t_bad
                if t - t_bad >= 3.0:
                    log["ended"] = cmd.state
                    phase, t_phase = "LAND", t
            else:
                t_bad = None
            if cmd.state == "ARRIVED":
                log["ended"] = "ARRIVED"
                phase, t_phase = "LAND", t
        while act_q and act_q[0][0] <= t + 1e-9:
            u_applied = act_q.pop(0)[1]
        if phase == "LAND":
            u_applied = np.zeros(2)
            z = max(0.07, z - 0.35 * dt)
        if phase in ("FOLLOW", "LAND") or (phase == "HOVER" and hover_push):
            push = _rot(yaw) @ np.asarray(err.push_body, float)          # body (forward, left) -> world
            target = u_applied + (push if phase in ("FOLLOW", "HOVER") else 0.0)
            v_new = target + (v - target) * a_true
            dv = v_new - v
            n = float(np.hypot(*dv))
            if n > err.acc_max * dt:
                v_new = v + dv * (err.acc_max * dt / n)
            v_new = v_new + rng.normal(0.0, err.vel_noise, 2)
            x = x + 0.5 * (v + v_new) * dt
            v = v_new
        log["phases"].setdefault(phase, round(t, 2))
        track.append((t, x[0], x[1], z, phase))
        if keep_frames and len(track) % 2 == 0:
            ii, jj = np.nonzero(nav.grid.occupied())
            occ_b = np.c_[nav.grid.origin[0] + ii * nav.grid.p.cell, nav.grid.origin[1] + jj * nav.grid.p.cell]
            Rinv = _rot(-eyaw)
            frames.append(dict(t=t, x=x.copy(), z=z, yaw=yaw, phase=phase, cmd=cmd, e=e.copy(), eyaw=eyaw,
                               occ=(occ_b - launch - e) @ Rinv.T + launch))     # shown in the TRUE frame
        if phase == "LAND" and z <= 0.07 + 1e-6:
            break
        t += dt
    T = np.array([[r[1], r[2], r[3]] for r in track])
    rows = [k for k, r in enumerate(track) if r[4] == "FOLLOW"]
    clr = world.clearance(T[rows, :2], height) if rows else np.array([np.inf])
    clr = np.where(np.isfinite(clr), clr, -1.0)                       # off the map = through the net: a touch, not NaN
    k_min = int(np.argmin(clr)) if rows else 0
    path_len = float(np.hypot(*np.diff(T[rows, :2], axis=0).T).sum()) if len(rows) > 1 else 0.0
    follow_s = (track[rows[-1]][0] - track[rows[0]][0]) if rows else 0.0
    log.update(time_s=t, follow_s=follow_s, avg_speed=path_len / max(follow_s, 1e-9), min_clearance=float(np.min(clr)),
               min_clearance_at=T[rows[k_min], :2].round(2).tolist() if rows else None,
               touched=bool(np.min(clr) < p.r_drone), max_est_error=max_err, final_est_error=float(np.hypot(*e)),
               land_xy=T[-1, :2].round(3).tolist(), land_err_to_last_wp=float(np.hypot(*(T[-1, :2] - wps[-1]))),
               push_est_body=nav.mppi.push.d.round(3).tolist(), follow_start_xy=T[rows[0], :2].round(3).tolist()
               if rows else None)
    if keep_frames:
        log.update(frames=frames, track=T, world=world, nav=nav, wps=wps, launch=launch)
    return log


def render(log: dict, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection
    cm, frames, T, nav, launch = log["world"].cm, log["frames"], log["track"], log["nav"], log["launch"]
    lo, hi = cm.bounds
    xs = np.arange(lo[0], hi[0] + 1e-9, cm.cell)
    ys = np.arange(lo[1], hi[1] + 1e-9, cm.cell)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    flat = np.c_[X.ravel(), Y.ravel(), np.full(X.size, 0.7)]
    walls = cm.distance(flat, "localization").reshape(X.shape) < 0
    solid = cm.distance(flat, "planning").reshape(X.shape) < 0
    low = (cm.distance(np.c_[X.ravel(), Y.ravel(), np.full(X.size, 0.15)], "localization").reshape(X.shape) < 0) & ~walls
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=85)
    wp = np.r_[launch[None], log["wps"]]

    def draw(n: int) -> None:
        ax.clear()
        f = frames[n]
        Rinv = _rot(-f["eyaw"])

        def tru(Q: np.ndarray) -> np.ndarray:                      # believed -> true frame, for drawing
            return (np.asarray(Q) - launch - f["e"]) @ Rinv.T + launch
        ax.scatter(X[solid & ~walls], Y[solid & ~walls], s=1, c="#9cc3e6", marker="s", linewidths=0)
        ax.scatter(X[walls], Y[walls], s=1, c="#9a9a9a", marker="s", linewidths=0)
        ax.scatter(X[low], Y[low], s=2, c="#f0a030", marker="s", linewidths=0)
        if len(f["occ"]):
            ax.scatter(f["occ"][:, 0], f["occ"][:, 1], s=2, c="k", marker="s", linewidths=0)
        ax.plot(*tru(wp).T, ":", color="#3060c0", lw=0.8, marker="o", ms=2.5)
        ax.plot(*tru(nav.path).T, "--", color="#3060c0", lw=0.8)
        c = f["cmd"]
        r = c.mppi if c is not None else None
        if r is not None and len(r.samples):
            wv = r.sample_w / max(float(r.sample_w.max()), 1e-12)
            order = np.argsort(wv)
            cols = [((0.85, 0.2, 0.2, 0.3) if r.sample_crashed[k] else
                     plt.cm.viridis(0.15 + 0.85 * wv[k], alpha=0.15 + 0.75 * wv[k])) for k in order]
            ax.add_collection(LineCollection([tru(r.samples[k]) for k in order], colors=cols, linewidths=0.6))
            ax.plot(*tru(r.plan).T, color="#d020c0", lw=2)
        k = int(round(f["t"] / nav.mppi.p.dt))
        ax.plot(T[: k + 1, 0], T[: k + 1, 1], color="#1f4fa0", lw=1.2)
        ax.add_patch(plt.Circle(f["x"], nav.mppi.p.r_drone, fill=False, color="#1f4fa0", lw=1.2))
        ax.plot([f["x"][0], f["x"][0] + 0.35 * math.cos(f["yaw"])], [f["x"][1], f["x"][1] + 0.35 * math.sin(f["yaw"])],
                color="#1f4fa0", lw=1.8)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        st = c.state if c is not None else "-"
        ax.set_title(f"t {f['t']:5.1f} s  {f['phase']:7s} {st:7s} z {f['z']:.2f} m  est err "
                     f"{100 * float(np.hypot(*f['e'])):.0f} cm {math.degrees(f['eyaw']):+.1f} deg",
                     fontsize=9, family="monospace")

    FuncAnimation(fig, draw, frames=range(0, len(frames), 2), interval=150).save(out, writer=PillowWriter(fps=10))
    draw(len(frames) - 1)
    fig.savefig(str(Path(out).with_suffix("")) + "_end.png")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seed = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--seed=")), 3)
    preset = next((a.split("=")[1] for a in sys.argv if a.startswith("--errors=")), "old_noise")
    speed = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--speed=")), None)
    matched = "--matched" in sys.argv
    use_yaw = "--nopush" not in sys.argv
    hl = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--hoverlearn=")), None)
    hp = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--hoverpush=")), None)
    out = args[2] if len(args) > 2 else None
    log = fly(args[0], args[1], err=PRESETS[preset], seed=seed, keep_frames=out is not None, speed=speed, matched=matched,
              use_yaw=use_yaw, hover_s=hl or hp or 2.0, hover_push=(hl or hp) is not None, hover_learn=hl is not None)
    if out:
        render(log, out)
    keep = ("ended", "follow_s", "avg_speed", "min_clearance", "min_clearance_at", "touched", "max_est_error",
            "final_est_error", "land_err_to_last_wp", "push_est_body", "follow_start_xy")
    print(json.dumps({"route": Path(args[1]).stem, "errors": preset, "seed": seed, "speed": speed or 0.45,
                      "model": "matched" if matched else "default", "push_est": use_yaw,
                      "hover": "learn" if hl else ("push" if hp else "still"),
                      **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in log.items() if k in keep}}),
          flush=True)
