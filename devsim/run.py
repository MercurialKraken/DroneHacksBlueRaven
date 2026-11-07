"""Fly refnav.core.Mission in a box world: a kinematic PX4 with real-ish limits, a 15 Hz ray-cast
depth camera, optional depth noise/dropout and VIO drift. Exit 0 = complete, 1 = other result, 2 = crash."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from refnav import core as C
from refnav import follow as F
from refnav.launch import flight_params
from .world import DEPTH_BORDER_PX, K, R_BODY_CAM, T_BODY_CAM, World, q_yaw

ROOT = Path(__file__).resolve().parents[1]

DT = 1 / 60                     # physics tick; mission every 3 (20 Hz), depth every 4 (15 Hz)
DEPTH_LATENCY_TICKS = 2         # ~33 ms, the sim models 30 ms
VXY_MAX, ACC_XY, XY_P = 1.5, 1.0, 0.95          # MPC_XY_VEL_MAX, MPC_ACC_HOR, MPC_XY_P
VZ_UP, VZ_DN, ACC_Z = 1.0, 0.7, 3.0
YAW_RATE = math.radians(60)
RADIUS = 0.25


class PX4:
    """Obeys Offboard/arm/land like tests/test_refnav_core.py::FakePX4, but tracks the setpoint
    from its ESTIMATE with speed/accel limits, so VIO drift moves the true aircraft."""

    def __init__(self, n, e, yaw, drift, rng):
        self.pos, self.vel = np.array([n, e, 0.0]), np.zeros(3)
        self.yaw, self.off = yaw, np.zeros(3)
        self.armed, self.nav = False, 4
        self.stream_since = None
        self.drift, self.rng = drift, rng

    def est(self):
        return self.pos + self.off

    def command(self, t, out):
        self.stream_since = (self.stream_since if self.stream_since is not None else t) if out.stream else None
        for c in out.commands:
            if c == "offboard" and self.stream_since is not None and t - self.stream_since >= 0.5:
                self.nav = C.NAV_OFFBOARD
            elif c == "arm":
                self.armed = True
            elif c == "land":
                self.nav = C.NAV_AUTO_LAND

    def physics(self, sp, dt, vff=None):
        want = np.zeros(3)
        if self.armed and self.nav == C.NAV_OFFBOARD and sp is not None:
            err = np.array(sp[:3]) - self.est()
            dist = math.hypot(err[0], err[1])
            if vff is not None:                  # PX4 adds a velocity feed-forward to the position loop's output
                want[:2] = np.array(vff) + XY_P * err[:2]
                sp_n = np.linalg.norm(want[:2])
                want[:2] *= min(1.0, VXY_MAX / sp_n) if sp_n > 1e-9 else 1.0
            elif dist > 1e-6:
                want[:2] = err[:2] / dist * min(VXY_MAX, XY_P * dist, math.sqrt(2 * ACC_XY * dist))
            want[2] = np.clip(1.0 * err[2], -VZ_UP, VZ_DN)
            yerr = float(F.wrap(sp[3] - self.yaw))
            self.yaw += np.clip(yerr, -YAW_RATE * dt, YAW_RATE * dt)
        elif self.armed and self.nav == C.NAV_AUTO_LAND:
            want[2] = VZ_DN
        else:
            self.vel[:] = 0.0
            return
        dv = want[:2] - self.vel[:2]
        n = np.linalg.norm(dv)
        self.vel[:2] += dv * min(1.0, ACC_XY * dt / n) if n > 1e-9 else 0.0
        self.vel[2] += np.clip(want[2] - self.vel[2], -ACC_Z * dt, ACC_Z * dt)
        self.pos += self.vel * dt
        if self.drift > 0:
            self.off[:2] += self.rng.normal(0.0, self.drift * math.sqrt(dt), 2)
        if self.pos[2] >= 0.0:
            self.pos[2], self.vel[2] = 0.0, min(self.vel[2], 0.0)
            if self.nav == C.NAV_AUTO_LAND:
                self.armed, self.vel[:] = False, 0.0

    def vehicle(self):
        return C.Vehicle(pos=tuple(self.est()), heading=self.yaw, pos_ok=True, q=q_yaw(self.yaw), vel=tuple(self.vel[:2]),
                         armed=self.armed, nav_state=self.nav)


def degrade(img, rng):
    """Stereo-like depth: 1.5 % long (calibration), noise growing with z^2, 40 % pixel dropout."""
    z = img * 1.015
    z = z + rng.normal(0.0, 1.0, z.shape).astype(np.float32) * 0.019 * z * z
    z[rng.random(z.shape) < 0.4] = np.nan
    return z


def load_params(path, height=None, hover=None):
    """The aircraft's params: the route file's, then flight.yaml, then main.py's height and hover (or the CLI's)."""
    from main import HEIGHT_M, HOVER_S
    with open(path) as f:
        doc = json.load(f)
    over = {**flight_params(ROOT), "height_m": HEIGHT_M if height is None else height,
            "hover_s": HOVER_S if hover is None else hover, "depth_border_px": DEPTH_BORDER_PX}
    prm, unknown = C.params_from_doc(doc, over)
    for k in unknown:
        print(f"ignored route param {k}", file=sys.stderr)
    return prm


def simulate(world, prm, noise=False, drift=0.0, seed=0, seconds=600.0, quiet=False):
    rng = np.random.default_rng(seed)
    m = C.Mission(prm)
    px4 = PX4(*world.start, drift=drift, rng=rng)
    out, pending, crash = None, [], None
    log = {"t": [], "true": [], "est": []}
    k = 0
    while k * DT < seconds and m.phase != "done":
        t = k * DT
        if k % 4 == 0 and not (noise and rng.random() < 0.01):          # 1 % dropped frames with --noise
            img = world.render(px4.pos, px4.yaw)
            pending.append((k + DEPTH_LATENCY_TICKS, t, degrade(img, rng) if noise else img, q_yaw(px4.yaw)))
        while pending and pending[0][0] <= k:
            _, stamp, img, q = pending.pop(0)
            m.on_depth(stamp, img, K, R_BODY_CAM, T_BODY_CAM, q, now=t)
        if k % 3 == 0:
            m.on_health(t, True)
            out = m.step(t, px4.vehicle())
            px4.command(t, out)
            for ev in out.events:
                if not quiet:
                    print(f"{t:7.2f}  {ev}")
            log["t"].append(t)
            log["true"].append(px4.pos.copy())
            log["est"].append(px4.est())
        px4.physics(out.setpoint if out.stream else None, DT, out.velocity if out.stream else None)
        hit = world.collision(px4.pos, RADIUS)
        if hit:
            crash = f"CRASH into {hit} at true n={px4.pos[0]:.2f} e={px4.pos[1]:.2f} up={-px4.pos[2]:.2f}"
            break
        k += 1
    result = crash or m.result or f"timeout after {seconds:.0f} s in phase {m.phase}"
    return m, px4, log, result, crash is not None


def plot(path, world, m, px4, log, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(7, 7))
    for lo, hi, name in zip(world.lo, world.hi, world.names):
        if name != "ceiling":
            ax.add_patch(Rectangle((lo[1], lo[0]), hi[1] - lo[1], hi[0] - lo[0], color="0.4", alpha=0.6))
    tr, es = np.array(log["true"]), np.array(log["est"])
    ax.plot(tr[:, 1], tr[:, 0], "b-", lw=1.5, label="true path")
    ax.plot(es[:, 1], es[:, 0], "r--", lw=1, label="estimated path")
    ax.add_patch(plt.Circle((tr[0, 1], tr[0, 0]), RADIUS, fill=False, color="g"))
    ax.add_patch(plt.Circle((tr[-1, 1], tr[-1, 0]), RADIUS, fill=False, color="k"))
    if m.targets:
        wp = np.array(m.targets)
        ax.plot(wp[:, 1], wp[:, 0], "o", ms=8, mfc="none", color="orange", label="waypoints")
        for i, (n, e) in enumerate(m.targets):
            ax.annotate(str(i + 1), (e, n), textcoords="offset points", xytext=(6, 6))
    ax.plot([], [], "go", mfc="none", label="start")
    ax.plot([], [], "ko", mfc="none", label="end")
    ax.set(xlabel="east (m)", ylabel="north (m)", aspect="equal", title=f"{world.name}\n{result}")
    ax.title.set_fontsize(9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=110, bbox_inches="tight")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", required=True)
    ap.add_argument("--route", required=True, help="route.json: {route: [[forward, left], ...], params: {...}}")
    ap.add_argument("--height", type=float, help="override height_m (default: main.py's HEIGHT_M)")
    ap.add_argument("--hover", type=float, help="override hover_s (default: main.py's HOVER_S)")
    ap.add_argument("--noise", action="store_true", help="stereo-like depth noise, dropout, 1%% frame drops")
    ap.add_argument("--drift", type=float, default=0.0, help="VIO random walk, m/sqrt(s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--plot")
    a = ap.parse_args(argv)
    world = World.load(a.world)
    m, px4, log, result, crashed = simulate(world, load_params(a.route, a.height, a.hover), a.noise, a.drift,
                                            a.seed, a.seconds)
    err = np.linalg.norm(px4.off[:2])
    print(f"result: {result}")
    print(f"points passed {m.follower.points_passed if m.follower else 0}/{len(m.targets)}; end true n={px4.pos[0]:.2f} e={px4.pos[1]:.2f}; "
          f"estimate error {err:.2f} m; sim time {log['t'][-1]:.1f} s")
    print("counts:", {k: v for k, v in m.counts.items() if v})
    if a.plot:
        plot(a.plot, world, m, px4, log, result)
        print(f"plot: {a.plot}")
    return 2 if crashed else 0 if result == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
