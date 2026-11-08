"""Arnav's MPPI (nav.local_nav) as refnav's FOLLOW stage (nav:=mppi): odometry and on-camera depth in, the
Mission.ext command (velocity and heading in the pad NED frame, and MPPI's state) out. PURE: no ROS.
refnav keeps take-off, hover, landing and the twists; this only replaces path following."""
from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np

from blueraven_skybeta.mission_core import _repo_root

if str(_repo_root()) not in sys.path:
    sys.path.insert(0, str(_repo_root()))
from nav.depth_points import depth_to_odom  # noqa: E402
from nav.local_grid import LocalGridParams  # noqa: E402
from nav.local_nav import LocalNav, LocalNavParams  # noqa: E402
from nav.mppi import MppiParams  # noqa: E402
from nav.mppi_ros import route_to_odom  # noqa: E402
from nav.pose_buffer import PoseBuffer  # noqa: E402
from refnav import depth as D  # noqa: E402

STEP_EVERY = 2              # nav.step at 10 Hz from the 20 Hz tick: MPPI's warm start shifts one 0.1 s step per call
DEPTH_STRIDE = 4            # 640x400 -> 160x100 samples
SPIKE_WIN, SPIKE_TOL_M = 5, 0.3   # a sample is kept only within 0.3 m of its 5x5 median (stereo speckle)


def despeckle(depth_m: np.ndarray, stride: int = DEPTH_STRIDE) -> np.ndarray:
    """NaN everywhere except the samples depth_to_odom reads at this stride that agree with their 5x5 median."""
    r = SPIKE_WIN // 2
    vs, us = np.arange(stride // 2, depth_m.shape[0], stride), np.arange(stride // 2, depth_m.shape[1], stride)
    zp = np.pad(np.nan_to_num(depth_m, nan=0.0), r)
    win = np.stack([zp[np.ix_(vs + r + dv, us + r + du)] for dv in range(-r, r + 1) for du in range(-r, r + 1)])
    c = depth_m[np.ix_(vs, us)]
    out = np.full(depth_m.shape, np.nan)
    with np.errstate(invalid="ignore"):
        out[np.ix_(vs, us)] = np.where(np.abs(c - np.median(win, axis=0)) <= SPIKE_TOL_M, c, np.nan)
    return out


class MppiFollow:
    def __init__(self, v_cruise: float, tau_s: float, dead_time_s: float = 0.0) -> None:
        self.nav_p = LocalNavParams(v_cruise=v_cruise)
        self.mppi_p = MppiParams(tau=tau_s, dead_time_s=dead_time_s)
        self.poses = PoseBuffer()
        self.nav: LocalNav | None = None
        self.pos: np.ndarray | None = None          # latest odom ENU position and velocity
        self.vel = np.zeros(2)
        self.ticks = 0
        self.ext: dict[str, Any] | None = None
        self.cmd = None                             # the latest NavCommand

    def on_odom(self, stamp: float, p: Any, q_xyzw: Any, v_flu: Any) -> None:
        x, y, z, w = q_xyzw
        self.poses.add_position(stamp, p)
        self.poses.add_attitude(stamp, (w, x, y, z))
        self.pos = np.asarray(p, float)
        self.vel = (D.quat_to_mat(w, x, y, z) @ np.asarray(v_flu, float))[:2]

    def on_depth(self, stamp: float, depth_m: np.ndarray, K: Any, T_base_cam: np.ndarray,
                 stride: int = DEPTH_STRIDE) -> bool:
        """One decoded depth frame (metres, NaN = none). -> False when it could not be placed (no pose at its stamp)."""
        if self.nav is None or self.pos is None:
            return False
        out = depth_to_odom(despeckle(depth_m, stride), K, T_base_cam, self.poses, stamp, stride=stride, scale=1.0)
        if out is None:
            return False
        self.nav.add_depth(out[0], out[1], float(self.pos[2]), stamp)
        return True

    def start(self, route: Any, pad: Any) -> None:
        """At follow start, from the take-off pose (base_link on the ground: obstacles start at least 0.12 m above it)."""
        self.nav = LocalNav(route_to_odom(route, pad.p0[:2], pad.yaw0), pad.p0[:2], pad.yaw0, nav=self.nav_p,
                            grid=LocalGridParams(floor_z=pad.p0[2]), mppi=self.mppi_p)

    def tick(self, now: float, pad: Any) -> dict[str, Any] | None:
        """Every 20 Hz tick in follow; steps MPPI on every STEP_EVERY-th. -> the Mission.ext command."""
        self.ticks += 1
        if self.nav is None or self.pos is None or (self.ticks - 1) % STEP_EVERY:
            return self.ext
        c = self.cmd = self.nav.step(self.pos[:2], self.vel, now)
        vn, ve = pad.vector(c.vx, c.vy)
        r = c.mppi
        self.ext = {"vn": vn, "ve": ve, "yaw": math.remainder(pad.yaw0 - c.yaw, math.tau), "state": c.state,
                    "depth_age": None if math.isinf(c.depth_age) else round(c.depth_age, 3),
                    "ms": None if r is None else round(r.ms, 1), "e_lat": None if r is None else round(r.e_lat, 3),
                    "crash_frac": None if r is None else round(r.crash_frac, 3),
                    "min_d_plan": None if r is None else round(r.min_d_plan, 3)}
        return self.ext
