"""Arnav's MPPI (nav.local_nav) as refnav's FOLLOW stage (nav:=mppi). PURE: no ROS.
refnav keeps take-off, hover, landing and the twists; MPPI only replaces path following.

MPPI runs in ITS OWN PROCESS (mppi_node), never inside refnav's control thread:
  refnav    -> /blueraven/mppi/start  {"id": follow-start time, "pad": [x, y, z, yaw]}  (its take-off pose, latched)
  mppi_node -> /blueraven/mppi/cmd    cmd_msg(): odom ENU velocity + heading, MPPI's state, tagged with that id
  refnav    CmdLink: a fresh command for ITS id -> Mission.ext (pad NED); a gap holds still; silence lands."""
from __future__ import annotations

import math
import sys
from collections import deque
from typing import Any

import numpy as np

from blueraven_skybeta.mission_core import _repo_root

if str(_repo_root()) not in sys.path:
    sys.path.insert(0, str(_repo_root()))
from nav.depth_points import depth_to_odom  # noqa: E402
from nav.local_grid import LocalGridParams  # noqa: E402
from nav.local_nav import LocalNav, LocalNavParams, NavCommand  # noqa: E402
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


CMD_STALE_S = 0.35          # MPPI steps at 10 Hz: older than this (3 missed steps) and refnav holds still
CMD_LAND_S = 1.0            # no fresh command for this long in follow: land
CMD_FIRST_S = 3.0           # follow started this long ago without one command for it: land
# VIO jump guard. In nav:=mppi refnav's tracking guard can't fire (its setpoint is the estimate itself), so MPPI
# watches the odometry: a step of more than JUMP_M + JUMP_MPS x the gap between samples is not flight (cuVSLAM
# jumped 2.3 m in a T4 hover facing a wall). MPPI then latches LOST (zero velocity) and refnav lands.
JUMP_M, JUMP_MPS, JUMP_GAP_S = 0.15, 0.6, 1.0
# Further VIO failure modes MPPI must not fly through (safety, not observed in flight so far): an estimate that
# FREEZES (MPPI would believe it stands still and keep flying blind), a heading jump, and messages that keep coming
# without new data. A live estimator never repeats itself to the micron (hover jitter 1.6 mm rms), so:
FREEZE_S, FREEZE_EPS_M = 0.5, 1e-6          # position AND heading unchanged for 0.5 s of advancing stamps: LOST
YAW_JUMP_DEG, YAW_JUMP_DPS = 15.0, 60.0     # heading step > 15 deg + 60 deg/s x gap (flight turns <= 30 deg/s): LOST
REPEAT_MAX = 30                              # this many odometry messages in a row without a newer stamp (~1 s): LOST
                                             # (counted, not timed: no clock comparison between processes)
# MPPI's caps in flight (the library's 0.6 m/s is far too fast for 0.10 m/s flying):
# Sim sweep (route 1, real push, 0.10 m/s, both response models): ground 0.15 touched the west net 6/12 (too little
# authority to claw back an unlearned push); 0.30/0.45 was 12/12 clean, min clearance 0.337 m (0.6/0.6: 0.315).
V_MAX_MIN, V_MAX_PER_CRUISE = 0.30, 3.0     # GROUND speed it may plan: max(0.30, 3 x cruise) = 0.30 at 0.10
U_MAX_MIN, U_MAX_PER_CRUISE = 0.45, 4.5     # the COMMAND, incl. leaning against the ~0.115 m/s push: 0.45


def yaw_enu(q_xyzw: Any) -> float:
    x, y, z, w = q_xyzw
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def start_msg(start_id: float, pad: Any) -> dict[str, Any]:
    """refnav -> mppi_node at follow start: the take-off pose the route is placed from (refnav's own pad)."""
    return {"id": float(start_id), "pad": [float(c) for c in (*pad.p0, pad.yaw0)]}


def cmd_msg(c: Any, start_id: float | None, now: float) -> dict[str, Any]:
    """mppi_node -> refnav: one MPPI step (a NavCommand) in the odom ENU frame. start_id None: not flying
    (the pre-flight loop), refnav never uses it."""
    r = c.mppi
    rnd = lambda x, n=3: None if x is None or not math.isfinite(x) else round(float(x), n)  # noqa: E731
    dist = getattr(r, "dist", None) if r is not None else None
    return {"start": start_id, "t": float(now), "vx": float(c.vx), "vy": float(c.vy), "yaw": float(c.yaw),
            "state": c.state, "depth_age": rnd(c.depth_age),
            "ms": None if r is None else rnd(r.ms, 1), "e_lat": None if r is None else rnd(r.e_lat),
            "crash_frac": None if r is None else rnd(r.crash_frac), "min_d_plan": None if r is None else rnd(r.min_d_plan),
            "dist": None if dist is None else [rnd(d) for d in dist], "why": getattr(c, "why", None)}


def to_ext(m: dict[str, Any], pad: Any) -> dict[str, Any]:
    """A command message -> Mission.ext: velocity and heading in refnav's pad NED frame."""
    vn, ve = pad.vector(m["vx"], m["vy"])
    ext = {k: v for k, v in m.items() if k not in ("vx", "vy", "start", "t")}
    ext.update(vn=vn, ve=ve, yaw=math.remainder(pad.yaw0 - m["yaw"], math.tau))
    return ext


class CmdLink:
    """refnav's end of the MPPI link. Only commands computed for THIS follow (its start id) and younger than
    CMD_STALE_S are flown; otherwise it holds still. why_land() says when the silence has gone on too long."""

    def __init__(self) -> None:
        self.start_id: float | None = None
        self.t_start = 0.0
        self.last: dict[str, Any] | None = None     # the newest command for this start
        self.n_cmds = 0
        self.n_ignored = 0

    def begin(self, now: float) -> float:
        self.start_id, self.t_start, self.last, self.n_cmds = float(now), now, None, 0
        return self.start_id

    def on_cmd(self, m: dict[str, Any]) -> bool:
        if self.start_id is None or m.get("start") != self.start_id:
            self.n_ignored += 1
            return False
        if self.last is not None and m["t"] <= self.last["t"]:
            return False                             # out of order: keep the newer one
        self.last, self.n_cmds = m, self.n_cmds + 1
        return True

    def age(self, now: float) -> float:
        return math.inf if self.last is None else now - self.last["t"]

    def ext(self, now: float, pad: Any) -> dict[str, Any]:
        if self.last is not None and self.age(now) <= CMD_STALE_S:
            return to_ext(self.last, pad)
        # hold still; keep MPPI's last depth age (aged) so refnav's own depth-silence rule still means depth
        da = None if self.last is None or self.last.get("depth_age") is None else self.last["depth_age"] + self.age(now)
        return {"vn": 0.0, "ve": 0.0, "yaw": None, "state": "NOCMD", "depth_age": da}

    def why_land(self, now: float) -> str | None:
        if self.start_id is None:
            return None
        if self.last is not None and self.last.get("state") == "LOST":
            return f"MPPI: VIO lost: {self.last.get('why') or 'see the mppi_node log'}"
        if self.last is None:
            return (f"MPPI sent no command in {CMD_FIRST_S:g} s of follow (mppi_node down?)"
                    if now - self.t_start > CMD_FIRST_S else None)
        a = self.age(now)
        return f"MPPI command silent for {a:.2f} s (CMD_LAND_S {CMD_LAND_S:g})" if a > CMD_LAND_S else None


GRID_Z_RESET_M = 0.10       # odometry height this far from where MPPI's grid was built: clear it (refnav's ext climb)
DEPTH_KEEP = 8              # depth frames mppi_node keeps (~0.27 s at 30 Hz), newest last


class DepthQueue:
    """mppi_node's depth frames. The callback only pushes (O(1)); the tick pops the NEWEST frame that odometry already
    covers (stamp <= the newest odometry stamp). The pose buffer interpolates and never extrapolates, so a frame newer
    than the newest pose can't be placed yet: it stays queued for the next tick instead of being dropped. Frames older
    than the one returned are discarded.
    Why: skybeta-10 route_mppi_full_160248: on-camera depth arrived AHEAD of odometry (each frame ~20 us after its
    capture's odometry stamp, odometry ~35 ms late). Keeping only the newest frame meant it was always a few ms newer
    than the newest pose: 'no pose' 386 -> 613, 1-2 frames placed per 5 s, SLOW/HOLD flapping (the crawl), then
    'depth silent for more than 3 s' -> land."""

    def __init__(self, keep: int = DEPTH_KEEP) -> None:
        self.q: deque = deque(maxlen=keep)

    def push(self, stamp: float, item: Any) -> None:
        self.q.append((float(stamp), item))

    def pop(self, odom_stamp: float | None) -> tuple[float, Any] | None:
        """-> (stamp, item) of the newest frame with stamp <= odom_stamp, else None (all frames wait)."""
        if odom_stamp is None:
            return None
        for i in range(len(self.q) - 1, -1, -1):
            if self.q[i][0] <= odom_stamp:
                got = self.q[i]
                for _ in range(i + 1):
                    self.q.popleft()
                return got
        return None


class MppiFollow:
    def __init__(self, v_cruise: float, tau_s: float, dead_time_s: float = 0.0) -> None:
        self.nav_p = LocalNavParams(v_cruise=v_cruise)
        v_max = max(V_MAX_MIN, V_MAX_PER_CRUISE * v_cruise)
        self.mppi_p = MppiParams(tau=tau_s, dead_time_s=dead_time_s, v_max=v_max, v_lat=min(MppiParams.v_lat, v_max),
                                 u_max=max(U_MAX_MIN, U_MAX_PER_CRUISE * v_cruise))
        self.lost: str | None = None                # why the odometry is no longer trusted (latched until start())
        self._odom_prev: tuple[float, np.ndarray, float] | None = None    # (stamp, position, ENU yaw)
        self._z_grid: float | None = None          # odometry height MPPI's grid was built at
        self._frozen_since: float | None = None
        self._repeats = 0
        self.poses = PoseBuffer()
        self.nav: LocalNav | None = None
        self.pos: np.ndarray | None = None          # latest odom ENU position, its stamp, and velocity
        self.stamp = 0.0
        self.vel = np.zeros(2)
        self.yaw: float | None = None               # measured ENU heading at the odometry stamp (MPPI's push estimate)
        self.ticks = 0
        self.ext: dict[str, Any] | None = None
        self.cmd = None                             # the latest NavCommand

    def on_odom(self, stamp: float, p: Any, q_xyzw: Any, v_flu: Any) -> None:
        x, y, z, w = q_xyzw
        pa = np.asarray(p, float)
        yaw_new = yaw_enu(q_xyzw)
        if self._odom_prev is not None and self.nav is not None and self.lost is None and np.isfinite(pa).all():
            gap = stamp - self._odom_prev[0]
            self._repeats = 0 if gap > 0.0 else self._repeats + 1
            if self._repeats >= REPEAT_MAX:
                self.lost = f"odometry repeating: {self._repeats} messages without a newer stamp"
            if gap > 0.0:                           # a repeated or older stamp is no new information
                step = float(np.linalg.norm(pa - self._odom_prev[1]))
                dyaw = abs(math.degrees(math.remainder(yaw_new - self._odom_prev[2], math.tau)))
                if gap <= JUMP_GAP_S and step > JUMP_M + JUMP_MPS * gap:
                    self.lost = f"odometry jumped {step:.2f} m in {gap:.2f} s (limit {JUMP_M + JUMP_MPS * gap:.2f} m)"
                elif gap <= JUMP_GAP_S and dyaw > YAW_JUMP_DEG + YAW_JUMP_DPS * gap:
                    self.lost = f"odometry heading jumped {dyaw:.0f} deg in {gap:.2f} s"
                elif step < FREEZE_EPS_M and dyaw < 1e-4:
                    self._frozen_since = self._odom_prev[0] if self._frozen_since is None else self._frozen_since
                    if stamp - self._frozen_since >= FREEZE_S:
                        self.lost = f"odometry frozen: identical pose for {stamp - self._frozen_since:.2f} s"
                else:
                    self._frozen_since = None
        if np.isfinite(pa).all():
            self._odom_prev = (stamp, pa, yaw_new)
        self.poses.add_position(stamp, p)
        self.poses.add_attitude(stamp, (w, x, y, z))
        self.pos, self.stamp = np.asarray(p, float), stamp
        if self.nav is not None and math.isfinite(self.pos[2]):
            if self._z_grid is None:
                self._z_grid = float(self.pos[2])
            elif abs(float(self.pos[2]) - self._z_grid) > GRID_Z_RESET_M:   # a height step: the old band's cells lie
                self.nav.clear_grid()
                self._z_grid = float(self.pos[2])
        self.vel = (D.quat_to_mat(w, x, y, z) @ np.asarray(v_flu, float))[:2]
        self.yaw = yaw_enu(q_xyzw)

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

    def start(self, route: Any, pad: Any, fence: dict | None = None) -> None:
        """At follow start, from the take-off pose (base_link on the ground: obstacles start at least 0.12 m above it).
        fence: the route file's params.fence ({"fwd": [..], "left": [..]} from the take-off pose: the arena's nets)."""
        self.nav = LocalNav(route_to_odom(route, pad.p0[:2], pad.yaw0), pad.p0[:2], pad.yaw0, nav=self.nav_p,
                            grid=LocalGridParams(floor_z=pad.p0[2]), mppi=self.mppi_p, fence=fence)
        self.lost, self._frozen_since, self._repeats = None, None, 0   # a new follow: the odometry guards start afresh
        self._z_grid = None

    def tick(self, now: float, pad: Any) -> dict[str, Any] | None:
        """Every 20 Hz tick in follow; steps MPPI on every STEP_EVERY-th. -> the Mission.ext command."""
        self.ticks += 1
        if self.nav is None or self.pos is None or (self.ticks - 1) % STEP_EVERY:
            return self.ext
        if self.lost is not None:                   # odometry no longer trusted: no MPPI, zero velocity, refnav lands
            c = self.cmd = NavCommand(0.0, 0.0, self.nav.yaw_cmd, "LOST", None, math.inf)
            c.why = self.lost
        else:
            c = self.cmd = self.nav.step(self.pos[:2], self.vel, now, pose_stamp=self.stamp, yaw=self.yaw)
        if pad is None:                             # mppi_node: the command goes out as cmd_msg, refnav converts it
            return None
        vn, ve = pad.vector(c.vx, c.vy)
        r = c.mppi
        self.ext = {"vn": vn, "ve": ve, "yaw": math.remainder(pad.yaw0 - c.yaw, math.tau), "state": c.state,
                    "depth_age": None if math.isinf(c.depth_age) else round(c.depth_age, 3),
                    "ms": None if r is None else round(r.ms, 1), "e_lat": None if r is None else round(r.e_lat, 3),
                    "crash_frac": None if r is None else round(r.crash_frac, 3),
                    "min_d_plan": None if r is None else round(r.min_d_plan, 3)}
        return self.ext
