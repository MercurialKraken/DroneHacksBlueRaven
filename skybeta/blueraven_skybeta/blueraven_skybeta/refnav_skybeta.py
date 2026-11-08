"""refnav.core.Mission through the organizers' flight interface, in a pad NED frame (map_route.MapFrame's).
Odometry and /skybeta/state in, one twist out per tick; its PX4 commands are dropped. PURE: no ROS."""
from __future__ import annotations

import math
import sys
from typing import Any

from blueraven_skybeta.frame_offset import twist_for_interface, yaw_enu_from_xyzw
from blueraven_skybeta.mission_core import _repo_root

if str(_repo_root()) not in sys.path:
    sys.path.insert(0, str(_repo_root()))
from refnav import core as C  # noqa: E402
from refnav import depth as D  # noqa: E402

KP_XY = 0.5                 # 1/s: under PX4's MPC_XY_P 0.95, as the twist reaches PX4's velocity loop after a dead time
KP_Z = 1.0                  # 1/s: PX4's MPC_Z_P 1.0, the loop a position setpoint gets
KP_YAW = 1.5                # 1/s on the heading error left after the yawspeed feed-forward
DESCEND_MPS = 0.35          # PX4's land detector needs a commanded descent >= 1.1 x LNDMC_Z_VEL_MAX (0.25)
GROUND_PRESS_MPS = 0.60     # on the ground GROUND_WAIT_S without PX4 disarming: press down firmer
GROUND_WAIT_S = 3.0
GROUND_BAND_M = 0.06        # "on the ground": within this of the pad's height
STALL_S, STALL_M = 1.0, 0.03  # or the descent stalled: sank < 3 cm in 1 s (the estimate's height can sit 0.1-0.2 m high)
ARRIVE_MPS, BRAKE_MAX_S = 0.10, 2.0  # brake to a stop before descending, at most 2 s
STATE_STALE_S = 1.0         # /skybeta/state is 5 Hz
ODOM_STALE_S = 1.0          # real skybeta-09 odometry arrives with gaps up to 0.39 s under load
HOLD_S = 0.25               # a shorter silence than STALE holds position (zero twist) instead of landing
CORR_MAX_MPS = 0.25         # position-loop share of the twist: past the 0.20 m off the path the reference waits at
XY_MAX_MPS = 0.5            # the 0.3 m/s routes plus their corner lead (<= 0.23 m/s); a jam never gets 1-1.5 m/s
VEL_LAG_S = 0.55            # twist -> velocity lag: 1 / MPC_XY_VEL_P_ACC 1.8 on the drone; the T4 sim fits 0.9 s + 0.6 s dead time
TRACK_LAND_M, TRACK_LAND_S = 0.40, 0.5  # this far from the setpoint for 0.5 s in follow (T4 cuVSLAM glitched 1.06 m for
                                        # 0.13 s): stop and land where it is
STUCK_S, STUCK_M, STUCK_ASKED_M = 3.0, 0.10, 0.45  # moved < 0.10 m in 3 s while commanded >= 0.45 m (a jam): land
LAND_HOLD_MPS = 0.15        # the descent's pull back over its landing point
_Q_X180 = (0.0, 1.0, 0.0, 0.0)  # FLU <-> FRD, and ENU-about-the-pad <-> NED-about-the-pad


def _qmul(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


class Pad:
    """North along the drone's nose on the pad, origin there, z down: the odometry pose it was anchored at
    until the climb began."""

    def __init__(self, x: float, y: float, z: float, yaw: float) -> None:
        self.p0, self.yaw0 = (x, y, z), yaw
        self.c, self.s = math.cos(yaw), math.sin(yaw)

    def vector(self, vx: float, vy: float) -> tuple[float, float]:
        """odom ENU (x, y) -> pad (north, east)."""
        return self.c * vx + self.s * vy, self.s * vx - self.c * vy

    def to_odom(self, vn: float, ve: float) -> tuple[float, float]:
        """pad (north, east) -> odom ENU (x, y): the same matrix, its own inverse."""
        return self.vector(vn, ve)


def moved(pad: Pad, before: Pad | None, m: float = 0.02, deg: float = 1.0) -> bool:
    """The pad has moved or turned by more than this since `before` (None: always)."""
    return (before is None or math.dist(pad.p0[:2], before.p0[:2]) > m
            or abs(math.degrees(math.remainder(pad.yaw0 - before.yaw0, math.tau))) > deg)


def on_odom(v: C.Vehicle, pad: Pad | None, phase: str, p: tuple[float, float, float],
            q_xyzw: tuple[float, float, float, float], v_flu: tuple[float, float, float]) -> Pad | None:
    """One odometry sample (ENU position, FLU -> ENU orientation, FLU body velocity) into the mission's Vehicle.
    -> the pad frame: this pose until the climb (moved or turned before it, the route starts from there), else the one it had."""
    qx, qy, qz, qw = q_xyzw
    if not all(math.isfinite(c) for c in (*p, *q_xyzw, *v_flu)):
        v.pos_ok = False
        return pad
    if phase in ("wait", "arm") or pad is None:
        pad = Pad(*p, yaw_enu_from_xyzw(*q_xyzw))
    h = -pad.yaw0 / 2
    q_pad_enu = _qmul(_Q_X180, (math.cos(h), 0.0, 0.0, math.sin(h)))
    v.q = _qmul(_qmul(q_pad_enu, (qw, qx, qy, qz)), _Q_X180)
    v.pos = (*pad.vector(p[0] - pad.p0[0], p[1] - pad.p0[1]), pad.p0[2] - p[2])
    v.heading = D.quat_yaw(*v.q)
    ve = D.quat_to_mat(qw, qx, qy, qz) @ v_flu
    v.vel = pad.vector(float(ve[0]), float(ve[1]))
    v.pos_ok = True
    return pad


def on_state(v: C.Vehicle, state: dict[str, Any], phase: str) -> None:
    """armed and nav_state as the mission reads them. Before Offboard the pilot's arming is the handover the
    mission waits for, not someone else flying: it reads as disarmed until Offboard is active."""
    nav = state.get("nav_state")
    armed = state.get("armed")
    v.armed = armed and (nav == C.NAV_OFFBOARD or phase not in ("wait", "arm")) if armed is not None else None
    v.nav_state = C.NAV_AUTO_LAND if (nav == C.NAV_OFFBOARD and phase == "land") else nav


def healthy(state: dict[str, Any] | None, t_state: float, now: float) -> bool:
    """The mission's estimator health: PX4 linked and its state fresh (it is called with each odometry sample)."""
    return state is not None and now - t_state <= STATE_STALE_S and state.get("px4_link") is True


def _clamp(x: float, y: float, cap: float) -> tuple[float, float]:
    n = math.hypot(x, y)
    return (x * cap / n, y * cap / n) if n > cap else (x, y)


def _to_interface(pad: Pad, vn: float, ve: float, yaw_offset_rad: float) -> tuple[float, float]:
    """pad (north, east) velocity -> the ENU twist the interface turns into that NED direction."""
    return twist_for_interface(*pad.to_odom(vn, ve), yaw_offset_rad)


class Track:
    """Follow-phase tracking guard: too far from the setpoint, or jammed (commanded to move, not moving)."""

    def __init__(self) -> None:
        self.hist: list[tuple[float, tuple[float, float], float]] = []    # (t, pad (n, e), commanded m/s)
        self.t_far: float | None = None

    def check(self, now: float, setpoint: tuple[float, ...], pos: tuple[float, ...], cmd_mps: float) -> str | None:
        err = math.dist(setpoint[:2], pos[:2])
        self.t_far = (self.t_far if self.t_far is not None else now) if err > TRACK_LAND_M else None
        if self.t_far is not None and now - self.t_far >= TRACK_LAND_S:
            return f"tracking error {err:.2f} m from the setpoint for {TRACK_LAND_S:g} s (TRACK_LAND_M {TRACK_LAND_M:g})"
        self.hist = [h for h in self.hist if now - h[0] <= STUCK_S] + [(now, pos[:2], cmd_mps)]
        if now - self.hist[0][0] < STUCK_S * 0.9:
            return None
        asked = sum(c * (t1 - t0) for (t0, _, c), (t1, _, _) in zip(self.hist, self.hist[1:]))
        moved = math.dist(self.hist[0][1], self.hist[-1][1])
        if asked >= STUCK_ASKED_M and moved < STUCK_M:
            return f"stuck: moved {moved:.2f} m in {STUCK_S:g} s while commanded {asked:.2f} m"
        return None


class Descent:
    """The mission's "land" as Chelsea's hover lands: straight down over `target`, pressing firmer on the ground.
    It goes on until PX4 disarms and stops for good once Offboard ends (the pilot or a failsafe has it)."""

    def __init__(self) -> None:
        self.target: tuple[float, float] | None = None      # pad (n, e) held during the descent
        self.t_ground: float | None = None
        self.t_start: float | None = None
        self.zs: list[tuple[float, float]] = []
        self.braked = self.over = False

    def twist(self, now: float, v: C.Vehicle, state: dict[str, Any] | None
              ) -> tuple[float, float, float] | None:
        """-> (pad north, pad east, ENU up) velocity, or None once Offboard has ended."""
        if self.over or not state or not (state.get("offboard_active") and state.get("armed")):
            self.over = True
            return None
        self.t_start = now if self.t_start is None else self.t_start
        vn = ve = 0.0
        if self.target is not None and v.pos is not None and self.t_ground is None:    # on the ground: no xy push
            vn, ve = _clamp(KP_XY * (self.target[0] - v.pos[0]), KP_XY * (self.target[1] - v.pos[1]), LAND_HOLD_MPS)
        if not self.braked and math.hypot(*v.vel) > ARRIVE_MPS and now - self.t_start < BRAKE_MAX_S:
            return vn, ve, 0.0
        self.braked = True
        if v.pos is not None:
            self.zs = [(t, z) for t, z in self.zs if now - t <= STALL_S] + [(now, v.pos[2])]
        stalled = bool(self.zs) and now - self.zs[0][0] >= STALL_S * 0.9 and self.zs[-1][1] - self.zs[0][1] < STALL_M
        on_ground = v.pos is not None and v.pos[2] >= -GROUND_BAND_M or stalled
        self.t_ground = (self.t_ground if self.t_ground is not None else now) if on_ground else None
        press = self.t_ground is not None and now - self.t_ground >= GROUND_WAIT_S
        if self.t_ground is not None:
            vn = ve = 0.0
        return vn, ve, -(GROUND_PRESS_MPS if press else DESCEND_MPS)


def twist(out: C.Output, v: C.Vehicle, pad: Pad, yaw_rate_dps: float, yaw_offset_rad: float,
          lag_s: float = VEL_LAG_S, speed_mps: float = 0.0) -> tuple[float, float, float, float] | None:
    """The twist (ENU vx, vy, vz, yaw rate) for a streaming mission output: feed-forward velocity, plus lag_s of the
    path's acceleration (PX4's velocity loop lags that long; the interface has no acceleration input), plus PX4's
    position loop, bounded."""
    if not out.stream or out.setpoint is None or v.pos is None or v.heading is None:
        return None
    n, e, d, yaw = out.setpoint
    fn, fe = out.velocity or (0.0, 0.0)
    an, ae = out.accel or (0.0, 0.0)
    cn, ce = _clamp(KP_XY * (n - v.pos[0]), KP_XY * (e - v.pos[1]), CORR_MAX_MPS)
    vn, ve = _clamp(fn + lag_s * an + cn, fe + lag_s * ae + ce, max(speed_mps, XY_MAX_MPS))
    vx, vy = _to_interface(pad, vn, ve, yaw_offset_rad)
    cap = math.radians(yaw_rate_dps)
    r = (out.yawspeed or 0.0) + KP_YAW * math.remainder(yaw - v.heading, math.tau)
    return vx, vy, -KP_Z * (d - v.pos[2]), -max(-cap, min(cap, r))


def command(out: C.Output, v: C.Vehicle, pad: Pad | None, descent: Descent, track: Track,
            state: dict[str, Any] | None, t_odom: float, t_state: float, now: float, yaw_rate_dps: float,
            yaw_offset_rad: float, lag_s: float = VEL_LAG_S, speed_mps: float = 0.0
            ) -> tuple[tuple[float, float, float, float] | None, str | None]:
    """-> (the twist to publish this tick or None, why to land now). In the air a silence over HOLD_S holds still,
    over the STALE limits it lands; on the ground nothing is sent, so the interface never enters Offboard on it."""
    stale = (f"odometry silent for {now - t_odom:.2f} s" if now - t_odom > ODOM_STALE_S else
             f"/skybeta/state silent for {now - t_state:.2f} s" if now - t_state > STATE_STALE_S else None)
    hold = now - t_odom > HOLD_S or now - t_state > HOLD_S
    if out.phase == "land":
        if descent.target is None and v.pos is not None:
            sp = out.setpoint
            near = sp is not None and math.dist(sp[:2], v.pos[:2]) <= TRACK_LAND_M
            descent.target = sp[:2] if near else v.pos[:2]          # a far setpoint (a jam, a failsafe): land here
        tw = descent.twist(now, v, state)
        if tw is None:
            return None, None
        vn, ve = (0.0, 0.0) if pad is None or now - t_odom > HOLD_S else tw[:2]
        return (*(_to_interface(pad, vn, ve, yaw_offset_rad) if pad else (0.0, 0.0)), tw[2], 0.0), None
    if (stale or hold) and out.phase in C.AIRBORNE:
        return (0.0, 0.0, 0.0, 0.0), stale
    if hold or pad is None:
        return None, None
    tw = twist(out, v, pad, yaw_rate_dps, yaw_offset_rad, lag_s, speed_mps)
    if out.phase != "follow" or tw is None:
        track.hist, track.t_far = [], None
        return tw, None
    why = track.check(now, out.setpoint, v.pos, math.hypot(tw[0], tw[1]))
    if why:
        descent.target = v.pos[:2]
        return (0.0, 0.0, 0.0, 0.0), why
    return tw, None
