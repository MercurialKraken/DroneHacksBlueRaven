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

KP_XY = KP_Z = 1.0          # 1/s: PX4's MPC_XY_P 0.95 and MPC_Z_P 1.0, the loop a position setpoint gets
KP_YAW = 1.5                # 1/s on the heading error left after the yawspeed feed-forward
DESCEND_MPS = 0.35          # PX4's land detector needs a commanded descent >= 1.1 x LNDMC_Z_VEL_MAX (0.25)
GROUND_PRESS_MPS = 0.60     # on the ground GROUND_WAIT_S without PX4 disarming: press down firmer
GROUND_WAIT_S = 3.0
GROUND_BAND_M = 0.06        # "on the ground": within this of the pad's height
STATE_STALE_S = 1.0         # /skybeta/state is 5 Hz
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


class Descent:
    """The mission's "land" as Chelsea's hover lands: straight down, pressing firmer on the ground, never a hold.
    It goes on until PX4 disarms and stops for good once Offboard ends (the pilot or a failsafe has it)."""

    def __init__(self) -> None:
        self.t_ground: float | None = None
        self.over = False

    def twist(self, now: float, v: C.Vehicle, state: dict[str, Any] | None
              ) -> tuple[float, float, float, float] | None:
        if self.over or not state or not (state.get("offboard_active") and state.get("armed")):
            self.over = True
            return None
        on_ground = v.pos is not None and v.pos[2] >= -GROUND_BAND_M
        self.t_ground = (self.t_ground if self.t_ground is not None else now) if on_ground else None
        press = self.t_ground is not None and now - self.t_ground >= GROUND_WAIT_S
        return 0.0, 0.0, -(GROUND_PRESS_MPS if press else DESCEND_MPS), 0.0


def twist(out: C.Output, v: C.Vehicle, pad: Pad, yaw_rate_dps: float, yaw_offset_rad: float
          ) -> tuple[float, float, float, float] | None:
    """The twist (ENU vx, vy, vz, yaw rate) for a streaming mission output: feed-forward plus PX4's position loop."""
    if not out.stream or out.setpoint is None or v.pos is None or v.heading is None:
        return None
    n, e, d, yaw = out.setpoint
    fn, fe = out.velocity or (0.0, 0.0)
    vx, vy = pad.to_odom(fn + KP_XY * (n - v.pos[0]), fe + KP_XY * (e - v.pos[1]))
    vx, vy = twist_for_interface(vx, vy, yaw_offset_rad)
    cap = math.radians(yaw_rate_dps)
    r = (out.yawspeed or 0.0) + KP_YAW * math.remainder(yaw - v.heading, math.tau)
    return vx, vy, -KP_Z * (d - v.pos[2]), -max(-cap, min(cap, r))
