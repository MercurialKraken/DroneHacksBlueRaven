"""The guide's frame caveat (competitor_guide.md section 1), as numbers. PURE: no ROS.

EKF2's NED frame and the vision `odom` frame differ by a constant yaw (and a translation). The flight
interface can correct poses for it (`odom_to_ned_yaw_offset_rad`), using its own conventions
(skybeta_flight_interface/frames.py, 0.3.3):

    yaw_ned_target = (pi/2 - yaw_enu) + offset                      # yaw_ned_from_quat_xyzw, then + offset
    position_ned   = R_down(+offset) . enu_to_ned(position_enu)     # setpoint_position_enu_to_ned

so `offset = PX4 heading - (pi/2 - vision heading)`, measured with the drone still (probe below).

The interface does NOT rotate twist setpoints by the offset (flight_interface.py: velocity = enu_to_ned(v)).
twist_for_interface pre-rotates our ENU twist so it lands in NED exactly like a position would. With the
default offset of 0 it is the identity.
"""
from __future__ import annotations

import math
from typing import Iterable


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def yaw_enu_from_xyzw(x: float, y: float, z: float, w: float) -> float:
    """Yaw of a ROS (ENU/FLU) orientation, about +z, counter-clockwise from +x."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_ned_from_wxyz(w: float, x: float, y: float, z: float) -> float:
    """Heading of a PX4 (NED/FRD) quaternion as px4_msgs orders it [w, x, y, z]; clockwise from north."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def offset_sample(yaw_vision_enu: float, yaw_px4_ned: float) -> float:
    """One sample of odom_to_ned_yaw_offset_rad, in the interface's convention."""
    return wrap(yaw_px4_ned - (math.pi / 2 - yaw_vision_enu))


def circular_stats(samples: Iterable[float]) -> tuple[float, float, int]:
    """(mean, standard deviation, count) of angles in radians, on the circle."""
    s = list(samples)
    if not s:
        return math.nan, math.nan, 0
    c = sum(math.cos(a) for a in s) / len(s)
    si = sum(math.sin(a) for a in s) / len(s)
    r = min(1.0, math.hypot(c, si))
    return math.atan2(si, c), math.sqrt(max(0.0, -2.0 * math.log(r))) if r > 0 else math.inf, len(s)


def twist_for_interface(vx: float, vy: float, offset_rad: float) -> tuple[float, float]:
    """Our ENU twist (vx, vy), pre-rotated so the interface's plain enu_to_ned turns it into the same NED
    direction a position setpoint gets: R_down(+offset) in NED is R_up(-offset) in ENU."""
    if offset_rad == 0.0:
        return vx, vy
    c, s = math.cos(-offset_rad), math.sin(-offset_rad)
    return c * vx - s * vy, s * vx + c * vy
