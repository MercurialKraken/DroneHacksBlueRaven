"""Re-anchoring for the EV bridge. PURE (no ROS).

When the VIO source jumps for good (a persistent jump or a cuVSLAM re-initialisation), the bridge does
NOT pass the jump to PX4 (no reset_counter bump, no step): it re-anchors the new track to the last pose
it published, with a fixed planar rigid offset (yaw about the world z axis, plus a translation, z
included), so the published track stays continuous. Global position is map_align's job; an EV jump is
never real. (T4: accepting a self-consistent garbage track made PX4 reset xy to (790, 11031) and yaw by
0.94 rad; 12 of 19 mid-course-stop landings.)

Poses are in the source's ROS world frame (as the bridge receives them): position (x, y, z),
orientation quaternion (x, y, z, w).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def yaw_of(q_xyzw) -> float:
    x, y, z, w = q_xyzw
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _qmul(a, b):                                        # xyzw
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


@dataclass
class Anchor:
    dyaw: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    n_rebased: int = 0

    def apply(self, pos, q_xyzw):
        c, s = math.cos(self.dyaw), math.sin(self.dyaw)
        x, y, z = pos
        out = (c * x - s * y + self.tx, s * x + c * y + self.ty, z + self.tz)
        qz = (0.0, 0.0, math.sin(self.dyaw / 2), math.cos(self.dyaw / 2))
        return out, _qmul(qz, q_xyzw)

    def rebase(self, last_out_pos, last_out_q, raw_pos, raw_q) -> None:
        """Choose the offset so that apply(raw) == the last published pose (position and yaw)."""
        self.dyaw = math.atan2(math.sin(yaw_of(last_out_q) - yaw_of(raw_q)), math.cos(yaw_of(last_out_q) - yaw_of(raw_q)))
        c, s = math.cos(self.dyaw), math.sin(self.dyaw)
        x, y, z = raw_pos
        self.tx = last_out_pos[0] - (c * x - s * y)
        self.ty = last_out_pos[1] - (s * x + c * y)
        self.tz = last_out_pos[2] - z
        self.n_rebased += 1
