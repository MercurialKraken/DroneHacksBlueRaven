"""Pose at a past instant: short buffers of position and attitude, interpolated at a depth frame's stamp.
Pure (numpy). Task D (1).

Why: a depth frame is published 70-96 ms after its capture stamp in the T4 sim (window-blend 144246Z:
log time - header stamp), so pairing it with the LATEST pose places its points where the drone was ~0.1 s
later: ~10 cm at 1 m/s, ~3 deg in a 30 deg/s turn.

Clocks. Everything here is in seconds of ONE clock, the ROS clock (sim time in the sims). Depth header
stamps are already ROS time. PX4's px4_msgs carry `timestamp_sample` in PX4 microseconds (time since PX4
boot), so ClockMap maps them into ROS time with offset = min over a window of (ROS receipt time - PX4
time): the minimum is the transport latency floor, so the mapping is late by at most that floor (~1 ms on
uXRCE-DDS). In SITL the offset is ~0 (measured in the T4 bags: 1 us). On the drone with skybeta,
/visual_slam/tracking/odometry is stamped in ROS time and needs no mapping.
"""
from __future__ import annotations

import bisect
import math
from collections import deque

import numpy as np


class ClockMap:
    """PX4 microseconds -> ROS seconds, from (receipt time, PX4 time) pairs."""

    def __init__(self, window: int = 200) -> None:
        self.diffs: deque[float] = deque(maxlen=window)
        self.offset: float | None = None

    def observe(self, ros_recv_s: float, px4_us: int) -> None:
        self.diffs.append(ros_recv_s - px4_us * 1e-6)
        self.offset = min(self.diffs)

    def to_ros(self, px4_us: int) -> float | None:
        return None if self.offset is None else px4_us * 1e-6 + self.offset


_NED_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
_FLU_FRD = np.diag([1.0, -1.0, -1.0])


def mat_to_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0); q = (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]); q = ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]); q = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]); q = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)
    n = math.sqrt(sum(v * v for v in q))
    return tuple(v / n for v in q)


def quat_wxyz_to_mat(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def enu_flu_to_ned_frd_wxyz(q_xyzw) -> tuple[float, float, float, float]:
    """An ENU <- FLU attitude (ROS odometry, x y z w) as PX4's NED <- FRD quaternion (w, x, y, z), the form
    refnav.depth.level_rotation takes. On the drone, skybeta's odom is ENU-like (x forward at start, y left,
    z up), which is all levelling needs: only roll and pitch about gravity matter there."""
    x, y, z, w = (float(v) for v in q_xyzw)
    return mat_to_wxyz(_NED_ENU @ quat_wxyz_to_mat(w, x, y, z) @ _FLU_FRD)


def slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    """Spherical interpolation between unit quaternions (any component order, as long as both agree)."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0.0:                                   # the short way round
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
    else:
        th = math.acos(d)
        q = (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)


class _Buffer:
    def __init__(self, horizon_s: float) -> None:
        self.horizon = horizon_s
        self.t: list[float] = []
        self.v: list[np.ndarray] = []

    def add(self, t: float, v) -> None:
        v = np.asarray(v, float)
        if self.t and t <= self.t[-1]:
            if t == self.t[-1]:
                self.v[-1] = v                    # same instant: keep the newest
                return
            if t < self.t[-1] - self.horizon:     # far in the past (clock reset?): start over
                self.t.clear(); self.v.clear()
            else:                                 # slightly out of order: insert in place
                i = bisect.bisect_left(self.t, t)
                self.t.insert(i, t); self.v.insert(i, v)
                return
        self.t.append(t); self.v.append(v)
        cut = bisect.bisect_left(self.t, self.t[-1] - self.horizon)
        if cut > 0:
            del self.t[:cut], self.v[:cut]

    def bracket(self, t: float, max_gap_s: float):
        """(v0, v1, a) around t, or None if t is outside the buffer or the samples around it are too far apart."""
        if not self.t or t < self.t[0] or t > self.t[-1]:
            return None
        i = bisect.bisect_left(self.t, t)
        if self.t[i] == t:
            return self.v[i], self.v[i], 0.0
        t0, t1 = self.t[i - 1], self.t[i]
        if t1 - t0 > max_gap_s:
            return None
        return self.v[i - 1], self.v[i], (t - t0) / (t1 - t0)

    def __len__(self) -> int:
        return len(self.t)


class PoseBuffer:
    """Position (linear) and attitude (slerp) over the last horizon_s, queried at an instant inside it.

    at(t) returns (position, quaternion) or None ("refuse"): t older than the buffer, newer than the newest
    sample (no extrapolation), or between two samples more than max_gap_s apart."""

    def __init__(self, horizon_s: float = 1.0, max_gap_s: float = 0.1) -> None:
        self.pos = _Buffer(horizon_s)
        self.att = _Buffer(horizon_s)
        self.max_gap_s = max_gap_s

    def add_position(self, t: float, p) -> None:
        self.pos.add(t, p)

    def add_attitude(self, t: float, q) -> None:
        self.att.add(t, q)

    def position_at(self, t: float) -> np.ndarray | None:
        b = self.pos.bracket(t, self.max_gap_s)
        return None if b is None else b[0] + b[2] * (b[1] - b[0])

    def attitude_at(self, t: float) -> np.ndarray | None:
        b = self.att.bracket(t, self.max_gap_s)
        return None if b is None else slerp(b[0], b[1], b[2])

    def at(self, t: float):
        p, q = self.position_at(t), self.attitude_at(t)
        return None if p is None or q is None else (p, q)

    def span(self) -> tuple[float, float] | None:
        if not len(self.pos) or not len(self.att):
            return None
        return max(self.pos.t[0], self.att.t[0]), min(self.pos.t[-1], self.att.t[-1])
