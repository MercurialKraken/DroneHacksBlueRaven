"""Hold each stereo frame until IMU covers its capture stamp, then release it with exactly the IMU
samples since the previous frame. PURE (no ROS, no cuVSLAM).

Why (T4 replica, bag hinson-settle45_20261107T101735Z): image and IMU callbacks share one executor.
After a tracking stall (233-1115 ms seen) the queued images can be delivered before the queued IMU.
The old node then tracked each frame at once and discarded every later-arriving IMU sample stamped
before that frame as "imu_dropped", although the samples were intact in the stream. That left holes
in the inertial data; on a fast turn the tracker lost a frame and reinitialised, healthy went false
and PX4 landed blind.

Here a frame waits until an IMU sample stamped at or after it has arrived (the IMU then covers the
whole interval), or until `wait_s` of wall time has passed, whichever is first. On timeout the frame
is released with whatever IMU is there and the timeout is counted, so a missing IMU stream can never
freeze tracking. Frames are released strictly in stamp order. An IMU sample stamped at or before
the last released frame cannot be used any more (the tracker has moved past it) and is counted as
late, which is what the old code counted as dropped.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any

Sample = tuple[int, Any, Any]                   # (t_ns, accel, gyro)


@dataclass
class Released:
    t_ns: int
    left: Any
    right: Any
    imu: list[Sample]                           # samples with last_frame < t <= t_ns, in stamp order
    timed_out: bool


@dataclass
class ImuFrameSync:
    wait_s: float = 0.08                        # longest a frame waits for IMU coverage (wall time)
    max_frames: int = 30                        # frames held at once (2 s at 15 Hz); oldest dropped beyond
    max_imu: int = 20000                        # IMU samples held at once (about 40 s at 500 Hz)
    imu: list[Sample] = field(default_factory=list)
    frames: list[tuple[int, Any, Any, float]] = field(default_factory=list)    # (t_ns, left, right, arrived)
    last_frame_ns: int = -1
    n: dict[str, int] = field(default_factory=lambda: {"imu_late": 0, "frames_waited": 0, "frames_timeout": 0,
                                                       "frames_stale": 0, "frames_overflow": 0, "imu_overflow": 0})

    def add_imu(self, t_ns: int, accel: Any, gyro: Any) -> None:
        t_ns = int(t_ns)
        if t_ns <= self.last_frame_ns:
            self.n["imu_late"] += 1
            return
        bisect.insort(self.imu, (t_ns, accel, gyro), key=lambda s: s[0])
        if len(self.imu) > self.max_imu:
            drop = len(self.imu) - self.max_imu
            del self.imu[:drop]
            self.n["imu_overflow"] += drop

    def add_frame(self, t_ns: int, left: Any, right: Any, now_s: float) -> None:
        t_ns = int(t_ns)
        if t_ns <= self.last_frame_ns:
            self.n["frames_stale"] += 1
            return
        bisect.insort(self.frames, (t_ns, left, right, float(now_s)), key=lambda f: f[0])
        while len(self.frames) > self.max_frames:
            self.frames.pop(0)
            self.n["frames_overflow"] += 1

    def covered(self, t_ns: int) -> bool:
        return bool(self.imu) and self.imu[-1][0] >= t_ns

    def ready(self, now_s: float) -> list[Released]:
        """Frames that may be tracked now, oldest first."""
        out: list[Released] = []
        while self.frames:
            t, left, right, arrived = self.frames[0]
            if self.covered(t):
                timed_out = False
            elif now_s - arrived >= self.wait_s:
                timed_out = True
            else:
                break                           # wait; later frames wait behind it (stamp order)
            self.frames.pop(0)
            if now_s > arrived:
                self.n["frames_waited"] += 1
            if timed_out:
                self.n["frames_timeout"] += 1
            k = bisect.bisect_right(self.imu, t, key=lambda s: s[0])
            batch, self.imu = self.imu[:k], self.imu[k:]
            self.last_frame_ns = t
            out.append(Released(t, left, right, batch, timed_out))
        return out


def legacy_release(imu_q: list[Sample], t_ns: int, last_frame_ns: int, last_imu_ns: int
                   ) -> tuple[list[Sample], list[Sample], int, int]:
    """The node's previous logic, kept for A/B tests: track the frame at once with whatever IMU is
    queued. -> (registered, kept, dropped_count, new_last_imu_ns)."""
    keep, reg, dropped = [], [], 0
    for e in sorted(imu_q, key=lambda e: e[0]):
        if e[0] > t_ns:
            keep.append(e)
        elif e[0] <= last_imu_ns or e[0] < last_frame_ns:
            dropped += 1
        else:
            reg.append(e)
            last_imu_ns = e[0]
    return reg, keep, dropped, last_imu_ns
