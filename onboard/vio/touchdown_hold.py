"""Touchdown hold for the EV bridge. PURE (no ROS).

Close to the floor the stereo camera stops tracking well (T4 bag hinson-follow-vio_20261107T125629Z:
cuVSLAM's position slid about 1.5 m in the last ~0.6 m of descent while PX4's own estimate was right;
PX4 then reset its xy to that bad vision). On FlyX a frozen VIO plus a 0.0 m rangefinder on the pad let
the EKF dead-reckon away, so PX4 never confirmed the landing.

Only while `landing` is true (the bridge sets it from PX4's nav_state == AUTO_LAND) and the downward
rangefinder reads below `engage_m`, the bridge publishes
the LAST pose seen while the range was still at or above `ref_m`, instead of each new vision pose, with
its x/y variance raised by `hold_var` and its z variance by `hold_z_var` (the held height is the height at
ref_m, so it must not be used; the starter EKF2_EV_CTRL=9 does not fuse EV height anyway, which comes from
the rangefinder). The drone is nearly stationary in xy on the way down, so the held pose stays close to
the truth and keeps the EKF anchored in x, y and yaw, also when the camera freezes over the pad (FlyX
run_005: EKF diverged after landing while vision was frozen, and PX4 never confirmed the landing). The hold ends when the range has
been above `release_m` for `release_s` (a take-off, e.g. the return leg).

The landing gate matters: cruise is at ~0.8 m and the range dips below 0.6-0.7 m over obstacles, so a
range threshold alone engaged mid-flight in replays (published error up to 2.1 m). Replays of the T4
landings (tools/replay_touchdown_hold.py), gated on landing: engage 0.7 / ref 0.8 cut the published error
at touchdown on hinson-follow-vio_125629Z from 1.24 m (max 1.54) to 0.17 m (max 0.34), and changed
nothing on two clean landings (follow_115005Z, follow_114334Z). engage 0.6 / ref 0.7 gave 0.35 m.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TouchdownHold:
    engage_m: float = 0.7
    ref_m: float = 0.8
    release_m: float = 0.9                  # above cruise height (~0.8-0.89 m) so it cannot flap in flight
    release_s: float = 0.3
    hold_var: float = 0.01                  # m^2 added to the x/y variance while holding
    hold_z_var: float = 100.0               # m^2 added to z: the held height is from ref_m, not the floor
    active: bool = False
    landing: bool = False                    # set by the caller: PX4 is in AUTO_LAND
    _ref: Any = None                         # (t, sample) last seen with range >= ref_m
    _held: Any = None
    _above_since: float | None = None
    _range: float | None = None
    n: dict[str, int] = field(default_factory=lambda: {"engaged": 0, "released": 0, "held_samples": 0})

    def on_range(self, t: float, distance: float) -> None:
        """Downward range in metres (0 or below the sensor minimum counts as on the floor)."""
        d = float(distance)
        self._range = d
        if not self.active:
            if self.landing and d < self.engage_m and self._ref is not None:
                self.active, self._held, self._above_since = True, self._ref[1], None
                self.n["engaged"] += 1
        else:
            if d > self.release_m:
                self._above_since = t if self._above_since is None else self._above_since
                if t - self._above_since >= self.release_s:
                    self.active, self._held, self._above_since = False, None, None
                    self.n["released"] += 1
            else:
                self._above_since = None

    def filter(self, t: float, sample: Any) -> tuple[Any, bool]:
        """-> (sample to publish, held?). `sample` is whatever the bridge publishes (pose etc.)."""
        if not self.active and self._range is not None and self._range >= self.ref_m:
            self._ref = (t, sample)
        if self.active:
            self.n["held_samples"] += 1
            return self._held, True
        return sample, False
