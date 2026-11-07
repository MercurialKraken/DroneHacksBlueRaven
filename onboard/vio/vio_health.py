"""Reject impossible VIO kinematics without turning one bad pose into a lost flight.

Each sample is judged against the last GOOD sample (not the last sample), so a one-frame spike is one
event and the pose coming back is not a second one:

* an isolated bad sample (jump over jump_m, speed over max_speed_mps, non-finite, or a measurement time
  that did not advance) is withheld on its own; the next good sample goes through;
* a sample whose reported position variance exceeds max_var_m2 is bad (cuVSLAM's covariance jumped from
  ~4e-4 to 3 m^2 as it degraded near the floor; such poses must never reach PX4);
* bad samples that agree with each other (a new, consistent position, each with sane variance) for
  persist_s are a re-anchor event: the reference moves there, the sample is passed and `rebased` is set
  for that call, so the bridge re-anchors the new track to its last published pose (onboard/vio/anchor.py)
  instead of passing the jump or bumping PX4's reset counter;
* more than burst_n bad samples within burst_s that do NOT settle into a consistent new position mean
  the source is broken: unhealthy until quiet_s of good samples.

Why (T4 bag hinson-follow_20261107T124323Z): cuVSLAM produced ONE pose 0.53 m off and came back the
next frame. The old check compared against the last sample, flagged both steps, and two events in 10 s
withheld vision for up to 15 s; PX4 drops EV after EKF2_NOAID_TOUT (1 s) and blind-landed. Even a single
event used to force quiet_s (5 s) of settling.
"""
from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class VioHealth:
    def __init__(self, *, max_speed_mps: float = 5.0, jump_m: float = 1.0, persist_s: float = 0.3,
                 burst_n: int = 2, burst_s: float = 1.0, quiet_s: float = 5.0, max_var_m2: float = 0.05) -> None:
        self.max_var = float(max_var_m2)
        self.rebased: bool = False                                        # set for the one call that re-anchors
        self.max_speed = float(max_speed_mps)
        self.jump_m = float(jump_m)
        self.persist_s = float(persist_s)
        self.burst_n = int(burst_n)
        self.burst_s = float(burst_s)
        self.quiet_s = float(quiet_s)
        self._good: tuple[float, tuple[float, ...]] | None = None        # last good (t, pos)
        self._pending: list[tuple[float, tuple[float, ...]]] = []         # consecutive bad samples since the last good one
        self._events: deque[float] = deque()                              # times of bad samples (burst window)
        self._broken_since: float | None = None                           # unhealthy, waiting for quiet_s of good samples
        self._quiet_from: float | None = None
        self.reason: str = "no samples"
        self.n_bad = 0
        self.n_withheld = 0
        self.n_rebased = 0

    def _problem(self, ref: tuple[float, tuple[float, ...]], t: float, pos: tuple[float, ...]) -> str | None:
        t0, p0 = ref
        dt = t - t0
        d = math.dist(p0, pos)
        if dt <= 0:
            return "measurement time did not advance" + (f" (moved {d:.2f} m)" if d > 0 else "")
        if d > self.jump_m:
            return f"jump {d:.2f} m"
        if d / dt > self.max_speed:
            return f"speed {d / dt:.1f} m/s"
        return None

    def update(self, t: float, pos: Sequence[float], var: float | None = None) -> tuple[bool, str]:
        """-> (ok, reason): ok False = withhold this sample. t in seconds (measurement time); var = the
        largest reported position variance of the sample (m^2), None if unknown."""
        self.rebased = False
        finite = math.isfinite(t) and all(math.isfinite(float(c)) for c in pos)
        p = tuple(float(c) for c in pos) if finite else ()
        high_var = var is not None and not (math.isfinite(var) and var <= self.max_var)
        if not finite:
            bad = "non-finite sample"
        elif high_var:
            bad = f"position variance {var:.3g} m2 > {self.max_var:g}"
        elif self._good is None:
            bad = None
        else:
            bad = self._problem(self._good, t, p)
        tb = t if math.isfinite(t) else (self._good[0] if self._good else 0.0)
        while self._events and tb - self._events[0] > self.burst_s:
            self._events.popleft()

        if bad is None:
            self._good, self._pending = (t, p), []
            if self._broken_since is not None:                 # recovering: need quiet_s of good samples
                self._quiet_from = self._quiet_from if self._quiet_from is not None else t
                if t - self._quiet_from < self.quiet_s:
                    self.reason = f"recovering ({t - self._quiet_from:.1f}/{self.quiet_s:.0f} s quiet)"
                    return False, self.reason
                self._broken_since = self._quiet_from = None
            self.reason = "ok"
            return True, self.reason

        self.n_bad += 1
        self._events.append(tb)
        self._quiet_from = None
        if finite and not high_var and not bad.startswith("measurement time"):
            # a real jump: consecutive bad samples that agree with each other for persist_s
            if self._pending and self._problem(self._pending[-1], t, p) is not None:
                self._pending = []                             # inconsistent with the previous bad one: start over
            self._pending.append((t, p))
            if t - self._pending[0][0] >= self.persist_s and len(self._pending) >= 2:
                self._good, self._pending = (t, p), []
                self._events.clear()
                self._broken_since = None
                self.n_rebased += 1
                self.rebased = True
                self.reason = f"re-anchored after a persistent {bad}"
                return True, self.reason
        if (len(self._events) > self.burst_n and not self._pending[1:]) or (high_var and len(self._events) > self.burst_n):
            self._broken_since = self._broken_since if self._broken_since is not None else tb
        if self._broken_since is not None:
            self.reason = f"broken: {len(self._events)} bad samples in {self.burst_s:.0f} s (last: {bad})"
        else:
            self.reason = f"outlier withheld: {bad}"
        self.n_withheld += 1
        return False, self.reason
