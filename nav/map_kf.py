"""Kalman filter for the map <- odom drift correction (Task D (2)). Pure: numpy + nav.map_align.

Replaces DriftCorrector's tau low-pass with an error-state Kalman filter.

Nominal state: T = map <- odom (SE2). Error state e = (ex, ey, eyaw), referenced AT THE BODY: the true
correction is T rotated by eyaw about the body's map position, then shifted by (ex, ey). This is exactly
the frame align() reports its delta in, so a fit is a direct measurement of e. After each update e is
injected into T and reset to 0 (P is kept).

  predict   P <- A P A^T + Q dt. Q holds VIO's drift rates (random walk). A moves the error's reference
            point from the old body position to the new one: a heading error eyaw about the old point is the
            same correction as eyaw about the new point plus a shift eyaw x (new - old).
  measure   align()'s solution theta, ONLY in the fit's well-conditioned eigen-directions of its information
            H (k of 3): z = U^T theta, R = r_scale * rms^2 * diag(1 / lambda) + U^T R_floor U. A straight
            corridor (no information along it) gives no update along it, and the filter's P along it keeps
            growing until the walls do say something.
  gate      Mahalanobis d^2 = z^T S^-1 z against chi-square (99.9 %, k dof). After gate_reset_n consecutive
            gated fits the filter assumes it was wrong (P += P_reset) so it can re-acquire after a real jump.
  clamp     the injected step is still limited by max_rate_xy / max_rate_yaw and the total yaw by
            max_total_yaw, as before: a last guard, rarely active.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from nav.map_align import SE2, AlignParams, AlignResult, align, wrap

CHI2_999 = {1: 10.83, 2: 13.82, 3: 16.27}


@dataclass
class KfParams:
    # Tuned on 5 T4 truth bags, pose at the depth stamp (2026-11-07; table in the Task D report): round 1 on
    # 144246Z/opt_140016Z/follow_115005Z with 144958Z/speedprof_154800Z held out, round 2 (r_scale, q_yaw) on all 5.
    q_xy: float = 5e-4                       # m^2/s: about 5 cm of drift (1 sigma) after 5 s
    q_yaw: float = math.radians(0.4) ** 2    # rad^2/s: about 0.9 deg after 5 s ((0.2 deg)^2 lags real drift: speedprof 0.087 m)
    r_scale: float = 1000.0                  # fit points are correlated: inflate the fit's own covariance
    r_floor_xy: float = 0.05                 # m: a single fit is no better than this (0.01-0.03 chased fit noise)
    r_floor_yaw: float = math.radians(1.3)   # rad: the fits' heading accuracy (1.2-1.5 deg rms on 10 truth bags)
    p0_xy: float = 0.02                      # m, at launch (the drone sits on the mark)
    p0_yaw: float = math.radians(0.5)
    min_eig: float = 20.0                    # same threshold align() solves with
    gate_reset_n: int = 8                    # ~1.6 s of consecutive gated fits at 5 Hz
    p_reset_xy: float = 0.15
    p_reset_yaw: float = math.radians(4.0)
    max_rate_xy: float = 0.15                # m/s, last clamp
    max_rate_yaw: float = math.radians(3.0)
    max_total_yaw: float = math.radians(20.0)
    max_dt: float = 1.0
    align: AlignParams = field(default_factory=AlignParams)


def observable_measurement(r: AlignResult, p: KfParams):
    """(U 3xk, z k, R kxk) from a fit, or None when no direction is constrained."""
    if r.theta is None:
        return None
    H = np.asarray(r.info, float)
    th = np.asarray(r.theta, float)
    yaw_ok = bool(r.observable[2])               # False: unobservable, or align() dropped a yaw that didn't earn its keep
    if not yaw_ok:
        ev, V = np.linalg.eigh(H[:2, :2])
        V = np.vstack([V, np.zeros((1, 2))])
        th = th.copy(); th[2] = 0.0
    else:
        ev, V = np.linalg.eigh(H)
    keep = ev > p.min_eig
    if not keep.any():
        return None
    U, lam = V[:, keep], ev[keep]
    s2 = r.rms_after ** 2 if math.isfinite(r.rms_after) else 0.05 ** 2
    R = p.r_scale * s2 * np.diag(1.0 / lam) + U.T @ np.diag([p.r_floor_xy ** 2, p.r_floor_xy ** 2, p.r_floor_yaw ** 2]) @ U
    return U, U.T @ th, R


class KalmanCorrector:
    """Same interface as map_align.DriftCorrector: update(t, odom_pose, pts_level) -> SE2 (map <- odom)."""

    def __init__(self, course_map, start: SE2, params: KfParams | None = None) -> None:
        self.cm, self.p = course_map, params or KfParams()
        self.start, self.T = start, SE2(start.x, start.y, start.yaw)
        self.P = np.diag([self.p.p0_xy ** 2, self.p.p0_xy ** 2, self.p.p0_yaw ** 2])
        self.t_last: float | None = None
        self.ref: tuple[float, float] | None = None              # body map position the error is referenced at
        self.n = {"frames": 0, "used": 0, "rejected_fit": 0, "rejected_gate": 0, "unobservable": 0, "resets": 0, "clamped": 0}
        self.last: AlignResult | None = None
        self.last_d2: float | None = None
        self.gated_run = 0

    def sigma(self) -> tuple[float, float, float]:
        """1-sigma of the correction: x (m), y (m), yaw (deg)."""
        s = np.sqrt(np.maximum(np.diag(self.P), 0.0))
        return float(s[0]), float(s[1]), math.degrees(float(s[2]))

    def predict(self, t: float, body_xy: tuple[float, float]) -> float:
        dt = 0.0 if self.t_last is None else max(0.0, min(t - self.t_last, self.p.max_dt))
        self.t_last = t
        if self.ref is not None:
            dx, dy = body_xy[0] - self.ref[0], body_xy[1] - self.ref[1]
            A = np.array([[1.0, 0.0, -dy], [0.0, 1.0, dx], [0.0, 0.0, 1.0]])
            self.P = A @ self.P @ A.T
        self.ref = (body_xy[0], body_xy[1])
        self.P = self.P + np.diag([self.p.q_xy, self.p.q_xy, self.p.q_yaw]) * dt
        return dt

    def update(self, t: float, odom_pose: tuple[float, float, float, float], pts_level: np.ndarray) -> SE2:
        self.n["frames"] += 1
        prior = self.T.pose(odom_pose)
        dt = self.predict(t, (prior[0], prior[1]))
        r = align(self.cm, pts_level, prior, self.p.align)
        self.last, self.last_d2 = r, None
        if not r.ok:
            self.n["rejected_fit"] += 1
            return self.T
        meas = observable_measurement(r, self.p)
        if meas is None:
            self.n["unobservable"] += 1
            return self.T
        U, z, R = meas
        Hk = U.T
        S = Hk @ self.P @ Hk.T + R
        Si = np.linalg.inv(S)
        d2 = float(z @ Si @ z)
        self.last_d2 = d2
        if d2 > CHI2_999[len(z)]:
            self.n["rejected_gate"] += 1
            self.gated_run += 1
            if self.gated_run >= self.p.gate_reset_n:
                self.P = self.P + np.diag([self.p.p_reset_xy ** 2, self.p.p_reset_xy ** 2, self.p.p_reset_yaw ** 2])
                self.n["resets"] += 1
                self.gated_run = 0
            return self.T
        self.gated_run = 0
        K = self.P @ Hk.T @ Si
        e = K @ z
        IKH = np.eye(3) - K @ Hk
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T                    # Joseph form: stays symmetric, PSD
        self._inject(e, prior, dt)
        self.n["used"] += 1
        return self.T

    def _inject(self, e: np.ndarray, prior, dt: float) -> None:
        ex, ey, eyaw = (float(v) for v in e)
        lim = self.p.max_rate_xy * dt
        n_xy = math.hypot(ex, ey)
        clamped = False
        if dt > 0 and n_xy > lim:
            ex, ey, clamped = ex * lim / n_xy, ey * lim / n_xy, True
        ylim = self.p.max_rate_yaw * dt
        if dt > 0 and abs(eyaw) > ylim:
            eyaw, clamped = math.copysign(ylim, eyaw), True
        tot = wrap(self.T.yaw + eyaw - self.start.yaw)
        if abs(tot) > self.p.max_total_yaw:
            eyaw -= tot - math.copysign(self.p.max_total_yaw, tot); clamped = True
        if clamped:
            self.n["clamped"] += 1
        bx, by = prior[0], prior[1]
        c, s = math.cos(eyaw), math.sin(eyaw)
        nx = c * (self.T.x - bx) - s * (self.T.y - by) + bx + ex
        ny = s * (self.T.x - bx) + c * (self.T.y - by) + by + ey
        self.T = SE2(nx, ny, wrap(self.T.yaw + eyaw))
