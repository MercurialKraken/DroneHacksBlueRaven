"""Map-based drift correction (todos.md Task 2). Pure: numpy + nav.course_map.

Each depth frame's points are placed in the course map with the current pose estimate and aligned
to the map's 'localization' distance field (walls; see AlignParams.field) by a robust Gauss-Newton fit
over (dx, dy, dyaw). Only directions the walls constrain are corrected: in a straight corridor the
walls say nothing about position along it, so that component is left alone.

The result drives a slowly changing `map <- odom` correction (DriftCorrector). It is never fed into
PX4's vision input (the bridge's 1 m jump check would trip and the mission would land); consumers
apply it to their own targets, i.e. they fly the route in `odom` after mapping it through the
correction.

Frames: `map` = course ENU (x east, y north, z up). `odom` = ENU from the launch pose
(nav/frames.py). Pose = (x, y, z, yaw) of base_link, yaw from +x counter-clockwise. Depth points
arrive in the LEVEL body frame of refnav.depth.DepthFrame: (forward, right, down) from base_link.

Measured offline on T4 bag hinson-follow_20261107T110923Z against truth: see tools/eval_map_align.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def level_to_map(pts_level: np.ndarray, pose: tuple[float, float, float, float]) -> np.ndarray:
    """(N,3) level-frame points (forward, right, down) -> (N,3) map points for a base_link pose."""
    x, y, z, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    f, r, d = pts_level[:, 0], pts_level[:, 1], pts_level[:, 2]
    return np.stack([x + f * c + r * s, y + f * s - r * c, z - d], axis=1)


@dataclass
class AlignParams:
    # 'localization' (walls only, no nets). Depth DOES see the nets, so with this field net-heavy views (the
    # first corridor, the north side) are rejected and get no correction. 'planning' (walls + nets) accepts
    # them but was compared on 4 T4 truth bags (2026-11-07): position error the same (rms 0.039-0.074 vs
    # 0.041-0.073 m), heading worse (follow_110923Z max 9.5 vs 5.1 deg), likely because the map's nets sit a
    # few cm from the sim's. Revisit if the net geometry in the SDF is confirmed.
    field: str = "localization"
    band_m: float = 0.5           # points within this of flight height (walls, not floor or ceiling)
    max_points: int = 800         # subsample for speed
    inlier_m: float = 0.25        # a point further than this from any mapped wall is not used
    huber_m: float = 0.05
    iters: int = 8
    min_inliers: int = 150
    min_eig_xy: float = 40.0      # information (sum of weighted gradient^2) below which a direction is unobservable
    min_eig_yaw: float = 20.0
    max_rms_m: float = 0.08       # a fit worse than this is rejected
    # A yaw step is kept only if it lowers the residual by this fraction versus the same fit without yaw
    # (T4, 13 bags: before the maze 5->6 turn, unproven +2.0-2.5 deg steps about a far origin turned 2-4 cm
    # into 10-11 cm along-track and stopped 4 runs at the real wall).
    yaw_min_gain: float = 0.10


@dataclass
class AlignResult:
    ok: bool
    delta: np.ndarray             # (dx, dy, dyaw) to apply to the prior, in map frame (rotation about the body)
    info: np.ndarray              # 3x3 information matrix at the solution
    inliers: int
    rms_before: float
    rms_after: float
    observable: np.ndarray        # bool (3,): which of x, y, yaw were corrected
    why: str = ""


def align(course_map, pts_level: np.ndarray, prior: tuple[float, float, float, float],
          p: AlignParams | None = None, rng: np.random.Generator | None = None) -> AlignResult:
    p = p or AlignParams()
    pts = pts_level[np.abs(pts_level[:, 2]) < p.band_m]
    if len(pts) > p.max_points:
        rng = rng or np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), p.max_points, replace=False)]
    zero = np.zeros(3)
    if len(pts) < p.min_inliers:
        return AlignResult(False, zero, np.zeros((3, 3)), len(pts), math.nan, math.nan, np.zeros(3, bool), "too few points")
    x0, y0, z0, yaw0 = prior
    th = np.zeros(3)
    rms0 = None
    for _ in range(p.iters):
        pose = (x0 + th[0], y0 + th[1], z0, yaw0 + th[2])
        P = level_to_map(pts, pose)
        d, g = course_map.query(P, p.field)
        m = np.isfinite(d) & (np.abs(d) < p.inlier_m)
        if m.sum() < p.min_inliers:
            return AlignResult(False, zero, np.zeros((3, 3)), int(m.sum()), math.nan, math.nan, np.zeros(3, bool), "too few inliers")
        d, gx, gy = d[m], g[m, 0], g[m, 1]
        ex, ey = P[m, 0] - pose[0], P[m, 1] - pose[1]
        J = np.stack([gx, gy, gx * (-ey) + gy * ex], axis=1)          # d(distance)/d(dx, dy, dyaw)
        w = np.where(np.abs(d) <= p.huber_m, 1.0, p.huber_m / np.maximum(np.abs(d), 1e-9))
        H = (J * w[:, None]).T @ J
        b = (J * w[:, None]).T @ d
        if rms0 is None:
            rms0 = float(np.sqrt(np.mean(d ** 2)))
        # solve only in observable directions (eigen-decomposition; tiny eigenvalues are left at 0)
        ev, V = np.linalg.eigh(H)
        keep = ev > min(p.min_eig_xy, p.min_eig_yaw)
        step = -(V[:, keep] @ ((V[:, keep].T @ b) / ev[keep])) if keep.any() else np.zeros(3)
        th = th + step
        if np.linalg.norm(step[:2]) < 1e-4 and abs(step[2]) < 1e-5:
            break
    pose = (x0 + th[0], y0 + th[1], z0, yaw0 + th[2])
    d_after = course_map.distance(level_to_map(pts, pose), p.field)
    m = np.isfinite(d_after) & (np.abs(d_after) < p.inlier_m)
    rms1 = float(np.sqrt(np.mean(d_after[m] ** 2))) if m.any() else math.nan
    # observability per axis: projection of each unit axis onto the well-conditioned eigen-space
    ev, V = np.linalg.eigh(H)
    good = V[:, ev > min(p.min_eig_xy, p.min_eig_yaw)]
    obs = np.array([np.linalg.norm(good.T @ np.eye(3)[i]) > 0.9 for i in range(3)])
    obs[:2] &= np.diag(H)[:2] > p.min_eig_xy
    obs[2] &= H[2, 2] > p.min_eig_yaw
    delta = np.where(obs, th, 0.0)
    if obs[2] and abs(delta[2]) > 1e-6:
        # is the yaw part earning its keep? compare against the same translation with no rotation
        pose_ny = (x0 + delta[0], y0 + delta[1], z0, yaw0)
        d_ny = course_map.distance(level_to_map(pts, pose_ny), p.field)
        m_ny = np.isfinite(d_ny) & (np.abs(d_ny) < p.inlier_m)
        rms_ny = float(np.sqrt(np.mean(d_ny[m_ny] ** 2))) if m_ny.any() else math.inf
        if not (rms1 <= (1.0 - p.yaw_min_gain) * rms_ny):
            delta[2] = 0.0
            obs = obs.copy(); obs[2] = False
    if not (rms1 <= p.max_rms_m):
        return AlignResult(False, zero, H, int(m.sum()), rms0, rms1, obs, f"rms {rms1:.3f} > {p.max_rms_m}")
    return AlignResult(True, delta, H, int(m.sum()), rms0, rms1, obs)


@dataclass
class SE2:
    """map <- odom as a planar rigid transform."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def apply(self, px: float, py: float) -> tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.x + c * px - s * py, self.y + s * px + c * py

    def pose(self, odom_pose: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x, y = self.apply(odom_pose[0], odom_pose[1])
        return x, y, odom_pose[2], wrap(odom_pose[3] + self.yaw)


def compose(a: SE2, b: SE2) -> SE2:
    """a . b (apply b first)."""
    x, y = a.apply(b.x, b.y)
    return SE2(x, y, wrap(a.yaw + b.yaw))


def inverse(a: SE2) -> SE2:
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    return SE2(-(c * a.x + s * a.y), -(-s * a.x + c * a.y), wrap(-a.yaw))


def correction_ned(start: SE2, T: SE2) -> tuple[float, float, float]:
    """The odom correction as refnav consumes it (/blueraven/odom_correction, frame px4_local_ned):
    corrected = R(yaw_C) * p_px4 + (x_C, y_C), heading_corrected = heading_px4 + yaw_C, with x north,
    y east and yaw clockwise from north (PX4's sense). start = map <- odom at launch, T = the
    corrector's current map <- odom. The correction is start^-1 . T expressed in NED."""
    C = compose(inverse(start), T)                      # odom <- odom, ENU (x east, y north, yaw CCW)
    return C.y, C.x, wrap(-C.yaw)


@dataclass
class CorrectorParams:
    tau_s: float = 2.0            # time constant of the correction
    max_rate_xy: float = 0.15     # m/s the correction may move
    max_rate_yaw: float = math.radians(3.0)
    max_jump_xy: float = 0.5      # a single measurement further than this is ignored (likely a wrong match)
    max_jump_yaw: float = math.radians(10.0)
    max_total_yaw: float = math.radians(6.0)    # the correction's yaw never exceeds this (raw drift seen: up to 9 deg)
    # The fits' heading accuracy floor is ~1.2-1.5 deg rms (T4, 10 truth bags), worse than a good raw heading
    # (0.3-0.7 deg). So yaw follows only a sustained innovation beyond yaw_deadband, and slowly.
    tau_yaw_s: float = 8.0
    yaw_deadband: float = math.radians(1.0)
    align: AlignParams = field(default_factory=AlignParams)


class DriftCorrector:
    """Keeps map <- odom up to date from depth frames. start = map <- odom at launch (nav.frames)."""

    def __init__(self, course_map, start: SE2, params: CorrectorParams | None = None) -> None:
        self.cm, self.T, self.p = course_map, start, params or CorrectorParams()
        self.start = start
        self.t_last: float | None = None
        self.n = {"frames": 0, "used": 0, "rejected_fit": 0, "rejected_jump": 0}
        self.last: AlignResult | None = None

    def update(self, t: float, odom_pose: tuple[float, float, float, float], pts_level: np.ndarray) -> SE2:
        dt = 0.0 if self.t_last is None else max(0.0, min(t - self.t_last, 1.0))
        self.t_last = t
        self.n["frames"] += 1
        prior = self.T.pose(odom_pose)
        r = align(self.cm, pts_level, prior, self.p.align)
        self.last = r
        if not r.ok:
            self.n["rejected_fit"] += 1
            return self.T
        dx, dy, dyaw = r.delta
        if math.hypot(dx, dy) > self.p.max_jump_xy or abs(dyaw) > self.p.max_jump_yaw:
            self.n["rejected_jump"] += 1
            return self.T
        k = 1.0 - math.exp(-dt / self.p.tau_s) if dt > 0 else 0.0
        k_yaw = 1.0 - math.exp(-dt / self.p.tau_yaw_s) if dt > 0 else 0.0
        dyaw_eff = math.copysign(max(0.0, abs(dyaw) - self.p.yaw_deadband), dyaw)
        sx, sy, syaw = k * dx, k * dy, k_yaw * dyaw_eff
        lim = self.p.max_rate_xy * dt
        n_xy = math.hypot(sx, sy)
        if n_xy > lim > 0:
            sx, sy = sx * lim / n_xy, sy * lim / n_xy
        syaw = max(-self.p.max_rate_yaw * dt, min(self.p.max_rate_yaw * dt, syaw))
        tot = wrap(self.T.yaw + syaw - self.start.yaw)                 # keep the total yaw correction bounded
        if abs(tot) > self.p.max_total_yaw:
            syaw -= tot - math.copysign(self.p.max_total_yaw, tot)
        # compose: rotate about the body by syaw, then translate; T' = D . T
        bx, by = prior[0], prior[1]
        c, s = math.cos(syaw), math.sin(syaw)
        nx = c * (self.T.x - bx) - s * (self.T.y - by) + bx + sx
        ny = s * (self.T.x - bx) + c * (self.T.y - by) + by + sy
        self.T = SE2(nx, ny, wrap(self.T.yaw + syaw))
        self.n["used"] += 1
        return self.T
