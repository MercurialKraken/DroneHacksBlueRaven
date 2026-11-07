"""Particle filter (Monte Carlo localization) over the map <- odom correction. Pure: numpy + nav.course_map.

The probabilistic sibling of nav.map_align.DriftCorrector, same inputs and same output type (SE2
map <- odom), plus an honest spread. Each particle is a candidate correction T_i. Per depth frame:

  predict  each particle's MAP pose of the body (T_i applied to the odom pose) random-walks in x, y and
           yaw by the elapsed time (drift is slow); the walk is about the body, so a yaw perturbation
           does not throw the body around a lever arm
  weight   the frame's level-frame points are placed with each particle's pose and scored against the
           course SDF: a truncated (robust) squared distance per point, tempered so that 150 correlated
           points do not count as 150 independent ones
  resample systematic, when the effective sample size drops below half

Frames whose best particle explains too little of the scan (net-only views etc.) skip the weight step:
the cloud just widens by the random walk, which is exactly what an unobservable stretch should do.
Output = weighted mean (circular for yaw), rate-limited so a consumer never sees a jump, and the
weighted std of x, y and yaw (the status topic's 'spread'; refnav can scale speed by it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nav.map_align import SE2, wrap


@dataclass
class MclParams:
    n: int = 300
    field: str = "localization"
    band_m: float = 0.5
    max_points: int = 150
    init_xy_m: float = 0.05
    init_yaw_deg: float = 1.0
    walk_xy: float = 0.03            # m per sqrt(s)
    walk_yaw_deg: float = 0.4        # deg per sqrt(s)
    sigma_m: float = 0.05            # per-point measurement noise
    trunc_m: float = 0.20            # a point further than this from any wall costs the same as at 0.20
    points_per_dof: float = 25.0     # tempering: n points count as n / points_per_dof independent ones
    min_points: int = 60
    min_explained: float = 0.35      # best particle must put this share of points within sigma*2 of a wall
    max_rate_xy: float = 0.15        # output rate limits (m/s, deg/s)
    max_rate_yaw_deg: float = 3.0


class ParticleCorrector:
    def __init__(self, course_map, start: SE2, params: MclParams | None = None, seed: int = 0) -> None:
        self.cm, self.p = course_map, params or MclParams()
        self.rng = np.random.default_rng(seed)
        n = self.p.n
        # particles: correction T_i = (x, y, yaw), map <- odom
        self.T = np.column_stack([np.full(n, start.x), np.full(n, start.y), np.full(n, start.yaw)])
        self.w = np.full(n, 1.0 / n)
        self.out = start
        self._init_pending = True
        self.t_last: float | None = None
        self.n = {"frames": 0, "weighted": 0, "skipped": 0, "resampled": 0}
        self.last_explained = math.nan
        self.spread = (0.0, 0.0, 0.0)

    # --- particle <-> body map pose
    def _poses(self, odom) -> np.ndarray:
        ox, oy, _, oyaw = odom
        c, s = np.cos(self.T[:, 2]), np.sin(self.T[:, 2])
        return np.column_stack([self.T[:, 0] + c * ox - s * oy, self.T[:, 1] + s * ox + c * oy, self.T[:, 2] + oyaw])

    def _set_from_poses(self, poses: np.ndarray, odom) -> None:
        ox, oy, _, oyaw = odom
        yaw = poses[:, 2] - oyaw
        c, s = np.cos(yaw), np.sin(yaw)
        self.T = np.column_stack([poses[:, 0] - (c * ox - s * oy), poses[:, 1] - (s * ox + c * oy), np.arctan2(np.sin(yaw), np.cos(yaw))])

    def update(self, t: float, odom: tuple[float, float, float, float], pts_level: np.ndarray) -> SE2:
        p = self.p
        dt = 0.0 if self.t_last is None else max(0.0, min(t - self.t_last, 1.0))
        self.t_last = t
        self.n["frames"] += 1
        poses = self._poses(odom)
        # predict: random walk of the body's map pose (initial spread on the first frame)
        if self._init_pending:
            sxy, syaw = p.init_xy_m, math.radians(p.init_yaw_deg)
            self._init_pending = False
        else:
            sxy, syaw = p.walk_xy * math.sqrt(dt), math.radians(p.walk_yaw_deg) * math.sqrt(dt)
        poses = poses + self.rng.normal(0.0, 1.0, poses.shape) * np.array([sxy, sxy, syaw])
        # weight
        pts = pts_level[np.abs(pts_level[:, 2]) < p.band_m]
        if len(pts) > p.max_points:
            pts = pts[self.rng.choice(len(pts), p.max_points, replace=False)]
        if len(pts) >= p.min_points:
            f, r, dn = pts[:, 0], pts[:, 1], pts[:, 2]
            c, s = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
            X = poses[:, 0][:, None] + f * c + r * s
            Y = poses[:, 1][:, None] + f * s - r * c
            Z = np.broadcast_to(odom[2] - dn, X.shape)
            d = self.cm.distance(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]), p.field).reshape(X.shape)
            d = np.where(np.isfinite(d), np.abs(d), p.trunc_m)
            explained = (d < 2 * p.sigma_m).mean(axis=1)
            self.last_explained = float(explained.max())
            if self.last_explained >= p.min_explained:
                cost = (np.minimum(d, p.trunc_m) ** 2).sum(axis=1) / (p.sigma_m ** 2)
                logw = np.log(self.w + 1e-300) - 0.5 * cost / max(1.0, len(pts) / p.points_per_dof)
                logw -= logw.max()
                w = np.exp(logw)
                self.w = w / w.sum()
                self.n["weighted"] += 1
            else:
                self.n["skipped"] += 1
        else:
            self.n["skipped"] += 1
        self._set_from_poses(poses, odom)
        # resample
        ess = 1.0 / float((self.w ** 2).sum())
        if ess < p.n / 2:
            u = (self.rng.random() + np.arange(p.n)) / p.n
            idx = np.minimum(np.searchsorted(np.cumsum(self.w), u), p.n - 1)
            self.T = self.T[idx]
            self.w = np.full(p.n, 1.0 / p.n)
            self.n["resampled"] += 1
        # estimate: weighted mean of the body's map pose, converted back to a correction
        poses = self._poses(odom)
        mx, my = float(self.w @ poses[:, 0]), float(self.w @ poses[:, 1])
        myaw = math.atan2(float(self.w @ np.sin(poses[:, 2])), float(self.w @ np.cos(poses[:, 2])))
        self.spread = (math.sqrt(float(self.w @ (poses[:, 0] - mx) ** 2)), math.sqrt(float(self.w @ (poses[:, 1] - my) ** 2)),
                       math.degrees(math.sqrt(float(self.w @ np.array([wrap(a - myaw) for a in poses[:, 2]]) ** 2))))
        # rate-limited output, measured on the body (as DriftCorrector)
        cur = self.out.pose(odom)
        dx, dy, dyaw = mx - cur[0], my - cur[1], wrap(myaw - cur[3])
        lim = p.max_rate_xy * dt                        # dt = 0 (first frame): the output does not move
        nxy = math.hypot(dx, dy)
        if nxy > lim:
            dx, dy = (dx * lim / nxy, dy * lim / nxy) if nxy > 0 else (0.0, 0.0)
        ylim = math.radians(p.max_rate_yaw_deg) * dt
        dyaw = max(-ylim, min(ylim, dyaw))
        tx, ty, tyaw = cur[0] + dx, cur[1] + dy, cur[3] + dyaw
        yaw = wrap(tyaw - odom[3]); c, s = math.cos(yaw), math.sin(yaw)
        self.out = SE2(tx - (c * odom[0] - s * odom[1]), ty - (s * odom[0] + c * odom[1]), yaw)
        return self.out
