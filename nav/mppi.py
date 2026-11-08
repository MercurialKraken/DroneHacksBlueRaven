"""MPPI (model predictive path integral) local controller around the startup planner's path. Pure numpy.

Each control cycle chooses a horizontal velocity-command sequence U (H x 2, map frame, m/s) for a drone holding
height z, by sampling K rollouts V = clip(U + eps) and averaging them with information-theoretic weights:

    U  <-  sum_k w_k V_k ,    w_k  propto  exp(-(S_k + gamma * C_k) / lam)

* S_k is the rollout's path cost (below).
* gamma = 0 (default): the base distribution IS the proposal, so the importance ratio is 1 and the weights are the
  plain softmax of -S/lam. Nothing pulls U toward the previous plan; smoothness comes from the temporally correlated
  noise and an explicit du cost. gamma > 0 adds Williams et al.'s control cost C_k = sum_t u_t . Sigma^-1 eps_t
  (base = zero-mean Gaussian). A uniform base is deliberately NOT offered: against a Gaussian proposal its
  importance correction exp(+|eps|^2 / 2 sigma^2) varies by ~e^+-5 across samples at H*2 = 60 dims and swamps S.
* lam is chosen every cycle by bisection so the effective sample size 1 / sum(w^2) is ess_target * K, which makes
  the update independent of how the costs happen to be scaled.
* Noise is Ornstein-Uhlenbeck per axis (AR(1) with stationary std sigma): smooth like Brownian increments but with
  bounded variance at the end of the horizon.

Model: PX4's velocity loop as a first-order lag,  v <- v + (u - v) * dt / tau,  x <- x + v * dt;  2D at height z.

Cost per step t (summed), then terminal:
    contour   w_contour * e_lat^2                    lateral distance to the path, projected within a window of s
                                                     around the current progress (never onto a far corridor)
    speed     w_speed * (|v| - v_ref(s))^2            the planner's speed profile at the projected s
    obstacle  w_obs * max(0, r_safe - d)^2  +  w_collide * [d < r_drone]
              d = min(map SDF 'planning' at (x, y, z), distance to the nearest depth point)   -- pessimistic;
              outside the map counts as a collision
    du        w_du * |u_t - u_(t-1)|^2
    terminal  -w_progress * (s_end - s_now)  +  w_contour_end * e_lat_end^2

Always-included candidates: the previous plan shifted (eps = 0), a brake (all zeros) and, if given, an external
command held over the horizon (e.g. the existing follower's), so by its own cost MPPI never does worse than those.
If every candidate collides, the output is the brake and `blocked` is set: the supervisor decides what next.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:                                          # depth obstacles only; the map-only path needs nothing beyond numpy
    from scipy.spatial import cKDTree
except ImportError:                           # pragma: no cover
    cKDTree = None


@dataclass(frozen=True)
class MppiParams:
    dt: float = 0.1              # s per rollout step
    horizon: int = 25            # steps (2.5 s)
    samples: int = 768           # K rollouts per cycle (special candidates included)
    sigma: float = 0.35          # m/s, stationary std of the OU exploration noise, per axis
    theta: float = 0.8           # 1/s, OU mean reversion (0 -> Brownian-like within the horizon)
    side_steps: tuple[float, ...] = (0.25, 0.45)   # m/s, deterministic sideways candidates (both sides)
    mode_split_m: float = 0.10   # best sample this far to one side of the path: average only that side
    gamma: float = 0.0           # control-cost weight; 0 = plain softmax (base = proposal)
    ess_target: float = 0.10     # fraction of K the temperature is tuned to keep as effective samples
    tau: float = 0.35            # s, PX4 velocity-loop lag (fit from flight data)
    v_max: float = 0.60          # m/s, command speed cap
    r_drone: float = 0.28        # m, collision radius (course.yaml limits)
    r_safe: float = 0.40         # m, soft barrier starts here (radius + margin)
    w_contour: float = 8.0       # lateral error beyond the deadband; soft, so stepping round an unmapped
    contour_deadband: float = 0.05   # obstacle costs less than stopping in front of it
    w_contour_end: float = 20.0
    w_speed: float = 6.0
    w_obs: float = 400.0
    w_collide: float = 1.0e4
    w_du: float = 2.0
    w_progress: float = 20.0     # per metre, capped at what the planner's speed profile reaches in the horizon
    w_goal: float = 30.0         # terminal distance to the path's end, once the end is within reach
    v_floor: float = 0.05        # m/s, for the speed profile's time integral (its ends are 0)
    window_back_m: float = 0.5   # path projection window around the current progress
    window_fwd_m: float = 1.0    # beyond the furthest a rollout can reach (v_max * horizon * dt)
    seed: int | None = 0


@dataclass
class MppiResult:
    u: tuple[float, float]                 # velocity command now (map frame, m/s)
    s: float                               # current progress along the path (m)
    e_lat: float                           # current lateral error (m)
    lam: float
    ess: float                             # effective sample size actually achieved
    best_cost: float
    blocked: bool                          # every candidate collides: u is a brake
    min_d_plan: float                      # smallest predicted clearance along the weighted plan (m)
    ms: float                              # compute time (ms)
    plan: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 2)))   # predicted positions (H x 2)


class Mppi:
    def __init__(self, course_map: Any, path_xy: Any, v_ref: Any, z: float, params: MppiParams | None = None,
                 field_name: str = "planning") -> None:
        self.p = params or MppiParams()
        self.cm, self.field, self.z = course_map, field_name, float(z)
        P = np.asarray(path_xy, dtype=float)
        if P.ndim != 2 or P.shape[1] != 2 or len(P) < 2:
            raise ValueError("path_xy must be (N>=2, 2)")
        self.px, self.py = P[:, 0], P[:, 1]
        self.ps = np.r_[0.0, np.cumsum(np.hypot(np.diff(self.px), np.diff(self.py)))]
        self.vref = np.asarray(v_ref, dtype=float)
        if self.vref.shape != (len(P),):
            raise ValueError("v_ref must have one speed per path point")
        self.length = float(self.ps[-1])
        v_mid = np.maximum(0.5 * (self.vref[1:] + self.vref[:-1]), self.p.v_floor)
        self.pt = np.r_[0.0, np.cumsum(np.diff(self.ps) / v_mid)]   # time to each path point at the profile
        H = self.p.horizon
        self.U = np.zeros((H, 2))
        self.s_now: float | None = None
        self.rng = np.random.default_rng(self.p.seed)
        a = math.exp(-self.p.theta * self.p.dt)
        self._ou = (a, self.p.sigma * math.sqrt(max(1.0 - a * a, 0.0)))
        self._depth_tree = None

    # ------------------------------------------------------------------ inputs
    def set_depth_points(self, pts_map: Any | None) -> None:
        """Obstacle points from depth, map frame (N,2) or (N,3); the caller filters to a height band. None clears."""
        if pts_map is None or len(pts_map) == 0:
            self._depth_tree = None
            return
        if cKDTree is None:
            raise RuntimeError("scipy is needed for depth obstacles")
        Q = np.asarray(pts_map, dtype=float)[:, :2]
        self._depth_tree = cKDTree(Q)

    def reset(self, s_hint: float | None = None) -> None:
        self.U[:] = 0.0
        self.s_now = s_hint

    # ------------------------------------------------------------------ pieces
    def _project(self, X: np.ndarray, s_center: float) -> tuple[np.ndarray, np.ndarray]:
        """(e_lat, s) of points X (..., 2) onto the path within [s_center - back, s_center + fwd]."""
        p = self.p
        reach = p.v_max * p.horizon * p.dt
        a = int(np.searchsorted(self.ps, s_center - p.window_back_m))
        b = int(np.searchsorted(self.ps, s_center + reach + p.window_fwd_m, side="right"))
        a, b = max(0, min(a, len(self.ps) - 2)), max(a + 2, min(b, len(self.ps)))
        Wx, Wy, Ws = self.px[a:b], self.py[a:b], self.ps[a:b]
        flat = X.reshape(-1, 2)
        d2 = (flat[:, 0, None] - Wx[None, :]) ** 2 + (flat[:, 1, None] - Wy[None, :]) ** 2
        j = np.argmin(d2, axis=1)
        e = np.sqrt(d2[np.arange(len(flat)), j]).reshape(X.shape[:-1])
        return e, Ws[j].reshape(X.shape[:-1])

    def clearance(self, X: np.ndarray) -> np.ndarray:
        """Pessimistic clearance of points X (..., 2): min of the map SDF at height z and the depth points."""
        flat = X.reshape(-1, 2)
        P3 = np.c_[flat, np.full(len(flat), self.z)]
        d = self.cm.distance(P3, field=self.field)
        d = np.where(np.isfinite(d), d, -1.0)                    # outside the map: treat as inside an obstacle
        if self._depth_tree is not None:
            dd, _ = self._depth_tree.query(flat, k=1)
            d = np.minimum(d, dd)
        return d.reshape(X.shape[:-1])

    def _rollout(self, x0: np.ndarray, v0: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.p
        K, H, _ = V.shape
        alpha = min(p.dt / p.tau, 1.0)
        X = np.empty((K, H, 2))
        Vel = np.empty((K, H, 2))
        x = np.broadcast_to(x0, (K, 2)).copy()
        v = np.broadcast_to(v0, (K, 2)).copy()
        for t in range(H):
            v = v + (V[:, t] - v) * alpha
            x = x + v * p.dt
            X[:, t], Vel[:, t] = x, v
        return X, Vel

    def _clip(self, V: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(V, axis=-1, keepdims=True)
        return V * np.minimum(1.0, self.p.v_max / np.maximum(n, 1e-9))

    def _costs(self, X: np.ndarray, Vel: np.ndarray, V: np.ndarray, u_prev: np.ndarray, s_now: float) -> tuple[np.ndarray, np.ndarray]:
        p = self.p
        e, s = self._project(X, s_now)
        d = self.clearance(X)
        speed = np.linalg.norm(Vel, axis=-1)
        vref = np.interp(s, self.ps, self.vref)
        dU = np.diff(np.concatenate([np.broadcast_to(u_prev, (V.shape[0], 1, 2)), V], axis=1), axis=1)
        e_db = np.maximum(0.0, e - p.contour_deadband)
        step = (p.w_contour * e_db ** 2
                + p.w_speed * (speed - vref) ** 2
                + p.w_obs * np.maximum(0.0, p.r_safe - d) ** 2
                + p.w_collide * (d < p.r_drone)
                + p.w_du * np.sum(dU ** 2, axis=-1))
        # progress, capped at where the planner's speed profile gets in the horizon: outrunning it earns nothing
        s_ref_end = float(np.interp(np.interp(s_now, self.ps, self.pt) + p.horizon * p.dt, self.pt, self.ps))
        gain = np.minimum(s[:, -1], s_ref_end) - s_now
        S = step.sum(axis=1) - p.w_progress * gain + p.w_contour_end * e_db[:, -1] ** 2
        if self.length - s_now < p.v_max * p.horizon * p.dt + 0.5:   # the end is within reach: arrive on it
            S = S + p.w_goal * ((X[:, -1, 0] - self.px[-1]) ** 2 + (X[:, -1, 1] - self.py[-1]) ** 2)
        return S, d

    def _weights(self, S: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Softmax of -S/lam with lam bisected (in log space) so ESS = ess_target * K."""
        K = len(S)
        Sn = S - S.min()
        target = max(1.0, self.p.ess_target * K)

        def w_of(lam: float) -> tuple[np.ndarray, float]:
            w = np.exp(-Sn / lam)
            w /= w.sum()
            return w, 1.0 / float(np.sum(w * w))

        spread = float(np.median(Sn)) + 1e-9
        lo, hi = math.log(spread * 1e-6), math.log(spread * 1e3)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            _, ess = w_of(math.exp(mid))
            lo, hi = (mid, hi) if ess < target else (lo, mid)
        lam = math.exp(hi)
        w, ess = w_of(lam)
        return w, lam, ess

    # ------------------------------------------------------------------ step
    def step(self, pos: Any, vel: Any, u_ext: Any | None = None) -> MppiResult:
        """One control cycle from the drone's map-frame position (x, y) and velocity (vx, vy)."""
        t0 = time.perf_counter()
        p = self.p
        x0, v0 = np.asarray(pos, float)[:2], np.asarray(vel, float)[:2]
        if self.s_now is None:                                       # first call: project onto the whole path
            d2 = (self.px - x0[0]) ** 2 + (self.py - x0[1]) ** 2
            self.s_now = float(self.ps[int(np.argmin(d2))])
        e_now, s_proj = self._project(x0[None, None, :], self.s_now)
        self.s_now = float(max(self.s_now, s_proj[0, 0]))            # progress never goes backwards
        H, K = p.horizon, p.samples

        # the path's frame at the current progress: tangent t and left normal n
        i = int(np.clip(np.searchsorted(self.ps, self.s_now), 1, len(self.ps) - 1))
        tng = np.array([self.px[i] - self.px[i - 1], self.py[i] - self.py[i - 1]])
        tng /= max(float(np.linalg.norm(tng)), 1e-9)
        nrm = np.array([-tng[1], tng[0]])
        v_here = max(float(np.interp(self.s_now, self.ps, self.vref)), 0.15)

        # candidates: special rows first, then the noisy samples
        special = [self.U.copy(), np.zeros((H, 2))]
        if u_ext is not None:
            special.append(np.broadcast_to(np.asarray(u_ext, float)[:2], (H, 2)).copy())
        for dv in p.side_steps:                                     # step aside for the first half, then go on
            for sgn in (1.0, -1.0):
                c = np.broadcast_to(v_here * tng, (H, 2)).copy()
                c[: H // 2] += sgn * dv * nrm
                special.append(c)
        n_noisy = max(1, K - len(special))
        eps = np.empty((n_noisy, H, 2))
        a, b = self._ou
        eps[:, 0] = self.rng.normal(0.0, p.sigma, (n_noisy, 2))
        for t in range(1, H):
            eps[:, t] = a * eps[:, t - 1] + b * self.rng.normal(0.0, 1.0, (n_noisy, 2))
        V = self._clip(np.concatenate([np.stack(special), self.U[None] + eps], axis=0))

        X, Vel = self._rollout(x0, v0, V)
        u_prev = self.U[0] if np.any(self.U) else v0
        S, d = self._costs(X, Vel, V, u_prev, self.s_now)
        if p.gamma > 0.0:                                            # Williams' control cost vs the zero-mean base
            E = V - self.U[None]
            S = S + p.gamma * np.einsum("kti,ti->k", E, self.U) / (p.sigma ** 2)

        blocked = bool(np.all(np.min(d, axis=1) < p.r_drone))
        if blocked:
            U_new = np.zeros((H, 2))
            w, lam, ess = np.full(len(S), 1.0 / len(S)), float("nan"), float(len(S))
        else:
            # one mode only: if the best rollout goes round something on one side, don't average it with the other
            side = ((X - (self.px[i], self.py[i])) @ nrm).mean(axis=1)
            kb = int(np.argmin(S))
            keep = np.ones(len(S), dtype=bool)
            if abs(side[kb]) > p.mode_split_m:
                keep = np.sign(side) == np.sign(side[kb])
            w_sub, lam, ess = self._weights(S[keep])
            w = np.zeros(len(S))
            w[keep] = w_sub
            U_new = self._clip(np.einsum("k,kti->ti", w, V))
        Xp, _ = self._rollout(x0, v0, U_new[None])
        u = (float(U_new[0, 0]), float(U_new[0, 1]))
        self.U = np.concatenate([U_new[1:], U_new[-1:]], axis=0)    # warm start: shift by one step
        return MppiResult(u=u, s=self.s_now, e_lat=float(e_now[0, 0]), lam=lam, ess=float(ess),
                          best_cost=float(S.min()), blocked=blocked,
                          min_d_plan=float(np.min(self.clearance(Xp[0]))),
                          ms=(time.perf_counter() - t0) * 1e3, plan=Xp[0])
