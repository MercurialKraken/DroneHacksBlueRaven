"""MPPI (model predictive path integral) local controller around the startup planner's path.

Array-agnostic: pass xp=numpy (default, CPU) or xp=cupy (GPU); everything per-sample runs in xp.

Each control cycle picks a horizontal velocity-command sequence U (H x 2, map frame, m/s) for a drone holding height
z. It samples K command sequences V_k near U, rolls each through a model of PX4's velocity loop, scores them with
a cost S_k and sets

    U  <-  sum_k w_k V_k ,    w_k = softmax(-S_k / lam).

Why the weights are a plain softmax: the base distribution is the sampling distribution itself (Williams' gamma = 0;
"the reference is the sampler"), so there is no importance ratio and no control-cost term. Any truncation, clipping
or fixed candidates in the sampler are therefore legal and never enter the weights.
lam is chosen every cycle by bisection so the effective sample size 1 / sum(w^2) is ess_target * K, which makes the
update independent of how the costs happen to be scaled.

MODEL. We can only command velocity (/skybeta/setpoint/twist; never /fmu/in). PX4's closed velocity loop is
modelled as a first-order lag with tau = 1 / MPC_XY_VEL_P_ACC (1.8 on the fleet's params, so tau ~ 0.55 s). For a
command u held over a step dt, the exact update is

    v' = u + (v - u) e^(-dt/tau)
    p' = p + u dt + (v - u) tau (1 - e^(-dt/tau)).

SAMPLING (Brownian in the path frame, change-limited). Each sample builds its commands from increments:
    V_t = box_t( V_(t-1) + clipnorm(U_t - U_(t-1) + delta_t, a_cmd dt) ),   V_(-1) = the last command sent,
    delta_t ~ N(0, sigma_inc^2) per axis, i.i.d.
so V - U is a random walk (Brownian), every sample changes by at most a_cmd * dt per step (PX4 adds no limit of
its own on offboard velocity setpoints), and box_t keeps the forward speed along the reference in [0, v_max]
(never backwards along the path; where the reference is stopped, e.g. at the path's end, backing up to v_back is
allowed, or clipping the noise at 0 would bias the average forward and creep the drone on), the sideways speed in [-v_lat, v_lat] and |V| <= v_max. A convex combination of
such sequences obeys the same limits, so U does too (the change limit only approximately where the path frame
turns at a corner, because the box is applied after it). Fixed candidates are always included: the previous plan
(delta = 0), a brake to zero, the reference velocities, and an external command if given, each reached under the
same change limit. So by its own cost MPPI never does worse than those.

REFERENCE (re-anchored, no running clock). Each cycle the reference starts at the drone's projection s_now onto the
path and advances along the planner's speed profile: p_ref(t), v_ref(t), with the path frame (tangent, left
normal) at p_ref(t). A detour therefore never leaves the drone "behind schedule" for more than one horizon.

COST per step t, summed until the rollout crashes:
    q_lat e_lat^2 + q_lon e_lon^2         position error to p_ref(t), split in the path frame
    q_vel |v - v_ref(t)|^2
    w_obs exp(-(d - r_drone) / obs_len)   d = clearance at the rollout's position (below)
    w_du |V_t - V_(t-1)|^2 / dt^2
  crash (d < r_drone, or outside the map) at step t:  CRASH * (H - t), and nothing after it counts
  terminal (no crash):  qf_lat e_lat^2 + qf_lon e_lon^2

CLEARANCE d: a 2D distance grid at flight height (Grid2D), one bilinear lookup per rollout step, on CPU or GPU.
Map-free flight feeds the live grid built from depth (nav/local_grid.py) every cycle with set_grid(). A simulator
with a course map can use grid_from_course_map(); set_depth_points() then mins depth points into it (pessimistic:
a false depth hit is a wall).

PATH. set_path() swaps in a new reference path (the live centerline) every cycle; the warm start U and the last
command carry over, the progress along the path restarts at the drone's projection.

LEFT/RIGHT MODES. When an obstacle sits on the path, detours to the left and to the right can score alike, and their
average flies straight into it (the plain-MPPI stall). Each sample's side is the sign of its mean lateral offset from
the reference. If both sides carry real weight (each > split_mass), only one side (plus the samples near the path)
is averaged. The side chosen stays chosen unless the other side's weight is switch_ratio times larger, and it is
forgotten after side_memory_s without a split.

If every candidate crashes (the brake included) the output is a zero-velocity command and `blocked` is set; the
supervisor decides what next.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:                                          # depth obstacles only; the map-only path needs nothing beyond numpy
    from scipy.ndimage import distance_transform_edt
except ImportError:                           # pragma: no cover
    distance_transform_edt = None


@dataclass
class Grid2D:
    """Signed clearance (m) on a regular x/y lattice: dist[i, j] at (origin[0] + i * cell, origin[1] + j * cell).
    Negative inside obstacles; anything outside the lattice counts as -1 (a crash)."""
    dist: np.ndarray
    origin: tuple[float, float]
    cell: float


def grid_from_course_map(cm: Any, z: float, field_name: str = "planning") -> Grid2D:
    """The course map's distance at height z on the map's own x/y lattice (simulators with a map)."""
    nx, ny = int(cm.shape[0]), int(cm.shape[1])
    ox, oy, cell = float(cm.origin[0]), float(cm.origin[1]), float(cm.cell)
    X, Y = np.meshgrid(ox + cell * np.arange(nx), oy + cell * np.arange(ny), indexing="ij")
    d = cm.distance(np.c_[X.ravel(), Y.ravel(), np.full(X.size, float(z))], field=field_name).reshape(nx, ny)
    if not np.isfinite(d).any():
        raise ValueError(f"height z={z} is outside the course map")
    return Grid2D(np.where(np.isfinite(d), d, -1.0), (ox, oy), cell)


@dataclass(frozen=True)
class MppiParams:
    dt: float = 0.1              # s per rollout step
    horizon: int = 30            # steps (3 s): at 0.45 m/s that is ~1.3 m of look-ahead
    samples: int = 1024          # K rollouts per cycle (fixed candidates included)
    sigma_inc: float = 0.10      # m/s per step, std of the Brownian increments per path-frame axis
    ess_target: float = 0.10     # fraction of K the temperature is tuned to keep as effective samples
    tau: float = 0.55            # s, PX4 velocity-loop lag = 1 / MPC_XY_VEL_P_ACC (1.8)
    dead_time_s: float = 0.0     # s, command -> PX4 starts responding (+ pose latency). T4 fit: tau 0.9 + 0.6 s.
                                 # MPPI plans from the state PREDICTED after the commands already in flight
    v_max: float = 0.60          # m/s, command speed cap (the organizers' interface clamps at 1.5)
    v_lat: float = 0.50          # m/s, sideways command cap in the path frame
    a_cmd: float = 1.5           # m/s^2, command change limit (half of PX4's MPC_ACC_HOR 3.0)
    v_stopped: float = 0.05      # m/s: where the reference is slower than this, samples may back up ...
    v_back: float = 0.20         # ... at up to this speed (else forward >= 0 along the path)
    r_drone: float = 0.28        # m, collision radius (course.yaml limits)
    obs_len: float = 0.05        # m, decay length of the exponential obstacle cost: it matters within ~10 cm of
                                 # r_drone (13% of w_obs at +10 cm), so open corridors aren't slowed by far walls
    w_obs: float = 20.0          # obstacle cost at d = r_drone, per step
    crash: float = 1.0e3         # per remaining step after a crash
    q_lat: float = 4.0
    q_lon: float = 4.0
    q_vel: float = 1.0
    qf_lat: float = 20.0
    qf_lon: float = 10.0
    w_du: float = 0.1
    v_floor: float = 0.05        # m/s, for the speed profile's time integral (its ends are 0)
    window_back_m: float = 0.5   # where the drone's own projection onto the path may land, around s_now
    window_fwd_m: float = 2.0
    side_m: float = 0.08         # m, mean lateral offset that counts as "left" or "right"
    split_mass: float = 0.10     # both sides must carry this much weight before one is dropped
    switch_ratio: float = 3.0    # the other side needs this many times the weight to take over
    side_memory_s: float = 1.0
    seed: int | None = 0


@dataclass
class MppiResult:
    u: tuple[float, float]                 # velocity command now (map frame, m/s)
    s: float                               # current progress along the path (m)
    e_lat: float                           # current signed lateral error (m, + = left of the path)
    lam: float
    ess: float                             # effective sample size actually achieved
    best_cost: float
    crash_frac: float                      # fraction of candidates that crash within the horizon
    side: int                              # committed detour side: +1 left, -1 right, 0 none
    blocked: bool                          # every candidate crashes: u is zero
    min_d_plan: float                      # smallest predicted clearance along the chosen plan (m)
    ms: float                              # compute time (ms)
    plan: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 2)))   # predicted positions (H x 2)
    # debug only (Mppi.debug_samples > 0): a subset of the rollouts and their weights, and the reference
    samples: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0, 2)))
    sample_w: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    sample_crashed: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, bool))
    reference: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 2)))


def _to_np(a: Any) -> np.ndarray:
    return a.get() if hasattr(a, "get") else np.asarray(a)   # cupy arrays have .get(); numpy arrays don't


class Mppi:
    def __init__(self, grid: Grid2D, path_xy: Any, v_ref: Any, params: MppiParams | None = None,
                 xp: Any = None) -> None:
        self.p = params or MppiParams()
        self.xp = xp if xp is not None else np
        H = self.p.horizon
        self.U = self.xp.zeros((H, 2))
        self.u_last: np.ndarray | None = None
        self._side, self._side_age = 0, 0
        self.v_cap = math.inf                                        # extra speed cap (set_speed_cap), e.g. no depth
        self._inflight: list[np.ndarray] = []                        # commands sent within the last dead_time_s
        self._vr_now = self.xp.full((self.p.horizon, 2), 1.0)       # the reference velocities this cycle
        self.debug_samples = 0                                       # >0: return that many rollouts (debug/vis only)
        self.rng = self.xp.random.default_rng(self.p.seed)
        self._depth_grid: np.ndarray | None = None
        self.set_grid(grid)
        self.set_path(path_xy, v_ref)

    def set_path(self, path_xy: Any, v_ref: Any) -> None:
        """A new reference path (N>=2 points) with one speed per point. Progress restarts at the drone's projection."""
        P = np.asarray(path_xy, dtype=float)
        vref = np.asarray(v_ref, dtype=float)
        if P.ndim != 2 or P.shape[1] != 2 or len(P) < 2:
            raise ValueError("path_xy must be (N>=2, 2)")
        if vref.shape != (len(P),):
            raise ValueError("v_ref must have one speed per path point")
        seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
        keep = np.r_[True, seg > 1e-9]                               # drop repeated points
        if keep.sum() < 2:
            raise ValueError("path_xy has fewer than 2 distinct points")
        self._keep = keep
        P = P[keep]
        self.px, self.py = P[:, 0], P[:, 1]
        self.ps = np.r_[0.0, np.cumsum(np.hypot(np.diff(self.px), np.diff(self.py)))]
        self.length = float(self.ps[-1])
        self._set_profile(vref[keep])
        self.s_now: float | None = None

    def _set_profile(self, vref: np.ndarray) -> None:
        self.vref = vref
        v_mid = np.maximum(0.5 * (self.vref[1:] + self.vref[:-1]), self.p.v_floor)
        self.pt = np.r_[0.0, np.cumsum(np.diff(self.ps) / v_mid)]   # time to each path point along the profile

    def set_speed_cap(self, v: float | None) -> None:
        """A hard speed cap on every candidate (None: only v_max), e.g. while depth is missing."""
        self.v_cap = math.inf if v is None else float(v)

    def set_speed_profile(self, v_ref: Any) -> None:
        """New speeds for the SAME path (one per point given to set_path); progress is kept."""
        v = np.asarray(v_ref, dtype=float)
        if v.shape != self._keep.shape:
            raise ValueError("v_ref must have one speed per point of the path given to set_path")
        self._set_profile(v[self._keep])

    # ------------------------------------------------------------------ map and depth
    def set_grid(self, grid: Grid2D) -> None:
        """The clearance grid to plan against (the live depth grid each cycle, or a map slice)."""
        dist = np.asarray(_to_np(grid.dist), dtype=float)
        if dist.ndim != 2 or min(dist.shape) < 2:
            raise ValueError("grid.dist must be 2D with at least 2 cells per axis")
        self.g_origin = (float(grid.origin[0]), float(grid.origin[1]))
        self.g_cell = float(grid.cell)
        self._map_grid = dist
        self._depth_grid = None
        self.grid = self.xp.asarray(dist)

    def set_depth_points(self, pts_map: Any | None) -> None:
        """Obstacle points from depth, map frame (N,2) or (N,3); the caller filters to a height band. None clears.

        The points are rasterised onto the slice's lattice and turned into a distance grid (exact Euclidean distance
        transform), then min'd with the map slice. Distances are reduced by half a cell's diagonal so the
        rasterisation can only make obstacles look nearer."""
        if pts_map is None or len(pts_map) == 0:
            self._depth_grid = None
            self.grid = self.xp.asarray(self._map_grid)
            return
        if distance_transform_edt is None:
            raise RuntimeError("scipy is needed for depth obstacles")
        Q = np.asarray(_to_np(pts_map), dtype=float)[:, :2]
        nx, ny = self._map_grid.shape
        i = np.rint((Q[:, 0] - self.g_origin[0]) / self.g_cell).astype(int)
        j = np.rint((Q[:, 1] - self.g_origin[1]) / self.g_cell).astype(int)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        free = np.ones((nx, ny), dtype=bool)
        free[i[ok], j[ok]] = False
        if free.all():
            self._depth_grid = None
            self.grid = self.xp.asarray(self._map_grid)
            return
        dd = distance_transform_edt(free) * self.g_cell - 0.7072 * self.g_cell
        self._depth_grid = dd
        self.grid = self.xp.asarray(np.minimum(self._map_grid, dd))

    def clearance(self, X: Any) -> Any:
        """Pessimistic clearance (m) at points X (..., 2), bilinear on the 2D grid; outside the map is -1."""
        xp = self.xp
        X = xp.asarray(X)
        G = self.grid
        nx, ny = G.shape
        u = (X[..., 0] - self.g_origin[0]) / self.g_cell
        v = (X[..., 1] - self.g_origin[1]) / self.g_cell
        inside = (u >= 0.0) & (u <= nx - 1) & (v >= 0.0) & (v <= ny - 1)
        u = xp.clip(u, 0.0, nx - 1.000001)
        v = xp.clip(v, 0.0, ny - 1.000001)
        i0 = xp.floor(u).astype(xp.int32)
        j0 = xp.floor(v).astype(xp.int32)
        fu, fv = u - i0, v - j0
        d = (G[i0, j0] * (1 - fu) * (1 - fv) + G[i0 + 1, j0] * fu * (1 - fv)
             + G[i0, j0 + 1] * (1 - fu) * fv + G[i0 + 1, j0 + 1] * fu * fv)
        return xp.where(inside, d, -1.0)

    def reset(self, s_hint: float | None = None) -> None:
        """Forget the plan, the last command and the detour side (e.g. after a pilot takeover)."""
        self.U[:] = 0.0
        self.u_last = None
        self._inflight = []
        self.s_now = s_hint
        self._side, self._side_age = 0, 0

    # ------------------------------------------------------------------ path
    def _project_point(self, x0: np.ndarray, s_center: float | None) -> tuple[float, float]:
        """(s, signed lateral error) of x0 on the path; within a window around s_center if given."""
        if s_center is None:
            a, b = 0, len(self.ps)
        else:
            a = int(np.searchsorted(self.ps, s_center - self.p.window_back_m))
            b = int(np.searchsorted(self.ps, s_center + self.p.window_fwd_m, side="right"))
            a = max(0, min(a, len(self.ps) - 2))
            b = max(a + 2, min(b + 1, len(self.ps)))
        Ax, Ay = self.px[a:b - 1], self.py[a:b - 1]
        Dx, Dy = self.px[a + 1:b] - Ax, self.py[a + 1:b] - Ay
        L2 = Dx * Dx + Dy * Dy
        f = np.clip(((x0[0] - Ax) * Dx + (x0[1] - Ay) * Dy) / L2, 0.0, 1.0)
        qx, qy = Ax + f * Dx, Ay + f * Dy
        dist = np.hypot(x0[0] - qx, x0[1] - qy)
        k = int(np.argmin(dist))
        s = float(self.ps[a + k] + f[k] * math.sqrt(L2[k]))
        side = (Dx[k] * (x0[1] - qy[k]) - Dy[k] * (x0[0] - qx[k])) / math.sqrt(L2[k])   # cross > 0: left
        return s, float(math.copysign(dist[k], side) if dist[k] > 0 else 0.0)

    def reference(self, s_now: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Re-anchored reference over the horizon: positions, velocities, tangents, left normals (each H x 2)."""
        p = self.p
        H = p.horizon
        t_now = float(np.interp(s_now, self.ps, self.pt))
        tt = t_now + p.dt * np.arange(0, H + 1)
        s_ref = np.interp(tt, self.pt, self.ps)                      # clamps at the path's end
        pr = np.c_[np.interp(s_ref, self.ps, self.px), np.interp(s_ref, self.ps, self.py)]
        vr = np.diff(pr, axis=0) / p.dt                              # (H, 2): mean velocity over each step
        k = np.clip(np.searchsorted(self.ps, s_ref[1:], side="right"), 1, len(self.ps) - 1)
        tg = np.c_[self.px[k] - self.px[k - 1], self.py[k] - self.py[k - 1]]
        tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-9)
        nm = np.c_[-tg[:, 1], tg[:, 0]]
        return pr[1:], vr, tg, nm

    # ------------------------------------------------------------------ pieces
    def _clipnorm(self, a: Any, r: float) -> Any:
        xp = self.xp
        n = xp.sqrt(xp.sum(a * a, axis=-1, keepdims=True))
        return a * xp.minimum(1.0, r / xp.maximum(n, 1e-12))

    def _box(self, v: Any, tg: Any, nm: Any, f_lo: float = 0.0) -> Any:
        """Path-frame box: forward speed in [f_lo, v_max], sideways in [-v_lat, v_lat], |v| <= v_max."""
        xp, p = self.xp, self.p
        vmax = min(p.v_max, self.v_cap)
        f = xp.clip(v @ tg, f_lo, vmax)
        l = xp.clip(v @ nm, -min(p.v_lat, vmax), min(p.v_lat, vmax))
        return self._clipnorm(f[..., None] * tg + l[..., None] * nm, vmax)

    def sample(self, u_last: Any, targets: Any, tg: Any, nm: Any, n_noisy: int) -> Any:
        """Candidate command sequences (n_special + n_noisy, H, 2): the targets reached under the change limit,
        then Brownian perturbations of the nominal plan U."""
        xp, p = self.xp, self.p
        H = p.horizon
        amax = p.a_cmd * p.dt
        dU = xp.diff(xp.concatenate([xp.asarray(u_last)[None], self.U], axis=0), axis=0)   # (H, 2)
        # noise drawn in the path frame of each step, mapped to the map frame
        z = self.rng.standard_normal((n_noisy, H, 2)) * p.sigma_inc
        delta = z[..., 0:1] * tg[None] + z[..., 1:2] * nm[None]
        S_ = targets.shape[0]
        V = xp.empty((S_ + n_noisy, H, 2))
        prev_s = xp.broadcast_to(xp.asarray(u_last), (S_, 2))
        prev_n = xp.broadcast_to(xp.asarray(u_last), (n_noisy, 2))
        # forward >= 0 only where the reference moves: where it is stopped (the path's end, waiting), clipping the
        # noise at 0 would bias the average forward and creep the drone on, so backing up (v_back) is allowed there
        f_lo = [0.0 if float(xp.sqrt(xp.sum(vr_t * vr_t))) > p.v_stopped else -p.v_back for vr_t in self._vr_now]
        for t in range(H):
            vs = self._box(prev_s + self._clipnorm(targets[:, t] - prev_s, amax), tg[t], nm[t], f_lo[t])
            vn = self._box(prev_n + self._clipnorm(dU[t] + delta[:, t], amax), tg[t], nm[t], f_lo[t])
            V[:S_, t], V[S_:, t] = vs, vn
            prev_s, prev_n = vs, vn
        return V

    def rollout(self, x0: Any, v0: Any, V: Any) -> tuple[Any, Any]:
        """Exact first-order-lag rollout of command sequences V (K, H, 2): positions and velocities after each step."""
        xp, p = self.xp, self.p
        K, H, _ = V.shape
        a = math.exp(-p.dt / p.tau)
        b = p.tau * (1.0 - a)
        X = xp.empty((K, H, 2))
        Vel = xp.empty((K, H, 2))
        x = xp.broadcast_to(xp.asarray(x0), (K, 2))
        v = xp.broadcast_to(xp.asarray(v0), (K, 2))
        for t in range(H):
            u = V[:, t]
            x = x + u * p.dt + (v - u) * b
            v = u + (v - u) * a
            X[:, t], Vel[:, t] = x, v
        return X, Vel

    def costs(self, X: Any, Vel: Any, V: Any, u_last: Any, ref: tuple[Any, Any, Any, Any]) -> dict[str, Any]:
        """Path cost S (K,) with early-terminating crashes, plus the pieces the caller needs."""
        xp, p = self.xp, self.p
        K, H, _ = X.shape
        pr, vr, tg, nm = ref
        E = X - pr[None]
        e_lon = xp.sum(E * tg[None], axis=-1)
        e_lat = xp.sum(E * nm[None], axis=-1)
        d = self.clearance(X)
        dV = xp.diff(xp.concatenate([xp.broadcast_to(xp.asarray(u_last), (K, 1, 2)), V], axis=1), axis=1)
        obs = p.w_obs * xp.exp(xp.minimum(-(d - p.r_drone) / p.obs_len, 50.0))
        step = (p.q_lat * e_lat ** 2 + p.q_lon * e_lon ** 2
                + p.q_vel * xp.sum((Vel - vr[None]) ** 2, axis=-1)
                + obs + p.w_du * xp.sum(dV ** 2, axis=-1) / (p.dt * p.dt))
        hit = d < p.r_drone
        crashed = xp.any(hit, axis=1)
        first = xp.where(crashed, xp.argmax(hit, axis=1), H)        # index of the first crashing step
        alive = xp.arange(H)[None, :] < first[:, None]
        S = xp.sum(xp.where(alive, step, 0.0), axis=1)
        term = p.qf_lat * e_lat[:, -1] ** 2 + p.qf_lon * e_lon[:, -1] ** 2
        S = S + xp.where(crashed, p.crash * (H - first), term)
        side = xp.mean(e_lat, axis=1)
        return {"S": S, "crashed": crashed, "first": first, "side": side, "d": d}

    def weights(self, S: Any) -> tuple[Any, float, float]:
        """Softmax of -S/lam with lam bisected (in log space) so ESS = ess_target * K."""
        xp = self.xp
        K = int(S.shape[0])
        Sn = S - xp.min(S)
        target = max(1.0, self.p.ess_target * K)

        def w_of(lam: float) -> tuple[Any, float]:
            w = xp.exp(-Sn / lam)
            w = w / xp.sum(w)
            return w, 1.0 / float(xp.sum(w * w))

        pos = Sn[Sn > 0]
        spread = float(xp.median(pos)) if int(pos.shape[0]) else 1.0
        lo, hi = math.log(spread * 1e-6), math.log(spread * 1e4)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            _, ess = w_of(math.exp(mid))
            lo, hi = (mid, hi) if ess < target else (lo, mid)
        lam = math.exp(hi)
        w, ess = w_of(lam)
        return w, lam, ess

    def _choose_side(self, w: Any, side: Any) -> int:
        """Detour side to commit to (+1 left, -1 right) when both sides carry weight, else 0 (average all)."""
        xp, p = self.xp, self.p
        mL = float(xp.sum(w[side > p.side_m]))
        mR = float(xp.sum(w[side < -p.side_m]))
        if min(mL, mR) <= p.split_mass:
            self._side_age += 1
            if self._side_age * p.dt > p.side_memory_s:
                self._side = 0
            return 0
        self._side_age = 0
        if self._side == +1 and mR > p.switch_ratio * mL:
            self._side = -1
        elif self._side == -1 and mL > p.switch_ratio * mR:
            self._side = +1
        elif self._side == 0:
            self._side = +1 if mL >= mR else -1
        return self._side

    # ------------------------------------------------------------------ step
    def step(self, pos: Any, vel: Any, u_ext: Any | None = None) -> MppiResult:
        """One control cycle from the drone's map-frame position (x, y) and velocity (vx, vy)."""
        t0 = time.perf_counter()
        xp, p = self.xp, self.p
        H, K = p.horizon, p.samples
        x0 = np.asarray(_to_np(pos), float)[:2]
        v0 = np.asarray(_to_np(vel), float)[:2]
        n_dead = int(round(p.dead_time_s / p.dt))
        if n_dead > 0:                                               # Smith-predictor style: where the drone will be
            q = self._inflight[-n_dead:]                             # when this cycle's command starts to act
            q = [np.zeros(2)] * (n_dead - len(q)) + q
            ad = math.exp(-p.dt / p.tau)
            bd = p.tau * (1.0 - ad)
            for uq in q:
                x0 = x0 + uq * p.dt + (v0 - uq) * bd
                v0 = uq + (v0 - uq) * ad
        s_proj, e_now = self._project_point(x0, self.s_now)
        self.s_now = s_proj if self.s_now is None else max(self.s_now, s_proj)   # progress never goes backwards
        if self.u_last is None:
            self.u_last = v0.copy()
        pr, vr, tg, nm = (xp.asarray(a) for a in self.reference(self.s_now))
        self._vr_now = vr
        u_last = xp.asarray(self.u_last)

        targets = [self.U, xp.zeros((H, 2)), vr]                     # previous plan, brake, reference
        if u_ext is not None:
            targets.append(xp.broadcast_to(xp.asarray(np.asarray(u_ext, float)[:2]), (H, 2)))
        targets = xp.stack(targets)
        n_noisy = max(1, K - int(targets.shape[0]))
        V = self.sample(u_last, targets, tg, nm, n_noisy)
        X, Vel = self.rollout(x0, v0, V)
        c = self.costs(X, Vel, V, u_last, (pr, vr, tg, nm))
        S, crashed = c["S"], c["crashed"]

        blocked = bool(xp.all(crashed))
        side = 0
        if blocked:
            U_new = xp.zeros((H, 2))
            lam, ess = float("nan"), float(K)
        else:
            w, lam, ess = self.weights(S)
            side = self._choose_side(w, c["side"])
            if side != 0:                                            # one mode only: drop the other side's samples
                keep = c["side"] * side > -p.side_m
                w_sub, lam, ess = self.weights(S[keep])
                w = xp.zeros_like(S)
                w[keep] = w_sub
            U_new = xp.einsum("k,kti->ti", w, V)
        Xp, _ = self.rollout(x0, v0, U_new[None])
        dbg = {}
        if self.debug_samples > 0:                                   # the best half by weight, the rest at random
            n = min(self.debug_samples, int(K if blocked else S.shape[0]))
            wv = _to_np(w) if not blocked else np.full(int(S.shape[0]), 1.0 / int(S.shape[0]))
            top = np.argsort(-wv)[: n // 2]
            rest = np.setdiff1d(np.arange(len(wv)), top)
            pick = np.r_[top, np.random.default_rng(0).choice(rest, n - len(top), replace=False)]
            dbg = {"samples": _to_np(X)[pick], "sample_w": wv[pick], "sample_crashed": _to_np(crashed)[pick],
                   "reference": _to_np(pr)}
        u = (float(U_new[0, 0]), float(U_new[0, 1]))
        self.u_last = np.array(u)
        if n_dead > 0:
            self._inflight = (self._inflight + [self.u_last.copy()])[-n_dead:]
        self.U = xp.concatenate([U_new[1:], U_new[-1:]], axis=0)    # warm start: shift by one step
        return MppiResult(u=u, s=self.s_now, e_lat=e_now, lam=lam, ess=float(ess),
                          best_cost=float(xp.min(S)), crash_frac=float(xp.mean(crashed.astype(float))),
                          side=side, blocked=blocked, min_d_plan=float(xp.min(self.clearance(Xp[0]))),
                          ms=(time.perf_counter() - t0) * 1e3, plan=_to_np(Xp[0]), **dbg)
