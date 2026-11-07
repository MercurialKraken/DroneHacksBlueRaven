"""Continuous path following. PURE (numpy only), no clock reads.

A smooth path (fillet arcs at corners, or a map-optimised curve) carries a speed profile. Each tick a reference point
slides along it at the profile speed, never past the space depth has observed free, and the nose faces a look point
further along. The lookahead grows with the speed the aircraft is making while on track and shrinks for turns,
cross-track error and estimator doubt; a heading error beyond face_tol slows the reference until the nose catches up.
Frame: the mission's NED, x north, y east, psi the NED yaw of the tangent, k signed curvature (+ right).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

STEP_M = 0.02           # path sample spacing
MIN_SPEED = 0.05        # the reference creeps at least this fast when allowed to move, so v=0 samples never stall it
PROJECT_BACK_M = 0.25   # the aircraft is matched to the path only this far behind its last match...
PROJECT_AHEAD_M = 0.25  # ...or ahead of the reference, so a U-turn never jumps to the parallel corridor
SPEED_GRID_M = 0.05     # the speed profile is solved on nodes this far apart
JERK_WEIGHT = 1.0       # flight time (s) traded for each (m/s3)^2 s of squared jerk
CUSP_DEG = 170.0        # a sharper corner is a reversal: no arc can turn it and still pass near the point
LEAD_M, LEAD_MAX_M = 0.03, 0.10  # the reference leads the aircraft by this much: further ahead, PX4's position gain
                                 # pulls straight at it, cutting corners and adding speed the profile never planned


def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


@dataclass(frozen=True)
class SE2:
    """A planar correction in NED: p' = R(yaw) p + (x, y), yaw positive clockwise from north like PX4's heading."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def vector(self, n: float, e: float) -> tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return c * n - s * e, s * n + c * e

    def point(self, n: float, e: float) -> tuple[float, float]:
        rn, re = self.vector(n, e)
        return rn + self.x, re + self.y

    def inverse(self) -> SE2:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return SE2(-(c * self.x + s * self.y), -(-s * self.x + c * self.y), -self.yaw)


@dataclass
class Path:
    s: np.ndarray               # distance along the path of each sample
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    k: np.ndarray
    clearance: np.ndarray | None = None   # distance to the nearest mapped surface at each sample, if a map built it

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def index(self, s):
        return np.clip(np.searchsorted(self.s, s), 0, len(self.s) - 1)

    def point(self, s: float) -> tuple[float, float]:
        i = self.index(s)
        return float(self.x[i]), float(self.y[i])

    def project(self, xy: tuple[float, float], s_lo: float, s_hi: float) -> tuple[float, float]:
        """(s, distance) of the closest sample within [s_lo, s_hi]."""
        a, b = int(self.index(s_lo)), int(self.index(s_hi)) + 1
        d = np.hypot(self.x[a:b] - xy[0], self.y[a:b] - xy[1])
        j = int(np.argmin(d))
        return float(self.s[a + j]), float(d[j])

    def downsample(self, step: float) -> list[list[float]]:
        every = max(1, round(step / STEP_M))
        return np.round(np.stack([self.x[::every], self.y[::every]], 1), 3).tolist()


def arclength(P: np.ndarray) -> np.ndarray:
    """Distance along the polyline P (N,2) at each of its points."""
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]


def resample(s0: np.ndarray, n: int, *values: np.ndarray) -> tuple[np.ndarray, ...]:
    """n evenly spaced distances over s0 and each of `values` interpolated there."""
    s = np.linspace(0.0, s0[-1], n)
    return (s, *(np.interp(s, s0, v) for v in values))


def fillet_path(corners: list[tuple[float, float]], radius: float, cut: float = math.inf) -> Path:
    """Straight legs joined by arcs of up to `radius`, each using at most half of either adjacent leg and passing within
    `cut` of its corner. A turn sharper than CUSP_DEG stops on its corner and turns back there (speed ~0)."""
    P = [np.asarray(c, float) for c in corners]
    P = [p for i, p in enumerate(P) if i == 0 or np.hypot(*(p - P[i - 1])) > 1e-6]
    if len(P) < 2:
        raise ValueError("a path needs two distinct points")
    pieces, start, k0 = [], P[0], 0.0           # ("line", a, b, k at a) | ("arc", centre, r, start_angle, sweep)
    for i in range(1, len(P) - 1):
        u1 = (P[i] - P[i - 1]) / np.hypot(*(P[i] - P[i - 1]))
        u2 = (P[i + 1] - P[i]) / np.hypot(*(P[i + 1] - P[i]))
        turn = math.atan2(u1[0] * u2[1] - u1[1] * u2[0], float(u1 @ u2))     # + = right turn in NED
        if abs(turn) < 1e-3:
            continue
        if abs(turn) > math.radians(CUSP_DEG):
            pieces.append(("line", start, P[i], k0))
            start, k0 = P[i], turn / STEP_M             # the whole turn within one sample: the profile stops there
            continue
        half = abs(turn) / 2.0
        room = 0.5 * min(np.hypot(*(P[i] - P[i - 1])), np.hypot(*(P[i + 1] - P[i])))
        r = min(radius, room / math.tan(half), cut * math.cos(half) / (1.0 - math.cos(half)))
        t = r * math.tan(half)
        a, b = P[i] - u1 * t, P[i] + u2 * t
        c = a + np.array([-u1[1], u1[0]]) * math.copysign(r, turn)          # the centre is on the inside of the turn
        pieces += [("line", start, a, k0), ("arc", c, r, math.atan2(*(a - c)[::-1]), turn)]
        start, k0 = b, 0.0
    pieces.append(("line", start, P[-1], k0))

    xs, ys, ps, ks, ds = [], [], [], [], []
    for pc in pieces:
        if pc[0] == "line":
            _, a, b, k_a = pc
            L = float(np.hypot(*(b - a)))
            if L > 1e-6:                                # two arcs sharing a short leg leave a zero-length line
                m = math.ceil(L / STEP_M)
                f = np.arange(m) / m
                if k_a:                                 # out of a cusp between the approach's samples: the aircraft
                    f = np.r_[0.0, f + 0.5 / m]         # projects onto the leg it is on, never ties with the way in
                xs.append(a[0] + (b[0] - a[0]) * f); ys.append(a[1] + (b[1] - a[1]) * f)
                ps.append(np.full(len(f), math.atan2(b[1] - a[1], b[0] - a[0])))
                ks.append(np.r_[k_a, np.zeros(len(f) - 1)])
                ds.append(np.diff(np.r_[f, 1.0]) * L if k_a else np.full(m, L / m))
        else:
            _, c, r, th0, sweep = pc
            m = math.ceil(abs(sweep) * r / STEP_M)
            th = th0 + sweep * np.arange(m) / m
            xs.append(c[0] + r * np.cos(th)); ys.append(c[1] + r * np.sin(th))
            ps.append(th + math.copysign(math.pi / 2, sweep)); ks.append(np.full(m, math.copysign(1.0 / r, sweep)))
            ds.append(np.full(m, abs(sweep) * r / m))
    xs.append([P[-1][0]]); ys.append([P[-1][1]]); ps.append([ps[-1][-1]]); ks.append([0.0])
    s = np.concatenate([[0.0], np.cumsum(np.concatenate(ds))])
    return Path(s, np.concatenate(xs), np.concatenate(ys), wrap(np.concatenate(ps)), np.concatenate(ks))


def polyline_path(x: np.ndarray, y: np.ndarray) -> Path:
    """A Path along a dense, smooth polyline: curvature of the circle through each three neighbours, heading along
    the chord across each point, resampled every STEP_M."""
    keep = np.r_[True, np.hypot(np.diff(x), np.diff(y)) > 1e-9]       # a repeated sample has no curvature
    x, y = np.asarray(x)[keep], np.asarray(y)[keep]
    if len(x) < 3:
        raise ValueError("a polyline path needs three distinct samples")
    P = np.stack([x, y], 1)
    a, b, c = P[:-2], P[1:-1], P[2:]
    cross = (b - a)[:, 0] * (c - b)[:, 1] - (b - a)[:, 1] * (c - b)[:, 0]
    k = 2.0 * cross / np.prod([np.hypot(*(q - p).T) for p, q in ((a, b), (b, c), (a, c))], axis=0)
    k = np.r_[k[0], k, k[-1]]
    d = np.vstack([P[1] - P[0], c - a, P[-1] - P[-2]])
    psi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    s0 = arclength(P)
    s, xs, ys, psis, ks = resample(s0, math.ceil(s0[-1] / STEP_M) + 1, x, y, psi, k)
    return Path(s, xs, ys, wrap(psis), ks)


def _accel_limit(u: np.ndarray, ds: float, accel: float) -> np.ndarray:
    """The largest v^2 profile under u that speeds up and brakes within accel: a backward then a forward pass."""
    u = u.copy()
    for i in range(len(u) - 2, -1, -1):
        u[i] = min(u[i], u[i + 1] + 2.0 * accel * ds)
    for i in range(1, len(u)):
        u[i] = min(u[i], u[i - 1] + 2.0 * accel * ds)
    return u


def speed_profile(path: Path, v_max: float, yaw_rate: float, a_lat: float, accel: float,
                  tight: tuple[float, float, float] | None = None) -> np.ndarray:
    """Speed at each sample: as fast as it can be with smooth acceleration, within v_max, the turn rate and lateral
    acceleration of the curvature, a crawl where the mapped clearance is short, accel along the path and 0 at both
    ends. tight = (clearance at which to crawl, at which full speed is allowed, crawl speed).

    On nodes SPEED_GRID_M apart u = v^2 solves, by ADMM,  min -g.u + JERK_WEIGHT * integral(jerk^2 dt)  within the
    bounds, where g = d(flight time)/du at the fastest feasible profile: slowing where it is already slow costs most."""
    k = np.abs(path.k)
    with np.errstate(divide="ignore"):
        v = np.minimum.reduce([np.full(len(k), v_max), yaw_rate / k, np.sqrt(a_lat / k)])
    if tight is not None and path.clearance is not None:
        lo, hi, crawl = tight
        f = np.clip((path.clearance - lo) / (hi - lo), 0.0, 1.0)
        v = np.minimum(v, np.where(np.isfinite(path.clearance), crawl + f * (v_max - crawl), v_max))

    n = max(3, math.ceil(path.length / SPEED_GRID_M) + 1)
    sg, h = np.linspace(0.0, path.length, n), path.length / (n - 1)
    cell = np.full(n - 1, np.inf)        # every sample bounds both nodes around it, so interpolating the nodes keeps it
    np.minimum.at(cell, np.clip(np.searchsorted(sg, path.s, "right") - 1, 0, n - 2), v ** 2)
    top = np.minimum(np.r_[cell, np.inf], np.r_[np.inf, cell])
    top[[0, -1]] = 0.0
    top = _accel_limit(top, h, accel)

    vf = np.maximum(np.sqrt(top), MIN_SPEED)
    g = h / (2.0 * vf ** 3)
    D2 = np.diff(np.eye(n), 2, axis=0) / h ** 2
    J = D2.T @ (D2 * (vf[1:-1] * h / 4.0)[:, None])    # jerk = v u''/2 and dt = ds/v: jerk^2 dt = v u''^2 ds / 4
    E = np.vstack([np.eye(n), np.diff(np.eye(n), axis=0)])     # u itself and its steps, each boxed
    lo = np.r_[np.zeros(n), np.full(n - 1, -2.0 * accel * h)]
    hi = np.r_[top, np.full(n - 1, 2.0 * accel * h)]
    rho = 30.0 * float(np.median(g))     # stiff: the fastest feasible profile it starts from is close to the answer
    solve = np.linalg.inv(2.0 * JERK_WEIGHT * J + rho * E.T @ E)
    z, y = E @ top, np.zeros(len(lo))
    for _ in range(300):
        Eu = E @ (solve @ (g + rho * E.T @ (z - y)))
        z = np.clip(Eu + y, lo, hi)
        y += Eu - z
    u = _accel_limit(z[:n], h, accel)     # ADMM ends within a residual of the step bounds: make them exact
    return np.sqrt(np.interp(path.s, sg, u))


@dataclass
class Limits:
    accel: float                # m/s2, the braking that keeps the reference able to stop inside observed space
    look_min: float             # lookahead at rest, m; it grows by look_s per m/s made while on track...
    look_s: float
    look_max: float             # ...up to this
    face_tol: float             # rad: heading error the reference still moves at full speed with
    yaw_slow: float             # rad: heading error at which forward speed reaches zero
    xtrack_slow: float          # m off the path where speed starts to drop...
    xtrack_stop: float          # ...and reaches zero
    arrive_m: float             # the route is complete within this of its last point


@dataclass
class Command:
    pos: tuple[float, float]        # reference point (n, e): where the aircraft should be now
    vel: tuple[float, float]        # feed-forward (vn, ve)
    acc: tuple[float, float]        # feed-forward (an, ae): speeding up along the path plus v^2 k towards the turn centre
    yaw: float                      # heading to hold (NED): towards the look point
    look: tuple[float, float]
    lead: float                     # how far ahead the look point is
    xtrack: float
    scale: float                    # 0..1: how much of the planned speed the aircraft is trusted with
    done: bool


class Follower:
    def __init__(self, path: Path, v: np.ndarray, lim: Limits, points: list[tuple[float, float]]) -> None:
        self.path, self.v, self.lim = path, v, lim
        self.s_ref = 0.0
        self.s_hat = 0.0                # the aircraft's projection on the path
        self.points_s: list[float] = []
        lo = 0.0
        for p in points:                # in route order, so a U-turn's far corridor is never matched
            lo = path.project(p, lo, path.length)[0]
            self.points_s.append(lo)

    @property
    def points_passed(self) -> int:
        return sum(1 for s in self.points_s if s <= self.s_hat + 2 * STEP_M)

    def ahead(self, length: float, step: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """Path samples from the aircraft's projection forward, always including the path end: (M,2) (n, e) and
        their distance from the projection."""
        P = self.path
        span = max(min(length, P.length - self.s_hat), 0.0)
        s = self.s_hat + np.linspace(0.0, span, max(2, math.ceil(span / step) + 1))
        i = P.index(s)
        return np.stack([P.x[i], P.y[i]], 1), s - self.s_hat

    def step(self, pos: tuple[float, float], vel: tuple[float, float], heading: float, dt: float,
             s_limit: float, doubt: float = 0.0) -> Command:
        """s_limit: the reference may not pass this path distance (observed free space). doubt in [0, 1]: estimator
        doubt, 1 = stop."""
        P, L = self.path, self.lim
        self.s_hat, xtrack = P.project(pos, self.s_hat - PROJECT_BACK_M, min(self.s_ref + PROJECT_AHEAD_M, P.length))
        trust = clamp01((L.xtrack_stop - xtrack) / (L.xtrack_stop - L.xtrack_slow)) * (1.0 - clamp01(doubt))
        made = min(math.hypot(*vel), float(self.v[P.index(self.s_hat)]))      # a coming turn lowers the plan here
        lead = min(L.look_max, L.look_min + L.look_s * made * trust)
        s_look = min(self.s_hat + lead, P.length)
        look = P.point(s_look)
        if math.hypot(look[0] - pos[0], look[1] - pos[1]) > L.look_min / 2:
            yaw = math.atan2(look[1] - pos[1], look[0] - pos[0])
        else:
            yaw = float(P.psi[P.index(s_look)])
        yaw_err = abs(float(wrap(yaw - heading)))

        scale = trust * clamp01(1.0 - (yaw_err - L.face_tol) / (L.yaw_slow - L.face_tol))
        v_cap = math.sqrt(2.0 * L.accel * max(s_limit - self.s_ref, 0.0)) if math.isfinite(s_limit) else math.inf
        i = int(P.index(self.s_ref))
        speed = min(max(float(self.v[i]), MIN_SPEED), v_cap) * scale
        s_new = min(self.s_ref + speed * dt, self.s_hat + LEAD_MAX_M, s_limit, P.length)
        # always ahead of the aircraft: a reference it overruns pulls it back and flies a loop
        ahead = self.s_hat + LEAD_M
        self.s_ref = max(self.s_ref, s_new, min(ahead, s_limit, P.length))

        i = int(P.index(self.s_ref))
        j = min(i + 1, len(self.v) - 1)
        capped = v_cap < float(self.v[i])
        vf = min(float(self.v[i]), v_cap) * scale
        ds = float(P.s[j] - P.s[i])
        a_t = 0.0 if capped or ds <= 1e-9 else vf * (float(self.v[j]) - float(self.v[i])) * scale / ds
        a_n = vf * vf * float(P.k[i])                  # + right: towards a right turn's centre
        c, sn = math.cos(P.psi[i]), math.sin(P.psi[i])
        ref = P.point(self.s_ref)
        done = self.s_ref >= P.length - 1e-6 and math.hypot(pos[0] - ref[0], pos[1] - ref[1]) < L.arrive_m
        return Command(ref, (vf * c, vf * sn), (a_t * c - a_n * sn, a_t * sn + a_n * c), yaw, look, lead, xtrack,
                       scale, done)
