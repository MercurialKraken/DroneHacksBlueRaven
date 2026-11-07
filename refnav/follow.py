"""Continuous path following. PURE (numpy only), no clock reads.

A smooth path (fillet arcs at corners) carries a speed profile. Each tick a reference point slides along it at the
profile speed, never past the space depth has observed free, and the nose faces a look point further along. The
lookahead grows with the speed the aircraft is making while on track and shrinks for turns, cross-track error and
estimator doubt. Sharp corners (no room for an arc) are a stop and turn in place.
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
    k: np.ndarray               # inf at a sharp corner
    corners: list[float]        # s of every sharp corner

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


def fillet_path(corners: list[tuple[float, float]], radius: float, min_radius: float = 0.05) -> Path:
    """Straight legs joined by arcs of up to `radius`, each using at most half of either adjacent leg. A corner with
    no room for an arc of min_radius stays sharp."""
    P = [np.asarray(c, float) for c in corners]
    P = [p for i, p in enumerate(P) if i == 0 or np.hypot(*(p - P[i - 1])) > 1e-6]
    if len(P) < 2:
        raise ValueError("a path needs two distinct points")
    pieces, start, sharp_after = [], P[0], set()        # ("line", a, b) | ("arc", centre, r, start_angle, sweep)
    for i in range(1, len(P) - 1):
        u1 = (P[i] - P[i - 1]) / np.hypot(*(P[i] - P[i - 1]))
        u2 = (P[i + 1] - P[i]) / np.hypot(*(P[i + 1] - P[i]))
        turn = math.atan2(u1[0] * u2[1] - u1[1] * u2[0], float(u1 @ u2))     # + = right turn in NED
        if abs(turn) < 1e-3:
            continue
        room = 0.5 * min(np.hypot(*(P[i] - P[i - 1])), np.hypot(*(P[i + 1] - P[i])))
        r = min(radius, room / math.tan(abs(turn) / 2.0)) if abs(turn) < math.pi - 1e-3 else 0.0
        if r < min_radius:
            pieces.append(("line", start, P[i]))
            sharp_after.add(len(pieces) - 1)
            start = P[i]
            continue
        t = r * math.tan(abs(turn) / 2.0)
        a, b = P[i] - u1 * t, P[i] + u2 * t
        c = a + np.array([-u1[1], u1[0]]) * math.copysign(r, turn)          # the centre is on the inside of the turn
        pieces += [("line", start, a), ("arc", c, r, math.atan2(*(a - c)[::-1]), turn)]
        start = b
    pieces.append(("line", start, P[-1]))

    xs, ys, ps, ks, ds, sharp_n = [], [], [], [], [], []
    for n, pc in enumerate(pieces):
        if pc[0] == "line":
            _, a, b = pc
            L = float(np.hypot(*(b - a)))
            if L > 1e-6:                                # two arcs sharing a short leg leave a zero-length line
                m = math.ceil(L / STEP_M)
                f = np.arange(m) / m
                xs.append(a[0] + (b[0] - a[0]) * f); ys.append(a[1] + (b[1] - a[1]) * f)
                ps.append(np.full(m, math.atan2(b[1] - a[1], b[0] - a[0]))); ks.append(np.zeros(m))
                ds.append(np.full(m, L / m))
        else:
            _, c, r, th0, sweep = pc
            m = math.ceil(abs(sweep) * r / STEP_M)
            th = th0 + sweep * np.arange(m) / m
            xs.append(c[0] + r * np.cos(th)); ys.append(c[1] + r * np.sin(th))
            ps.append(th + math.copysign(math.pi / 2, sweep)); ks.append(np.full(m, math.copysign(1.0 / r, sweep)))
            ds.append(np.full(m, abs(sweep) * r / m))
        if n in sharp_after:
            sharp_n.append(sum(len(d) for d in ds))     # the sample at the corner point starts the next piece
    xs.append([P[-1][0]]); ys.append([P[-1][1]]); ps.append([ps[-1][-1]]); ks.append([0.0])
    s = np.concatenate([[0.0], np.cumsum(np.concatenate(ds))])
    k = np.concatenate(ks)
    k[[min(i, len(k) - 1) for i in sharp_n]] = math.inf
    return Path(s, np.concatenate(xs), np.concatenate(ys), wrap(np.concatenate(ps)), k, [float(s[i]) for i in sharp_n])


def speed_profile(path: Path, v_max: float, yaw_rate: float, a_lat: float, a_acc: float, a_brake: float) -> np.ndarray:
    """Fastest speed at each sample: turn rate and lateral acceleration on curves, zero at sharp corners and the ends,
    then limited by braking (backward pass) and speeding up (forward pass)."""
    k = np.abs(path.k)
    with np.errstate(divide="ignore"):
        v = np.minimum.reduce([np.full(len(k), v_max), yaw_rate / k, np.sqrt(a_lat / k)])
    v[0] = v[-1] = 0.0
    ds = np.diff(path.s)
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a_brake * ds[i]))
    for i in range(1, len(v)):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * a_acc * ds[i - 1]))
    return v


@dataclass
class Limits:
    accel: float                # m/s2, the braking that keeps the reference able to stop inside observed space
    look_min: float             # lookahead at rest, m; it grows by look_s per m/s made while on track...
    look_s: float
    look_max: float             # ...up to this
    face_tol: float             # rad: a sharp corner is left only once the nose is this close to the next leg
    yaw_slow: float             # rad: heading error at which forward speed reaches zero
    xtrack_slow: float          # m off the path where speed starts to drop...
    xtrack_stop: float          # ...and reaches zero
    end_tol: float              # m: the route is complete within this of its last point


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

    def next_sharp(self, s: float) -> float:
        return next((c for c in self.path.corners if c >= s - 1e-9), math.inf)

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
        corner = self.next_sharp(self.s_ref)
        at_corner = corner - self.s_ref < STEP_M
        s_look = min(corner + L.look_min if at_corner else self.s_hat + lead, P.length)
        look = P.point(s_look)
        if math.hypot(look[0] - pos[0], look[1] - pos[1]) > L.look_min / 2:
            yaw = math.atan2(look[1] - pos[1], look[0] - pos[0])
        else:
            yaw = float(P.psi[P.index(s_look)])
        yaw_err = abs(float(wrap(yaw - heading)))

        scale = trust * clamp01(1.0 - (yaw_err - L.face_tol) / (L.yaw_slow - L.face_tol))
        if at_corner and yaw_err > L.face_tol:
            scale = 0.0
        v_cap = math.sqrt(2.0 * L.accel * max(s_limit - self.s_ref, 0.0)) if math.isfinite(s_limit) else math.inf
        i = int(P.index(self.s_ref))
        speed = min(max(float(self.v[i]), MIN_SPEED), v_cap) * scale
        gate = math.inf if at_corner and scale > 0.0 else corner      # a sharp corner is passed only once faced
        s_new = min(self.s_ref + speed * dt, self.s_hat + LEAD_MAX_M, s_limit, P.length, gate)
        # always ahead of the aircraft: a reference it overruns pulls it back and flies a loop
        ahead = self.s_hat + LEAD_M
        self.s_ref = max(self.s_ref, s_new, min(ahead, s_limit, P.length, gate))

        i = int(P.index(self.s_ref))
        j = min(i + 1, len(self.v) - 1)
        capped = v_cap < float(self.v[i])
        vf = min(float(self.v[i]), v_cap) * scale
        a_t = 0.0 if capped or j == i else vf * (float(self.v[j]) - float(self.v[i])) * scale / float(P.s[j] - P.s[i])
        a_n = vf * vf * (float(P.k[i]) if math.isfinite(P.k[i]) else 0.0)      # + right: towards a right turn's centre
        c, sn = math.cos(P.psi[i]), math.sin(P.psi[i])
        ref = P.point(self.s_ref)
        done = self.s_ref >= P.length - 1e-6 and math.hypot(pos[0] - ref[0], pos[1] - ref[1]) < L.end_tol
        return Command(ref, (vf * c, vf * sn), (a_t * c - a_n * sn, a_t * sn + a_n * c), yaw, look, lead, xtrack,
                       scale, done)
