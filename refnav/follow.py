"""Continuous path following. PURE (numpy only), no clock reads.

A smooth path (fillet arcs at corners) carries a speed profile; each tick a reference point slides
along it at the profile speed, never more than ref_lead_max_m ahead of the aircraft, and the nose
faces a look point further along. Sharp corners (no room for an arc) are a stop and turn in place.
Frame: the mission's NED, x north, y east, psi the NED yaw of the tangent, k signed curvature (+ right).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

STEP_M = 0.02


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Path:
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    k: np.ndarray               # inf at a sharp corner
    corners: list[float]        # s of every sharp corner

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def index(self, s: float) -> int:
        return int(min(max(round(s / STEP_M), 0), len(self.s) - 1))

    def point(self, s: float) -> tuple[float, float]:
        i = self.index(s)
        return float(self.x[i]), float(self.y[i])

    def project(self, xy: tuple[float, float], s_lo: float, s_hi: float) -> tuple[float, float]:
        """(s, distance) of the closest path point within [s_lo, s_hi], so a U-turn cannot jump corridors."""
        a, b = self.index(s_lo), self.index(s_hi) + 1
        d = np.hypot(self.x[a:b] - xy[0], self.y[a:b] - xy[1])
        j = int(np.argmin(d))
        return float(self.s[a + j]), float(d[j])


def fillet_path(corners: list[tuple[float, float]], radius: float) -> Path:
    """Straight legs joined by arcs of up to `radius`, each arc using at most half of either adjacent leg."""
    P = [np.asarray(c, float) for c in corners]
    P = [p for i, p in enumerate(P) if i == 0 or np.hypot(*(p - P[i - 1])) > 1e-6]
    if len(P) < 2:
        raise ValueError("a path needs two distinct points")
    pieces = []                 # ("line", a, b) | ("arc", centre, r, start_angle, sweep)
    start = P[0]
    sharp = []
    for i in range(1, len(P) - 1):
        u1 = (P[i] - P[i - 1]) / np.hypot(*(P[i] - P[i - 1]))
        u2 = (P[i + 1] - P[i]) / np.hypot(*(P[i + 1] - P[i]))
        turn = math.atan2(u1[0] * u2[1] - u1[1] * u2[0], float(u1 @ u2))     # + = right turn in NED
        if abs(turn) < 1e-3:
            continue
        room = 0.5 * min(np.hypot(*(P[i] - P[i - 1])), np.hypot(*(P[i + 1] - P[i])))
        r = min(radius, room / math.tan(abs(turn) / 2.0)) if abs(turn) < math.pi - 1e-3 else 0.0
        if r < 0.05:
            pieces.append(("line", start, P[i]))
            sharp.append(len(pieces))
            start = P[i]
            continue
        t = r * math.tan(abs(turn) / 2.0)
        a, b = P[i] - u1 * t, P[i] + u2 * t
        side = np.array([-u1[1], u1[0]]) * (1.0 if turn > 0 else -1.0)      # towards the centre
        c = a + side * r
        pieces.append(("line", start, a))
        pieces.append(("arc", c, r, math.atan2(*(a - c)[::-1]), turn))
        start = b
    pieces.append(("line", start, P[-1]))

    xs, ys, ps, ks, s_sharp, s0 = [], [], [], [], [], 0.0
    for n, pc in enumerate(pieces):
        if n in sharp:
            s_sharp.append(s0)
        if pc[0] == "line":
            _, a, b = pc
            L = float(np.hypot(*(b - a)))
            m = max(1, int(round(L / STEP_M)))
            f = np.arange(m) / m
            xs.append(a[0] + (b[0] - a[0]) * f); ys.append(a[1] + (b[1] - a[1]) * f)
            ps.append(np.full(m, math.atan2(b[1] - a[1], b[0] - a[0]))); ks.append(np.zeros(m))
        else:
            _, c, r, th0, sweep = pc
            L = abs(sweep) * r
            m = max(1, int(round(L / STEP_M)))
            th = th0 + sweep * np.arange(m) / m
            xs.append(c[0] + r * np.cos(th)); ys.append(c[1] + r * np.sin(th))
            ps.append(th + math.copysign(math.pi / 2, sweep)); ks.append(np.full(m, math.copysign(1.0 / r, sweep)))
        s0 += m * STEP_M
    xs.append([P[-1][0]]); ys.append([P[-1][1]]); ps.append([ps[-1][-1]]); ks.append([0.0])
    x, y = np.concatenate(xs), np.concatenate(ys)
    psi = np.array([_wrap(a) for a in np.concatenate(ps)])
    k = np.concatenate(ks)
    s = np.arange(len(x)) * STEP_M
    for sc in s_sharp:
        k[min(int(round(sc / STEP_M)), len(k) - 1)] = math.inf
    return Path(s, x, y, psi, k, s_sharp)


def speed_profile(path: Path, v_max: float, yaw_rate: float, a_lat: float, a_acc: float, a_brake: float) -> np.ndarray:
    """Fastest speed at each sample: turn rate and lateral acceleration on curves, zero at sharp corners and the end,
    then limited by braking (backward pass) and accelerating (forward pass)."""
    k = np.abs(path.k)
    with np.errstate(divide="ignore"):
        v = np.minimum.reduce([np.full(len(k), v_max), yaw_rate / k, np.sqrt(a_lat / k)])
    v[~np.isfinite(path.k)] = 0.0
    v[0] = v[-1] = 0.0
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a_brake * STEP_M))
    for i in range(1, len(v)):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * a_acc * STEP_M))
    return v


@dataclass
class Command:
    pos: tuple[float, float]        # reference point (n, e)
    vel: tuple[float, float]        # feed-forward (vn, ve)
    acc: tuple[float, float]        # feed-forward (an, ae): speeding up along the path plus v^2 k towards the turn centre
    yaw: float                      # heading to hold (NED)
    s_ref: float
    s_hat: float
    look: tuple[float, float]
    xtrack: float
    scale: float
    done: bool


class Follower:
    """Advances a reference along the path. Speed scales down with cross-track and heading error;
    the reference waits at a sharp corner until the nose faces the next leg."""

    def __init__(self, path: Path, v: np.ndarray, *, ref_lead_max_m: float, look_min_m: float, look_s: float,
                 look_max_m: float, face_tol_rad: float, xtrack_slow_m: float, xtrack_stop_m: float,
                 yaw_slow_rad: float, end_tol_m: float) -> None:
        self.path, self.v = path, v
        self.ref_lead_max_m, self.look_min_m, self.look_s, self.look_max_m = ref_lead_max_m, look_min_m, look_s, look_max_m
        self.face_tol, self.xtrack_slow, self.xtrack_stop, self.yaw_slow = face_tol_rad, xtrack_slow_m, xtrack_stop_m, yaw_slow_rad
        self.end_tol = end_tol_m
        self.s_ref = 0.0
        self.s_hat = 0.0

    def _next_sharp(self, s: float) -> float:
        return next((c for c in self.path.corners if c >= s - 1e-9), math.inf)

    def step(self, pos: tuple[float, float], heading: float, dt: float, scale_in: float = 1.0) -> Command:
        P = self.path
        self.s_hat, xtrack = P.project(pos, self.s_hat - 0.2, min(self.s_ref + 0.6, P.length))
        vi = float(self.v[P.index(self.s_ref)])
        look_s = min(self.s_ref + min(self.look_max_m, max(self.look_min_m, self.look_s * vi)), P.length)
        corner = self._next_sharp(self.s_ref)
        at_corner = corner - self.s_ref < STEP_M
        if at_corner:
            look_s = min(corner + self.look_min_m, P.length)        # nose leads into the next leg
        look = P.point(look_s)
        d = math.hypot(look[0] - pos[0], look[1] - pos[1])
        yaw = math.atan2(look[1] - pos[1], look[0] - pos[0]) if d > 0.15 else float(P.psi[P.index(look_s)])
        yaw_err = abs(_wrap(yaw - heading))

        scale = scale_in
        scale *= min(1.0, max(0.0, (self.xtrack_stop - xtrack) / max(self.xtrack_stop - self.xtrack_slow, 1e-6)))
        scale *= min(1.0, max(0.0, 1.0 - (yaw_err - self.face_tol) / max(self.yaw_slow - self.face_tol, 1e-6)))
        if at_corner and yaw_err > self.face_tol:
            scale = 0.0
        speed = max(vi, 0.05) * scale if scale > 0.0 else 0.0     # never stall at a v=0 corner sample once facing
        s_new = min(self.s_ref + speed * dt, self.s_hat + self.ref_lead_max_m, P.length)
        if not at_corner:
            s_new = min(s_new, corner)
        self.s_ref = max(self.s_ref, s_new)
        i = P.index(self.s_ref)
        vf = float(self.v[i]) * scale
        j = min(i + 1, len(self.v) - 1)
        a_t = vf * (float(self.v[j]) - float(self.v[i])) * scale / STEP_M if j > i else 0.0
        a_n = vf * vf * (float(P.k[i]) if math.isfinite(P.k[i]) else 0.0)      # + right: towards the centre of a right turn
        c, sn = math.cos(P.psi[i]), math.sin(P.psi[i])
        ref = P.point(self.s_ref)
        done = self.s_ref >= P.length - 1e-6 and math.hypot(pos[0] - ref[0], pos[1] - ref[1]) < self.end_tol
        return Command(ref, (vf * c, vf * sn), (a_t * c - a_n * sn, a_t * sn + a_n * c), yaw, self.s_ref, self.s_hat, look,
                       xtrack, scale, done)
