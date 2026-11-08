"""Smooth paths through corner points, with clearance from the course map. Runtime: numpy only.

The shared path format for the route guide (pure pursuit) and MPPI: both consume the same
SmoothPath, so a planner change shows up identically in either follower.

    path = smooth_path(corners_xyz, sdf=CombinedMap(course, live), need=0.38)
    path.length, path.s, path.xyz, path.psi, path.k, path.clearance       # sampled every ds
    s = path.closest_s(position, s_lo=s_prev - 0.2, s_hi=s_prev + 0.6)  # windowed projection
    path.point_at(s), path.heading_at(s), path.curvature_at(s)
    checkpoint_crossings(path, course_yaml["checkpoints"], drone_radius=0.29)

Construction:
  * Each interior corner becomes a circular arc (a fillet) tangent to both legs. Its radius is the
    largest that (1) fits: the tangent length r*tan(theta/2) is at most half of each adjacent leg
    (the whole leg at the route's two ends), (2) stays clear: every arc sample is at least `need`
    from any surface in `sdf`, and (3) is at most `max_radius` (or a per-corner override).
    Clearance is easy to prove on an arc, and curvature is constant along it.
  * A corner with no clear radius stays sharp: the path passes through the corner point, the heading
    jumps there, and its curvature is inf. A follower whose speed is yaw_rate_max / k stops and
    rotates in place, which is what a 1 m corridor's corner needs.
  * Heights: arcs stay at their corner's z; straight legs ramp between corner heights.
  * psi is the path tangent's heading, measured from +x (east) counter-clockwise, in radians.
    k is signed: positive for a left (counter-clockwise) turn.

The sdf is anything with distance(points, field) (CourseMap, CombinedMap). Without one, radii are
limited by geometry and max_radius only, and clearance is None.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Corner:
    index: int                  # index into the input points
    point: np.ndarray           # the corner point (x, y, z)
    turn_rad: float             # signed heading change, positive = left
    radius: float               # fillet radius used (0 = sharp)
    limit: str                  # what limited it: "straight", "max", "leg", "clearance", "override"
    s: float                    # arc length at the middle of the turn


@dataclass
class SmoothPath:
    xyz: np.ndarray             # (M, 3) samples
    s: np.ndarray               # (M,) arc length, increasing
    psi: np.ndarray             # (M,) heading, wrapped to (-pi, pi]
    k: np.ndarray               # (M,) signed curvature (inf at a sharp corner)
    clearance: np.ndarray | None  # (M,) sdf distance at each sample, or None
    corners: list[Corner] = field(default_factory=list)
    _psi_u: np.ndarray | None = None   # unwrapped heading, for interpolation

    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def x(self) -> np.ndarray:
        return self.xyz[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.xyz[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.xyz[:, 2]

    @property
    def radii(self) -> list[float]:
        """Fillet radius per interior corner, in route order (0 = sharp: rotate in place)."""
        return [c.radius for c in self.corners]

    def sample(self, s: Any) -> dict[str, np.ndarray]:
        """{x, y, z, psi, k} at arc length(s) s."""
        p = self.point_at(s)
        return {"x": p[..., 0], "y": p[..., 1], "z": p[..., 2], "psi": self.heading_at(s), "k": self.curvature_at(s)}

    def project(self, xy: Any, s_lo: float = -math.inf, s_hi: float = math.inf) -> float:
        """Same as closest_s: the windowed projection of a position onto the path."""
        return self.closest_s(xy, s_lo, s_hi)

    def point_at(self, s: Any) -> np.ndarray:
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        return np.stack([np.interp(s, self.s, self.xyz[:, a]) for a in range(3)], axis=-1)

    def heading_at(self, s: Any) -> np.ndarray:
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        return _wrap(np.interp(s, self.s, self._psi_u))

    def curvature_at(self, s: Any) -> np.ndarray:
        """Curvature of the nearest sample (inf marks a sharp corner)."""
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        i = np.clip(np.searchsorted(self.s, s), 0, len(self.s) - 1)
        return self.k[i]

    def closest_s(self, p: Any, s_lo: float = -math.inf, s_hi: float = math.inf) -> float:
        """Arc length of the path point nearest to p (x, y[, z] - judged in x/y), searching only
        s in [s_lo, s_hi] so a U-turn can't jump to the other corridor."""
        p = np.asarray(p, dtype=float).reshape(-1)[:2]
        lo = int(np.searchsorted(self.s, max(s_lo, 0.0), side="left"))
        hi = int(np.searchsorted(self.s, min(s_hi, self.length), side="right"))
        if hi <= lo:
            return float(np.clip(s_lo, 0.0, self.length))
        seg = self.xyz[lo:hi, :2]
        i = lo + int(np.argmin(np.sum((seg - p) ** 2, axis=1)))
        best_s, best_d = float(self.s[i]), math.inf
        for j in (i - 1, i):                                 # refine on the two segments around it
            if j < max(lo, 0) or j + 1 >= min(hi + 1, len(self.s)):
                continue
            a, b = self.xyz[j, :2], self.xyz[j + 1, :2]
            ab = b - a
            L2 = float(ab @ ab)
            t = 0.0 if L2 == 0 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
            d = float(np.sum((a + t * ab - p) ** 2))
            if d < best_d:
                best_d, best_s = d, float(self.s[j] + t * (self.s[j + 1] - self.s[j]))
        return best_s


def _wrap(a: Any) -> Any:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _fillet(P: np.ndarray, a: np.ndarray, b: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Arc of radius r at corner P (2D) from direction a to direction b (unit vectors).
    Returns (start, end, centre, start angle, signed sweep)."""
    turn = math.atan2(a[0] * b[1] - a[1] * b[0], float(a @ b))
    T = r * math.tan(abs(turn) / 2)
    A, B = P - a * T, P + b * T
    n = np.array([-a[1], a[0]]) * (1.0 if turn > 0 else -1.0)   # towards the centre
    C = A + n * r
    return A, B, C, math.atan2(A[1] - C[1], A[0] - C[0]), turn


def _arc_points(C: np.ndarray, r: float, ang0: float, sweep: float, n: int) -> np.ndarray:
    t = ang0 + sweep * np.linspace(0.0, 1.0, n)
    return np.stack([C[0] + r * np.cos(t), C[1] + r * np.sin(t)], axis=1)


def max_fillet_radius(P: np.ndarray, a: np.ndarray, b: np.ndarray, z: float, room: float, sdf: Any,
                      need: float, cap: float, field_name: str = "planning", step: float = 0.01) -> tuple[float, str]:
    """Largest radius <= min(cap, room limit) whose arc keeps `need` clearance in `sdf`.
    `room`: the most tangent length the adjacent legs allow."""
    turn = math.atan2(a[0] * b[1] - a[1] * b[0], float(a @ b))
    if abs(turn) < 1e-6:
        return 0.0, "straight"
    r_room = room / math.tan(abs(turn) / 2) if abs(turn) < math.pi - 1e-6 else 0.0
    r_max, limit = (cap, "max") if cap <= r_room else (r_room, "leg")
    if sdf is None or r_max <= 0:
        return max(r_max, 0.0), limit
    r = r_max
    while r > 1e-9:                                          # largest radius whose own arc is clear
        A, B, C, ang0, sweep = _fillet(P, a, b, r)
        n = max(3, int(math.ceil(abs(sweep) * r / 0.02)) + 1)
        pts = _arc_points(C, r, ang0, sweep, n)
        d = sdf.distance(np.column_stack([pts, np.full(len(pts), z)]), field_name)
        if np.all(np.isfinite(d) & (d >= need)):
            return r, limit
        limit = "clearance"
        r -= step
    return 0.0, "clearance"


def smooth_path(points: Any, sdf: Any = None, need: float = 0.38, max_radius: float = 1.0, ds: float = 0.02,
                radius_overrides: dict[int, float] | None = None, field_name: str = "planning") -> SmoothPath:
    """Points: (N, 3) corner points (x, y, z) in `map`, N >= 2. See the module docstring."""
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 2:
        raise ValueError("points must be (N >= 2, 3)")
    keep = np.r_[True, np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1) > 1e-9]
    P = P[keep]                                              # drop repeated points
    if len(P) < 2:
        raise ValueError("the route has no length")
    overrides = radius_overrides or {}
    N = len(P)
    leg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
    dirs = [_unit(P[i + 1, :2] - P[i, :2]) for i in range(N - 1)]

    # corner arcs: (A, B, C, ang0, sweep, r) per interior corner; None = sharp or straight
    arcs: list[tuple | None] = [None] * N
    corners: list[Corner] = []
    for i in range(1, N - 1):
        a, b = dirs[i - 1], dirs[i]
        room = min(leg[i - 1] * (1.0 if i == 1 else 0.5), leg[i] * (1.0 if i == N - 2 else 0.5))
        if i in overrides:
            r, limit = min(float(overrides[i]), max_fillet_radius(P[i, :2], a, b, P[i, 2], room, None, need,
                                                                  math.inf)[0]), "override"
        else:
            r, limit = max_fillet_radius(P[i, :2], a, b, P[i, 2], room, sdf, need, max_radius, field_name)
        turn = math.atan2(a[0] * b[1] - a[1] * b[0], float(a @ b))
        if r > 1e-9 and abs(turn) > 1e-6:
            arcs[i] = (*_fillet(P[i, :2], a, b, r), r)
        corners.append(Corner(i, P[i].copy(), turn, r if arcs[i] else 0.0, limit, 0.0))

    xy, z, psi, k = [], [], [], []

    def add_line(p0: np.ndarray, p1: np.ndarray, z0: float, z1: float, include_start: bool) -> None:
        L = float(np.linalg.norm(p1 - p0))
        n = max(1, int(math.ceil(L / ds)))
        t = np.linspace(0.0, 1.0, n + 1)[0 if include_start else 1:]
        h = math.atan2(p1[1] - p0[1], p1[0] - p0[0]) if L > 0 else (psi[-1] if psi else 0.0)
        xy.extend(p0 + np.outer(t, p1 - p0))
        z.extend(z0 + t * (z1 - z0))
        psi.extend([h] * len(t))
        k.extend([0.0] * len(t))

    def add_arc(C: np.ndarray, r: float, ang0: float, sweep: float, zc: float) -> None:
        n = max(2, int(math.ceil(abs(sweep) * r / ds)))
        t = np.linspace(0.0, 1.0, n + 1)[1:]
        ang = ang0 + sweep * t
        xy.extend(np.stack([C[0] + r * np.cos(ang), C[1] + r * np.sin(ang)], axis=1))
        z.extend([zc] * len(t))
        psi.extend(ang + math.copysign(math.pi / 2, sweep))
        k.extend([math.copysign(1.0 / r, sweep)] * len(t))

    cur, cur_z = P[0, :2], P[0, 2]
    first = True
    for i in range(1, N):
        if i < N - 1 and arcs[i] is not None:
            A, B, C, ang0, sweep, r = arcs[i]
            add_line(cur, A, cur_z, P[i, 2], include_start=first)
            add_arc(C, r, ang0, sweep, P[i, 2])
            cur, cur_z = B, P[i, 2]
        else:
            add_line(cur, P[i, :2], cur_z, P[i, 2], include_start=first)
            if 0 < i < N - 1:                               # a sharp corner: rotate in place here
                k[-1] = math.inf
            cur, cur_z = P[i, :2], P[i, 2]
        first = False

    xy_a = np.asarray(xy)
    xyz = np.column_stack([xy_a, np.asarray(z)])
    step = np.linalg.norm(np.diff(xy_a, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(step)]
    psi_a = np.asarray(psi)
    # a sharp corner's sample still carries the incoming heading; the next sample the outgoing one
    psi_u = np.unwrap(psi_a)
    path = SmoothPath(xyz=xyz, s=s, psi=_wrap(psi_a), k=np.asarray(k, dtype=float),
                      clearance=None if sdf is None else np.asarray(sdf.distance(xyz, field_name), dtype=float),
                      corners=corners, _psi_u=psi_u)
    for c in corners:                                        # arc length at the middle of each turn
        c.s = path.closest_s(c.point if c.radius == 0 else _turn_mid(arcs[c.index]))
    return path


def build_path(corners: Any, course_map: Any = None, drone_radius: float = 0.29, margin: float = 0.10,
               z: float = 1.0, max_radius: float = 1.0, ds: float = 0.02,
               radius_overrides: dict[int, float] | None = None) -> SmoothPath:
    """The route guide's entry point. corners: (N, 2) x, y at height z, or (N, 3) with their own
    heights, in the course frame. Clearance needed = drone_radius + margin."""
    C = np.asarray(corners, dtype=float)
    if C.ndim != 2 or C.shape[1] not in (2, 3):
        raise ValueError("corners must be (N, 2) or (N, 3)")
    if C.shape[1] == 2:
        C = np.column_stack([C, np.full(len(C), float(z))])
    return smooth_path(C, sdf=course_map, need=drone_radius + margin, max_radius=max_radius, ds=ds,
                       radius_overrides=radius_overrides)


def _turn_mid(arc: tuple) -> np.ndarray:
    A, B, C, ang0, sweep, r = arc
    a = ang0 + sweep / 2
    return np.array([C[0] + r * math.cos(a), C[1] + r * math.sin(a)])


def checkpoint_crossings(path: SmoothPath, checkpoints: list[dict[str, Any]], drone_radius: float) -> list[dict[str, Any]]:
    """For each checkpoint line in course.yaml form ({id, line, direction} or {id, options: [...]}):
    how far past the line the path's centre gets (within the line's extent), and whether the whole
    drone crosses it (that distance >= drone_radius). Circles (the pad) are reported by the
    closest approach of the path's end to the centre."""
    out = []
    for cp in checkpoints:
        for opt in cp.get("options", [cp]):
            name = cp.get("id", "?") + (f" {opt['name']}" if opt is not cp else "")
            if "line" in opt:
                (x0, y0), (x1, y1) = opt["line"]
                d = _unit(np.asarray(opt["direction"], dtype=float))
                a = np.array([x0, y0])
                along = _unit(np.array([x1 - x0, y1 - y0]))
                L = math.hypot(x1 - x0, y1 - y0)
                rel = path.xyz[:, :2] - a
                u = rel @ along                               # position along the line
                past = rel @ d                                # distance past it, in the travel direction
                within = (u >= -drone_radius) & (u <= L + drone_radius)
                beyond = float(past[within].max()) if within.any() else -math.inf
                out.append({"checkpoint": name, "past_m": round(beyond, 3),
                            "whole_drone_across": bool(beyond >= drone_radius)})
            elif "centre" in opt and "outer_radius" in opt:
                c = np.asarray(opt["centre"], dtype=float)
                d_end = float(np.linalg.norm(path.xyz[-1, :2] - c))
                out.append({"checkpoint": name, "end_from_centre_m": round(d_end, 3),
                            "whole_drone_inside": bool(d_end + drone_radius <= float(opt["outer_radius"]))})
    return out
