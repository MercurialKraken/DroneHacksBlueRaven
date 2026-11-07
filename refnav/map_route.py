"""The route's corners nudged into clear space on the course map, then smoothed by nav.path_geometry. Numpy only.

A corner moves only where the map says the airframe would pass closer than `need` to a surface, and never more
than `max_shift` from where the route put it, so the path stays the route's own; a straight leg that grazes a
wall gets one extra corner pushed off it. The result is a refnav.follow.Path in the mission's NED frame.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from nav.path_geometry import build_path

from . import follow as F

if TYPE_CHECKING:
    from collections.abc import Callable

    from nav.course_map import CourseMap


class MapFrame:
    """PX4 local NED (origin and north at the launch pose) to and from the course map frame (ENU)."""

    def __init__(self, launch: dict[str, float]) -> None:
        a = math.radians(launch["yaw_deg"] - 90.0)      # the course's rotation of the launch's odom ENU frame
        self.c, self.s, self.a = math.cos(a), math.sin(a), a
        self.x0, self.y0 = launch["x"], launch["y"]

    def to_map(self, n: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.c * e - self.s * n + self.x0, self.s * e + self.c * n + self.y0

    def to_ned(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx, dy = x - self.x0, y - self.y0
        return -self.s * dx + self.c * dy, self.c * dx + self.s * dy


def clear_corners(P: np.ndarray, cm: CourseMap, z: float, need: float, max_shift: float,
                  step: float = 0.01, iters: int = 80, passes: int = 3) -> np.ndarray:
    """Interior corners climb the map's distance field while closer than `need`, within max_shift of the route's;
    then each straight leg whose closest sample is under `need` gets a corner there, pushed off the same way."""
    def climb(pts: np.ndarray, fixed: np.ndarray, movable: np.ndarray) -> np.ndarray:
        pts = pts.copy()
        for _ in range(iters):
            d, g = cm.query(np.c_[pts, np.full(len(pts), z)])
            low = movable & np.isfinite(d) & (d < need)
            if not low.any():
                break
            gxy = g[low, :2] / np.maximum(np.linalg.norm(g[low, :2], axis=1, keepdims=True), 1e-9)
            pts[low] += step * gxy
            off = pts - fixed
            far = np.linalg.norm(off, axis=1) > max_shift
            pts[far] = fixed[far] + off[far] / np.linalg.norm(off[far], axis=1, keepdims=True) * max_shift
        return pts

    inner = np.r_[False, np.ones(len(P) - 2, bool), False]
    P = climb(P, P.copy(), inner)
    for _ in range(passes):                             # a leg can graze two surfaces: split it again until clear
        out = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            f = np.linspace(0.0, 1.0, max(2, math.ceil(np.hypot(*(b - a)) / 0.02)))[1:-1]
            if len(f):
                q = a + (b - a) * f[:, None]
                d = cm.distance(np.c_[q, np.full(len(q), z)])
                j = int(np.nanargmin(d)) if np.isfinite(d).any() else -1
                if j >= 0 and d[j] < need - 0.01:
                    out.append(climb(q[j:j + 1], q[j:j + 1].copy(), np.array([True]))[0])
            out.append(b)
        if len(out) == len(P):
            break
        P = np.array(out)
    return P


def builder(cm: CourseMap, launch: dict[str, float], z: float, drone_radius: float, margin: float,
            max_radius: float, max_shift: float) -> Callable[[list[tuple[float, float]]], F.Path]:
    """A path builder for refnav.core.Mission: NED route corners in, a NED follow.Path out."""
    frame = MapFrame(launch)
    need = drone_radius + margin

    def build(corners_ned: list[tuple[float, float]]) -> F.Path:
        C = np.asarray(corners_ned, float)
        xy = np.stack(frame.to_map(C[:, 0], C[:, 1]), 1)
        xy = clear_corners(xy, cm, z, need, max_shift)
        sp = build_path(xy, cm, drone_radius=drone_radius, margin=margin, z=z, max_radius=max_radius, ds=F.STEP_M)
        keep = np.r_[True, np.diff(sp.s) > 1e-6]               # arcs and legs share their joining sample
        s, k_enu = sp.s[keep], sp.k[keep]
        n, e = frame.to_ned(sp.xyz[keep, 0], sp.xyz[keep, 1])
        psi = F.wrap(np.pi / 2 - (sp.psi[keep] - frame.a))  # ENU heading (ccw from east) -> NED yaw (cw from north)
        k, sharp = F.soften_kinks(s, psi, -k_enu)            # left-positive -> right-positive
        return F.Path(s.copy(), n, e, psi, k, sharp)

    return build
