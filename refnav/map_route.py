"""The route as an elastic band on the course map: as straight as it can be while keeping clear of every mapped
surface and within max_shift of the route's own line, so it still passes the route's points. Numpy only.

The result is a refnav.follow.Path in the mission's NED frame, carrying its clearance for the speed profile.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from . import follow as F

if TYPE_CHECKING:
    from collections.abc import Callable

    from nav.course_map import CourseMap


class MapFrame:
    """PX4 local NED (origin and north at the launch pose) to and from the course map frame (ENU)."""

    def __init__(self, launch: dict[str, float]) -> None:
        a = math.radians(launch["yaw_deg"] - 90.0)      # the course's rotation of the launch's odom ENU frame
        self.c, self.s = math.cos(a), math.sin(a)
        self.x0, self.y0 = launch["x"], launch["y"]

    def to_map(self, n: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.c * e - self.s * n + self.x0, self.s * e + self.c * n + self.y0

    def to_ned(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx, dy = x - self.x0, y - self.y0
        return -self.s * dx + self.c * dy, self.c * dx + self.s * dy


DRONE_RADIUS_M = 0.28  # 7 in props: the airframe's reach from its centre
BAND_STEP_M = 0.05      # band point spacing
BAND_BEND = 30.0        # stiffness against bending, relative to the pull towards the constraints
BAND_ITERS = 300


def elastic_band(P: np.ndarray, cm: CourseMap, z: float, clear: float, max_shift: float) -> np.ndarray:
    """The route polyline P (map frame, end points fixed) bent as little as it can be (least squared second
    differences) while every point is at least `clear` from the map's surfaces, or on the corridor's centre where
    that is narrower, and within max_shift of its place on P. ADMM: an exact bending solve, then a projection."""
    s = F.arclength(P)
    _, hx, hy = F.resample(s, max(3, round(s[-1] / BAND_STEP_M) + 1), P[:, 0], P[:, 1])
    home = np.stack([hx, hy], 1)
    n, zs = len(home), np.full(len(home), z)
    D2 = np.diff(np.eye(n), 2, axis=0)
    A = np.eye(n) + BAND_BEND * D2.T @ D2
    A[[0, -1]] = np.eye(n)[[0, -1]]                     # the end points stay put
    bend = np.linalg.inv(A)

    def project(p: np.ndarray) -> np.ndarray:
        d, g = cm.query(np.c_[p, zs])
        g = np.nan_to_num(g[:, :2])
        need = np.nan_to_num(np.clip(clear - d, 0.0, None))
        # moving `need` up the gradient gains less than that once it crosses the corridor's centre: stop there
        rise = np.nan_to_num(cm.distance(np.c_[p + need[:, None] * g, zs]) - d)
        p = p + np.clip((need + rise) / 2.0, 0.0, None)[:, None] * g
        off = p - home
        return home + off * np.minimum(1.0, max_shift / np.maximum(np.hypot(*off.T), 1e-12))[:, None]

    q, u = home, np.zeros_like(home)                    # q: the projected band, u: the pull it has kept resisting
    for _ in range(BAND_ITERS):
        x = bend @ (q - u)
        q = project(x + u)
        u += x - q
    return x


def builder(cm: CourseMap, launch: dict[str, float], z: float, clear: float,
            max_shift: float) -> Callable[[list[tuple[float, float]]], F.Path]:
    """A path builder for refnav.core.Mission: NED route points in, a NED follow.Path out."""
    frame = MapFrame(launch)

    def build(points_ned: list[tuple[float, float]]) -> F.Path:
        C = np.asarray(points_ned, float)
        band = elastic_band(np.stack(frame.to_map(C[:, 0], C[:, 1]), 1), cm, z, clear, max_shift)
        path = F.polyline_path(*frame.to_ned(band[:, 0], band[:, 1]))
        x, y = frame.to_map(path.x, path.y)
        path.clearance = cm.distance(np.c_[x, y, np.full(len(x), z)])
        return path

    return build
