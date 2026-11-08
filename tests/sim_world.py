"""A tiny 2.5D world for map-free navigation tests: boxes with heights, a forward depth camera, a lag-model drone.

Not a physics sim: it checks the planning loop (depth -> grid -> MPPI -> commands) against ground truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

CELL = 0.02
WALL = 1.73          # walls are 68 in tall


@dataclass
class World:
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    boxes: list[tuple[float, float, float, float, float]] = field(default_factory=list)   # x0, x1, y0, y1, height
    polys: list[tuple[np.ndarray, float]] = field(default_factory=list)                    # convex polygon, height
    nets: list[tuple[float, float, float, float]] = field(default_factory=list)            # x0, x1, y0, y1: solid to
                                                                                           # the drone, invisible to depth

    def __post_init__(self) -> None:
        self.xs = np.arange(self.x_range[0], self.x_range[1] + 1e-9, CELL)
        self.ys = np.arange(self.y_range[0], self.y_range[1] + 1e-9, CELL)
        X, Y = np.meshgrid(self.xs, self.ys, indexing="ij")
        self.h = np.zeros(X.shape)
        for x0, x1, y0, y1, hgt in self.boxes:
            self.h[(X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)] = hgt
        for P, hgt in self.polys:
            inside = np.ones(X.shape, bool)
            for a, b in zip(P, np.roll(P, -1, axis=0)):
                inside &= (b[0] - a[0]) * (Y - a[1]) - (b[1] - a[1]) * (X - a[0]) >= 0.0
            self.h[inside] = np.maximum(self.h[inside], hgt)
        self.net = np.zeros(X.shape, bool)
        for x0, x1, y0, y1 in self.nets:
            self.net[(X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)] = True

    def height_at(self, P: np.ndarray) -> np.ndarray:
        i = np.rint((P[:, 0] - self.xs[0]) / CELL).astype(int)
        j = np.rint((P[:, 1] - self.ys[0]) / CELL).astype(int)
        out = (i < 0) | (i >= len(self.xs)) | (j < 0) | (j >= len(self.ys))
        h = np.full(len(P), 10.0)                                  # outside the world: a tall wall
        h[~out] = self.h[i[~out], j[~out]]
        return h

    def clearance(self, xy: np.ndarray, min_height: float = 0.0) -> np.ndarray:
        """True 2D distance from points to obstacles taller than min_height (nets included)."""
        xy = np.atleast_2d(xy)
        occ = (self.h > min_height) | self.net
        X, Y = np.meshgrid(self.xs, self.ys, indexing="ij")
        Q = np.c_[X[occ], Y[occ]]
        d = np.empty(len(xy))
        for k in range(0, len(xy), 256):
            blk = xy[k:k + 256]
            d[k:k + 256] = np.sqrt(((blk[:, None, :] - Q[None, :, :]) ** 2).sum(-1)).min(axis=1)
        return d - 0.5 * CELL

    def depth_points(self, cam: np.ndarray, yaw: float, hfov_deg: float = 72.0, vfov_deg: float = 55.0,
                     n_h: int = 200, n_v: int = 9, max_range: float = 5.0, noise_k: float = 0.0,
                     dropout: float = 0.0, outliers: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
        """Returns of a forward camera at cam (x, y, z) facing yaw: (N, 3) points (floor and obstacle hits).

        Stereo-like errors if asked: range noise with std noise_k * r^2 along the ray, a fraction `dropout` of returns
        missing, and a fraction `outliers` replaced by a random range along their ray. Nets are invisible."""
        az = yaw + np.radians(np.linspace(-hfov_deg / 2, hfov_deg / 2, n_h))
        el = np.radians(np.linspace(-vfov_deg / 2, vfov_deg / 2, n_v))
        A, E = np.meshgrid(az, el, indexing="ij")
        d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
        r = np.arange(0.1, max_range + 1e-9, CELL)
        P = cam[None, None, :] + r[None, :, None] * d[:, None, :]            # rays x steps x 3
        h = self.height_at(P.reshape(-1, 2 + 1)[:, :2]).reshape(P.shape[:2])
        hit = (P[..., 2] <= h) | (P[..., 2] <= 0.0)
        any_hit = hit.any(axis=1)
        k = np.argmax(hit, axis=1)
        rays = np.nonzero(any_hit)[0]
        rr = r[k[any_hit]]
        if noise_k or dropout or outliers:
            g = rng if rng is not None else np.random.default_rng(0)
            rr = rr + g.normal(0.0, 1.0, len(rr)) * noise_k * rr ** 2
            out = g.random(len(rr)) < outliers
            rr = np.where(out, g.uniform(0.2, max_range, len(rr)), rr)
            keep = (g.random(len(rr)) >= dropout) & (rr > 0.1)
            rays, rr = rays[keep], rr[keep]
        return cam[None, :] + rr[:, None] * d[rays]


    def depth_image(self, R_world_opt: np.ndarray, cam: np.ndarray, K: np.ndarray, w: int, h: int,
                    max_range: float = 5.0) -> np.ndarray:
        """A pinhole depth image (metres along the optical axis, NaN = no return) from cam, optical axes R_world_opt."""
        v, u = np.mgrid[0:h, 0:w]
        d_opt = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((h, w))], -1).reshape(-1, 3)
        d = d_opt @ R_world_opt.T
        r = np.arange(0.1, max_range + 1e-9, CELL)                          # along the optical axis
        P = cam[None, None, :] + r[None, :, None] * d[:, None, :]
        hit = (P[..., 2] <= self.height_at(P.reshape(-1, 3)[:, :2]).reshape(P.shape[:2])) | (P[..., 2] <= 0.0)
        z = np.where(hit.any(axis=1), r[np.argmax(hit, axis=1)], np.nan)
        return z.reshape(h, w)


def start_world(course_yaml: Path) -> World:
    """The rebuilt start from a course folder's course.yaml (rebuilt_start): wall A, the diagonal wall B with the CP2 gap
    between their ends (each wall's thickness on the side away from the gap), the cone (as a box), the barrier, and
    the nets (which depth can't see)."""
    rs = yaml.safe_load(Path(course_yaml).read_text())["rebuilt_start"]
    a, b, th = np.array(rs["wall_a"]), np.array(rs["wall_b"]), rs["wall_thickness"]
    polys = []
    for wall, gap_end in ((a, b[1]), (b, a[1])):
        u = (wall[1] - wall[0]) / np.hypot(*(wall[1] - wall[0]))
        n = np.array([-u[1], u[0]])
        n = -n if np.dot(gap_end - wall[1], n) > 0 else n                  # away from the gap
        P = np.array([wall[0], wall[1], wall[1] + n * th, wall[0] + n * th])
        e1, e2 = P[1] - P[0], P[2] - P[1]
        polys.append((P if e1[0] * e2[1] - e1[1] * e2[0] > 0 else P[::-1], rs["wall_height"]))   # counter-clockwise
    (cx, cy), half = rs["cone"], rs["cone_size"] / 2
    return World((-3.3, 1.2), (0.2, 3.3),
                 boxes=[(-3.048, 1.0, 0.41, 0.51, WALL),                   # the barrier
                        (cx - half, cx + half, cy - half, cy + half, rs["cone_height"])],
                 polys=polys, nets=[(-3.3, -3.048, 0.2, 3.3), (-3.3, 1.2, 3.048, 3.3)])


@dataclass
class Drone:
    x: np.ndarray
    yaw: float
    z: float = 0.7
    v: np.ndarray = field(default_factory=lambda: np.zeros(2))
    tau: float = 0.55
    yaw_rate_dps: float = 90.0

    def step(self, u: tuple[float, float], yaw_cmd: float, dt: float) -> None:
        a = math.exp(-dt / self.tau)
        u = np.asarray(u, float)
        self.x = self.x + u * dt + (self.v - u) * self.tau * (1 - a)
        self.v = u + (self.v - u) * a
        dy = (yaw_cmd - self.yaw + math.pi) % (2 * math.pi) - math.pi
        lim = math.radians(self.yaw_rate_dps) * dt
        self.yaw += float(np.clip(dy, -lim, lim))

    def cam(self) -> np.ndarray:
        return np.array([self.x[0] + 0.104 * math.cos(self.yaw), self.x[1] + 0.104 * math.sin(self.yaw), self.z])
