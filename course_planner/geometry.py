"""Course map loading and rasterisation.

Everything course-specific lives in the JSON map file. This module only knows
generic shapes: wall footprints (polygons), openings cut into walls, a tunnel
(two side shells along an axis), round boulders and the net boundary.
Frame: x = East, y = North (ENU), metres.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def load_map(path: str | Path) -> dict:
    with open(path) as f:
        m = json.load(f)
    if m.get("version") != 2:
        raise ValueError("expected course map version 2")
    return m


def points_in_polygon(xs: np.ndarray, ys: np.ndarray, poly) -> np.ndarray:
    """Even-odd ray casting, vectorised over points."""
    poly = np.asarray(poly, dtype=float)
    inside = np.zeros(xs.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        crosses = (y1 > ys) != (y2 > ys)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = x1 + (ys - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (xs < x_at)
    return inside


def segment_distance(xs, ys, a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = b - a
    t = np.clip(((xs - a[0]) * d[0] + (ys - a[1]) * d[1]) / max(d @ d, 1e-12), 0, 1)
    return np.hypot(xs - (a[0] + t * d[0]), ys - (a[1] + t * d[1]))


@dataclass
class Grid:
    res: float
    x0: float
    y0: float
    occ: np.ndarray            # True = obstacle (2-D, below the walls' tops)
    clearance: np.ndarray      # metres from each cell centre to the nearest obstacle

    @property
    def shape(self):
        return self.occ.shape

    def to_cell(self, x, y):
        return int(round((y - self.y0) / self.res)), int(round((x - self.x0) / self.res))

    def to_xy(self, r, c):
        return self.x0 + c * self.res, self.y0 + r * self.res

    def inside(self, r, c):
        return 0 <= r < self.shape[0] and 0 <= c < self.shape[1]

    def clearance_at(self, x, y):
        r, c = self.to_cell(x, y)
        if not self.inside(r, c):
            return 0.0
        return float(self.clearance[r, c])


def _distance_field(occ: np.ndarray, res: float) -> np.ndarray:
    try:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(~occ) * res
    except ImportError:  # small two-pass chamfer fallback, ~2 % error
        INF = 1e9
        d = np.where(occ, 0.0, INF)
        a, b = 1.0, 2 ** 0.5
        h, w = d.shape
        for r in range(h):
            for c in range(w):
                for dr, dc, k in ((-1, -1, b), (-1, 0, a), (-1, 1, b), (0, -1, a)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and d[rr, cc] + k < d[r, c]:
                        d[r, c] = d[rr, cc] + k
        for r in range(h - 1, -1, -1):
            for c in range(w - 1, -1, -1):
                for dr, dc, k in ((1, 1, b), (1, 0, a), (1, -1, b), (0, 1, a)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and d[rr, cc] + k < d[r, c]:
                        d[r, c] = d[rr, cc] + k
        return d * res


def rasterise(m: dict, res: float = 0.025, open_openings: tuple[str, ...] = ()) -> Grid:
    """Build the 2-D obstacle grid. Openings named in open_openings are carved out."""
    b = m["bounds"]
    x0, y0 = b["x_min"], b["y_min"]
    ncols = int(round((b["x_max"] - x0) / res)) + 1
    nrows = int(round((b["y_max"] - y0) / res)) + 1
    cols, rows = np.meshgrid(np.arange(ncols), np.arange(nrows))
    xs, ys = x0 + cols * res, y0 + rows * res
    occ = np.zeros((nrows, ncols), dtype=bool)

    # The net: the outermost ring of cells is an obstacle.
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True

    half = res * 0.75  # thin walls must still occupy at least one cell
    for wall in m["walls"]:
        poly = wall["footprint_xy"]
        hit = points_in_polygon(xs, ys, poly)
        for i in range(len(poly)):
            hit |= segment_distance(xs, ys, poly[i], poly[(i + 1) % len(poly)]) <= half
        occ |= hit

    for op in m.get("openings", []):
        if op["name"] not in open_openings:
            continue
        c = np.asarray(op["centre_xy"]); u = np.asarray(op["along_wall_unit"])
        along = (xs - c[0]) * u[0] + (ys - c[1]) * u[1]
        across = -(xs - c[0]) * u[1] + (ys - c[1]) * u[0]
        occ &= ~((np.abs(along) <= op["width_m"] / 2) & (np.abs(across) <= 0.15))

    t = m.get("tunnel")
    if t:
        a, e = np.asarray(t["entry_xy"]), np.asarray(t["exit_xy"])
        axis = (e - a) / np.linalg.norm(e - a)
        side = np.array([-axis[1], axis[0]]) * (t["inner_width_m"] / 2 + t["shell_thickness_m"] / 2)
        for s in (side, -side):
            occ |= segment_distance(xs, ys, a + s, e + s) <= max(t["shell_thickness_m"] / 2, half)

    for bo in m.get("boulders", []):
        occ |= np.hypot(xs - bo["centre_xy"][0], ys - bo["centre_xy"][1]) <= bo["radius_m"]

    return Grid(res, x0, y0, occ, _distance_field(occ, res))
