"""Rolling 2D obstacle grid at flight height, built live from depth points. Map-free; pure numpy (scipy optional).

Frame: the odometry frame (cuVSLAM's, i.e. the take-off frame). The caller transforms each depth frame's points
into it with the drone's pose at the frame's capture time and passes the camera's position too.

Per depth frame:
1. Points in the obstacle band and min_range_m..max_range_m from the camera are obstacle HITS: their cells gain l_hit
   (log-odds), once per cell per frame. The band runs from floor_clear_m above the floor (so boulders and anything
   else the drone could fly over are obstacles to go round, not things the rangefinder bumps over) up to
   top_margin_m above the drone (so a tunnel's ceiling isn't a wall). Returns below the drone's own band
   (z - low_band_m) only count within low_range_m: far floor-adjacent returns are mostly stereo noise on the floor.
   floor_z is the odometry-frame floor height, taken at INIT from the landed drone.
2. For CLEARING, the in-band points are binned by bearing from the camera and the nearest per bin is a ray. Cells along
   the ray, short of the hit by hit_margin_cells, lose l_miss once per frame. An in-band return beyond max_range_m
   clears its ray to max_range_m (less the same margin). Out-of-band returns (floor, ceiling) clear nothing: a floor
   return says nothing about flight height. A 2D ray can pass OVER a low obstacle, so cells nearer than where the
   camera's lowest ray reaches the band's bottom ((z_cam - floor_z - floor_clear_m) / tan(vfov/2)) are never
   cleared: a boulder that drops out of view under the camera stays marked.
3. Log-odds are clamped to [l_min, l_max]. Cells the camera no longer sees keep their value, so side walls stay
   after they leave the field of view. mem_tau_s fades everything towards unknown, which bounds what odometry drift
   can do to old walls.

The window is size_m square, centred on the drone, and shifts by whole cells once the drone is recentre_m from its
centre (cells shifted in are unknown).

Output (grid2d): the signed clearance on the window for nav.mppi / nav.centerline. Occupied cells (log-odds >=
l_occ) are obstacles; unknown and free cells are both free space (nav.centerline penalises unknown separately, via
known()). Distances are reduced by half a cell's diagonal so rasterisation never makes a wall look further away.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from nav.mppi import Grid2D

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:                                   # pragma: no cover
    distance_transform_edt = None


@dataclass(frozen=True)
class LocalGridParams:
    size_m: float = 10.0
    cell: float = 0.05
    floor_z: float = 0.0              # odometry-frame floor height (set at INIT from the landed drone)
    floor_clear_m: float = 0.12       # obstacles start this far above the floor (boulders count)
    top_margin_m: float = 0.30        # ...and end this far above the drone (a tunnel ceiling doesn't)
    low_band_m: float = 0.25          # returns more than this below the drone ...
    low_range_m: float = 2.0          # ... only count this close (far floor returns are noisy)
    min_range_m: float = 0.25         # closer returns are ignored (props, the frame, stereo's minimum range)
    max_range_m: float = 4.0          # stereo depth error grows with range squared: no hits beyond this
    bearing_bin_deg: float = 1.0
    vfov_deg: float = 50.0            # camera vertical FOV, set a little NARROWER than the real one (OAK-D S2 ~55):
                                      # cells nearer than the lowest ray reaches the band's bottom are never
                                      # cleared, so a boulder that drops out of view under the camera stays marked
    l_hit: float = 0.85
    l_miss: float = -0.40
    l_min: float = -2.0
    l_max: float = 3.5
    l_occ: float = 0.6
    hit_margin_cells: int = 2
    recentre_m: float = 2.0
    mem_tau_s: float = 60.0
    max_clear_m: float = 3.0          # distances saturate here (also the no-obstacle value)


def _edt_cells(free: np.ndarray, max_cells: int) -> np.ndarray:
    """Distance in cells from each free cell to the nearest non-free one (0 on those)."""
    if distance_transform_edt is not None:
        return distance_transform_edt(free)
    # fallback without scipy: 8-neighbour chamfer propagation, saturating at max_cells
    d = np.where(free, float(max_cells), 0.0)
    s2 = math.sqrt(2.0)
    for _ in range(max_cells):
        p = np.pad(d, 1, constant_values=float(max_cells))
        nb = np.minimum.reduce([p[:-2, 1:-1] + 1, p[2:, 1:-1] + 1, p[1:-1, :-2] + 1, p[1:-1, 2:] + 1,
                                p[:-2, :-2] + s2, p[:-2, 2:] + s2, p[2:, :-2] + s2, p[2:, 2:] + s2])
        new = np.minimum(d, nb)
        if np.array_equal(new, d):
            break
        d = new
    return d


class LocalGrid:
    def __init__(self, params: LocalGridParams | None = None, centre: Any = (0.0, 0.0)) -> None:
        self.p = params or LocalGridParams()
        self.n = int(round(self.p.size_m / self.p.cell))
        self.L = np.zeros((self.n, self.n), dtype=np.float32)
        self.origin = np.zeros(2)                     # world position of cell (0, 0)'s centre
        self._place(np.asarray(centre, dtype=float)[:2])
        self.t_last: float | None = None
        self.frames = 0

    # ------------------------------------------------------------------ window
    def _place(self, centre: np.ndarray) -> None:
        half = 0.5 * (self.n - 1) * self.p.cell
        self.origin = np.round((centre - half) / self.p.cell) * self.p.cell

    @property
    def centre(self) -> np.ndarray:
        return self.origin + 0.5 * (self.n - 1) * self.p.cell

    def follow(self, xy: Any) -> None:
        """Shift the window (by whole cells) once the drone is recentre_m from its centre."""
        xy = np.asarray(xy, dtype=float)[:2]
        if np.max(np.abs(xy - self.centre)) < self.p.recentre_m:
            return
        old = self.origin.copy()
        self._place(xy)
        di, dj = (int(round(v)) for v in (self.origin - old) / self.p.cell)
        L = np.zeros_like(self.L)
        n = self.n
        si, sj = slice(max(di, 0), min(n + di, n)), slice(max(dj, 0), min(n + dj, n))
        ti, tj = slice(max(-di, 0), min(n - di, n)), slice(max(-dj, 0), min(n - dj, n))
        if si.start < si.stop and sj.start < sj.stop:
            L[ti, tj] = self.L[si, sj]
        self.L = L

    def cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ij = np.rint((xy - self.origin) / self.p.cell).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.n) & (ij[:, 1] >= 0) & (ij[:, 1] < self.n)
        return ij[:, 0], ij[:, 1], ok

    # ------------------------------------------------------------------ update
    def integrate(self, pts_odom: Any, sensor_xyz: Any, z_drone: float, t: float | None = None) -> dict[str, int]:
        """One depth frame: points (N, 3) and the camera's position, both in the odometry frame."""
        p = self.p
        if t is not None and self.t_last is not None and t > self.t_last and p.mem_tau_s > 0:
            self.L *= np.float32(math.exp(-(t - self.t_last) / p.mem_tau_s))
        if t is not None:
            self.t_last = t
        sensor = np.asarray(sensor_xyz, dtype=float)[:3]
        self.follow(sensor[:2])
        P = np.asarray(pts_odom, dtype=float).reshape(-1, 3)
        P = P[np.isfinite(P).all(axis=1)]
        rel = P[:, :2] - sensor[:2]
        rng = np.hypot(rel[:, 0], rel[:, 1])
        z = P[:, 2]
        band = (z >= p.floor_z + p.floor_clear_m) & (z <= z_drone + p.top_margin_m)
        low = z < z_drone - p.low_band_m
        keep = band & (rng >= p.min_range_m) & (~low | (rng <= p.low_range_m))
        P, rel, rng = P[keep], rel[keep], rng[keep]
        self.frames += 1
        if len(P) == 0:
            return {"hits": 0, "rays": 0}

        # clearing rays: the nearest in-band return per bearing bin
        bearing = np.arctan2(rel[:, 1], rel[:, 0])
        b = np.floor(bearing / math.radians(p.bearing_bin_deg)).astype(np.int64)
        order = np.lexsort((rng, b))
        first = np.r_[True, b[order][1:] != b[order][:-1]]
        ray_dir = rel[order][first] / rng[order][first][:, None]
        ray_len = np.minimum(rng[order][first], p.max_range_m) - p.hit_margin_cells * p.cell
        blind = max(sensor[2] - p.floor_z - p.floor_clear_m, 0.0) / math.tan(math.radians(p.vfov_deg) / 2.0)
        steps = np.arange(blind, max(float(ray_len.max()), 0.0) + 1e-9, 0.5 * p.cell)
        if len(steps) and len(ray_len):
            pts = sensor[None, None, :2] + ray_dir[:, None, :] * steps[None, :, None]
            use = steps[None, :] <= ray_len[:, None]
            i, j, ok = self.cells(pts[use])
            if ok.any():
                flat = np.unique(i[ok] * self.n + j[ok])
                self.L.ravel()[flat] += np.float32(p.l_miss)

        # hits: in-band returns within range, once per cell
        near = rng <= p.max_range_m
        i, j, ok = self.cells(P[near, :2])
        n_hits = 0
        if ok.any():
            flat = np.unique(i[ok] * self.n + j[ok])
            self.L.ravel()[flat] += np.float32(p.l_hit)
            n_hits = len(flat)
        np.clip(self.L, p.l_min, p.l_max, out=self.L)
        return {"hits": n_hits, "rays": int(len(ray_dir))}

    # ------------------------------------------------------------------ outputs
    def occupied(self) -> np.ndarray:
        return self.L >= self.p.l_occ

    def known(self) -> np.ndarray:
        return np.abs(self.L) > 1e-3

    def grid2d(self) -> Grid2D:
        p = self.p
        occ = self.occupied()
        max_cells = int(math.ceil(p.max_clear_m / p.cell))
        if not occ.any():
            d = np.full(occ.shape, p.max_clear_m)
        else:
            d = np.minimum(_edt_cells(~occ, max_cells) * p.cell, p.max_clear_m)
        d = d - 0.7072 * p.cell
        return Grid2D(d, (float(self.origin[0]), float(self.origin[1])), p.cell)
