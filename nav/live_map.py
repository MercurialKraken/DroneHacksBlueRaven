"""Live obstacles from depth, combined with the static course map. Runtime: numpy only.

    course = CourseMap.load(course_dir() / "course_sdf.npz")
    live = LiveObstacleLayer(course)                    # same 5 cm grid as the course map
    live.integrate(depth_m, K, T_map_cam)               # every depth frame, in the `map` frame
    cmap = CombinedMap(course, live)                    # drop-in for CourseMap in a planner
    d, g = cmap.query(points)                           # min(static planning field, live obstacles)
    ok = cmap.clearance_ok(setpoints, need=0.38)        # finite and at least `need` from anything
    seen = live.observed(points)                        # has depth actually looked at this space?

What the live layer records, and why:
  * Only returns the static map does NOT explain. A depth point within `explain_tol_m` of a mapped
    surface is the known course (walls, net, floor) and adds nothing; anything else (a wall that
    moved, a prop, a person inside the net) is evidence of an unmapped obstacle. In a well-aligned
    run that set stays small, which keeps the layer cheap.
  * A voxel becomes an obstacle after `min_hits` hits in separate frames, so a one-frame stereo
    phantom does not stop the drone. Rays that pass through a voxel count against it, so a phantom
    or a person who walked away is cleared again.
  * Before projection, depth goes through a small min filter (`min_filter_px`). Scattered returns
    on net cords grow into a surface, while pixels with no return anywhere in the window stay
    invalid.
  * No return is UNKNOWN, never free: invalid pixels carve nothing. `observed()` says which voxels
    any ray has reached; the distance query does not inflate unknown space (the static map is the
    prior there), so a planner that must not fly into unseen space asks `observed()` as well.

Frames: `map` is the course file's frame (x east, y north, z up, metres; see courses/<name>/course.yaml).
T_map_cam is a 4x4 transform from the camera's OPTICAL frame (x right, y down, z forward) to map.
Depth is metres along the optical axis; 0, negative or non-finite means no return.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .course_map import CourseMap


@dataclass
class LiveParams:
    explain_tol_m: float = 0.12        # |static distance| below this: the return is a mapped surface
    min_range_m: float = 0.30          # closer returns are unreliable (stereo minimum range ~0.40)
    max_range_m: float = 3.50          # sim depth is coarse beyond ~2 m (0.5 m steps at 3 m)
    depth_scale: float = 1.0           # multiply depth by this (sim depth reads ~1.5 % long: 0.985)
    min_filter_px: int = 5             # odd window of the depth min filter; 1 disables it
    hit_stride_px: int = 4             # use every n-th pixel for hits
    free_stride_px: int = 8            # and for carving free space: at fx ~400 px, rays stay under a
                                       # voxel apart to ~2.5 m (96 % of stride 4's voxels, 3x faster)
    min_hits: int = 2                  # frames with a hit before a voxel counts as an obstacle
    miss_weight: float = 0.5           # each frame a ray passes through cancels this many hits
    truncate_m: float = 0.80           # live distances are capped here (the static map covers beyond)
    max_obstacle_voxels: int = 4000    # above this the map is probably misaligned: refuse, don't stall
    # Pose check per frame. With a good pose, returns near a mapped surface land ON it. If more than
    # max_misfit of the returns within misfit_band_m of a surface miss it by over explain_tol_m, the
    # pose disagrees with the map (a VIO jump) and the frame is refused instead of drawing real walls
    # in the wrong place. Judged only when at least misfit_min_points such returns exist.
    misfit_band_m: float = 0.50
    max_misfit: float = 0.50
    misfit_min_points: int = 200
    # Misfit is judged against WALLS only (wall_field, the course map's 'localization' field): floor
    # returns stay "on the floor" whatever the xy error and would dilute it, and nets sag and move.
    # A return whose nearest mapped surface is a vertical non-wall (a net) is explained within
    # net_tol_m instead of explain_tol_m; the floor keeps the tight tolerance, so low objects show.
    wall_field: str = "localization"
    net_tol_m: float = 0.25


@dataclass
class FrameStats:
    hits: int = 0                      # depth points used
    explained: int = 0                 # ... that lay on a mapped surface
    unexplained: int = 0               # ... that did not (evidence of an unmapped obstacle)
    off_map: int = 0                   # ... outside the map's grid (e.g. seen through the net)
    carved: int = 0                    # voxels a ray passed through this frame
    obstacles: int = 0                 # obstacle voxels after this frame
    misfit: float | None = None        # share of near-surface returns that miss it (None: too few to judge)
    rejected: bool = False             # True: the frame changed nothing (pose disagrees with the map)
    notes: list[str] = field(default_factory=list)


class LiveObstacleLayer:
    def __init__(self, course: CourseMap, static_field: str = "planning", params: LiveParams | None = None) -> None:
        if static_field not in course.fields:
            raise KeyError(f"course map has no field {static_field!r}")
        self.course = course
        self.static_field = static_field
        self.p = params or LiveParams()
        self.wall_field = self.p.wall_field if self.p.wall_field in course.fields else static_field
        if self.p.min_filter_px < 1 or self.p.min_filter_px % 2 == 0:
            raise ValueError("min_filter_px must be a positive odd integer")
        self.shape = course.shape
        self.origin = course.origin
        self.cell = course.cell
        self.hits = np.zeros(self.shape, dtype=np.uint16)
        self.misses = np.zeros(self.shape, dtype=np.uint16)
        self.seen = np.zeros(self.shape, dtype=bool)
        # flat views of the same memory: per-frame updates use linear voxel indices, no sorting
        self._hits = self.hits.reshape(-1)
        self._misses = self.misses.reshape(-1)
        self._seen = self.seen.reshape(-1)
        self._mark = np.zeros(self.hits.size, dtype=bool)  # scratch for de-duplicating indices
        self._occupied: np.ndarray | None = None          # (M, 3) voxel indices, cached
        self._field: CourseMap | None = None              # truncated live distance grid, cached
        self._kernel = distance_kernel(self.cell, self.p.truncate_m)

    # ------------------------------------------------------------------ input
    def integrate(self, depth_m: Any, K: Any, T_map_cam: Any) -> FrameStats:
        """One depth frame. K is the 3x3 intrinsic matrix of the depth image."""
        st = FrameStats()
        D = np.asarray(depth_m, dtype=np.float32) * np.float32(self.p.depth_scale)
        D = np.where(np.isfinite(D) & (D > 0), D, np.inf).astype(np.float32)
        D = min_filter(D, self.p.min_filter_px)
        K = np.asarray(K, dtype=float).reshape(3, 3)
        T = np.asarray(T_map_cam, dtype=float).reshape(4, 4)
        origin = T[:3, 3]

        hit_pts = self._points(D, K, T, self.p.hit_stride_px)
        st.hits = len(hit_pts)
        if st.hits:
            idx, inside = self._index(hit_pts)
            d_all = np.full(len(hit_pts), np.nan)
            d_all[inside] = self.course.distance(hit_pts[inside], self.static_field)
            inside &= np.isfinite(d_all)                  # past the outermost voxel centres: off the map
            st.off_map = int(np.count_nonzero(~inside))
            d_static = d_all[inside]
            pts_in = hit_pts[inside]
            d_wall = (self.course.distance(pts_in, self.wall_field) if self.wall_field != self.static_field
                      else d_static)
            tol = np.full(len(d_static), self.p.explain_tol_m)
            not_wall = np.abs(d_wall) > np.abs(d_static) + self.cell    # nearest surface: floor or net
            # only returns between the two tolerances need the surface's orientation
            maybe_net = not_wall & (np.abs(d_static) > self.p.explain_tol_m) & (np.abs(d_static) <= self.p.net_tol_m)
            if maybe_net.any():
                _, g = self.course.query(pts_in[maybe_net], self.static_field)
                t = tol[maybe_net]
                t[np.abs(g[:, 2]) < 0.7] = self.p.net_tol_m              # vertical, not a wall: a net
                tol[maybe_net] = t
            explained = np.abs(d_static) <= tol
            st.explained = int(np.count_nonzero(explained))
            st.unexplained = int(np.count_nonzero(~explained))
            # near a wall, and not already explained by the floor or a net (e.g. floor at a wall's foot)
            near = (np.abs(d_wall) <= self.p.misfit_band_m) & ~(explained & not_wall)
            n_near = int(np.count_nonzero(near))
            if n_near >= self.p.misfit_min_points:
                st.misfit = float(np.count_nonzero(near & (np.abs(d_wall) > self.p.explain_tol_m))) / n_near
                if st.misfit > self.p.max_misfit:
                    st.rejected = True
                    st.obstacles = len(self.obstacle_voxels())
                    st.notes.append(f"frame refused: {st.misfit:.0%} of returns near a mapped surface miss it; "
                                    "the pose probably disagrees with the map")
                    return st
            lin = np.ravel_multi_index(idx[inside].T, self.shape)
            self._seen[lin] = True
            new = self._unique(lin[~explained])                    # one hit per voxel per frame
            if len(new):
                self._hits[new] = np.minimum(self._hits[new].astype(np.int32) + 1, 65535)

        free_pts = self._points(D, K, T, self.p.free_stride_px)
        if len(free_pts):
            carved = self._carve(origin, free_pts)
            st.carved = len(carved)
            if len(carved):
                self._seen[carved] = True
                self._misses[carved] = np.minimum(self._misses[carved].astype(np.int32) + 1, 65535)

        self._occupied, self._field = None, None
        occ = self.obstacle_voxels()
        st.obstacles = len(occ)
        if len(occ) > self.p.max_obstacle_voxels:
            st.notes.append(f"{len(occ)} obstacle voxels exceed max_obstacle_voxels: the map or pose is "
                            "probably misaligned; live distances are withheld until it recovers")
        return st

    def reset(self) -> None:
        self.hits[:] = 0
        self.misses[:] = 0
        self.seen[:] = False
        self._mark[:] = False
        self._occupied, self._field = None, None

    # ---------------------------------------------------------------- queries
    def obstacle_voxels(self) -> np.ndarray:
        if self._occupied is None:
            score = self.hits.astype(np.float32) - self.p.miss_weight * self.misses.astype(np.float32)
            self._occupied = np.argwhere((self.hits >= self.p.min_hits) & (score >= self.p.min_hits))
        return self._occupied

    @property
    def trusted(self) -> bool:
        """False when there are more obstacle voxels than max_obstacle_voxels. Then the map or the
        pose is probably misaligned, and the combined map falls back to the static map alone: a
        supervisor should treat that as degraded perception."""
        return len(self.obstacle_voxels()) <= self.p.max_obstacle_voxels

    def obstacle_points(self) -> np.ndarray:
        """Obstacle voxel centres in `map`, (M, 3)."""
        return self.origin + self.obstacle_voxels() * self.cell

    def observed(self, points: Any) -> np.ndarray:
        """True where some ray has reached the point's voxel. Off the grid: False."""
        idx, inside = self._index(np.atleast_2d(np.asarray(points, dtype=float)))
        out = np.zeros(len(idx), dtype=bool)
        out[inside] = self.seen[tuple(idx[inside].T)]
        return out

    def as_course_map(self) -> CourseMap | None:
        """The truncated live distance field as a CourseMap (field "live"); None if there is
        nothing to report or too much to trust."""
        occ = self.obstacle_voxels()
        if len(occ) == 0 or len(occ) > self.p.max_obstacle_voxels:
            return None
        if self._field is None:
            F = stamp_field(self.shape, occ, self._kernel, self.p.truncate_m)
            self._field = CourseMap({"live": F}, self.origin, self.cell, {"source": "LiveObstacleLayer"})
        return self._field

    # ---------------------------------------------------------------- helpers
    def _points(self, D: np.ndarray, K: np.ndarray, T: np.ndarray, stride: int) -> np.ndarray:
        h, w = D.shape
        v, u = np.mgrid[stride // 2:h:stride, stride // 2:w:stride]
        z = D[v, u]
        ok = np.isfinite(z) & (z >= self.p.min_range_m) & (z <= self.p.max_range_m)
        z, u, v = z[ok].astype(float), u[ok], v[ok]
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        P = np.stack([x, y, z], axis=1)
        return P @ T[:3, :3].T + T[:3, 3]

    def _index(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.rint((pts - self.origin) / self.cell).astype(np.intp)
        inside = np.all((idx >= 0) & (idx < np.array(self.shape)), axis=1)
        return idx, inside

    def _unique(self, lin: np.ndarray) -> np.ndarray:
        """De-duplicate linear voxel indices with the scratch mask (no sort)."""
        if len(lin) == 0:
            return lin
        self._mark[lin] = True
        out = np.flatnonzero(self._mark)
        self._mark[out] = False
        return out

    def _carve(self, origin: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Linear indices of the voxels strictly between the camera and each hit (the last cell
        before a hit is kept). Rays are marched in grid units, in float32."""
        g0 = ((origin - self.origin) / self.cell).astype(np.float32)
        d = ((pts - origin) / self.cell).astype(np.float32)        # ray to the hit, in cells
        L = np.linalg.norm(d, axis=1)
        u = d / L[:, None]
        step = np.float32(0.9)
        n = int(np.ceil(L.max() / step))
        t = (np.arange(n, dtype=np.float32) + np.float32(0.5)) * step
        keep = t[None, :] < (L[:, None] - 1.0)
        g = g0 + u[:, None, :] * t[None, :, None]
        ijk = np.rint(g[keep]).astype(np.int32)
        inside = np.all((ijk >= 0) & (ijk < np.array(self.shape, dtype=np.int32)), axis=1)
        if not inside.any():
            return np.zeros(0, dtype=np.intp)
        return self._unique(np.ravel_multi_index(ijk[inside].T, self.shape))


class CombinedMap:
    """The static course map and the live obstacle layer behind CourseMap's interface.

    For `combine_field` (planning) the grid is min(static, live), rebuilt only when the live layer
    changes, so each query is a single trilinear lookup. Other fields (localization) pass straight
    through to the static map: live obstacles are by definition not part of the mapped course.
    Off the course grid: nan."""

    def __init__(self, course: CourseMap, live: LiveObstacleLayer | None = None,
                 combine_field: str = "planning") -> None:
        if combine_field not in course.fields:
            raise KeyError(f"course map has no field {combine_field!r}")
        self.course = course
        self.live = live
        self.combine_field = combine_field
        self.fields = course.fields
        self.cell = course.cell
        self.origin = course.origin
        self.shape = course.shape
        self.bounds = course.bounds
        self._key: CourseMap | None = None
        self._combined: CourseMap | None = None

    def _grid(self) -> CourseMap | None:
        live = self.live.as_course_map() if self.live is not None else None
        if live is None:
            return None
        if self._key is not live:
            F = np.minimum(self.course.fields[self.combine_field], live.fields["live"])
            self._combined = CourseMap({self.combine_field: F}, self.course.origin, self.course.cell,
                                       {"source": "CombinedMap"})
            self._key = live
        return self._combined

    def query(self, points: Any, field: str = "planning") -> tuple[np.ndarray, np.ndarray]:
        if field == self.combine_field:
            grid = self._grid()
            if grid is not None:
                return grid.query(points, field)
        return self.course.query(points, field)

    def distance(self, points: Any, field: str = "planning") -> np.ndarray:
        return self.query(points, field)[0]

    def gradient(self, points: Any, field: str = "planning") -> np.ndarray:
        return self.query(points, field)[1]

    def clearance_ok(self, points: Any, need: float, field: str = "planning") -> np.ndarray:
        """True where the point is on the map and at least `need` metres from any known or live
        obstacle. Off the map (nan) is never ok."""
        d = self.distance(points, field)
        return np.isfinite(d) & (d >= need)


def distance_kernel(cell: float, truncate_m: float) -> np.ndarray:
    """Distances (m) from the centre voxel of a cube reaching truncate_m in each direction."""
    r = int(np.ceil(truncate_m / cell))
    o = np.arange(-r, r + 1)
    X, Y, Z = np.meshgrid(o, o, o, indexing="ij")
    return (np.sqrt(X * X + Y * Y + Z * Z) * cell).astype(np.float32)


def stamp_field(shape: tuple[int, int, int], voxels: np.ndarray, kernel: np.ndarray, truncate_m: float) -> np.ndarray:
    """Distance to the nearest of `voxels` (index triples) on a grid, capped at truncate_m."""
    F = np.full(shape, truncate_m, dtype=np.float32)
    r, n = kernel.shape[0] // 2, np.array(shape)
    for v in voxels:
        lo = np.maximum(v - r, 0)
        hi = np.minimum(v + r + 1, n)
        k0 = lo - (v - r)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        ks = tuple(slice(a, a + (b - c)) for a, b, c in zip(k0, hi, lo))
        np.minimum(F[sl], kernel[ks], out=F[sl])
    return F


class ObstacleCloud:
    """Live obstacles received from another process (the live map node's point cloud), usable as
    CombinedMap(course, ObstacleCloud(course, points)) without running a LiveObstacleLayer."""

    def __init__(self, course: CourseMap, points: Any, truncate_m: float = 0.80, trusted: bool = True) -> None:
        self.course = course
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        idx = np.rint((pts - course.origin) / course.cell).astype(np.intp)
        ok = np.all((idx >= 0) & (idx < np.array(course.shape)), axis=1)
        self.voxels = np.unique(idx[ok], axis=0) if ok.any() else np.zeros((0, 3), dtype=np.intp)
        self.truncate_m = float(truncate_m)
        self.trusted = bool(trusted)
        self._field: CourseMap | None = None

    def as_course_map(self) -> CourseMap | None:
        if not self.trusted or len(self.voxels) == 0:
            return None
        if self._field is None:
            F = stamp_field(self.course.shape, self.voxels, distance_kernel(self.course.cell, self.truncate_m),
                            self.truncate_m)
            self._field = CourseMap({"live": F}, self.course.origin, self.course.cell, {"source": "ObstacleCloud"})
        return self._field


class PoseGate:
    """Decides whether the pose is trustworthy enough to map with. Pure: the caller supplies times.

    Refuses while the estimator's health is unhealthy or stale, for recover_s after it last was, and
    for recover_s after the camera position jumps faster than max_speed_mps between checks (a VIO
    relocalisation can move the estimate 0.6 m while the drone stands still)."""

    def __init__(self, recover_s: float = 3.0, stale_s: float = 1.0, max_speed_mps: float = 2.5) -> None:
        self.recover_s, self.stale_s, self.max_speed = float(recover_s), float(stale_s), float(max_speed_mps)
        self.health_t: float | None = None
        self.healthy = False
        self.bad_until = -np.inf
        self.last: tuple[float, np.ndarray] | None = None

    def on_health(self, t: float, ok: bool) -> None:
        if not ok:
            self.bad_until = max(self.bad_until, t + self.recover_s)
        self.health_t, self.healthy = t, bool(ok)

    def check(self, t: float, position: Any) -> tuple[bool, str]:
        p = np.asarray(position, dtype=float).reshape(3)
        if self.health_t is None:
            return False, "no estimator health received"
        if t - self.health_t > self.stale_s:
            self.bad_until = max(self.bad_until, t + self.recover_s)
            return False, f"estimator health stale ({t - self.health_t:.1f} s)"
        if not self.healthy:
            return False, "estimator unhealthy"
        if self.last is not None and t > self.last[0]:
            speed = float(np.linalg.norm(p - self.last[1])) / (t - self.last[0])
            if speed > self.max_speed:
                self.bad_until = max(self.bad_until, t + self.recover_s)
                self.last = (t, p)
                return False, f"pose jumped ({speed:.1f} m/s)"
        self.last = (t, p)
        if t < self.bad_until:
            return False, f"recovering ({self.bad_until - t:.1f} s left)"
        return True, "ok"


def min_filter(D: np.ndarray, w: int) -> np.ndarray:
    """Sliding-window minimum over a w x w window (inf = no return). Pure numpy."""
    if w <= 1:
        return D
    r = w // 2
    P = np.pad(D, r, mode="constant", constant_values=np.inf)
    out = np.full_like(D, np.inf)
    h, wd = D.shape
    for dy in range(w):
        for dx in range(w):
            np.minimum(out, P[dy:dy + h, dx:dx + wd], out=out)
    return out
