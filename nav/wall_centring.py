"""Keep the drone centred between the REAL corridor walls, which the published map only approximates.
Pure: numpy only, no ROS, no clock reads.

The physical course is boards and tarp: walls stand where they were put, not exactly where the OBJ says,
and there are two copies of it. Flying the map's centre line then leaves the real clearance to chance at
the tight spots. This module measures where the walls really are, in a horizontal slice of the depth
image at flight height, and returns a sideways offset for the path reference.

Per depth frame, on a straight stretch of corridor:

1. Wall points ahead of the drone (ahead_lo_m..ahead_hi_m along the path, within max_side_m of it, in a
   height band around the drone) are expressed in the path's frame: along-track a, cross-track e (left +).
2. The map's own walls are found along the same stretch by marching sideways in the walls-only field
   (`localization`: no floor, net or pad): e_L(a) > 0 on the left, e_R(a) < 0 on the right. The stretch is
   a corridor only if the map has walls within corridor_max_m on BOTH sides along most of it.
3. A measured point within match_m of a mapped wall is a sighting of it; the residual r = e_point - e_wall.
   Per side, the median residual says how far that real wall is from the mapped one, as the drone sees it.
4. The offset is the mean of the two sides' residuals (only one side seen: that side's residual).

NETS. Depth goes through netting: a few returns from the strands, the rest from whatever is behind it
(people, the hall's walls). A net must never be read as a wall that has moved. Two guards, per side:
* the map's own nets: a side whose nearest mapped surface is in the planning field but not in the walls-only
  field (a net, or anything else that isn't a wall) is never used; the solid side alone is;
* see-through: if more than net_through_frac of a side's returns land more than match_m BEHIND the mapped
  wall, or its matched returns scatter more than max_spread_m (median absolute deviation; a board gives
  ~1-2 cm), that side is treated as see-through (a net, mesh, a gap) and ignored. This also catches a
  real wall built of mesh where the map has a board.

The pose error cancels: a point measured from a drone that believes it is `err` off lands `err` off in the
map, so r = (real wall - mapped wall) + err on both sides, and steering the believed position to the mapped
centre + offset puts the TRUE drone in the middle of the REAL walls (one side seen: at the mapped distance
from that real wall). Points further than match_m from any mapped wall (people behind a net, an
obstacle in the corridor) are not used here.

The offset is low-passed, capped at cap_m, held for hold_s without a measurement and then decayed to zero.
No measurement while turning (yaw rate over max_yaw_rate_dps) or where the path bends in the window.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CentringParams:
    band_lo_m: float = -0.20            # height band around the drone (map z) for wall points
    band_hi_m: float = 0.25
    ahead_lo_m: float = 0.55            # the camera sees a 1 m corridor's side walls from ~0.6 m ahead
    ahead_hi_m: float = 1.60
    max_side_m: float = 0.90            # points further sideways than this are not corridor walls
    corridor_max_m: float = 0.80        # the map must have a wall within this on both sides...
    corridor_frac: float = 0.7          # ...along this fraction of the window
    match_m: float = 0.25               # a point within this of a mapped wall is a sighting of it
    min_points: int = 15                # sightings needed per side
    max_bend_deg: float = 12.0          # path heading change over the window
    max_yaw_rate_dps: float = 20.0      # no measurement while turning
    net_through_frac: float = 0.25      # a side with more returns than this far BEHIND its wall is see-through
    max_spread_m: float = 0.045         # ...or whose matched returns scatter more than this (MAD): a net
    solid_tol_m: float = 0.06           # a side is a solid wall if the walls-only and full maps agree on it
    wall_field: str = "localization"    # solid walls only (no floor, net, pad)
    any_field: str = "planning"         # everything, nets included
    step_m: float = 0.02                # sideways march step to find the mapped walls
    cap_m: float = 0.15                 # largest offset applied
    tau_s: float = 0.8                  # low-pass time constant
    hold_s: float = 1.5                 # keep the last offset this long without a measurement...
    decay_s: float = 2.0                # ...then fade it to zero over about this long


@dataclass
class Measurement:
    ok: bool
    why: str
    offset: float = 0.0                 # raw (unfiltered) offset, metres, left positive
    resid_left: float | None = None
    resid_right: float | None = None
    n_left: int = 0
    n_right: int = 0
    left: str = ""                      # per side: "used", or why not
    right: str = ""


@dataclass
class WallCentring:
    course: Any                         # nav.course_map.CourseMap
    p: CentringParams = field(default_factory=CentringParams)
    offset: float = 0.0                 # filtered offset to apply, metres, left of the path positive
    t_meas: float = -math.inf
    t_last: float | None = None
    last: Measurement | None = None

    # ---------------------------------------------------------------- measure
    def mapped_walls(self, c: np.ndarray, t_hat: np.ndarray, z: float, along: np.ndarray,
                     field_name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Lateral position of the map's walls (left > 0, right < 0) at each along-track distance from c on a
        straight line with unit tangent t_hat; nan where there is none within max_side_m."""
        p = self.p
        fname = field_name or p.wall_field
        n_hat = np.array([-t_hat[1], t_hat[0]])
        steps = np.arange(p.step_m, p.max_side_m + 1e-9, p.step_m)
        out = []
        for sign in (1.0, -1.0):
            base = c[None, None, :] + along[:, None, None] * t_hat[None, None, :]
            pts = base + sign * steps[None, :, None] * n_hat[None, None, :]
            q = np.concatenate([pts.reshape(-1, 2), np.full((pts.shape[0] * pts.shape[1], 1), z)], axis=1)
            d = self.course.distance(q, fname).reshape(len(along), len(steps))
            hit = d <= p.step_m / 2
            first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
            e = np.where(first >= 0, sign * steps[np.clip(first, 0, None)], np.nan)
            out.append(e)
        return out[0], out[1]

    def measure(self, pts_map: Any, drone_xyz: Any, path: Any, s_proj: float, yaw_rate: float = 0.0) -> Measurement:
        """One depth frame. pts_map: (N, 3) points in the map frame, transformed with the drone's believed pose
        at the frame's capture time. drone_xyz: that believed position. path: nav.path_geometry.SmoothPath."""
        p = self.p
        if abs(yaw_rate) > math.radians(p.max_yaw_rate_dps):
            return Measurement(False, "turning")
        s0, s1 = s_proj + p.ahead_lo_m, s_proj + p.ahead_hi_m
        if s1 > path.length:
            return Measurement(False, "near the end of the route")
        psi = np.unwrap(np.asarray(path.heading_at(np.linspace(s_proj, s1, 8)), dtype=float))
        if float(np.max(np.abs(psi - psi[0]))) > math.radians(p.max_bend_deg):
            return Measurement(False, "the path bends ahead")
        c = np.asarray(path.point_at(s_proj), dtype=float)[:2]
        t_hat = np.array([math.cos(psi[0]), math.sin(psi[0])])
        n_hat = np.array([-t_hat[1], t_hat[0]])
        z = float(np.asarray(drone_xyz, dtype=float)[2])

        along = np.arange(p.ahead_lo_m, p.ahead_hi_m + 1e-9, 0.10)
        anyL, anyR = self.mapped_walls(c, t_hat, z, along, p.any_field)       # whatever bounds the corridor
        wallL, wallR = self.mapped_walls(c, t_hat, z, along, p.wall_field)    # solid walls only
        bounded = (np.abs(anyL) <= p.corridor_max_m) & (np.abs(anyR) <= p.corridor_max_m)
        if np.mean(bounded) < p.corridor_frac:
            return Measurement(False, "not a corridor here")
        solidL = bounded & (np.abs(wallL - anyL) <= p.solid_tol_m)           # nan compares False: not solid
        solidR = bounded & (np.abs(wallR - anyR) <= p.solid_tol_m)

        P = np.asarray(pts_map, dtype=float).reshape(-1, 3)
        dz = P[:, 2] - z
        P = P[(dz >= p.band_lo_m) & (dz <= p.band_hi_m)]
        rel = P[:, :2] - c
        a, e = rel @ t_hat, rel @ n_hat
        keep = (a >= p.ahead_lo_m) & (a <= p.ahead_hi_m) & (np.abs(e) <= p.max_side_m + p.match_m + 1.0)
        a, e = a[keep], e[keep]

        def side(sign: float, solid: np.ndarray, wall: np.ndarray) -> tuple[float | None, int, str]:
            if np.mean(solid) < p.corridor_frac:
                return None, 0, "not a mapped solid wall (net?)"
            w = np.interp(a, along, np.where(solid, wall, np.nan))
            on = (sign * e > 0) & np.isfinite(w)
            r = sign * (e[on] - w[on])                                     # > 0: behind the mapped wall
            matched = np.abs(r) <= p.match_m
            through = r > p.match_m
            n, nt = int(matched.sum()), int(through.sum())
            if n < p.min_points:
                return None, n, f"too few points ({n})"
            if nt > p.net_through_frac * (n + nt):
                return None, n, f"see-through: {nt}/{n + nt} returns behind it (net?)"
            res = sign * r[matched]                                         # back to lateral (left +)
            spread = float(np.median(np.abs(res - np.median(res))))
            if spread > p.max_spread_m:
                return None, n, f"scattered returns, spread {spread:.3f} m (net?)"
            return float(np.median(res)), n, "used"

        rl, nl, wl_why = side(1.0, solidL, wallL)
        rr, nr, wr_why = side(-1.0, solidR, wallR)
        if rl is None and rr is None:
            return Measurement(False, f"no usable wall (left: {wl_why}; right: {wr_why})", n_left=nl, n_right=nr,
                               left=wl_why, right=wr_why)
        off = (rl + rr) / 2 if (rl is not None and rr is not None) else (rl if rl is not None else rr)
        return Measurement(True, "both walls" if (rl is not None and rr is not None) else "one wall",
                           float(off), rl, rr, nl, nr, wl_why, wr_why)

    # ----------------------------------------------------------------- filter
    def update(self, t: float, m: Measurement | None) -> float:
        """Advance the filtered offset to time t with this frame's measurement (None or not ok: no news)."""
        p = self.p
        dt = 0.0 if self.t_last is None else max(0.0, t - self.t_last)
        self.t_last = t
        if m is not None:
            self.last = m
        if m is not None and m.ok:
            target = max(-p.cap_m, min(p.cap_m, m.offset))
            if self.t_meas == -math.inf:                 # first measurement: start from it
                self.offset = target
            else:                                        # low-pass over the time between MEASUREMENTS
                dt_m = max(0.0, t - self.t_meas)
                self.offset += (target - self.offset) * (1.0 - math.exp(-dt_m / p.tau_s))
            self.t_meas = t
        elif t - self.t_meas > p.hold_s:
            self.offset *= math.exp(-dt / p.decay_s)
        self.offset = max(-p.cap_m, min(p.cap_m, self.offset))
        return self.offset


def points_from_depth(depth: Any, K: Any, T_frame_cam: Any, stride: int = 4, min_m: float = 0.2,
                      max_m: float = 3.0, scale: float = 1.0) -> np.ndarray:
    """Pixels of a depth image -> (N, 3) points in the frame T_frame_cam maps the optical frame into.
    depth: (H, W) array (16UC1 millimetres: scale=0.001; 32FC1 metres: scale=1). K: 3x3 intrinsics."""
    D = np.asarray(depth)
    K = np.asarray(K, dtype=float).reshape(3, 3)
    T = np.asarray(T_frame_cam, dtype=float)
    v, u = np.mgrid[stride // 2:D.shape[0]:stride, stride // 2:D.shape[1]:stride]
    z = D[v, u].astype(float) * scale
    ok = np.isfinite(z) & (z >= min_m) & (z <= max_m)
    z, u, v = z[ok], u[ok], v[ok]
    P = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    return P @ T[:3, :3].T + T[:3, 3]
