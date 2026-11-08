"""Depth image -> horizontal range slice at flight height (a LaserScan's worth of numbers).

Pure numpy, no ROS. For each bearing bin, the NEAREST horizontal range of any depth pixel whose
3D point lies within a height band around the camera. Min (not mean or median) because a slanted
wall is nearest at one height only. Raw: no hole filling, no smoothing across bins.

Frames: the depth image is in the camera OPTICAL frame (x right, y down, z forward). Points go to
the body frame (FLU: x forward, y left, z up) with the static TF base <- camera, then optionally
to a gravity-levelled body frame with R_level (roll and pitch removed, yaw kept), so a pitched
drone doesn't cut the floor or ceiling into the band. Bearings follow ROS LaserScan: 0 = forward,
counter-clockwise (left) positive. Ranges are measured in the horizontal plane from the body
origin (not the camera).

Bin values (REP 117 style):
  finite  nearest in-band return within [range_min, range_max]
  +inf    the bin had in-band returns, but every one was beyond range_max
  -inf    the bin had in-band returns, but every one was closer than range_min (blind zone)
  NaN     no usable in-band return at all: UNKNOWN, never free space. Depth holes (0 / NaN
          pixels, e.g. through a net or on blank texture) stay NaN; nothing is filled in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# camera optical (x right, y down, z forward) -> body FLU, for a camera looking straight ahead
R_BODY_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


@dataclass(frozen=True)
class SliceConfig:
    band_m: float = 0.20  # keep points within +-band_m of the camera's height
    range_min: float = 0.20
    range_max: float = 3.0
    angle_increment_deg: float = 1.5
    min_pixels: int = 2  # a bin needs this many in-band returns, else NaN (rejects lone specks)
    depth_scale_16u: float = 0.001  # 16UC1 is millimetres


@dataclass
class Slice:
    ranges: np.ndarray  # float32, one per bin, see module docstring
    angle_min: float  # rad, centre of the first bin
    angle_increment: float  # rad
    range_min: float
    range_max: float
    counts: np.ndarray  # in-band returns per bin (diagnostics)
    stamp: float | None = None

    @property
    def angle_max(self) -> float:
        return self.angle_min + self.angle_increment * (len(self.ranges) - 1)

    @property
    def bearings(self) -> np.ndarray:
        return self.angle_min + self.angle_increment * np.arange(len(self.ranges))


def _check_rotation(R: np.ndarray, name: str) -> np.ndarray:
    R = np.asarray(R, float)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        raise ValueError(f"{name} must be a finite 3x3 rotation")
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or np.linalg.det(R) < 0.999:
        raise ValueError(f"{name} is not a proper rotation")
    return R


def depth_to_metres(depth: np.ndarray, scale_16u: float = 0.001) -> np.ndarray:
    """16UC1 (mm, 0 = no data) or 32FC1 (m, 0/NaN/inf = no data) -> float32 metres, NaN = none."""
    d = np.asarray(depth)
    if d.ndim != 2:
        raise ValueError("depth must be a 2-D image")
    if d.dtype == np.uint16:
        m = d.astype(np.float32) * np.float32(scale_16u)
        m[d == 0] = np.nan
        return m
    if d.dtype in (np.float32, np.float64):
        m = d.astype(np.float32)
        m[~np.isfinite(m) | (m <= 0)] = np.nan
        return m
    raise ValueError(f"unsupported depth dtype {d.dtype} (want uint16 mm or float32 m)")


def horizontal_slice(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    R_base_cam: np.ndarray = R_BODY_OPTICAL,
    t_base_cam=(0.0, 0.0, 0.0),
    R_level: np.ndarray | None = None,
    config: SliceConfig = SliceConfig(),
    stamp: float | None = None,
) -> Slice:
    """One depth frame -> one horizontal slice. See the module docstring for the conventions."""
    if not (fx > 0 and fy > 0 and np.isfinite([fx, fy, cx, cy]).all()):
        raise ValueError("bad intrinsics")
    R = _check_rotation(R_base_cam, "R_base_cam")
    t = np.asarray(t_base_cam, float).reshape(3)
    if R_level is not None:
        L = _check_rotation(R_level, "R_level")
        R, t = L @ R, L @ t
    z = depth_to_metres(depth, config.depth_scale_16u)
    h, w = z.shape
    v, u = np.nonzero(np.isfinite(z))
    zz = z[v, u].astype(np.float64)
    pc = np.stack([(u - cx) / fx * zz, (v - cy) / fy * zz, zz])  # optical, 3 x N
    pb = R @ pc + t[:, None]  # body (levelled if R_level)

    # field of view in bearing, from the image corners at unit depth, rounded out to whole bins
    inc = np.deg2rad(config.angle_increment_deg)
    corners = np.array([[0, w - 1, 0, w - 1], [0, 0, h - 1, h - 1]], float)
    rays = R @ np.stack([(corners[0] - cx) / fx, (corners[1] - cy) / fy, np.ones(4)])
    edge = np.arctan2(rays[1], rays[0])
    lo = np.floor(edge.min() / inc) * inc
    n = int(np.round((np.ceil(edge.max() / inc) * inc - lo) / inc)) + 1

    band = np.abs(pb[2] - t[2]) <= config.band_m
    x, y = pb[0, band], pb[1, band]
    rng = np.hypot(x, y)
    idx = np.clip(np.round((np.arctan2(y, x) - lo) / inc).astype(int), 0, n - 1)

    counts = np.bincount(idx, minlength=n)
    ok = (rng >= config.range_min) & (rng <= config.range_max)
    best = np.full(n, np.inf)
    np.minimum.at(best, idx[ok], rng[ok])
    n_ok = np.bincount(idx[ok], minlength=n)
    n_far = np.bincount(idx[rng > config.range_max], minlength=n)
    n_near = np.bincount(idx[rng < config.range_min], minlength=n)

    out = np.full(n, np.nan, np.float32)
    enough = counts >= config.min_pixels
    sel = enough & (n_ok > 0)
    out[sel] = best[sel]
    out[enough & (n_ok == 0) & (n_far > 0) & (n_near == 0)] = np.inf
    out[enough & (n_ok == 0) & (n_near > 0)] = -np.inf
    return Slice(out, float(lo), float(inc), config.range_min, config.range_max, counts, stamp)


def rotation_level(roll: float, pitch: float) -> np.ndarray:
    """Body FLU -> gravity-levelled FLU (yaw kept) for a body with this roll and pitch (rad,
    FLU convention: roll about x, pitch about y, positive = nose DOWN in FLU)."""
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return Ry @ Rx
