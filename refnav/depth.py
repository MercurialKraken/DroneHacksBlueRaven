"""Free space along the path ahead, from one depth frame. PURE (numpy only).

The frame arrives with its own calibration: the pinhole intrinsics from CameraInfo and the
camera's pose on the body from TF (base_link <- camera optical frame, REP-103: optical x right,
y down, z forward; body x forward, y left, z up). The vehicle attitude levels the points, so
"ahead" and "at flight height" mean the same thing while the airframe is tilted.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    Intrinsics = tuple[float, float, float, float]      # pinhole (fx, fy, cx, cy) in pixels

# body FLU -> body FRD (PX4's body frame)
_FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])


def quat_to_mat(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_yaw(w: float, x: float, y: float, z: float) -> float:
    """Rotation about the parent frame's vertical axis, rad (PX4's heading for an FRD -> NED attitude)."""
    return math.atan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z))


def quat_up_cos(w: float, x: float, y: float, z: float) -> float:
    """Cosine of the tilt: the body's vertical axis against the parent frame's."""
    return 1.0 - 2.0 * (x * x + y * y)


def level_rotation(q_wxyz: Sequence[float]) -> np.ndarray:
    """Rotation taking body-FLU vectors into the LEVEL heading frame (x forward horizontal,
    y right horizontal, z down), i.e. PX4's attitude with its yaw removed. q = PX4's
    vehicle_attitude quaternion (FRD body -> NED), w first."""
    R = quat_to_mat(*q_wxyz)                                   # NED <- FRD
    yaw = quat_yaw(*q_wxyz)
    c, s = math.cos(yaw), math.sin(yaw)
    Rz_inv = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz_inv @ R @ _FLU_TO_FRD


def decode_depth(encoding: str, height: int, width: int, step: int, data: bytes) -> np.ndarray:
    """sensor_msgs/Image depth -> float32 metres (NaN = no return). 32FC1 is metres,
    16UC1 is millimetres (0 = invalid); anything else raises ValueError."""
    enc = encoding.lower()
    if enc == "32fc1":
        a = np.frombuffer(data, np.float32).reshape(height, step // 4)[:, :width].astype(np.float32)
        a = a.copy()
        a[~np.isfinite(a) | (a <= 0.0)] = np.nan
        return a
    if enc in ("16uc1", "mono16"):
        a = np.frombuffer(data, np.uint16).reshape(height, step // 2)[:, :width].astype(np.float32) * 1e-3
        a[a <= 0.0] = np.nan
        return a
    raise ValueError(f"unsupported depth encoding {encoding!r} (need 32FC1 metres or 16UC1 millimetres)")


class DepthFrame:
    """One depth frame, sampled every `stride` pixels inside a `border_px` margin, in the LEVEL
    heading frame (forward, right, down) relative to the body origin. Keeps, per sampled pixel, its
    ray and whether the camera returned ANY depth for it - so the frame can say what it observed
    (a return, near or far) apart from what it did not (no return: unknown, never free space)."""

    def __init__(self, depth_m: ArrayLike, K: Intrinsics, R_body_cam: ArrayLike, t_body_cam: ArrayLike,
                 R_level_body: ArrayLike, stride: int = 8, border_px: int = 0) -> None:
        fx, fy, cx, cy = K
        d = np.asarray(depth_m, np.float32)
        h, w = d.shape
        b = int(border_px)
        vs, us = np.mgrid[b:h - b:stride, b:w - b:stride] if b > 0 else np.mgrid[0:h:stride, 0:w:stride]
        z = d[vs, us].ravel()
        u, v = us.ravel().astype(np.float32), vs.ravel().astype(np.float32)
        rays_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=1)    # optical z = 1
        R = np.asarray(R_level_body) @ np.asarray(R_body_cam)
        self.dirs: np.ndarray = rays_cam @ R.T                     # a point at optical depth z is origin + z * dir
        self.origin: np.ndarray = np.asarray(R_level_body) @ np.asarray(t_body_cam, float)
        self.z: np.ndarray = z
        self.valid: np.ndarray = np.isfinite(z) & (z > 0.0)
        self.valid_frac: float = float(self.valid.mean()) if z.size else 0.0

    def points(self, rmin: float = 0.3, rmax: float = 6.0) -> np.ndarray:
        sel = self.valid & (self.z > rmin) & (self.z < rmax)
        return self.origin + self.dirs[sel] * self.z[sel, None]


def path_clearance(frame: DepthFrame, points: np.ndarray | None, path_fr: np.ndarray, along: np.ndarray,
                   half_width_m: float, obstacle_band_m: float, min_hits: int, min_coverage: float,
                   blind_m: float = 0.5) -> tuple[float, bool]:
    """Free distance along a path ahead: path_fr (M,2) samples (forward, right) in the level frame, `along` their path
    distances. Ends at the first sample with an obstacle beside it, or one the camera did not observe (the rays
    towards it mostly returned nothing). -> (free distance, True if an obstacle ended it)."""
    pts = np.empty((0, 3)) if points is None else points[np.abs(points[:, 2]) < obstacle_band_m]
    fwd = frame.dirs[:, 0]
    ahead = fwd > 1e-6
    az = np.arctan2(frame.dirs[:, 1], np.where(ahead, fwd, 1.0))
    el = np.abs(frame.dirs[:, 2]) / np.where(ahead, fwd, 1.0)
    for (pf, pr), s in zip(path_fr, along):
        if len(pts) and int((np.hypot(pts[:, 0] - pf, pts[:, 1] - pr) < half_width_m).sum()) >= min_hits:
            return float(s), True
        d = math.hypot(pf, pr)
        if d < blind_m:
            continue
        window = ahead & (np.abs(az - math.atan2(pr, pf)) < math.atan2(half_width_m, d)) & (el * d < obstacle_band_m)
        if not window.any() or frame.valid[window].mean() < min_coverage:
            return float(s), False
    return float(along[-1]) if len(along) else 0.0, False
