"""'Is there a wall straight ahead, and how far?' from one depth image. PURE: numpy only, no ROS.

The check looks only at the CORRIDOR the drone would sweep flying straight ahead: points whose sideways offset from
the drone's centre line is within `half_width_m` (drone radius + margin) and whose height is within a band around the
camera (the floor and anything far above are ignored). The range is a low percentile of those points' forward
distance, so a few noisy pixels can't create or hide a wall, and too few points means 'nothing seen'.

Frames: the depth image is in an optical frame (x right, y down, z forward). The camera's mounting relative to the
drone's centre (base_link) is a pure translation on skybeta-09 (tf_static: oak 0.1053 m forward, 0.0304 m up, no
rotation; the depth is aligned to the LEFT camera, 0.0376 m left of the OAK's centre)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraMount:
    forward_m: float = 0.1053     # camera ahead of the drone's centre (Darion's tape: 4 in = 0.102 m)
    left_m: float = 0.0376        # the depth image's camera, left of the drone's centre line
    up_m: float = 0.0304          # camera above the drone's centre


@dataclass
class CorridorParams:
    half_width_m: float = 0.34    # drone radius 0.29 + 0.05 margin, each side of the centre line
    up_m: float = 0.40            # count points up to this far above the camera
    down_m: float = 0.20          # ...and down to this far below it (the floor at 0.4 m flight height is excluded)
    min_range_m: float = 0.15     # stereo can't see closer; nearer returns are noise
    max_range_m: float = 4.0
    percentile: float = 10.0      # robust 'nearest': the 10th percentile of the corridor's forward distances
    min_points: int = 40          # fewer corridor points than this (after the stride) = nothing seen
    stride: int = 4               # use every 4th pixel each way (640x400 -> 160x100)


@dataclass
class Ahead:
    range_m: float                # forward distance from the DRONE'S CENTRE to the nearest corridor surface; inf if none
    points: int                   # corridor points counted
    valid_frac: float             # fraction of the sampled pixels with a valid depth (texture/holes indicator)


def decode(encoding: str, height: int, width: int, step: int, data: bytes) -> np.ndarray:
    """A sensor_msgs/Image depth payload -> float32 metres, NaN where invalid."""
    if encoding in ("16UC1", "mono16"):
        a = np.frombuffer(data, dtype="<u2", count=height * step // 2).reshape(height, step // 2)[:, :width]
        d = a.astype(np.float32) * 0.001
        d[a == 0] = np.nan
        return d
    if encoding == "32FC1":
        d = np.frombuffer(data, dtype="<f4", count=height * step // 4).reshape(height, step // 4)[:, :width].astype(np.float32)
        d[~np.isfinite(d) | (d <= 0)] = np.nan
        return d
    raise ValueError(f"unsupported depth encoding {encoding!r}")


def ahead(depth_m: np.ndarray, K: tuple[float, float, float, float], mount: CameraMount = CameraMount(),
          p: CorridorParams = CorridorParams()) -> Ahead:
    """depth_m: HxW metres (NaN invalid); K = (fx, fy, cx, cy) of that image."""
    fx, fy, cx, cy = K
    s = max(1, int(p.stride))
    d = depth_m[::s, ::s]
    vs, us = np.mgrid[0:depth_m.shape[0]:s, 0:depth_m.shape[1]:s]
    ok = np.isfinite(d) & (d >= p.min_range_m) & (d <= p.max_range_m)
    valid_frac = float(ok.mean()) if ok.size else 0.0
    z = d[ok]
    x = (us[ok] - cx) / fx * z            # right of the camera
    y = (vs[ok] - cy) / fy * z            # below the camera
    lateral = x - mount.left_m            # right of the drone's centre line (the camera sits left of it)
    keep = (np.abs(lateral) <= p.half_width_m) & (y >= -p.up_m) & (y <= p.down_m)
    n = int(keep.sum())
    if n < p.min_points:
        return Ahead(float("inf"), n, valid_frac)
    return Ahead(float(np.percentile(z[keep], p.percentile)) + mount.forward_m, n, valid_frac)
