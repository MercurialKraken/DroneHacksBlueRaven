"""Stereo depth from a rectified mono pair (OpenCV SGBM). Pure: no ROS, no clocks.

Why: the real drone publishes no depth image (disabled to save USB bandwidth; see issues.md), so
map_align drift correction and any live obstacle check have nothing to work with on hardware. This
computes depth on the Jetson from the two camera images, at reduced resolution, a few times a second.

Contract
  input   left, right: rectified mono8 images of equal size (sim /stereo/left|right/image are
          rectified pinhole, which is how cuvslam_lib_node.py uses them)
          fx, fy, cx, cy: left intrinsics at the input resolution (CameraInfo P for rectified)
          baseline_m: distance between the optical centres (TF oak_left->oak_right, or -P[3]/P[0]
          of the right CameraInfo)
  output  DepthResult.depth_m: float32 metres along the left optical axis at the OUTPUT resolution,
          NaN where there is no trustworthy estimate (never 0 or "far")
          DepthResult.fx/fy/cx/cy: intrinsics at the output resolution
Depth = fx * baseline / disparity. A disparity of d pixels at the output scale means
depth resolution ~ depth^2 / (fx * baseline) per pixel, so far depth is coarse by construction.

OpenCV is required. Whether the drone's skybeta container ships cv2 is NOT yet confirmed (the team
rule is numpy-only at runtime, no new pip packages); compute_depth raises RuntimeError if cv2 is
missing instead of silently returning nothing.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class StereoConfig:
    scale: float = 0.5  # resize factor applied to both images before matching
    num_disparities: int = 64  # search range at the OUTPUT scale; multiple of 16
    block_size: int = 5  # odd, 3..11
    uniqueness: int = 10  # SGBM uniquenessRatio (percent)
    speckle_window: int = 100  # SGBM speckleWindowSize (pixels at output scale), 0 disables
    speckle_range: int = 2  # SGBM speckleRange (disparity units)
    disp12_max_diff: int = 1  # left-right consistency check (pixels), <0 disables
    min_disparity_px: float = 1.0  # smaller disparities are too far/noisy to trust
    min_depth_m: float = 0.3
    max_depth_m: float = 6.0

    def __post_init__(self):
        if not (0.1 <= self.scale <= 1.0):
            raise ValueError("scale must be in [0.1, 1.0]")
        if self.num_disparities <= 0 or self.num_disparities % 16:
            raise ValueError("num_disparities must be a positive multiple of 16")
        if self.block_size % 2 == 0 or not (3 <= self.block_size <= 11):
            raise ValueError("block_size must be odd and in 3..11")
        if not (0 < self.min_depth_m < self.max_depth_m):
            raise ValueError("need 0 < min_depth_m < max_depth_m")
        if self.min_disparity_px <= 0:
            raise ValueError("min_disparity_px must be > 0")


@dataclass(frozen=True)
class DepthResult:
    depth_m: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    scale: float
    valid_fraction: float
    elapsed_ms: float
    max_range_m: float  # deepest depth this config can represent (fx*B/min_disparity)
    min_range_m: float  # nearest depth inside the disparity search range


def _cv2():
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "perception.stereo_depth needs OpenCV (python3 -c 'import cv2'); check the drone container"
        ) from e
    return cv2


def _check_inputs(left, right, fx, fy, cx, cy, baseline_m):
    for name, img in (("left", left), ("right", right)):
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 2:
            raise ValueError(f"{name} must be a 2-D uint8 (mono8) image")
    if left.shape != right.shape:
        raise ValueError(f"left {left.shape} and right {right.shape} differ in size")
    if min(left.shape) < 32:
        raise ValueError("images too small for stereo matching")
    vals = (fx, fy, cx, cy, baseline_m)
    if not all(math.isfinite(float(v)) for v in vals):
        raise ValueError("intrinsics and baseline must be finite")
    if fx <= 0 or fy <= 0:
        raise ValueError("fx and fy must be > 0")
    if not (0.005 <= baseline_m <= 1.0):
        raise ValueError(f"implausible baseline {baseline_m} m (expected 0.005..1.0)")


def scaled_intrinsics(fx, fy, cx, cy, scale):
    """Intrinsics after resizing by `scale` (pixel-centre convention)."""
    return fx * scale, fy * scale, (cx + 0.5) * scale - 0.5, (cy + 0.5) * scale - 0.5


def compute_depth(left, right, fx, fy, cx, cy, baseline_m, config: StereoConfig | None = None):
    """Rectified mono pair -> DepthResult. Raises ValueError on bad input, RuntimeError without cv2."""
    c = config or StereoConfig()
    _check_inputs(left, right, fx, fy, cx, cy, baseline_m)
    cv2 = _cv2()
    t0 = time.perf_counter()
    if c.scale != 1.0:
        size = (max(32, round(left.shape[1] * c.scale)), max(32, round(left.shape[0] * c.scale)))
        L = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
        R = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
        sx, sy = size[0] / left.shape[1], size[1] / left.shape[0]
    else:
        L, R, sx, sy = left, right, 1.0, 1.0
    fxs, fys = fx * sx, fy * sy
    cxs, cys = (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5
    b = c.block_size
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=c.num_disparities,
        blockSize=b,
        P1=8 * b * b,
        P2=32 * b * b,
        disp12MaxDiff=c.disp12_max_diff,
        uniquenessRatio=c.uniqueness,
        speckleWindowSize=c.speckle_window,
        speckleRange=c.speckle_range,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = sgbm.compute(L, R).astype(np.float32) / 16.0  # SGBM returns fixed-point disparity * 16
    valid = disp >= c.min_disparity_px
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (fxs * baseline_m) / disp
    valid &= np.isfinite(depth) & (depth >= c.min_depth_m) & (depth <= c.max_depth_m)
    # the leftmost num_disparities columns have no full search range: SGBM leaves them unmatched,
    # but make it explicit so a lucky match there is never trusted
    valid[:, : c.num_disparities] = False
    depth = np.where(valid, depth, np.nan).astype(np.float32)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return DepthResult(
        depth_m=depth,
        fx=fxs,
        fy=fys,
        cx=cxs,
        cy=cys,
        scale=float(sx),
        valid_fraction=float(valid.mean()),
        elapsed_ms=elapsed,
        max_range_m=float(fxs * baseline_m / c.min_disparity_px),
        min_range_m=float(fxs * baseline_m / c.num_disparities),
    )


def depth_to_16uc1_mm(depth_m: np.ndarray) -> np.ndarray:
    """Metres (NaN = invalid) -> uint16 millimetres (0 = invalid), the sim's /camera/depth encoding."""
    d = np.asarray(depth_m, dtype=np.float64)
    out = np.zeros(d.shape, dtype=np.uint16)
    ok = np.isfinite(d) & (d > 0)
    out[ok] = np.clip(np.rint(d[ok] * 1000.0), 1, 65535).astype(np.uint16)
    return out


def config_snapshot(config: StereoConfig | None = None) -> dict:
    return asdict(config or StereoConfig())
