"""Depth from a stereo pair already on the drone (Task C). PURE: numpy + OpenCV, no ROS.

The organizers' skybeta stack publishes the OAK's left/right images (1024x640 bgr8 at 30 Hz, raw and
unrectified) but no depth. This turns one pair into a depth image in the form refnav, map_align and
live_map already consume: a pinhole (rectified) image whose CameraInfo K is all they need, 16UC1 mm or
32FC1 m, in an optical frame (REP-103: x right, y down, z forward).

Steps, each at a reduced resolution (scale), because a few Hz is enough and the Jetson is shared:
  1. to grey and shrink (INTER_AREA);
  2. undistort + rectify with the cameras' own calibration: the CameraInfo R/P when they already hold a
     stereo rectification (right P[0,3] != 0, as in the sim), else cv2.stereoRectify from K/D plus the
     right<-left extrinsic (from TF);
  3. semi-global matching (cv2.StereoSGBM, 3-way mode), disparity -> depth z = f * B / d;
  4. depth outside [min_m, max_m] or with too little disparity is invalid (NaN, 0 in 16UC1).

The rectified left camera is rotated from the raw one by R1 (X_rect = R1 @ X_raw). When R1 is not the
identity the depth image lives in a new frame, and the node publishes TF raw_left <- rect (rotation R1^T).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except ImportError:                       # the module still imports (and its pure helpers still work)
    cv2 = None


def _need_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is needed for rectification and stereo matching")


# --- image decoding -------------------------------------------------------------------------------------

_CHANNELS = {"mono8": 1, "8uc1": 1, "bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4}


def to_gray(encoding: str, height: int, width: int, step: int, data: bytes) -> np.ndarray:
    """sensor_msgs/Image (8-bit) -> uint8 HxW grey. step may include row padding."""
    enc = encoding.lower()
    if enc not in _CHANNELS:
        raise ValueError(f"unsupported image encoding {encoding!r} (need mono8, bgr8, rgb8, bgra8 or rgba8)")
    c = _CHANNELS[enc]
    if step < width * c or len(data) < step * height:
        raise ValueError(f"image buffer too small: {len(data)} bytes for {height} rows of step {step}")
    a = np.frombuffer(data, np.uint8, count=step * height).reshape(height, step)[:, :width * c]
    if c == 1:
        return np.ascontiguousarray(a)
    a = a.reshape(height, width, c).astype(np.float32)
    if enc.startswith("bgr"):
        b, g, r = a[..., 0], a[..., 1], a[..., 2]
    else:
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return np.clip(0.299 * r + 0.587 * g + 0.114 * b + 0.5, 0, 255).astype(np.uint8)   # ITU-R BT.601, as OpenCV


# --- calibration ----------------------------------------------------------------------------------------

@dataclass
class CameraModel:
    """The fields of sensor_msgs/CameraInfo this needs (row-major K 3x3, R 3x3, P 3x4)."""
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray = field(default_factory=lambda: np.zeros(5))
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    P: np.ndarray | None = None
    model: str = "plumb_bob"
    frame_id: str = ""

    @staticmethod
    def from_lists(width, height, K, D=(), R=(), P=(), model="plumb_bob", frame_id="") -> "CameraModel":
        K = np.asarray(K, float).reshape(3, 3)
        R = np.asarray(R, float).reshape(3, 3) if len(R) == 9 and any(R) else np.eye(3)
        P = np.asarray(P, float).reshape(3, 4) if len(P) == 12 and any(P) else None
        return CameraModel(int(width), int(height), K, np.asarray(D, float).ravel(), R, P, model, frame_id)

    def scaled(self, s: float) -> "CameraModel":
        """The same camera at s times the resolution (pixel centres kept: c' = (c + 0.5) s - 0.5)."""
        S = np.array([[s, 0.0, 0.5 * s - 0.5], [0.0, s, 0.5 * s - 0.5], [0.0, 0.0, 1.0]])
        P = None if self.P is None else S @ self.P
        return CameraModel(int(round(self.width * s)), int(round(self.height * s)), S @ self.K, self.D.copy(),
                           self.R.copy(), P, self.model, self.frame_id)


def _cv_dist(m: CameraModel) -> np.ndarray:
    if m.model == "equidistant":
        raise ValueError("fisheye (equidistant) calibration is not supported")
    return m.D if m.D.size else np.zeros(5)


class StereoRectifier:
    """Undistort + rectify + shrink a raw stereo pair, with the calibration's rectified pinhole.

    Uses the CameraInfo stereo rectification (R, P) when the right P carries a baseline; otherwise needs the
    right<-left extrinsic (R_rl, t_rl: X_right = R_rl @ X_left + t_rl, metres) and runs cv2.stereoRectify.
    """

    def __init__(self, left: CameraModel, right: CameraModel, scale: float = 0.5,
                 R_rl: np.ndarray | None = None, t_rl: np.ndarray | None = None) -> None:
        _need_cv2()
        if (left.width, left.height) != (right.width, right.height):
            raise ValueError("left and right images differ in size")
        if not 0.1 <= scale <= 1.0:
            raise ValueError("scale must be in [0.1, 1]")
        self.scale = scale
        L, Rt = left.scaled(scale), right.scaled(scale)
        self.size = (L.width, L.height)
        have_cal = (left.P is not None and right.P is not None and abs(right.P[0, 3]) > 1e-9)
        if have_cal:
            self.source = "camera_info"
            R1, R2, P1, P2 = L.R, Rt.R, L.P, Rt.P
        else:
            if R_rl is None or t_rl is None:
                raise ValueError("camera_info holds no stereo rectification: the right<-left extrinsic is needed")
            self.source = "stereoRectify"
            R1, R2, P1, P2, *_ = cv2.stereoRectify(L.K, _cv_dist(L), Rt.K, _cv_dist(Rt), self.size,
                                                   np.asarray(R_rl, float), np.asarray(t_rl, float).reshape(3, 1),
                                                   flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
        self.R1, self.R2 = np.asarray(R1, float), np.asarray(R2, float)
        self.P1, self.P2 = np.asarray(P1, float), np.asarray(P2, float)
        self.f = float(self.P1[0, 0])
        self.baseline_m = float(-self.P2[0, 3] / self.P2[0, 0])
        if self.baseline_m <= 0:
            raise ValueError(f"baseline {self.baseline_m:.4f} m: is the right camera on the right?")
        if abs(self.P1[0, 2] - self.P2[0, 2]) > 0.5 or abs(self.P1[1, 2] - self.P2[1, 2]) > 0.5:
            raise ValueError("left/right rectified principal points differ; disparity would be offset")
        self.K_out = self.P1[:, :3].copy()
        # already rectified (the sim): no remap, and the depth stays in the left camera's own frame
        self.identity = (np.allclose(self.R1, np.eye(3), atol=1e-9) and not np.any(_cv_dist(L))
                         and np.allclose(L.K, self.K_out, atol=1e-6))
        right_identity = (np.allclose(self.R2, np.eye(3), atol=1e-9) and not np.any(_cv_dist(Rt))
                          and np.allclose(Rt.K, self.P2[:, :3], atol=1e-6))
        m = cv2.CV_16SC2
        self.maps_l = None if self.identity else cv2.initUndistortRectifyMap(L.K, _cv_dist(L), self.R1, self.K_out, self.size, m)
        self.maps_r = None if right_identity else \
            cv2.initUndistortRectifyMap(Rt.K, _cv_dist(Rt), self.R2, self.P2[:, :3], self.size, m)

    def _prep(self, gray: np.ndarray, maps) -> np.ndarray:
        if gray.shape[1] != self.size[0] or gray.shape[0] != self.size[1]:
            gray = cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA)
        if maps is None:
            return gray
        return cv2.remap(gray, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def rectify(self, gray_l: np.ndarray, gray_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._prep(gray_l, self.maps_l), self._prep(gray_r, self.maps_r)

    def raw_from_rect_quat_xyzw(self) -> tuple[float, float, float, float]:
        """TF raw left optical <- rectified left optical (rotation R1^T), as a quaternion x, y, z, w."""
        return quat_xyzw(self.R1.T)


def check_baseline(baseline_m: float, expected_m: float, tol_m: float) -> str | None:
    """None if the derived baseline is plausible, else why not (the node then refuses to start).
    expected_m = 0 disables the check. The OAK on skybeta-09 is 75.3 mm; the T4 sim's is 75.7 mm."""
    if expected_m <= 0 or abs(baseline_m - expected_m) <= tol_m:
        return None
    return (f"derived stereo baseline {baseline_m:.3f} m is not {expected_m:.3f} +- {tol_m:.3f} m: "
            "wrong extrinsic or calibration source; refusing to publish depth")


def quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (x / n, y / n, z / n, w / n)


# --- matching -------------------------------------------------------------------------------------------

@dataclass
class DepthParams:
    min_m: float = 0.35           # nearest depth to resolve; sets the disparity search range
    max_m: float = 6.0            # beyond this, one pixel of disparity is > ~0.5 m of depth: unusable
    block: int = 5                # SGBM block size (odd)
    uniqueness: int = 10
    speckle_window: int = 100
    speckle_range: int = 2
    lr_check_px: int = 1          # disp12MaxDiff: left/right consistency (occlusions)


class StereoDepth:
    """Rectified grey pair -> depth in metres (float32, NaN = invalid)."""

    def __init__(self, f_px: float, baseline_m: float, p: DepthParams | None = None) -> None:
        _need_cv2()
        self.p = p or DepthParams()
        self.fB = float(f_px) * float(baseline_m)
        n = int(math.ceil(self.fB / self.p.min_m / 16.0)) * 16           # disparity range must be a multiple of 16
        self.num_disp = max(16, n)
        b = self.p.block | 1
        self.sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=self.num_disp, blockSize=b,
                                          P1=8 * b * b, P2=32 * b * b, disp12MaxDiff=self.p.lr_check_px,
                                          uniquenessRatio=self.p.uniqueness, speckleWindowSize=self.p.speckle_window,
                                          speckleRange=self.p.speckle_range, preFilterCap=31,
                                          mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.min_disp = self.fB / self.p.max_m

    def disparity(self, rect_l: np.ndarray, rect_r: np.ndarray) -> np.ndarray:
        return self.sgbm.compute(rect_l, rect_r).astype(np.float32) / 16.0

    def depth(self, rect_l: np.ndarray, rect_r: np.ndarray) -> np.ndarray:
        return disparity_to_depth(self.disparity(rect_l, rect_r), self.fB, self.p.min_m, self.p.max_m)


def disparity_to_depth(disp: np.ndarray, fB: float, min_m: float, max_m: float) -> np.ndarray:
    d = np.asarray(disp, np.float32)
    z = np.full(d.shape, np.nan, np.float32)
    ok = d > (fB / max_m)
    z[ok] = fB / d[ok]
    z[(z < min_m) | (z > max_m)] = np.nan
    return z


def to_16uc1(depth_m: np.ndarray) -> np.ndarray:
    """metres (NaN invalid) -> uint16 millimetres (0 invalid), as the depth consumers decode it."""
    z = np.asarray(depth_m, np.float32)
    mm = np.where(np.isfinite(z) & (z > 0), np.rint(z * 1000.0), 0.0)
    return np.clip(mm, 0, 65535).astype(np.uint16)


# --- the whole frame ------------------------------------------------------------------------------------

class StereoDepthPipeline:
    def __init__(self, left: CameraModel, right: CameraModel, scale: float = 0.5, p: DepthParams | None = None,
                 R_rl: np.ndarray | None = None, t_rl: np.ndarray | None = None) -> None:
        self.rect = StereoRectifier(left, right, scale, R_rl, t_rl)
        self.matcher = StereoDepth(self.rect.f, self.rect.baseline_m, p)

    def __call__(self, gray_l: np.ndarray, gray_r: np.ndarray) -> np.ndarray:
        rl, rr = self.rect.rectify(gray_l, gray_r)
        z = self.matcher.depth(rl, rr)
        z[:, :self.matcher.num_disp] = np.nan          # no right-image overlap there: SGBM can't match
        return z

    @property
    def K(self) -> np.ndarray:
        return self.rect.K_out

    @property
    def size(self) -> tuple[int, int]:
        return self.rect.size
