"""Depth image -> points in the odometry frame, placed with the drone's pose at the frame's CAPTURE time. Pure (numpy).

The on-camera depth (Hinson: /oak/stereo/image_raw, 16UC1 millimetres, 0 = no data) arrives 70-100 ms after it was
captured. Pairing it with the latest pose would put its points where the drone was ~0.1 s later (~5 cm at 0.45 m/s,
~4 deg in a 45 deg/s turn), so the shell keeps a nav.pose_buffer.PoseBuffer of odometry and this module asks it for the
pose at the image's header stamp; if the buffer can't answer (stale odometry, a gap), the frame is dropped, never
placed with a guess.

Frames: optical (x right, y down, z forward) -> base_link (FLU, the static TF from the camera driver) -> odom (ENU,
cuVSLAM's, the take-off frame). points_from_depth is Chelsea's (nav/wall_centring.py, cd284bf), kept identical so the
two consumers agree.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from nav.pose_buffer import PoseBuffer, quat_wxyz_to_mat

# base_link (FLU) <- optical frame, rotation only: forward = z_opt, left = -x_opt, up = -y_opt
R_FLU_OPTICAL = np.array([[0.0, 0.0, 1.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0]])


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


def pose_matrix(p: Any, q_wxyz: Any) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_mat(*np.asarray(q_wxyz, dtype=float))
    T[:3, 3] = np.asarray(p, dtype=float)[:3]
    return T


def depth_to_odom(depth: Any, K: Any, T_base_cam: Any, poses: PoseBuffer, stamp: float, stride: int = 2,
                  scale: float = 0.001, min_m: float = 0.25, max_m: float = 4.5) -> tuple[np.ndarray, np.ndarray] | None:
    """(points (N, 3) in odom, camera position (3,) in odom) for one depth frame, or None if the pose buffer can't
    give the pose at `stamp` (the frame is then dropped).

    T_base_cam: 4x4 base_link <- camera OPTICAL frame (static TF). stride 2 on 320x200 keeps ~16k pixels before the
    ~35% valid fraction: plenty for a 5 cm grid out to 4 m."""
    pose = poses.at(stamp)
    if pose is None:
        return None
    T_odom_cam = pose_matrix(*pose) @ np.asarray(T_base_cam, dtype=float)
    pts = points_from_depth(depth, K, T_odom_cam, stride=stride, min_m=min_m, max_m=max_m, scale=scale)
    return pts, T_odom_cam[:3, 3].copy()
