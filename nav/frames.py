"""The `map` <- `odom` transform at launch, from maps/course.yaml. Pure (numpy; pyyaml to read the file).

`map` is the course file's frame (x east, y north, z up). `odom` is PX4's local frame as
onboard/adapters/px4_odom.py publishes it: ENU with x = PX4 east and y = PX4 north, origin at
base_link where the estimator started. PX4 north is the drone's heading at launch, so

    map <- odom = translate(launch x, y, base_link_z) . rotate_z(launch yaw - 90 deg)

Facing course north (yaw 90 deg) at launch, the rotation is the identity. This is the starting
value of the transform the drift correction (todos.md Task 2) keeps up to date.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def load_launch(config: str | Path, key: str = "launch") -> dict[str, float]:
    """{x, y, yaw_deg, z} of a launch entry in course.yaml: 'launch' (the published mark, real
    drone) or 'launch_sim' (where the simulator spawns)."""
    import yaml                                   # ships with every ROS 2 install
    cfg: dict[str, Any] = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    if key not in cfg:
        have = sorted(k for k in cfg if k.startswith("launch"))
        raise KeyError(f"{config} has no {key!r} entry (have {have})")
    L = cfg[key]
    return {"x": float(L["x"]), "y": float(L["y"]), "yaw_deg": float(L["yaw_deg"]),
            "z": float(L.get("base_link_z", 0.0))}


def map_from_odom(launch: dict[str, float]) -> np.ndarray:
    """4x4 transform taking odom coordinates to map coordinates."""
    a = math.radians(launch["yaw_deg"] - 90.0)
    T = np.eye(4)
    T[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    T[:3, 3] = (launch["x"], launch["y"], launch["z"])
    return T


def correction_in_odom(x_north: float, y_east: float, yaw_cw: float) -> np.ndarray:
    """The drift correction published on /blueraven/odom_correction, as a 4x4 transform in `odom`.

    The message is an SE2 correction C in PX4 local NED (map_align_node / refnav):
        p_corrected = R_ned(yaw_cw) . p_px4 + (x_north, y_east),   heading_corrected = heading + yaw_cw
    with yaw_cw positive clockwise seen from above (PX4's heading sense). `odom` is the same frame
    in ENU (x = east, y = north), where that is a counter-clockwise rotation by -yaw_cw and a
    translation of (y_east, x_north). Apply it as T_map_cam = T_map_odom . C . T_odom_cam."""
    c, s = math.cos(-yaw_cw), math.sin(-yaw_cw)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = (y_east, x_north)
    return T


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Rotation about the quaternion frame's z axis (the yaw), in radians."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_z(angle_rad: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation about z, as ROS messages order it."""
    return 0.0, 0.0, math.sin(angle_rad / 2), math.cos(angle_rad / 2)
