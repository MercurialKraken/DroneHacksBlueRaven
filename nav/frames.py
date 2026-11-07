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


def quaternion_z(angle_rad: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation about z, as ROS messages order it."""
    return 0.0, 0.0, math.sin(angle_rad / 2), math.cos(angle_rad / 2)
