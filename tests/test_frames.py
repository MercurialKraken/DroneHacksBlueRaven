"""nav.frames: the launch pose from course.yaml and map <- odom. A temporary YAML, no course numbers."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("yaml")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nav.frames import load_launch, map_from_odom, quaternion_z  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "course.yaml"
    p.write_text("launch: {x: 1.0, y: 2.0, yaw_deg: 90.0, base_link_z: 0.07}\n"
                 "launch_sim: {x: 1.0, y: 2.2, yaw_deg: 0.0}\n", encoding="utf-8")
    return p


def test_load_launch_entries(cfg):
    assert load_launch(cfg) == {"x": 1.0, "y": 2.0, "yaw_deg": 90.0, "z": 0.07}
    assert load_launch(cfg, "launch_sim")["z"] == 0.0
    with pytest.raises(KeyError, match="launch_other"):
        load_launch(cfg, "launch_other")


def test_facing_north_is_a_pure_translation(cfg):
    T = map_from_odom(load_launch(cfg))
    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(T @ [0, 0, 0, 1], [1.0, 2.0, 0.07, 1])          # odom origin = base_link at launch
    assert np.allclose(T @ [0, 1, 0, 1], [1.0, 3.0, 0.07, 1])          # 1 m PX4-north = 1 m course-north


def test_facing_east_rotates_px4_north_onto_course_east(cfg):
    T = map_from_odom(load_launch(cfg, "launch_sim"))                  # launched facing +x (yaw 0)
    assert np.allclose(T @ [0, 1, 0, 1], [2.0, 2.2, 0.0, 1])           # PX4 north (forward) -> course +x
    assert np.allclose(T @ [1, 0, 0, 1], [1.0, 1.2, 0.0, 1])           # PX4 east (right)   -> course -y


def test_quaternion_matches_the_rotation():
    x, y, z, w = quaternion_z(math.radians(-90.0))
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z)]])
    T = map_from_odom({"x": 0.0, "y": 0.0, "yaw_deg": 0.0, "z": 0.0})
    assert np.allclose(R, T[:2, :2])
