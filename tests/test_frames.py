"""nav.frames: the launch pose from course.yaml and map <- odom. A temporary YAML, no course numbers."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("yaml")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nav.frames import (MAP_FROM_ODOM_INIT_TOPIC, MAP_FROM_ODOM_TOPIC, correction_in_odom,  # noqa: E402
                        load_launch, map_from_odom, map_from_startup_odom, pose_msg_fields, pose_xyz_yaw,
                        quaternion_z, transform_from_pose_fields, yaw_from_quaternion)


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


def test_correction_in_odom_matches_the_ned_definition():
    """map_align's rule in NED: p' = R_ned(yaw) p + (xN, yE), heading' = heading + yaw (clockwise).
    The same correction applied in odom (ENU) must give the same corrected point and heading."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        xN, yE, yaw = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.3, 0.3)
        n, e, h = rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-math.pi, math.pi)
        c, s = math.cos(yaw), math.sin(yaw)
        n2, e2 = c * n - s * e + xN, s * n + c * e + yE                  # NED rule
        h2 = h + yaw
        T = correction_in_odom(xN, yE, yaw)
        p = T @ [e, n, 0.0, 1.0]                                            # the same point in ENU
        assert np.allclose(p[:2], [e2, n2])
        fwd_enu = [math.sin(h), math.cos(h), 0.0]                           # heading h, as an ENU vector
        f2 = T[:3, :3] @ fwd_enu
        assert np.allclose(f2, [math.sin(h2), math.cos(h2), 0.0])


LAUNCH_N = {"x": -2.5, "y": 0.9, "yaw_deg": 90.0, "z": 0.07}            # on the mark, facing course north


def test_startup_odom_at_the_mark_maps_forward_to_the_launch_heading():
    T = map_from_startup_odom(LAUNCH_N, (0.0, 0.0, 0.0, 0.0))            # stack started on the mark
    assert np.allclose(T @ [0, 0, 0, 1], [-2.5, 0.9, 0.07, 1])             # odom origin = the mark
    assert np.allclose(T @ [1, 0, 0, 1], [-2.5, 1.9, 0.07, 1])             # 1 m forward = 1 m north
    assert np.allclose(T @ [0, 1, 0, 1], [-3.5, 0.9, 0.07, 1])             # 1 m left (west) = -x


def test_startup_odom_started_elsewhere_still_puts_the_drone_on_the_mark():
    # the stack started 1.2 m away and 30 deg rotated, then the drone was carried to the mark
    now = (1.2, -0.4, 0.03, math.radians(30.0))
    T = map_from_startup_odom(LAUNCH_N, now)
    base_now = pose_xyz_yaw(*now)
    assert np.allclose((T @ base_now)[:3, 3], [-2.5, 0.9, 0.07])           # the drone is on the mark...
    fwd = T @ base_now @ [1, 0, 0, 1]                                      # ...and its nose points north
    assert np.allclose(fwd[:3], [-2.5, 1.9, 0.07])
    later = pose_xyz_yaw(*now) @ pose_xyz_yaw(0.5, 0.0, 0.7, 0.0)          # flown 0.5 m forward, 0.7 m up
    assert np.allclose((T @ later)[:3, 3], [-2.5, 1.4, 0.77])
    assert np.allclose(np.linalg.inv(T) @ T, np.eye(4))                    # round trip


def test_map_from_odom_message_layout_round_trips_and_pins_the_yaw_sign():
    """The drone-frames interface (adapter <-> map_align): position = the odom origin in the map,
    orientation = yaw counter-clockwise about +z. Both nodes go through these two functions."""
    assert MAP_FROM_ODOM_INIT_TOPIC == "/blueraven/map_from_odom_init"
    assert MAP_FROM_ODOM_TOPIC == "/blueraven/map_from_odom"
    T = map_from_startup_odom(LAUNCH_N, (1.2, -0.4, 0.03, math.radians(30.0)))
    f = pose_msg_fields(T)
    assert np.allclose(f[:3], (T @ [0, 0, 0, 1])[:3])                       # the odom origin, in map
    assert np.allclose(transform_from_pose_fields(*f), T)                     # round trip
    # yaw sign: a frame rotated +90 deg (CCW) maps odom +x onto map +y
    g = pose_msg_fields(pose_xyz_yaw(0.0, 0.0, 0.0, math.radians(90.0)))
    assert np.allclose(transform_from_pose_fields(*g) @ [1, 0, 0, 1], [0, 1, 0, 1])
    assert g[5] > 0 and g[6] > 0                                              # qz, qw: +90 deg about +z


def test_yaw_from_quaternion():
    for a in (-2.0, -0.3, 0.0, 0.7, 3.0):
        x, y, z, w = quaternion_z(a)
        assert yaw_from_quaternion(x, y, z, w) == pytest.approx(a)


def test_quaternion_matches_the_rotation():
    x, y, z, w = quaternion_z(math.radians(-90.0))
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z)]])
    T = map_from_odom({"x": 0.0, "y": 0.0, "yaw_deg": 0.0, "z": 0.0})
    assert np.allclose(R, T[:2, :2])
