"""depth_scan.slice against ray-cast depth images of known geometry (walls, a corridor, a slanted
wall, net holes, a floor seen while pitched). No ROS."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depth_scan.slice import (  # noqa: E402
    R_BODY_OPTICAL,
    SliceConfig,
    depth_to_metres,
    horizontal_slice,
    rotation_level,
)

W, H = 320, 200  # the drone's on-camera depth (decimation 2)
FX = FY = 160.0  # about 90 deg horizontal field of view
CX, CY = (W - 1) / 2, (H - 1) / 2
T_CAM = np.array([0.104, 0.038, 0.03])  # left camera in base_link, as measured on skybeta-09


def raycast(planes, R_world_cam=R_BODY_OPTICAL, t=T_CAM):
    """planes: [(normal, offset)] with normal . p = offset in the world (levelled body) frame.
    Returns the optical-axis depth image (metres, 0 = no hit)."""
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u, float)], -1) @ R_world_cam.T
    best = np.full((H, W), np.inf)
    for n, c in planes:
        n = np.asarray(n, float)
        den = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (c - n @ t) / den
        s[(den == 0) | (s <= 0)] = np.inf
        best = np.minimum(best, s)
    best[~np.isfinite(best)] = 0.0
    return best.astype(np.float32)


def bin_bounds(sl, fn):
    """For a bin-wise expected range function of bearing, the [min, max] over each bin."""
    half = sl.angle_increment / 2
    b = sl.bearings
    a, c = fn(b - half), fn(b + half)
    return np.minimum(a, c), np.maximum(a, c)


def test_wall_ahead_ranges_follow_the_geometry():
    depth = raycast([((1, 0, 0), 1.5)])
    sl = horizontal_slice(depth, FX, FY, CX, CY, t_base_cam=T_CAM)
    fin = np.isfinite(sl.ranges)
    assert fin.sum() > 45  # most of the ~90 deg field of view
    lo, hi = bin_bounds(sl, lambda b: 1.5 / np.cos(b))
    r = sl.ranges[fin]
    # the bin min may come from anywhere in the bin, plus a hair for the 3D band
    assert np.all(r >= np.minimum(lo, hi)[fin] - 0.01)
    assert np.all(r <= np.maximum(lo, hi)[fin] + 0.01)
    centre = np.argmin(np.abs(sl.bearings))
    assert sl.ranges[centre] == pytest.approx(1.5, abs=0.01)


def test_corridor_sides_finite_far_end_beyond_range():
    depth = raycast([((0, 1, 0), 0.6), ((0, -1, 0), 0.6), ((1, 0, 0), 4.0)])
    sl = horizontal_slice(depth, FX, FY, CX, CY, t_base_cam=T_CAM)
    b = sl.bearings
    centre = np.abs(b) < np.deg2rad(5)
    assert np.all(np.isposinf(sl.ranges[centre]))  # far wall at 4 m > range_max 3 m
    left = (b > np.deg2rad(30)) & np.isfinite(sl.ranges)
    right = (b < np.deg2rad(-30)) & np.isfinite(sl.ranges)
    assert left.sum() > 5 and right.sum() > 5
    # perpendicular distance to each wall from the BODY origin
    assert np.allclose(sl.ranges[left] * np.sin(b[left]), 0.6, atol=0.03)
    assert np.allclose(-sl.ranges[right] * np.sin(b[right]), 0.6, atol=0.03)


def test_slanted_wall_reports_its_nearest_height():
    a = np.deg2rad(30)  # top leans towards the drone: x + tan(a) z = 1.2 (z relative to camera)
    n = np.array([np.cos(a), 0.0, np.sin(a)])
    off = 1.2 * np.cos(a) + n[2] * T_CAM[2]
    sl = horizontal_slice(raycast([(n, off)]), FX, FY, CX, CY, t_base_cam=T_CAM)
    centre = np.argmin(np.abs(sl.bearings))
    expect_min = 1.2 - np.tan(a) * 0.20  # at the top of the +-0.20 m band
    assert sl.ranges[centre] == pytest.approx(expect_min, abs=0.02)
    assert sl.ranges[centre] < 1.2 - 0.05  # clearly nearer than the wall at camera height


def test_net_holes_stay_unknown_not_free():
    depth = raycast([((1, 0, 0), 1.5)])
    depth[:, 150:170] = 0.0  # the stereo found nothing here (through a net, blank texture)
    sl = horizontal_slice(depth, FX, FY, CX, CY, t_base_cam=T_CAM)
    # columns 150-169 span about +-3.6 deg from the optical axis; the camera sits 3.8 cm left of
    # the body origin, so from the body that hole is centred about 1.45 deg left
    shift = np.arctan2(T_CAM[1], 1.5)
    hole = np.abs(sl.bearings - shift) < np.deg2rad(2.0)
    assert hole.any() and np.all(np.isnan(sl.ranges[hole]))
    assert not np.any(np.isposinf(sl.ranges[hole]))


def test_floor_is_outside_the_band_when_level():
    depth = raycast([((0, 0, 1), -0.45 + T_CAM[2])])  # floor 0.45 m below the camera
    sl = horizontal_slice(depth, FX, FY, CX, CY, t_base_cam=T_CAM)
    assert not np.any(np.isfinite(sl.ranges))


def test_pitch_puts_floor_in_band_unless_levelled():
    L = rotation_level(0.0, np.deg2rad(15))  # nose down 15 deg
    R_world_cam = L @ R_BODY_OPTICAL
    floor = ((0, 0, 1), -0.45 + (L @ T_CAM)[2])
    depth = raycast([floor], R_world_cam=R_world_cam, t=L @ T_CAM)
    raw = horizontal_slice(depth, FX, FY, CX, CY, t_base_cam=T_CAM)
    lev = horizontal_slice(depth, FX, FY, CX, CY, t_base_cam=T_CAM, R_level=L)
    assert np.isfinite(raw.ranges).sum() > 10  # the floor would look like a wall
    assert not np.any(np.isfinite(lev.ranges))


def test_wall_on_the_left_has_positive_bearings():
    sl = horizontal_slice(raycast([((0, 1, 0), 0.5)]), FX, FY, CX, CY, t_base_cam=T_CAM)
    fin = np.isfinite(sl.ranges)
    assert fin.any() and np.all(sl.bearings[fin] > 0)


def test_too_close_is_minus_inf():
    sl = horizontal_slice(raycast([((1, 0, 0), 0.25)]), FX, FY, CX, CY, t_base_cam=T_CAM)
    centre = np.argmin(np.abs(sl.bearings))
    assert sl.ranges[centre] == pytest.approx(0.25, abs=0.01)  # 0.25 m from the body: in range
    sl2 = horizontal_slice(raycast([((1, 0, 0), 0.15)]), FX, FY, CX, CY, t_base_cam=T_CAM)
    assert np.isneginf(sl2.ranges[np.argmin(np.abs(sl2.bearings))])


def test_16uc1_millimetres_match_32fc1_metres():
    d = raycast([((1, 0, 0), 1.5), ((0, 1, 0), 0.7)])
    a = horizontal_slice(d, FX, FY, CX, CY, t_base_cam=T_CAM)
    b = horizontal_slice(np.round(d * 1000).astype(np.uint16), FX, FY, CX, CY, t_base_cam=T_CAM)
    both = np.isfinite(a.ranges) & np.isfinite(b.ranges)
    assert both.sum() > 40 and np.allclose(a.ranges[both], b.ranges[both], atol=0.002)
    assert np.array_equal(np.isnan(a.ranges), np.isnan(b.ranges))


def test_depth_conversion_and_validation():
    m = depth_to_metres(np.array([[0, 1500]], np.uint16))
    assert np.isnan(m[0, 0]) and m[0, 1] == pytest.approx(1.5)
    f = depth_to_metres(np.array([[0.0, np.inf, -1.0, 2.0]], np.float32))
    assert np.isnan(f[0, :3]).all() and f[0, 3] == 2.0
    with pytest.raises(ValueError):
        depth_to_metres(np.zeros((2, 2), np.int32))
    with pytest.raises(ValueError):
        horizontal_slice(np.zeros((H, W), np.uint16), FX, FY, CX, CY, R_base_cam=np.eye(3) * 2)
    with pytest.raises(ValueError):
        horizontal_slice(np.zeros((H, W), np.uint16), 0.0, FY, CX, CY)


def test_empty_frame_is_all_unknown():
    sl = horizontal_slice(np.zeros((H, W), np.uint16), FX, FY, CX, CY, stamp=12.5)
    assert np.all(np.isnan(sl.ranges)) and sl.stamp == 12.5


def test_fast_enough_for_30_hz():
    d = raycast([((0, 1, 0), 0.6), ((0, -1, 0), 0.6), ((1, 0, 0), 2.0)])
    horizontal_slice(d, FX, FY, CX, CY, t_base_cam=T_CAM)
    t0 = time.perf_counter()
    for _ in range(20):
        horizontal_slice(d, FX, FY, CX, CY, t_base_cam=T_CAM)
    assert (time.perf_counter() - t0) / 20 < 0.03


def test_config_band_and_min_pixels():
    depth = raycast([((1, 0, 0), 1.5)])
    thin = horizontal_slice(
        depth, FX, FY, CX, CY, t_base_cam=T_CAM, config=SliceConfig(band_m=0.0005, min_pixels=50)
    )
    assert np.isnan(thin.ranges).all()  # too few pixels per bin in a 1 mm band


def _img(arr, encoding, pad=0, big=False):
    class M:  # sensor_msgs/Image double
        pass

    m = M()
    a = arr.astype(arr.dtype.newbyteorder(">" if big else "<"))
    rows = a.view(np.uint8).reshape(arr.shape[0], -1)
    m.data = np.hstack([rows, np.zeros((arr.shape[0], pad), np.uint8)]).tobytes()
    m.height, m.width = arr.shape
    m.step = rows.shape[1] + pad
    m.encoding, m.is_bigendian = encoding, int(big)
    return m


def test_image_to_array_padding_and_byte_order():
    from depth_scan.node import image_to_array

    d = (np.arange(12, dtype=np.uint16).reshape(3, 4) * 300).astype(np.uint16)
    assert np.array_equal(image_to_array(_img(d, "16UC1", pad=6)), d)
    assert np.array_equal(image_to_array(_img(d, "16UC1", big=True)), d)
    f = np.linspace(0.5, 2.0, 12, dtype=np.float32).reshape(3, 4)
    assert np.allclose(image_to_array(_img(f, "32FC1", pad=4)), f)
    with pytest.raises(ValueError):
        image_to_array(_img(d, "rgb8"))


def test_standard_optical_frame_quaternion_matches():
    from depth_scan.node import _quat_to_R

    # base_link <- *_optical_frame, the usual ROS static transform (x right/y down/z forward)
    assert np.allclose(_quat_to_R(-0.5, 0.5, -0.5, 0.5), R_BODY_OPTICAL)


def test_node_reads_only_depth_and_never_maps_or_px4():
    src = (Path(__file__).resolve().parents[1] / "depth_scan" / "node.py").read_text()
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    for bad in ("course_map", "map_align", "maps/", "create_client", "VehicleCommand"):
        assert bad not in code
    assert '"/oak/stereo/image_raw"' in code  # the on-camera depth, not the camera images
    for f in ("slice.py", "__init__.py"):
        s = (Path(__file__).resolve().parents[1] / "depth_scan" / f).read_text()
        assert "import rclpy" not in s and "course_map" not in s
