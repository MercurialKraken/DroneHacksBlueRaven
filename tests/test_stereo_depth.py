"""onboard.depth.stereo_depth: decode, calibration, rectification and depth on synthetic stereo."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
cv2 = pytest.importorskip("cv2")
from onboard.depth.stereo_depth import (CameraModel, DepthParams, StereoDepthPipeline, StereoRectifier,  # noqa: E402
                                        disparity_to_depth, quat_xyzw, to_16uc1, to_gray)
from refnav.depth import decode_depth, quat_to_mat  # noqa: E402

# the sim's calibration (T4 bags): 640x400, already rectified, baseline 31.866 / 421.016 = 0.0757 m
SIM_K = [421.016, 0, 319.248, 0, 421.016, 200.659, 0, 0, 1]
SIM_L = CameraModel.from_lists(640, 400, SIM_K, [0] * 5, [1, 0, 0, 0, 1, 0, 0, 0, 1],
                               [421.016, 0, 319.248, 0, 0, 421.016, 200.659, 0, 0, 0, 1, 0], frame_id="oak_left")
SIM_R = CameraModel.from_lists(640, 400, SIM_K, [0] * 5, [1, 0, 0, 0, 1, 0, 0, 0, 1],
                               [421.016, 0, 319.248, -31.866, 0, 421.016, 200.659, 0, 0, 0, 1, 0], frame_id="oak_right")


def texture(h, w, seed=0):
    """Random blobs at several scales: matchable everywhere, like a printed target."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)
    for k in (2, 4, 8, 16):
        small = rng.random((h // k + 2, w // k + 2)).astype(np.float32)
        img += cv2.resize(small, (w + 2 * k, h + 2 * k), interpolation=cv2.INTER_CUBIC)[k:k + h, k:k + w] / k
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 220 + 20).astype(np.uint8)


def shift(img, d):
    """The right view of a fronto-parallel plane: x_right = x_left - d (sub-pixel)."""
    M = np.float32([[1, 0, -d], [0, 1, 0]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# --- decoding ------------------------------------------------------------------------------------------------

def test_to_gray_matches_opencv_and_honours_row_padding():
    rng = np.random.default_rng(1)
    bgr = rng.integers(0, 256, (6, 5, 3), np.uint8)
    padded = np.zeros((6, 5 * 3 + 7), np.uint8)
    padded[:, :15] = bgr.reshape(6, 15)
    g = to_gray("bgr8", 6, 5, 22, padded.tobytes())
    assert np.abs(g.astype(int) - cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(int)).max() <= 1
    g2 = to_gray("rgb8", 6, 5, 22, np.ascontiguousarray(padded).tobytes())
    assert np.abs(g2.astype(int) - cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY).astype(int)).max() <= 1
    mono = rng.integers(0, 256, (4, 3), np.uint8)
    assert (to_gray("mono8", 4, 3, 3, mono.tobytes()) == mono).all()


def test_to_gray_rejects_bad_input():
    with pytest.raises(ValueError):
        to_gray("16UC1", 2, 2, 4, bytes(8))
    with pytest.raises(ValueError):
        to_gray("bgr8", 4, 4, 12, bytes(10))


# --- calibration ---------------------------------------------------------------------------------------------

def test_scaling_keeps_pixel_centres():
    m = SIM_L.scaled(0.5)
    assert (m.width, m.height) == (320, 200)
    assert m.K[0, 0] == pytest.approx(210.508) and m.K[0, 2] == pytest.approx((319.248 + 0.5) * 0.5 - 0.5)
    assert SIM_R.scaled(0.5).P[0, 3] == pytest.approx(-15.933)


def test_sim_calibration_is_used_as_is():
    r = StereoRectifier(SIM_L, SIM_R, scale=0.5)
    assert r.source == "camera_info" and r.identity and r.maps_l is None and r.maps_r is None
    assert r.baseline_m == pytest.approx(31.866 / 421.016)
    assert r.f == pytest.approx(210.508)
    assert r.raw_from_rect_quat_xyzw() == pytest.approx((0, 0, 0, 1))


def test_without_a_stereo_calibration_the_extrinsic_is_required():
    L = CameraModel.from_lists(640, 400, SIM_K)
    with pytest.raises(ValueError):
        StereoRectifier(L, CameraModel.from_lists(640, 400, SIM_K))


# a raw OAK-like pair: 1024x640, radial-tangential distortion, the right camera 7.5 cm to the right and
# rotated by about a degree (what the organizers' raw /oak/*/image_raw + camera_info look like)
RAW_K = np.array([[560.0, 0, 515.0], [0, 560.0, 318.0], [0, 0, 1]])
RAW_KR = np.array([[562.0, 0, 509.0], [0, 561.0, 323.0], [0, 0, 1]])
RAW_DL = np.array([-0.08, 0.02, 0.0005, -0.0004, 0.0])
RAW_DR = np.array([-0.075, 0.018, -0.0003, 0.0006, 0.0])
R_RL = cv2.Rodrigues(np.array([0.004, -0.012, 0.006]))[0]
T_RL = np.array([-0.075, 0.0006, 0.0011])          # X_right = R_rl X_left + t: the right camera sits at +x


def raw_models():
    L = CameraModel.from_lists(1024, 640, RAW_K.ravel(), RAW_DL, frame_id="oak_left_optical")
    R = CameraModel.from_lists(1024, 640, RAW_KR.ravel(), RAW_DR, frame_id="oak_right_optical")
    return L, R


def test_rectified_correspondences_share_rows_and_give_the_right_disparity():
    L, R = raw_models()
    r = StereoRectifier(L, R, scale=0.5, R_rl=R_RL, t_rl=T_RL)
    assert r.source == "stereoRectify" and not r.identity
    assert r.baseline_m == pytest.approx(np.linalg.norm(T_RL), rel=1e-3)
    Ls, Rs = L.scaled(0.5), R.scaled(0.5)
    rng = np.random.default_rng(2)
    X = np.column_stack([rng.uniform(-1, 1, 50), rng.uniform(-0.6, 0.6, 50), rng.uniform(0.8, 4.0, 50)])  # left optical
    ul = cv2.projectPoints(X, np.zeros(3), np.zeros(3), Ls.K, Ls.D)[0].reshape(-1, 2)
    ur = cv2.projectPoints(X, cv2.Rodrigues(R_RL)[0], T_RL, Rs.K, Rs.D)[0].reshape(-1, 2)
    rl = cv2.undistortPoints(ul.reshape(-1, 1, 2), Ls.K, Ls.D, R=r.R1, P=r.P1).reshape(-1, 2)
    rr = cv2.undistortPoints(ur.reshape(-1, 1, 2), Rs.K, Rs.D, R=r.R2, P=r.P2).reshape(-1, 2)
    assert np.abs(rl[:, 1] - rr[:, 1]).max() < 0.05                         # rows agree (pixels)
    z_rect = (X @ r.R1.T)[:, 2]                                              # depth in the rectified frame
    assert np.abs((rl[:, 0] - rr[:, 0]) - r.f * r.baseline_m / z_rect).max() < 0.05


def test_the_rectification_maps_move_a_raw_pixel_where_the_calibration_says():
    L, R = raw_models()
    r = StereoRectifier(L, R, scale=0.5, R_rl=R_RL, t_rl=T_RL)
    Ls = L.scaled(0.5)
    for u, v in ((100.0, 80.0), (256.0, 160.0), (430.0, 270.0)):
        img = np.zeros((Ls.height, Ls.width), np.uint8)
        cv2.circle(img, (int(u), int(v)), 3, 255, -1)
        out, _ = r.rectify(img, img)
        ys, xs = np.nonzero(out > 100)
        want = cv2.undistortPoints(np.array([[[u, v]]]), Ls.K, Ls.D, R=r.R1, P=r.P1).ravel()
        assert abs(xs.mean() - want[0]) < 1.0 and abs(ys.mean() - want[1]) < 1.0


# --- the real skybeta-09 calibration (arnav-claude's read-only checks, 2026-11-07) ----------------------------
# Both CameraInfo: 1024x640, rational_polynomial with 14 zero coefficients, R = I. The baseline sits on the LEFT P
# (P_left[0,3] = -48.381 -> 75.3 mm) and P_right[0,3] = 0 (reversed from ROS), and cy differs by 21 px: CameraInfo is
# no rectification. The static TF holds the factory extrinsic: tf2_echo oak_left_camera_optical_frame
# oak_right_camera_optical_frame (the right camera expressed in the left) = t [0.075, -0.001, -0.002] m,
# RPY [0.325, -0.338, 0.207] deg.
SKY_KL = [642.443, 0, 502.110, 0, 642.161, 309.373, 0, 0, 1]
SKY_KR = [639.476, 0, 510.784, 0, 639.036, 330.313, 0, 0, 1]
SKY_L = CameraModel.from_lists(1024, 640, SKY_KL, [0.0] * 14, [1, 0, 0, 0, 1, 0, 0, 0, 1],
                               [642.443, 0, 502.110, -48.381, 0, 642.161, 309.373, 0, 0, 0, 1, 0],
                               "rational_polynomial", "oak_left_camera_optical_frame")
SKY_R = CameraModel.from_lists(1024, 640, SKY_KR, [0.0] * 14, [1, 0, 0, 0, 1, 0, 0, 0, 1],
                               [639.476, 0, 510.784, 0, 0, 639.036, 330.313, 0, 0, 0, 1, 0],
                               "rational_polynomial", "oak_right_camera_optical_frame")


def skybeta_extrinsic():
    """TF left <- right (as tf2_echo printed it) -> X_right = R_rl X_left + t_rl, what the node looks up."""
    r, p, y = (math.radians(v) for v in (0.325, -0.338, 0.207))
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    R_lr = Rz @ Ry @ Rx                                       # tf2 RPY: fixed-axis roll, pitch, yaw
    t_lr = np.array([0.075, -0.001, -0.002])
    return R_lr.T, -R_lr.T @ t_lr


def test_skybeta_camera_info_is_not_taken_as_a_rectification():
    R_rl, t_rl = skybeta_extrinsic()
    with pytest.raises(ValueError):                           # without the extrinsic it must refuse, not guess
        StereoRectifier(SKY_L, SKY_R, scale=0.5)
    r = StereoRectifier(SKY_L, SKY_R, scale=0.5, R_rl=R_rl, t_rl=t_rl)
    assert r.source == "stereoRectify" and not r.identity
    assert r.baseline_m == pytest.approx(0.07503, abs=2e-4)


def test_skybeta_rectified_rows_line_up():
    """Points seen by both real cameras land on the same rectified row, with the disparity their depth implies;
    unrectified, the same points sit tens of pixels apart vertically (cy gap + the factory rotation)."""
    R_rl, t_rl = skybeta_extrinsic()
    r = StereoRectifier(SKY_L, SKY_R, scale=0.5, R_rl=R_rl, t_rl=t_rl)
    Ls, Rs = SKY_L.scaled(0.5), SKY_R.scaled(0.5)
    rng = np.random.default_rng(5)
    X = np.column_stack([rng.uniform(-1.2, 1.2, 80), rng.uniform(-0.7, 0.7, 80), rng.uniform(0.6, 4.0, 80)])
    ul = cv2.projectPoints(X, np.zeros(3), np.zeros(3), Ls.K, None)[0].reshape(-1, 2)
    ur = cv2.projectPoints(X, cv2.Rodrigues(R_rl)[0], t_rl, Rs.K, None)[0].reshape(-1, 2)
    assert np.abs(ul[:, 1] - ur[:, 1]).max() > 8.0              # raw rows disagree (half resolution)
    rl = cv2.undistortPoints(ul.reshape(-1, 1, 2), Ls.K, None, R=r.R1, P=r.P1).reshape(-1, 2)
    rr = cv2.undistortPoints(ur.reshape(-1, 1, 2), Rs.K, None, R=r.R2, P=r.P2).reshape(-1, 2)
    assert np.abs(rl[:, 1] - rr[:, 1]).max() < 0.05
    z_rect = (X @ r.R1.T)[:, 2]
    assert np.abs((rl[:, 0] - rr[:, 0]) - r.f * r.baseline_m / z_rect).max() < 0.05


@pytest.mark.parametrize("z_true", [1.0, 2.0, 3.0])
def test_skybeta_textured_plane(z_true):
    """End to end with the real calibration: un-rectify a synthetic pair into raw skybeta images, recover depth."""
    R_rl, t_rl = skybeta_extrinsic()
    pipe = StereoDepthPipeline(SKY_L, SKY_R, scale=0.5, R_rl=R_rl, t_rl=t_rl)
    r = pipe.rect
    w, h = r.size
    left_rect = texture(h, w, seed=11)
    right_rect = shift(left_rect, r.f * r.baseline_m / z_true)

    def unrectify(rect_img, cam, Rr, Pr):
        cs = cam.scaled(0.5)
        us, vs = np.meshgrid(np.arange(cs.width, dtype=np.float32), np.arange(cs.height, dtype=np.float32))
        pts = np.stack([us.ravel(), vs.ravel()], 1).reshape(-1, 1, 2)
        m = cv2.undistortPoints(pts, cs.K, None, R=Rr, P=Pr).reshape(cs.height, cs.width, 2).astype(np.float32)
        return cv2.remap(rect_img, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    raw_l = cv2.resize(unrectify(left_rect, SKY_L, r.R1, r.P1), (1024, 640), interpolation=cv2.INTER_LINEAR)
    raw_r = cv2.resize(unrectify(right_rect, SKY_R, r.R2, r.P2), (1024, 640), interpolation=cv2.INTER_LINEAR)
    z = pipe(raw_l, raw_r)
    roi = z[h // 4:3 * h // 4, w // 3:2 * w // 3]
    assert np.isfinite(roi).mean() > 0.8
    fB = r.f * r.baseline_m
    assert abs(fB / np.nanmedian(roi) - fB / z_true) < 0.25


def test_baseline_check():
    from onboard.depth.stereo_depth import check_baseline
    assert check_baseline(0.0753, 0.075, 0.01) is None
    assert "0.000" in check_baseline(0.0, 0.075, 0.01)
    assert check_baseline(0.12, 0.075, 0.01) is not None
    assert check_baseline(0.12, 0.0, 0.01) is None                  # 0 disables the check


# --- depth ---------------------------------------------------------------------------------------------------

def test_disparity_to_depth_and_its_limits():
    z = disparity_to_depth(np.array([[-1.0, 0.0, 10.0, 100.0, 1.0]]), fB=16.0, min_m=0.35, max_m=6.0)
    assert np.isnan(z[0, 0]) and np.isnan(z[0, 1])
    assert z[0, 2] == pytest.approx(1.6) and np.isnan(z[0, 3]) and np.isnan(z[0, 4])   # 0.16 m too near, 16 m too far


@pytest.mark.parametrize("z_true", [1.0, 2.0, 3.0])
def test_a_textured_plane_in_the_sim_calibration(z_true):
    pipe = StereoDepthPipeline(SIM_L, SIM_R, scale=0.5)
    left = texture(400, 640, seed=int(z_true * 10))
    d_full = 421.016 * (31.866 / 421.016) / z_true                         # disparity at full resolution
    z = pipe(left, shift(left, d_full))
    roi = z[40:160, 120:300]
    assert np.isfinite(roi).mean() > 0.9
    # SGBM's sub-pixel disparity is locked towards whole pixels by up to ~0.2 px, so the depth error grows as
    # z^2 / (f B): at half resolution in the sim (f B = 15.9 px m) 0.25 px is 1.6 cm at 1 m, 14 cm at 3 m
    fB = pipe.rect.f * pipe.rect.baseline_m
    assert abs(fB / np.nanmedian(roi) - fB / z_true) < 0.25


@pytest.mark.parametrize("z_true", [1.0, 2.0, 3.0])
def test_a_textured_plane_seen_by_a_raw_distorted_pair(z_true):
    """Build raw images by un-rectifying a synthetic rectified pair, then check the pipeline recovers depth."""
    L, R = raw_models()
    pipe = StereoDepthPipeline(L, R, scale=0.5, R_rl=R_RL, t_rl=T_RL)
    r = pipe.rect
    w, h = r.size
    left_rect = texture(h, w, seed=7)
    right_rect = shift(left_rect, r.f * r.baseline_m / z_true)

    def unrectify(rect_img, cam, Rr, Pr):
        """Raw (half-size) image whose rectification is rect_img: sample rect_img at each raw pixel's rectified spot."""
        cs = cam.scaled(0.5)
        us, vs = np.meshgrid(np.arange(cs.width, dtype=np.float32), np.arange(cs.height, dtype=np.float32))
        pts = np.stack([us.ravel(), vs.ravel()], 1).reshape(-1, 1, 2)
        m = cv2.undistortPoints(pts, cs.K, cs.D, R=Rr, P=Pr).reshape(cs.height, cs.width, 2).astype(np.float32)
        return cv2.remap(rect_img, m[..., 0], m[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    raw_l = cv2.resize(unrectify(left_rect, L, r.R1, r.P1), (1024, 640), interpolation=cv2.INTER_LINEAR)
    raw_r = cv2.resize(unrectify(right_rect, R, r.R2, r.P2), (1024, 640), interpolation=cv2.INTER_LINEAR)
    z = pipe(raw_l, raw_r)
    roi = z[h // 4:3 * h // 4, w // 3:2 * w // 3]
    assert np.isfinite(roi).mean() > 0.8
    fB = r.f * r.baseline_m
    assert abs(fB / np.nanmedian(roi) - fB / z_true) < 0.25          # disparity pixels, as above


def test_16uc1_round_trips_through_the_consumers_decoder():
    z = np.array([[np.nan, 0.4, 1.2345, 70.0]], np.float32)
    mm = to_16uc1(z)
    assert mm.dtype == np.uint16 and list(mm[0]) == [0, 400, 1234, 65535]
    back = decode_depth("16UC1", 1, 4, 8, mm.tobytes())
    assert np.isnan(back[0, 0]) and back[0, 1] == pytest.approx(0.4) and back[0, 2] == pytest.approx(1.234)


def test_quaternion_round_trips():
    for rv in ((0.0, 0.0, 0.0), (0.004, -0.012, 0.006), (2.5, 0.3, -1.0), (math.pi - 1e-3, 0.0, 0.0)):
        Rm = cv2.Rodrigues(np.array(rv))[0]
        x, y, z, w = quat_xyzw(Rm)
        assert np.allclose(quat_to_mat(w, x, y, z), Rm, atol=1e-9)


def test_params_set_the_search_range():
    pipe = StereoDepthPipeline(SIM_L, SIM_R, scale=0.5, p=DepthParams(min_m=0.35))
    assert pipe.matcher.num_disp % 16 == 0 and pipe.matcher.num_disp * 0.35 >= pipe.rect.f * pipe.rect.baseline_m
