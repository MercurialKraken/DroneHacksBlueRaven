"""perception.stereo_depth on synthetic rectified pairs with known disparity (no ROS).

A textured plane at disparity d (input pixels) has depth fx * B / d. The right image of a rectified
pair satisfies right(x) = left(x + d), so we build pairs by cropping one wide texture twice.
Numbers: OAK-D-like baseline 7.5 cm, fx 400 px at 640x400."""

import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception import stereo_depth as S  # noqa: E402

W, H = 640, 400
FX = FY = 400.0
CX, CY = W / 2 - 0.5, H / 2 - 0.5
B = 0.075


def texture(width, height, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.random((height, width)).astype(np.float32)
    t = cv2.GaussianBlur(t, (0, 0), 1.2)
    t = (t - t.min()) / (t.max() - t.min())
    return np.clip(t * 200 + 30, 0, 255).astype(np.uint8)


def plane_pair(d, seed=0):
    tex = texture(W + 200, H, seed)
    left = tex[:, 100 : 100 + W]
    right = tex[:, 100 + d : 100 + d + W]
    return np.ascontiguousarray(left), np.ascontiguousarray(right)


def depth_of(d):
    return FX * B / d


def interior(depth, nd):
    return depth[20:-20, nd + 10 : -20]


@pytest.mark.parametrize("d", [24, 40, 60, 100])
def test_plane_depth_recovered_at_half_scale(d):
    left, right = plane_pair(d)
    cfg = S.StereoConfig(scale=0.5, num_disparities=64)
    r = S.compute_depth(left, right, FX, FY, CX, CY, B, cfg)
    vals = interior(r.depth_m, cfg.num_disparities)
    assert np.isfinite(vals).mean() > 0.9
    assert np.nanmedian(vals) == pytest.approx(depth_of(d), rel=0.02)


@pytest.mark.parametrize("d", [20, 40])
def test_plane_depth_recovered_at_full_scale(d):
    left, right = plane_pair(d, seed=3)
    cfg = S.StereoConfig(scale=1.0, num_disparities=64)
    r = S.compute_depth(left, right, FX, FY, CX, CY, B, cfg)
    vals = interior(r.depth_m, cfg.num_disparities)
    assert np.nanmedian(vals) == pytest.approx(depth_of(d), rel=0.02)


def test_near_and_far_halves_are_separated():
    nl, nr = plane_pair(80, seed=1)  # 0.375 m
    fl, fr = plane_pair(30, seed=2)  # 1.0 m
    left = np.hstack([nl[:, : W // 2], fl[:, W // 2 :]])
    right = np.hstack([nr[:, : W // 2], fr[:, W // 2 :]])
    cfg = S.StereoConfig(scale=0.5)
    r = S.compute_depth(left, right, FX, FY, CX, CY, B, cfg)
    w = r.depth_m.shape[1]
    near = r.depth_m[20:-20, cfg.num_disparities + 10 : w // 2 - 30]
    far = r.depth_m[20:-20, w // 2 + 30 : -20]
    assert np.nanmedian(near) == pytest.approx(depth_of(80), rel=0.03)
    assert np.nanmedian(far) == pytest.approx(depth_of(30), rel=0.03)


def test_textureless_scene_gives_no_depth_rather_than_a_guess():
    flat = np.full((H, W), 128, np.uint8)
    r = S.compute_depth(flat, flat, FX, FY, CX, CY, B, S.StereoConfig())
    assert r.valid_fraction < 0.02
    assert not np.any(np.isfinite(r.depth_m) & (r.depth_m <= 0))


def test_left_border_without_full_search_range_is_never_trusted():
    left, right = plane_pair(40)
    cfg = S.StereoConfig(scale=0.5, num_disparities=64)
    r = S.compute_depth(left, right, FX, FY, CX, CY, B, cfg)
    assert not np.isfinite(r.depth_m[:, : cfg.num_disparities]).any()


def test_out_of_range_depth_is_dropped():
    left, right = plane_pair(100)  # 0.30 m at input; keep only >= 0.5 m
    cfg = S.StereoConfig(scale=0.5, min_depth_m=0.5)
    r = S.compute_depth(left, right, FX, FY, CX, CY, B, cfg)
    assert np.nanmin(r.depth_m) >= 0.5 if np.isfinite(r.depth_m).any() else True
    assert r.valid_fraction < 0.05


def test_output_intrinsics_and_range_limits():
    left, right = plane_pair(40)
    cfg = S.StereoConfig(scale=0.5, num_disparities=64, min_disparity_px=1.0)
    r = S.compute_depth(left, right, FX, FY, CX, CY, B, cfg)
    assert r.depth_m.shape == (H // 2, W // 2)
    assert (r.fx, r.fy) == (pytest.approx(200.0), pytest.approx(200.0))
    assert r.cx == pytest.approx((CX + 0.5) * 0.5 - 0.5)
    assert r.max_range_m == pytest.approx(200.0 * B / 1.0)
    assert r.min_range_m == pytest.approx(200.0 * B / 64)
    assert r.elapsed_ms > 0 and r.depth_m.dtype == np.float32


def test_16uc1_conversion_matches_sim_encoding():
    d = np.array([[np.nan, 0.0, -1.0], [1.2346, 2.0, 99.0]], np.float32)
    mm = S.depth_to_16uc1_mm(d)
    assert mm.dtype == np.uint16
    assert mm.tolist() == [[0, 0, 0], [1235, 2000, 65535]]


@pytest.mark.parametrize(
    "bad",
    [
        dict(left=np.zeros((H, W), np.float32)),  # not mono8
        dict(right=np.zeros((H, W - 2), np.uint8)),  # size mismatch
        dict(baseline_m=0.0),
        dict(baseline_m=float("nan")),
        dict(fx=0.0),
        dict(left=np.zeros((20, 20), np.uint8), right=np.zeros((20, 20), np.uint8)),
    ],
)
def test_bad_inputs_raise(bad):
    left, right = plane_pair(40)
    args = dict(left=left, right=right, fx=FX, fy=FY, cx=CX, cy=CY, baseline_m=B)
    args.update(bad)
    with pytest.raises(ValueError):
        S.compute_depth(**args)


@pytest.mark.parametrize(
    "cfg",
    [
        dict(num_disparities=50),
        dict(block_size=4),
        dict(scale=0.05),
        dict(min_depth_m=3.0, max_depth_m=1.0),
        dict(min_disparity_px=0),
    ],
)
def test_invalid_config_raises(cfg):
    with pytest.raises(ValueError):
        S.StereoConfig(**cfg)


def test_missing_opencv_is_an_explicit_error(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_cv2(name, *a, **k):
        if name == "cv2":
            raise ImportError("no cv2")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_cv2)
    left, right = plane_pair(40)
    with pytest.raises(RuntimeError, match="OpenCV"):
        S.compute_depth(left, right, FX, FY, CX, CY, B)
