"""vision.floor_line without ROS: synthetic 640x400 mono frames (the lab's stereo size) in the exact
sensor_msgs/Image byte layout. Pins: a clean floor stripe is found with the right image-space
offset and lean; blank, competing, wrong-polarity and black frames are NOT reported as a line;
bad input raises; the output never carries a command."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision import floor_line as L  # noqa: E402

W, H = 640, 400
NOW = 10_000_000_000
FRESH = NOW - 40_000_000


def scene(floor=90, wall=130, horizon=0.30, noise=3, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((H, W), floor, np.float32)
    img[: int(H * horizon)] = wall
    img += rng.normal(0, noise, img.shape)
    return img


def stripe(img, x_bottom, x_top, value, y_top=0.40, half_width=9):
    """Paint a straight stripe from the bottom row up to y_top (fractions of the image)."""
    for y in range(int(H * y_top), H):
        t = (H - 1 - y) / (H - 1 - int(H * y_top))
        cx = (x_bottom + (x_top - x_bottom) * t) * W
        img[y, max(0, int(cx - half_width)) : int(cx + half_width)] = value
    return img


def u8(img):
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def mono8(img, pad=0):
    rows = np.zeros((H, W + pad), np.uint8)
    rows[:, :W] = u8(img)
    return "mono8", H, W, W + pad, rows.tobytes(), False


def run(msg, config=None, stamp=FRESH):
    enc, h, w, step, data, big = msg
    return L.floor_line_message(enc, h, w, step, data, stamp, NOW, "left_optical", config, big)


def test_centred_straight_line():
    doc = run(mono8(stripe(scene(), 0.5, 0.5, 200)))
    assert doc["schema"] == L.SCHEMA and doc["fresh"]
    assert doc["detected"] and doc["status"] == "supported" and doc["confidence"] > 0.55
    assert doc["bottom_offset"] == pytest.approx(0.0, abs=0.03)
    assert doc["image_lean_deg"] == pytest.approx(0.0, abs=2.0)
    assert len(doc["points_normalized"]) == 12


def test_line_to_the_right_has_positive_offset():
    doc = run(mono8(stripe(scene(), 0.70, 0.70, 200)))
    assert doc["detected"]
    assert doc["bottom_offset"] == pytest.approx(0.40, abs=0.03)


def test_line_leaning_right_going_away():
    doc = run(mono8(stripe(scene(), 0.45, 0.60, 200)))
    assert doc["detected"]
    assert doc["image_lean_deg"] > 5.0
    left = run(mono8(stripe(scene(), 0.55, 0.40, 200)))
    assert left["detected"] and left["image_lean_deg"] < -5.0


def test_blank_floor_is_unknown():
    doc = run(mono8(scene()))
    assert not doc["detected"] and doc["status"] == "unknown"
    assert doc["bottom_offset"] is None and doc["points_normalized"] == []


def test_two_equal_parallel_lines_are_ambiguous():
    img = stripe(stripe(scene(), 0.35, 0.35, 200), 0.65, 0.65, 200)
    doc = run(mono8(img))
    assert not doc["detected"] and "competing_lines" in doc["rejection_reasons"]


def test_polarity_is_respected():
    dark = mono8(stripe(scene(floor=150, wall=180), 0.5, 0.5, 40))
    assert not run(dark)["detected"]
    assert run(dark, L.LineConfig(polarity="dark"))["detected"]


def test_black_frame_is_low_light_and_not_a_line():
    doc = run(mono8(np.zeros((H, W), np.float32)))
    assert doc["low_light"]["detected"] and not doc["detected"]


def test_bright_walls_alone_do_not_hide_a_floor_line():
    # regression: the prototype's wall-edge check compared with the floor median and rejected this
    doc = run(mono8(stripe(scene(floor=80, wall=170), 0.5, 0.5, 200)))
    assert doc["detected"]


def test_stripe_continuing_up_the_wall_is_rejected_as_wall_edge():
    doc = run(mono8(stripe(scene(), 0.5, 0.5, 220, y_top=0.0)))
    assert not doc["detected"]
    assert "line_wall_edge_ambiguous" in doc["rejection_reasons"]


def test_line_cropped_at_image_side_is_not_trusted():
    doc = run(mono8(stripe(scene(), 0.005, 0.005, 200, half_width=8)))
    assert not doc["detected"]


def test_row_padding_mono16_bigendian_and_colour_decode_alike():
    img = stripe(scene(), 0.5, 0.5, 200)
    ref = run(mono8(img))
    padded = run(mono8(img, pad=13))
    m16 = u8(img).astype(">u2") * 257
    big16 = run(("mono16", H, W, W * 2, m16.tobytes(), True))
    bgr = np.repeat(u8(img)[:, :, None], 3, axis=2)
    colour = run(("bgr8", H, W, W * 3, bgr.tobytes(), False))
    for other in (padded, big16, colour):
        assert other["detected"] and other["bottom_offset"] == pytest.approx(
            ref["bottom_offset"], abs=1e-3
        )


@pytest.mark.parametrize(
    "bad",
    [
        (
            "16UC1",
            H,
            W,
            W * 2,
            bytes(H * W * 2),
            False,
        ),  # depth is not an intensity image
        ("mono8", H, W, W, bytes(H * W - 1), False),  # short buffer
        ("mono8", H, W, W - 1, bytes(H * (W - 1)), False),  # row shorter than width
        ("mono8", 8, 8, 8, bytes(64), False),  # too small to search
    ],
)
def test_malformed_images_raise(bad):
    with pytest.raises(ValueError):
        run(bad)


def test_stale_frame_is_flagged():
    doc = run(mono8(stripe(scene(), 0.5, 0.5, 200)), stamp=NOW - 400_000_000)
    assert doc["fresh"] is False


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        L.LineConfig(polarity="green")
    with pytest.raises(ValueError):
        replace(L.LineConfig(), roi_top=1.0)


def test_output_is_strict_json_and_never_a_command():
    for img in (stripe(scene(), 0.5, 0.5, 200), scene(), np.zeros((H, W), np.float32)):
        doc = run(mono8(img))
        json.dumps(doc, allow_nan=False)
        assert doc["control_output"] == "DISABLED"
        assert not ({"setpoint", "velocity", "command"} & set(doc))


# --- cluttered floors -------------------------------------------------------------------------
# The sim floors are white with scattered black dots (local sim) or multicoloured terrazzo chips
# (lab sim); the venue tarp has dark printed spots. Clutter touching the tape used to fuse with it
# into one "branched" component that the gates rejected. These pin the spine-pruning recovery and,
# just as important, that clutter alone never becomes a line.

DARK = L.LineConfig(polarity="dark")


def cluttered_floor(kind, rng):
    import cv2

    img = np.full((H, W), 110.0, np.float32)
    img[: int(H * 0.30)] = 150
    floor = slice(int(H * 0.30), H)
    if kind == "dots":
        img[floor] = 215
        for _ in range(140):
            y, x = int(rng.integers(int(H * 0.32), H)), int(rng.integers(0, W))
            r = int(3 + (y - H * 0.3) / H * 22)
            cv2.ellipse(img, (x, y), (r, max(2, int(r * 0.55))), 0, 0, 360, 40, -1)
    elif kind == "terrazzo":
        img[floor] = 180
        for _ in range(500):
            y, x = int(rng.integers(int(H * 0.32), H)), int(rng.integers(0, W))
            r = int(2 + (y - H * 0.3) / H * 12)
            shade = float(rng.choice([60, 90, 140, 230]))
            cv2.ellipse(
                img, (x, y), (r, max(1, int(r * 0.6))), int(rng.integers(0, 180)), 0, 360, shade, -1
            )
    return img


def tape(img, x_bottom, x_top, value=30, half_width=11, y_top=0.40):
    y0 = int(H * y_top)
    for y in range(y0, H):
        t = (H - 1 - y) / (H - 1 - y0)
        cx = (x_bottom + (x_top - x_bottom) * t) * W
        hw = int(half_width * (0.5 + 0.5 * (1 - t)))  # wider near the camera
        img[y, max(0, int(cx - hw)) : int(cx + hw)] = value
    return img


def cluttered_case(kind, seed, with_line=True):
    rng = np.random.default_rng(seed)
    img = cluttered_floor(kind, rng)
    xb, xt = 0.5 + 0.1 * rng.standard_normal(), 0.5 + 0.1 * rng.standard_normal()
    if with_line:
        img = tape(img, xb, xt)
    img += rng.normal(0, 3, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8), 2 * xb - 1


@pytest.mark.parametrize("kind,min_rate", [("dots", 0.75), ("terrazzo", 0.9)])
def test_tape_on_cluttered_floor_is_found_where_it_is(kind, min_rate):
    hits, errors = 0, []
    for seed in range(20):
        gray, true_offset = cluttered_case(kind, seed)
        doc = L.detect_floor_line(gray, DARK)
        if doc["detected"]:
            hits += 1
            errors.append(abs(doc["bottom_offset"] - true_offset))
    assert hits / 20 >= min_rate
    assert max(errors) < 0.02  # pruning must not drag the line sideways


@pytest.mark.parametrize("kind", ["dots", "terrazzo"])
def test_clutter_alone_never_becomes_a_line(kind):
    false = sum(
        L.detect_floor_line(cluttered_case(kind, s, False)[0], DARK)["detected"] for s in range(30)
    )
    assert false == 0


def test_clean_tape_does_not_use_pruning():
    img = np.full((H, W), 110.0, np.float32)
    img[int(H * 0.30) :] = 200
    doc = L.detect_floor_line(np.clip(tape(img, 0.5, 0.55), 0, 255).astype(np.uint8), DARK)
    assert doc["detected"] and doc["clutter_pruned"] is False


def test_fused_blob_without_a_stripe_is_not_pruned_into_a_line():
    # one large dark irregular blob in the floor region: lots of rows, no straight narrow spine
    import cv2

    img = np.full((H, W), 110.0, np.float32)
    img[int(H * 0.30) :] = 200
    rng = np.random.default_rng(5)
    for _ in range(60):
        y, x = int(rng.integers(int(H * 0.40), H)), int(rng.integers(220, 420))
        cv2.circle(img, (x, y), int(rng.integers(8, 20)), 30, -1)
    doc = L.detect_floor_line(np.clip(img, 0, 255).astype(np.uint8), DARK)
    assert not doc["detected"]
