"""tools/flight_check without ROS: hand-encoded CDR messages in a minimal MCAP file. Pins the
decoders' byte layout, the dropout/jump/freeze detection, setpoint tracking maths, and that the
tool is read-only (no rclpy, no publisher)."""

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "flight_check"))
sys.path.insert(0, str(ROOT / "tools"))
import analyze_flight as A  # noqa: E402
import flight_log as F  # noqa: E402

# ----------------------------------------------------------------- tiny CDR / MCAP writers


class W:
    def __init__(self):
        self.b = bytearray(b"\x00\x01\x00\x00")

    def _al(self, n):
        while (len(self.b) - 4) % n:
            self.b.append(0)

    def i32(self, v):
        self._al(4)
        self.b += struct.pack("<i", v)

    def u32(self, v):
        self._al(4)
        self.b += struct.pack("<I", v)

    def f64(self, *vs):
        self._al(8)
        self.b += struct.pack("<%dd" % len(vs), *vs)

    def s(self, txt):
        raw = txt.encode() + b"\x00"
        self.u32(len(raw))
        self.b += raw

    def header(self, t, frame):
        self.i32(int(t))
        self.u32(int(round((t - int(t)) * 1e9)))
        self.s(frame)


def odom(t, p, frame="odom", child="base_link"):
    w = W()
    w.header(t, frame)
    w.s(child)
    w.f64(*p)
    w.f64(0, 0, 0, 1)
    w.f64(*([0.0] * 36))
    w.f64(0.1, 0.2, 0.3)
    w.f64(0, 0, 0.5)
    w.f64(*([0.0] * 36))
    return bytes(w.b)


def pose(t, p, frame="odom"):
    w = W()
    w.header(t, frame)
    w.f64(*p)
    w.f64(0, 0, 0, 1)
    return bytes(w.b)


def twist(t, v, frame="base_link"):
    w = W()
    w.header(t, frame)
    w.f64(*v)
    w.f64(0, 0, 0)
    return bytes(w.b)


def px4_odom(ts_us, sample_us):
    w = W()
    w._al(8)
    w.b += struct.pack("<QQ", ts_us, sample_us)
    w.b += bytes(64)  # the rest of VehicleOdometry is not read
    return bytes(w.b)


def string(txt):
    w = W()
    w.s(txt)
    return bytes(w.b)


def mcap(path, channels):
    """channels: [(topic, schema, [(log_time_s, bytes), ...])] -> minimal MCAP readable by
    tools/mcap_read.py (schema, channel and message records, no chunks)."""

    def rec(op, body):
        return bytes([op]) + struct.pack("<Q", len(body)) + body

    def st(x):
        x = x.encode()
        return struct.pack("<I", len(x)) + x

    out = bytearray(b"\x89MCAP0\r\n")
    for i, (topic, schema, msgs) in enumerate(channels, start=1):
        out += rec(0x03, struct.pack("<H", i) + st(schema) + st("ros2msg") + struct.pack("<I", 0))
        out += rec(0x04, struct.pack("<HH", i, i) + st(topic) + st("cdr") + struct.pack("<I", 0))
        for seq, (t, data) in enumerate(msgs):
            ns = int(t * 1e9)
            out += rec(0x05, struct.pack("<HIQQ", i, seq, ns, ns) + data)
    out += b"\x89MCAP0\r\n"
    Path(path).write_bytes(bytes(out))


# ----------------------------------------------------------------- decoders


def test_odometry_round_trip_with_odd_length_strings():
    m = F.decode("nav_msgs/msg/Odometry", odom(12.25, (1.5, -2.0, 0.7), "odom_x", "base"))
    assert m["stamp"] == pytest.approx(12.25)
    assert (m["frame"], m["child"]) == ("odom_x", "base")
    assert m["pos"] == (1.5, -2.0, 0.7)
    assert m["quat"] == (0, 0, 0, 1)
    assert m["lin"] == (0.1, 0.2, 0.3) and m["ang"] == (0, 0, 0.5)


def test_pose_twist_string_round_trip():
    assert F.decode("geometry_msgs/msg/PoseStamped", pose(3.0, (1, 2, 3)))["pos"] == (1, 2, 3)
    t = F.decode("geometry_msgs/msg/TwistStamped", twist(3.0, (0.3, 0, -0.2)))
    assert t["lin"] == (0.3, 0, -0.2) and t["frame"] == "base_link"
    assert F.decode("std_msgs/msg/String", string("OFFBOARD"))["data"] == "OFFBOARD"
    assert F.decode("sensor_msgs/msg/Image", string("x")) is None


def test_truncated_and_big_endian_are_refused():
    with pytest.raises(ValueError):
        F.decode("nav_msgs/msg/Odometry", odom(1.0, (0, 0, 0))[:60])
    with pytest.raises(ValueError):
        F.Cdr(b"\x00\x00\x00\x00")


# ----------------------------------------------------------------- metrics


def test_topic_stats_rate_and_gaps():
    t = np.r_[np.arange(0, 1, 0.1), np.arange(1.5, 2.0, 0.1)]
    s = F.topic_stats(t, gap_s=0.25)
    assert s["count"] == 15
    assert s["max_gap_s"] == pytest.approx(0.6, abs=1e-6)
    assert [g["t"] for g in s["gaps"]] == [pytest.approx(0.9)]
    assert F.topic_stats([], 0.25)["count"] == 0
    assert F.topic_stats(np.arange(0, 10, 1.0), gap_s=0.25)["gaps"] == []  # a 1 Hz topic
    assert len(F.topic_stats(np.r_[0, 1, 2, 6, 7], gap_s=0.25)["gaps"]) == 1


def test_smooth_flight_has_no_events():
    t = np.arange(0, 5, 1 / 30)
    p = np.c_[0.5 * t, np.sin(t) * 0.2, np.full_like(t, 0.7)]
    assert F.vio_events(t, p) == []


def test_jump_gap_freeze_and_nan_are_found():
    t = np.arange(0, 4, 1 / 30)
    p = np.c_[0.3 * t, np.zeros_like(t), np.full_like(t, 0.7)]
    p[60:, 0] += 12.0  # the bench test's kind of jump
    p[90:110] = p[90]  # tracking stopped, output held for 0.63 s
    p[20] = np.nan
    t = np.r_[t[:40], t[40:] + 0.5]  # a 0.53 s dropout after sample 39
    ev = F.vio_events(t, p)
    kinds = [e["kind"] for e in ev]
    big = [e for e in ev if e["kind"] == "jump" and e["step_m"] > 1]
    assert len(big) == 1 and "gap" in kinds and "freeze" in kinds
    # the catch-up step when output resumes after the freeze is reported too
    assert any(e["kind"] == "jump" and e["t"] > 3.5 for e in ev)
    assert "non_finite" in kinds
    assert big[0]["step_m"] == pytest.approx(12.0, abs=0.02)
    assert next(e for e in ev if e["kind"] == "freeze")["duration_s"] >= 0.6


def test_hovering_still_is_not_a_freeze_unless_bit_identical():
    t = np.arange(0, 3, 1 / 30)
    rng = np.random.default_rng(1)
    p = np.c_[rng.normal(0, 1e-4, (t.size, 2)), np.full_like(t, 0.7)]
    assert F.vio_events(t, p) == []


def test_pose_tracking_constant_offset_and_hold():
    t = np.arange(0, 2, 0.05)
    odo = np.c_[t, np.zeros_like(t), np.full_like(t, 0.7)]
    sp_t = np.arange(0, 1, 0.1)  # setpoints stop at 0.9 s
    sp = np.c_[sp_t + 0.3, np.full_like(sp_t, 0.4), np.full_like(sp_t, 0.8)]
    r = F.pose_tracking(t, odo, sp_t, sp, hold_s=0.52)
    # compared only while a setpoint at most 0.52 s old exists: 0.0 .. 1.4 s
    assert r["compared"] == int(np.sum(t <= 1.42))
    assert r["vertical_m"]["median"] == pytest.approx(0.1)
    assert r["horizontal_m"]["n"] == r["compared"]
    assert F.pose_tracking(t, odo, [], np.zeros((0, 3)))["compared"] == 0


def test_twist_tracking_is_yaw_frame_independent():
    t = np.arange(0, 3, 1 / 30)
    odo = np.c_[0.3 * t, 0.0 * t, np.full_like(t, 0.7)]  # flies +x at 0.3 m/s (world)
    sp_t = np.arange(0, 3, 0.05)
    body = np.c_[np.zeros_like(sp_t), np.full_like(sp_t, 0.3), np.zeros_like(sp_t)]  # same speed,
    r = F.twist_tracking(t, odo, sp_t, body)  # expressed in a frame yawed by 90 degrees
    assert r["compared"] > 50
    assert r["horizontal_speed_error_mps"]["max"] < 0.02
    assert r["flown_horizontal_mps"]["median"] == pytest.approx(0.3, abs=0.01)


def test_state_changes_only():
    out = F.state_changes([0, 1, 2, 3], ["IDLE", "IDLE", "OFFBOARD", "OFFBOARD"])
    assert out == [{"t": 0.0, "state": "IDLE"}, {"t": 2.0, "state": "OFFBOARD"}]


# ----------------------------------------------------------------- end to end


def _flight(tmp_path, jump=False, frame_sp="odom"):
    t = np.arange(0, 6, 1 / 30)
    p = np.c_[0.3 * t, np.zeros_like(t), np.full_like(t, 0.7)]
    if jump:
        p[100:, 1] += 15.0
    sp_t = np.arange(0, 6, 0.1)
    path = tmp_path / "flight.mcap"
    mcap(
        path,
        [
            (
                A.DEFAULTS["odom"],
                "nav_msgs/msg/Odometry",
                [(100 + a, odom(100 + a - 0.05, b)) for a, b in zip(t, p)],
            ),
            (
                A.DEFAULTS["pose_sp"],
                "geometry_msgs/msg/PoseStamped",
                [
                    (100 + a, pose(100 + a - 0.05, (0.3 * a + 0.1, 0.0, 0.7), frame_sp))
                    for a in sp_t
                ],
            ),
            (
                A.DEFAULTS["twist_sp"],
                "geometry_msgs/msg/TwistStamped",
                [(100 + a, twist(100 + a - 0.05, (0.3, 0.0, 0.0))) for a in sp_t],
            ),
            (
                A.DEFAULTS["state"],
                "std_msgs/msg/String",
                [
                    (100.0, string("ARMED")),
                    (100.5, string("OFFBOARD")),
                    (101.0, string("OFFBOARD")),
                ],
            ),
            ("/oak/left/image_raw", "sensor_msgs/msg/Image", [(100.0, string("x"))]),
            (
                "/fmu/in/vehicle_visual_odometry",
                "px4_msgs/msg/VehicleOdometry",
                [
                    (
                        100 + a,
                        px4_odom(
                            int((5 + a) * 1e6), int((5 + a - (0.075 if a < 3 else 0.3)) * 1e6)
                        ),
                    )
                    for a in t[::2]
                ],
            ),
        ],
    )
    return path


def test_end_to_end_clean_flight(tmp_path):
    rep = A.analyze(A.load(_flight(tmp_path)), dict(A.DEFAULTS), F.Thresholds())
    assert rep["vio"]["events"] == []
    assert rep["pose_tracking"]["horizontal_m"]["median"] == pytest.approx(0.1, abs=0.02)
    assert rep["twist_tracking"]["horizontal_speed_error_mps"]["median"] < 0.02
    assert [s["state"] for s in rep["state_changes"]] == ["ARMED", "OFFBOARD"]
    assert rep["topics"]["/oak/left/image_raw"]["count"] == 1
    assert rep["frames"]["odom"] == ["odom"] and rep["notes"] == []
    lat = rep["latency"]
    assert lat[A.DEFAULTS["odom"]]["median_ms"] == pytest.approx(50, abs=1)
    assert lat[A.DEFAULTS["odom"]]["first_slow_second_t"] is None
    ev = lat["/fmu/in/vehicle_visual_odometry"]
    assert ev["kind"].startswith("px4") and ev["first_slow_second_t"] == pytest.approx(3, abs=1.01)
    assert ev["over_limit_fraction"] == pytest.approx(0.5, abs=0.05)


def test_end_to_end_jump_and_frame_warning_and_files(tmp_path):
    path = _flight(tmp_path, jump=True, frame_sp="map")
    out = tmp_path / "f"
    assert A.main([str(path), "--out", str(out)]) == 0
    rep = json.loads(Path(f"{out}_report.json").read_text())
    jumps = [e for e in rep["vio"]["events"] if e["kind"] == "jump"]
    assert len(jumps) == 1 and jumps[0]["t"] == pytest.approx(100 / 30, abs=0.05)
    assert any("different frame ids" in n for n in rep["notes"])
    assert "VIO:" in Path(f"{out}_report.txt").read_text()


def test_missing_topics_are_reported_not_crashed(tmp_path):
    path = tmp_path / "x.mcap"
    mcap(path, [("/other", "std_msgs/msg/String", [(1.0, string("a"))])])
    rep = A.analyze(A.load(path), dict(A.DEFAULTS), F.Thresholds())
    assert "vio" not in rep and any("not recorded" in n for n in rep["notes"])


def test_tool_is_read_only():
    for f in (ROOT / "tools" / "flight_check").glob("*.py"):
        src = f.read_text()
        for bad in ("rclpy", "create_publisher", "/fmu/in", "subprocess", "os.system"):
            assert bad not in src, f"{f.name} contains {bad}"


def test_thresholds_are_sane():
    th = F.Thresholds()
    assert th.jump_speed_mps > 1.5  # above the skybeta twist cap, so real flight never trips it
    assert 0 < th.gap_s < 1 and math.isfinite(th.hold_s)


def test_px4_vehicle_odometry_age_decoding():
    m = F.decode("px4_msgs/msg/VehicleOdometry", px4_odom(10_300_000, 10_225_000))
    assert m["px4_timestamp_us"] - m["px4_sample_us"] == 75_000


def test_latency_finds_overload_onset():
    t = np.arange(0, 40, 1 / 30)
    age = np.where(t < 21, 0.075, 0.3)  # Arnav's field bag shape: 75 ms, then 200-400 ms
    r = F.latency(t, age, limit_s=0.2)
    assert r["median_ms"] == pytest.approx(75, abs=1)  # 21 of 40 s at 75 ms
    assert r["first_slow_second_t"] == pytest.approx(21, abs=1.01)
    assert r["over_limit_fraction"] == pytest.approx(19 / 40, abs=0.01)
    assert F.latency(t, np.full_like(t, 0.075))["first_slow_second_t"] is None
    assert F.latency([], [])["n"] == 0


def test_clock_mismatch_is_noted_not_reported(tmp_path):
    path = tmp_path / "c.mcap"
    mcap(
        path,
        [
            (
                A.DEFAULTS["odom"],
                "nav_msgs/msg/Odometry",
                [(100 + i / 30, odom(5000 + i / 30, (0, 0, 0.7))) for i in range(60)],
            )
        ],
    )
    rep = A.analyze(A.load(path), dict(A.DEFAULTS), F.Thresholds())
    assert A.DEFAULTS["odom"] not in rep["latency"]
    assert any("different clocks" in n for n in rep["notes"])


def test_backwards_stamps_are_reported_but_not_a_jump():
    t = np.arange(0, 2, 1 / 30)
    t[30:] -= 0.2  # delivery got slower; stamps step back once
    p = np.c_[0.01 * np.arange(t.size), np.zeros_like(t), np.full_like(t, 0.7)]
    kinds = [e["kind"] for e in F.vio_events(t, p)]
    assert kinds == ["stamp_backwards"]
