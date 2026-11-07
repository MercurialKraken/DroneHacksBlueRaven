"""nav.map_kf (Kalman drift correction) and nav.pose_buffer (pose at the depth stamp)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nav.map_align import SE2, AlignResult  # noqa: E402
from nav.map_kf import KalmanCorrector, KfParams, observable_measurement  # noqa: E402
from nav.pose_buffer import ClockMap, PoseBuffer, slerp  # noqa: E402
from test_map_align import ROOM_WALLS, corridor, room, wall_points  # noqa: E402


def odom_for(truth, T):
    """The odom pose that T maps onto truth."""
    c, s = math.cos(-T.yaw), math.sin(-T.yaw)
    ox, oy = truth[0] - T.x, truth[1] - T.y
    return (c * ox - s * oy, s * ox + c * oy, truth[2], truth[3] - T.yaw)


# --- the filter --------------------------------------------------------------------------------------------

def test_converges_to_a_constant_offset_and_its_covariance_shrinks():
    cm = room()
    truth = (1.4, 1.2, 0.8, math.radians(20))
    true_T = SE2(0.12, -0.08, math.radians(3))
    odom = odom_for(truth, true_T)
    pts = wall_points(ROOM_WALLS, truth)
    kf = KalmanCorrector(cm, SE2(), KfParams(p0_xy=0.2, p0_yaw=math.radians(5)))
    s0 = kf.sigma()
    sig = []
    for k in range(100):                                       # 20 s at 5 Hz
        kf.update(k * 0.2, odom, pts)
        sig.append(kf.sigma())
    est = kf.T.pose(odom)
    assert math.hypot(est[0] - truth[0], est[1] - truth[1]) < 0.02
    assert abs(math.degrees(est[3] - truth[3])) < 0.5
    assert sig[-1][0] < s0[0] / 2 and sig[-1][2] < s0[2] / 2  # good fits shrink the covariance
    assert kf.n["used"] > 90 and kf.n["rejected_gate"] == 0


def test_never_moves_along_a_corridors_unobservable_axis():
    cm = corridor()
    truth = (0.0, 0.5, 0.8, 0.0)
    true_T = SE2(0.30, -0.08, 0.0)                              # 0.30 m along the corridor: unknowable
    odom = odom_for(truth, true_T)
    pts = wall_points([(-6, 0, 6, 0), (-6, 1, 6, 1)], truth)
    kf = KalmanCorrector(cm, SE2(), KfParams(p0_xy=0.2))
    sx0 = kf.sigma()[0]
    for k in range(60):
        kf.update(k * 0.2, odom, pts)
    est = kf.T.pose(odom)
    assert abs(est[1] - truth[1]) < 0.02                        # across: corrected
    assert abs(kf.T.x) < 1e-6                                   # along: never touched
    assert kf.sigma()[0] > sx0                                  # and its uncertainty keeps growing
    assert kf.sigma()[1] < 0.06


def test_gates_out_an_outlier_fit():
    cm = room()
    truth = (1.4, 1.2, 0.8, math.radians(20))
    pts = wall_points(ROOM_WALLS, truth)
    kf = KalmanCorrector(cm, SE2())
    for k in range(50):                                         # settle: odom = truth
        kf.update(k * 0.2, truth, pts)
    T_before = SE2(kf.T.x, kf.T.y, kf.T.yaw)
    # one frame whose fit says 0.24 m off. (With the 5 cm fit floor 0.22 m is ~4 sigma and passes at d2 15.4;
    # beyond align()'s 0.25 m inlier band the fit no longer sees the jump at all, so this is the gate's regime.)
    jumped = (truth[0] - 0.24, truth[1], truth[2], truth[3])
    kf.update(10.2, jumped, pts)
    assert kf.n["rejected_gate"] == 1 and kf.last_d2 > 16.27
    assert math.hypot(kf.T.x - T_before.x, kf.T.y - T_before.y) < 1e-9


def test_a_sustained_real_jump_is_reacquired_after_the_gate_resets():
    cm = room()
    truth = (1.4, 1.2, 0.8, math.radians(20))
    pts = wall_points(ROOM_WALLS, truth)
    kf = KalmanCorrector(cm, SE2())
    for k in range(50):
        kf.update(k * 0.2, truth, pts)
    shifted = (truth[0] - 0.24, truth[1], truth[2], truth[3])  # VIO really jumped 0.24 m and stays there
    for k in range(60):
        kf.update(10.0 + k * 0.2, shifted, pts)
    assert kf.n["rejected_gate"] >= 1                           # first gated...
    est = kf.T.pose(shifted)                                    # ...then P grows with Q until the fit is believed
    assert abs(est[0] - truth[0]) < 0.03


def test_the_gate_reset_inflates_the_covariance():
    kf = KalmanCorrector(room(), SE2(), KfParams(gate_reset_n=3))
    r = AlignResult(True, np.zeros(3), np.diag([1e4, 1e4, 1e3]), 500, 0.03, 0.02, np.array([True, True, True]), "",
                    np.array([0.5, 0.0, 0.0]))                  # a far, confident fit, every frame
    import nav.map_kf as M
    orig = M.align
    M.align = lambda *a, **k: r
    try:
        p0 = kf.P[0, 0]
        for k in range(3):
            kf.update(k * 0.2, (1.0, 1.0, 0.8, 0.0), np.zeros((0, 3)))
    finally:
        M.align = orig
    assert kf.n["rejected_gate"] == 3 and kf.n["resets"] == 1 and kf.P[0, 0] > p0 + 0.15 ** 2 * 0.9


def test_measurement_uses_only_the_well_conditioned_directions():
    H = np.diag([400.0, 3.0, 50.0])                             # y unobservable
    r = AlignResult(True, np.zeros(3), H, 300, 0.05, 0.03, np.array([True, False, True]), "", np.array([0.1, 0.0, 0.02]))
    U, z, R = observable_measurement(r, KfParams())
    assert U.shape == (3, 2) and np.allclose(np.abs(U[1]), 0.0)
    assert R.shape == (2, 2) and np.all(np.linalg.eigvalsh(R) > 0)
    r2 = AlignResult(True, np.zeros(3), H, 300, 0.05, 0.03, np.array([True, False, False]), "", np.array([0.1, 0.0, 0.0]))
    U2, _, _ = observable_measurement(r2, KfParams())               # align() dropped the yaw: no yaw component
    assert np.allclose(U2[2], 0.0)


def test_covariance_moves_with_the_body():
    kf = KalmanCorrector(room(), SE2(), KfParams(q_xy=0.0, q_yaw=0.0, p0_xy=0.0, p0_yaw=math.radians(2)))
    kf.predict(0.0, (0.0, 0.0))
    kf.predict(0.2, (3.0, 0.0))                                 # 3 m east: a 2 deg heading error is now 0.105 m north
    assert math.sqrt(kf.P[1, 1]) == pytest.approx(3.0 * math.radians(2), rel=1e-6)


# --- the pose buffer ---------------------------------------------------------------------------------------

def test_interpolation_is_exact_at_samples_and_linear_between():
    b = PoseBuffer(horizon_s=1.0, max_gap_s=0.1)
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([math.cos(math.radians(10)), 0.0, 0.0, math.sin(math.radians(10))])   # 20 deg about z, w first
    for k in range(11):
        t = 10.0 + 0.05 * k
        b.add_position(t, (k * 0.1, -k * 0.05, 1.0))
        b.add_attitude(t, slerp(q0, q1, k / 10))
    p, q = b.at(10.25)
    assert np.allclose(p, (0.5, -0.25, 1.0))
    p, q = b.at(10.275)
    assert np.allclose(p, (0.55, -0.275, 1.0))
    yaw = math.degrees(2 * math.atan2(q[3], q[0]))
    assert yaw == pytest.approx(11.0, abs=1e-6)


def test_an_exact_stamp_returns_that_sample_untouched():
    """On the drone, depth and odometry share the capture stamp: the buffer must return that very sample."""
    b = PoseBuffer()
    for k in range(30):                                          # 30 Hz odometry with irregular stamps
        t = 100.0 + k / 30.0 + (0.001 if k % 3 else 0.0)
        b.add_position(t, (k * 0.01, 0.5, 1.0)); b.add_attitude(t, (math.sin(k * 0.01), 0.0, 0.0, math.cos(k * 0.01)))
    t17 = 100.0 + 17 / 30.0 + 0.001
    p, q = b.at(t17)
    assert np.array_equal(p, np.array([0.17, 0.5, 1.0])) and np.array_equal(q, np.array([math.sin(0.17), 0.0, 0.0, math.cos(0.17)]))


def test_drone_depth_stamps_20us_off_the_odometry_land_on_that_sample():
    """skybeta-09 walk bag (arnav-claude): depth stamps sit ~20 us from an odometry sample, never bit-identical
    (0 of 241). Interpolating 20 us into a 33 ms interval must give that sample to well under a millimetre."""
    b = PoseBuffer()
    for k in range(60):                                          # 30 Hz odometry, moving 0.5 m/s and turning 90 deg/s
        t = 200.0 + k / 30.0
        yaw = math.radians(90.0) * (k / 30.0)
        b.add_position(t, (0.5 * k / 30.0, 0.0, 1.0)); b.add_attitude(t, (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))
    for k, off in ((35, 20e-6), (44, -20e-6), (52, 35e-6)):     # inside the 1 s horizon (the last 30 samples)
        p, q = b.at(200.0 + k / 30.0 + off)
        assert abs(p[0] - 0.5 * k / 30.0) < 1e-4                  # 0.5 m/s x 35 us = 0.02 mm
        yaw = 2 * math.atan2(q[2], q[3])
        assert abs(yaw - math.radians(90.0) * (k / 30.0)) < math.radians(0.01)


def test_a_depth_frame_far_from_any_odometry_is_refused():
    """The walk bag had one depth frame 3.1 s from any odometry sample: refuse it, whichever side it falls on."""
    b = PoseBuffer(horizon_s=1.0, max_gap_s=0.1)
    for k in range(20):
        t = 300.0 + k / 30.0
        b.add_position(t, (k, 0, 0)); b.add_attitude(t, (0, 0, 0, 1))
    newest = 300.0 + 19 / 30.0
    assert b.at(newest + 3.1) is None                            # odometry stopped (bag edge or a stack restart)
    assert b.at(300.0 - 3.1) is None                             # before the buffer
    b.add_position(newest + 3.1, (99, 0, 0)); b.add_attitude(newest + 3.1, (0, 0, 0, 1))
    assert b.at(newest + 1.5) is None                            # inside a 3.1 s hole between two samples


def test_map_from_odom_round_trips_through_the_pose_message_layout():
    """The /blueraven/map_from_odom layout (position x, y; yaw about +z as w = cos(yaw/2), z = sin(yaw/2)) keeps the
    transform and its yaw sign: a point mapped with the decoded transform lands where the original puts it."""
    from nav.map_align import SE2
    for T in (SE2(-2.51, 1.11, math.radians(90)), SE2(0.3, -4.0, math.radians(-170)), SE2(1.0, 2.0, math.radians(3.5))):
        w, z = math.cos(T.yaw / 2), math.sin(T.yaw / 2)               # what the node writes
        back = SE2(T.x, T.y, math.atan2(2 * w * z, 1 - 2 * z * z))     # what a reader decodes (yaw from the quaternion)
        for p_odom in ((1.0, 0.0), (0.0, 1.0), (2.5, -1.2)):
            assert np.allclose(back.apply(*p_odom), T.apply(*p_odom), atol=1e-12)


def test_refuses_out_of_range_and_across_gaps():
    b = PoseBuffer(horizon_s=1.0, max_gap_s=0.1)
    for k in range(30):
        t = 5.0 + 0.05 * k
        b.add_position(t, (k, 0, 0)); b.add_attitude(t, (1, 0, 0, 0))
    assert b.at(5.0 + 0.05 * 29 + 0.01) is None                 # newer than the newest: no extrapolation
    assert b.at(5.0 + 0.05 * 29 - 1.2) is None                  # older than the 1 s horizon
    b.add_position(8.0, (99, 0, 0)); b.add_attitude(8.0, (1, 0, 0, 0))
    assert b.at(7.9) is None                                    # between samples 1.5 s apart


def test_slerp_takes_the_short_way():
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = -np.array([math.cos(0.1), 0.0, 0.0, math.sin(0.1)])    # the same rotation with the opposite sign
    q = slerp(q0, q1, 0.5)
    assert abs(abs(q[0]) - math.cos(0.05)) < 1e-9


def test_odometry_attitude_levels_the_same_as_px4s():
    """skybeta odometry (ENU <- FLU) converted for refnav's levelling must level points like PX4's attitude does."""
    from nav.pose_buffer import enu_flu_to_ned_frd_wxyz, mat_to_wxyz, quat_wxyz_to_mat
    from refnav.depth import level_rotation
    NED_ENU = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, -1]])
    FLU_FRD = np.diag([1.0, -1, -1])
    rng = np.random.default_rng(4)
    for _ in range(20):
        roll, pitch, yaw = rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-math.pi, math.pi)
        cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
        R_enu_flu = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                     @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
        w, x, y, z = mat_to_wxyz(R_enu_flu)
        q_px4 = mat_to_wxyz(NED_ENU @ R_enu_flu @ FLU_FRD)              # what PX4 would report for the same attitude
        q_conv = enu_flu_to_ned_frd_wxyz((x, y, z, w))
        assert np.allclose(quat_wxyz_to_mat(*q_conv), quat_wxyz_to_mat(*q_px4), atol=1e-9)
        L = level_rotation(q_conv)
        g_level = L @ (R_enu_flu.T @ np.array([0.0, 0.0, -1.0]))          # gravity, from the body into the level frame
        assert np.allclose(g_level, [0.0, 0.0, 1.0], atol=1e-9)          # level frame z points down


def test_clock_map_takes_the_latency_floor():
    c = ClockMap()
    assert c.to_ros(1_000_000) is None
    for recv, px4_us, lat in ((100.010, 0, 0.010), (100.105, 100_000, 0.005), (100.203, 200_000, 0.003)):
        c.observe(recv, px4_us)
    assert c.to_ros(300_000) == pytest.approx(100.0 + 0.3 + 0.003)
