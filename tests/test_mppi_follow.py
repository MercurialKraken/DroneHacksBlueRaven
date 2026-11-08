"""nav:=mppi: MPPI (blueraven_skybeta.mppi_follow) as refnav's follower, flown through the same fake organizers' stack
as tests/test_refnav_skybeta.py, with pinhole depth images rendered from the tape course's start (tests/sim_world)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skybeta" / "blueraven_skybeta"))
from blueraven_skybeta import mppi_follow as M  # noqa: E402
from blueraven_skybeta import refnav_skybeta as S  # noqa: E402

from nav.depth_points import R_FLU_OPTICAL  # noqa: E402
from refnav import core as C  # noqa: E402
from sim_world import WALL, World, start_world  # noqa: E402
from test_local_nav import TAPE, crossed, tape_course  # noqa: E402
from test_refnav_skybeta import DT, LAUNCH_PARAMS, Stack, _rz  # noqa: E402

STAGE1 = json.loads((TAPE / "routes" / "route_v6_start2_stage1_sim.json").read_text())
W, H = 80, 50                                           # the drone's 640x400 at 1/8: K scaled, every pixel used
K = np.array([[50.2, 0.0, 39.2], [0.0, 50.2, 24.2], [0.0, 0.0, 1.0]])
T_BASE_CAM = np.eye(4)
T_BASE_CAM[:3, :3], T_BASE_CAM[:3, 3] = R_FLU_OPTICAL, (0.105, 0.038, 0.031)


@pytest.mark.parametrize("p0, yaw0", [((0.0, 0.0, 0.0), 0.0), ((1.2, -0.7, 0.1), 2.3), ((-3.0, 2.0, 0.0), -1.0)])
def test_route_to_odom_is_the_pads_forward_left(p0, yaw0):
    pad, route = S.Pad(*p0, yaw0), STAGE1["route"]
    want = [np.add(p0[:2], pad.to_odom(f, -l)) for f, l in route]
    assert np.allclose(M.route_to_odom(route, p0, yaw0), want, atol=1e-12)


def test_despeckle_keeps_surfaces_and_drops_lone_spikes_and_holes():
    d = np.full((40, 40), 2.0)
    d[18, 18] = 0.9                                     # a lone spike on a sampled pixel (stride 4: rows 2, 6, ...)
    d[10, 10] = np.nan
    d[26:, :] = 1.0                                     # a real step
    out = M.despeckle(d)
    assert np.isnan(out[18, 18]) and np.isnan(out[10, 10]) and out[14, 14] == 2.0 and out[30, 30] == 1.0
    assert np.isnan(out[15, 15])                        # only the stride's samples
    sparse = np.full((40, 40), np.nan)
    sparse[2::4, 2::4] = 2.0                            # returns without neighbours are speckle
    assert np.isnan(M.despeckle(sparse)).all()


# ---------------------------------------------------------------------------------------------- the core branch
def _following(ext=None):
    prm, _ = C.params_from_doc(STAGE1, {**LAUNCH_PARAMS, "nav": "mppi"})
    m = C.Mission(prm)
    m.phase, m.t_phase, m.sp = "follow", 0.0, [1.0, 2.0, -0.7, 0.3]
    m.offboard_seen = True
    m.health.on_msg(0.0, True)
    v = C.Vehicle(pos=(1.5, 2.5, -0.7), heading=0.3, pos_ok=True, armed=True, nav_state=C.NAV_OFFBOARD)
    m.ext = ext
    return m, v


def _ext(state="GO", vn=0.3, ve=-0.1, yaw=0.5, age=0.05):
    return {"vn": vn, "ve": ve, "yaw": yaw, "state": state, "depth_age": age}


def test_the_mppi_velocity_is_the_feed_forward_at_the_aircrafts_own_position():
    m, v = _following(_ext())
    m.step(0.0, v)
    m.health.on_msg(0.05, True)
    out = m.step(0.05, v)
    assert out.phase == "follow" and out.velocity == (0.3, -0.1) and out.setpoint[:3] == (1.5, 2.5, -0.7)
    assert 0.3 < out.setpoint[3] < 0.5 and out.yawspeed > 0                    # slewed towards MPPI's heading
    tw = S.twist(out, v, S.Pad(0.0, 0.0, 0.0, 0.0), 45.0, 0.0)
    assert tw[:2] == pytest.approx(S.Pad(0, 0, 0, 0).to_odom(0.3, -0.1)) and tw[2] == pytest.approx(0.0)


@pytest.mark.parametrize("state", ["WAIT", "ARRIVED", "BLOCKED", "STUCK"])
def test_mppi_states_other_than_go_and_slow_command_no_velocity(state):
    m, v = _following(_ext(state))
    assert m.step(0.05, v).velocity == (0.0, 0.0)


@pytest.mark.parametrize("state, t, why", [("ARRIVED", 0.05, "route complete"), ("STUCK", 0.05, "MPPI stuck"),
                                           ("BLOCKED", 2.1, "MPPI blocked"), ("SLOW", 3.1, "MPPI: depth silent")])
def test_arrived_stuck_blocked_for_2_s_and_3_s_without_depth_land(state, t, why):
    m, v = _following()
    k = 1
    while k * DT <= t + 1e-9 and m.phase == "follow":
        m.health.on_msg(k * DT, True)
        m.ext = _ext(state, age=k * DT)                 # depth last seen at follow start
        m.step(k * DT, v)
        k += 1
    assert m.phase == "land" and m.land_reason.startswith(why), (m.phase, m.land_reason)
    assert (k - 1) * DT >= t - DT


@pytest.mark.parametrize("state", ["BLOCKED", "SLOW"])
def test_a_short_block_and_slow_flight_on_fresh_enough_depth_keep_following(state):
    m, v = _following()
    for k in range(1, 38):                               # 1.85 s
        m.health.on_msg(k * DT, True)
        m.ext = _ext(state, age=0.6)
        m.step(k * DT, v)
    assert m.phase == "follow"


def test_no_mppi_command_for_3_s_after_follow_starts_lands():
    m, v = _following(None)
    for k in range(1, 70):
        m.health.on_msg(k * DT, True)
        m.step(k * DT, v)
    assert m.phase == "land" and "depth silent" in m.land_reason


def test_nav_must_be_refnav_or_mppi():
    with pytest.raises(ValueError, match="nav must be"):
        C.params_from_doc(STAGE1, {**LAUNCH_PARAMS, "nav": "other"})


# ---------------------------------------------------------------------------------------------- flights
def flight(offset_deg: float = 0.0, depth_until: float = math.inf, limit_s: float = 120.0, world=None):
    """The stage-1 flight from the sim spawn: odometry starts at the spawn facing x (map = spawn + Rz(launch yaw) odom),
    so truth is odometry. Depth at 15 Hz, stamped at capture and delivered 70 ms later, as on the drone."""
    w0, start, yaw_map, _, _ = tape_course()
    w = world or w0
    off = math.radians(offset_deg)
    prm, _ = C.params_from_doc(STAGE1, {**LAUNCH_PARAMS, "nav": "mppi"})
    m, sk, v, pad, descent = C.Mission(prm), Stack(off, 0.0), C.Vehicle(), None, S.Descent()
    mf = M.MppiFollow(prm.speed_mps, 0.3)                                   # the Stack's velocity lag
    state, t_state, t_odom, t_arm = None, -math.inf, -math.inf, None
    R_map = _rz(yaw_map)
    traj, states, sent, pending = [], [], [], []
    t = DT
    while t < limit_s and m.phase != "done":
        p, q, vf = sk.odom()
        pad, t_odom = S.on_odom(v, pad, m.phase, p, q, vf), t
        mf.on_odom(t, p, q, vf)
        m.on_health(t, S.healthy(state, t_state, t))
        if round(t / DT) % 4 == 0:
            state, t_state = sk.state(), t
            S.on_state(v, state, m.phase)
        p_map = np.r_[np.array(start) + (R_map @ np.array(p))[:2], p[2]]
        if m.phase in C.AIRBORNE:
            traj.append(p_map[:2])
        if round(t / DT) % 4 != 1 and t < depth_until:                      # 15 Hz
            R_cam = R_map @ _rz(sk.yaw_enu) @ T_BASE_CAM[:3, :3]
            cam = p_map + R_map @ _rz(sk.yaw_enu) @ T_BASE_CAM[:3, 3]
            pending.append((t + 0.07, t, w.depth_image(R_cam, cam, K, W, H)))
        while pending and pending[0][0] <= t + 1e-9:
            _, stamp, img = pending.pop(0)
            mf.on_depth(stamp, img, K, T_BASE_CAM, stride=1)
        if m.phase == "follow":
            if mf.nav is None:
                mf.start(prm.route, pad)
            m.ext = mf.tick(t, pad)
            if m.ext:
                states.append(m.ext["state"])
        out = m.step(t, v)
        tw, land = S.command(out, v, pad, descent, state, t_odom, t_state, t, prm.yaw_rate_dps, off)
        if land:
            m.request_land(land)
        if tw is not None:
            sk.twist(t, tw)
            sent.append((t, out.phase, tw))
        if m.phase == "arm" and not sk.armed and t_arm is None and sent and t - sent[-1][0] < 1.0:
            sk.armed, sk.nav, t_arm = True, 1, t
        if t_arm is not None and sk.nav == 1 and t - t_arm >= 0.5 and t - sk.t_twist < 1.0:
            sk.nav = C.NAV_OFFBOARD
        sk.step(t)
        t += DT
    return m, sk, np.array(traj), states, w, mf


@pytest.mark.parametrize("offset_deg", [0.0, -90.0])
def test_stage1_is_flown_through_the_1m_cp2_gap_and_landed(offset_deg):
    m, sk, traj, states, w, mf = flight(offset_deg)
    assert m.result == "complete", (m.result, m.why, states[-5:])
    assert not sk.armed and sk.landed
    assert w.clearance(traj, min_height=0.35).min() > mf.mppi_p.r_drone      # the cone is flown over
    assert crossed(traj, tape_course()[4])
    assert {"GO", "ARRIVED"} <= set(states)


def test_depth_lost_in_flight_slows_then_lands_3_s_later():
    m, sk, traj, states, w, mf = flight(depth_until=8.0)                 # follow runs 6.3-16 s with depth
    assert m.land_reason.startswith("MPPI: depth silent"), (m.land_reason, states[-5:])
    assert "SLOW" in states and not sk.armed


def test_a_wall_across_the_gap_blocks_or_sticks_and_lands_without_contact():
    w = start_world(TAPE / "course.yaml")
    shut = World(w.x_range, w.y_range, boxes=[*w.boxes, (-1.45, -0.80, 1.90, 2.40, WALL)], polys=w.polys, nets=w.nets)
    m, sk, traj, states, _, mf = flight(world=shut)
    assert m.land_reason.startswith(("MPPI stuck", "MPPI blocked")), (m.land_reason, states[-5:])
    assert shut.clearance(traj, min_height=0.35).min() > mf.mppi_p.r_drone and not sk.armed

