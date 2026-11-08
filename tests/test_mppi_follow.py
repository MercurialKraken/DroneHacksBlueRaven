"""nav:=mppi: MPPI (blueraven_skybeta.mppi_follow) as refnav's follower, flown through the same fake organizers' stack
as tests/test_refnav_skybeta.py, with pinhole depth images rendered from the tape course's start (tests/sim_world)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skybeta" / "blueraven_skybeta"))
from blueraven_skybeta import mppi_follow as M  # noqa: E402
from blueraven_skybeta import refnav_skybeta as S  # noqa: E402

from nav.depth_points import R_FLU_OPTICAL  # noqa: E402
from refnav import core as C  # noqa: E402
from sim_world import WALL, World, start_world  # noqa: E402
from test_refnav_skybeta import DT, LAUNCH_PARAMS, Stack, _rz  # noqa: E402

TAPE = ROOT / "courses" / "start-2026-11-07-tape"
STAGE1 = json.loads((TAPE / "routes" / "route_v6_start2_stage1_sim.json").read_text())
W, H = 80, 50                                           # the drone's 640x400 at 1/8: K scaled, every pixel used
K = np.array([[50.2, 0.0, 39.2], [0.0, 50.2, 24.2], [0.0, 0.0, 1.0]])
T_BASE_CAM = np.eye(4)
T_BASE_CAM[:3, :3], T_BASE_CAM[:3, 3] = R_FLU_OPTICAL, (0.105, 0.038, 0.031)


def tape_course():
    """The tape course (CP2 gap 1.0 m), its sim spawn, Chelsea's stage-1 route in map coordinates, and CP2's line."""
    y = yaml.safe_load((TAPE / "course.yaml").read_text())
    L, cp2 = y["launch_sim"], y["checkpoints"][0]["line"]
    yaw = math.radians(L["yaw_deg"])
    wps = M.route_to_odom(STAGE1["route"], (L["x"], L["y"]), yaw)
    return start_world(TAPE / "course.yaml"), (L["x"], L["y"]), yaw, wps, np.array(cp2)


def crossed(traj: np.ndarray, line: np.ndarray) -> bool:
    """The trajectory crossed the segment between the line's ends."""
    p, q = line
    side = np.sign((q[0] - p[0]) * (traj[:, 1] - p[1]) - (q[1] - p[1]) * (traj[:, 0] - p[0]))
    k = int(np.argmax(side != side[0]))
    return k > 0 and 0.0 < float(np.dot(traj[k] - p, q - p) / np.dot(q - p, q - p)) < 1.0


@pytest.mark.parametrize("p0, yaw0", [((0.0, 0.0, 0.0), 0.0), ((1.2, -0.7, 0.1), 2.3), ((-3.0, 2.0, 0.0), -1.0)])
def test_route_to_odom_is_the_pads_forward_left(p0, yaw0):
    pad, route = S.Pad(*p0, yaw0), STAGE1["route"]
    want = [np.add(p0[:2], pad.to_odom(f, -l)) for f, l in route]
    assert np.allclose(M.route_to_odom(route, p0[:2], yaw0), want, atol=1e-12)


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


@pytest.mark.parametrize("state", ["ARRIVED", "STUCK", "NOCMD"])
def test_mppi_states_that_end_or_lack_a_command_send_no_velocity(state):
    m, v = _following(_ext(state))
    assert m.step(0.05, v).velocity == (0.0, 0.0)


@pytest.mark.parametrize("state", ["WAIT", "HOLD", "BLOCKED"])
def test_mppi_holds_send_only_their_push_compensation_bounded(state):
    m, v = _following(_ext(state, vn=-0.06, ve=0.09))                 # minus a ~0.11 m/s push: passed as is
    assert m.step(0.05, v).velocity == pytest.approx((-0.06, 0.09))
    m, v = _following(_ext(state, vn=0.3, ve=-0.1))                  # anything larger: cut to EXT_HOLD_MAX_MPS
    out = m.step(0.05, v).velocity
    assert math.hypot(*out) == pytest.approx(C.EXT_HOLD_MAX_MPS) and out[0] / out[1] == pytest.approx(-3.0)


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
def flight(offset_deg: float = 0.0, depth_until: float = math.inf, limit_s: float = 200.0, world=None,
           plant: tuple[float, float] = (0.3, 0.0), jitter_m: float = 0.0, link_s: float = 0.05,
           mppi_dies_after_s: float = math.inf, odom_jump_after_s: float = math.inf,
           odom_freeze_after_s: float = math.inf):
    """The stage-1 flight from the sim spawn: odometry starts at the spawn facing x (map = spawn + Rz(launch yaw) odom),
    so truth is odometry. Depth at 15 Hz, stamped at capture and delivered 70 ms later, as on the drone.
    MPPI is mppi_node's side of the link (pad None, cmd_msg as JSON, delivered link_s later); refnav's side is
    CmdLink, exactly as refnav_node wires it. mppi_dies_after_s: mppi_node stops sending that long into follow."""
    w0, start, yaw_map, _, _ = tape_course()
    w = world or w0
    off = math.radians(offset_deg)
    prm, _ = C.params_from_doc(STAGE1, {**LAUNCH_PARAMS, "nav": "mppi"})
    m, sk, v, pad, descent, track = C.Mission(prm), Stack(off, 0.0, tau=plant[0], delay=plant[1]), C.Vehicle(), None, S.Descent(), S.Track()
    lag = plant[0] + plant[1]                                               # vel_lag_s: the Stack's lag + dead time
    mf = M.MppiFollow(0.15, plant[0], plant[1])                              # MPPI models the plant
    state, t_state, t_odom, t_arm = None, -math.inf, -math.inf, None
    R_map = _rz(yaw_map)
    traj, states, sent, pending = [], [], [], []
    link, wire, mppi_id, traj_t = M.CmdLink(), [], None, []
    frozen = None
    rng = np.random.default_rng(7)
    t = DT
    while t < limit_s and m.phase != "done":
        p, q, vf = sk.odom()
        p = tuple(np.add(p, np.r_[rng.normal(0.0, jitter_m, 2), 0.0])) if jitter_m else p     # VIO jitter per sample
        p_est = p                                                           # what cuVSLAM reports (traj stays truth)
        if link.start_id is not None and t - link.t_start >= odom_jump_after_s:
            p_est = (p[0] + 1.6, p[1] - 1.6, p[2])                          # a 2.3 m jump that stays (Darion's T4)
        if link.start_id is not None and t - link.t_start >= odom_freeze_after_s:   # the estimate freezes
            frozen = frozen or (p_est, q, vf)
            p_est, q, vf = frozen
        pad, t_odom = S.on_odom(v, pad, m.phase, p_est, q, vf), t
        mf.on_odom(t, p_est, q, vf)
        m.on_health(t, S.healthy(state, t_state, t))
        if round(t / DT) % 4 == 0:
            state, t_state = sk.state(), t
            S.on_state(v, state, m.phase)
        p_map = np.r_[np.array(start) + (R_map @ np.array(p))[:2], p[2]]
        if m.phase in C.AIRBORNE:
            traj.append(p_map[:2])
            traj_t.append(t)
        if round(t / DT) % 4 != 1 and t < depth_until:                      # 15 Hz
            R_cam = R_map @ _rz(sk.yaw_enu) @ T_BASE_CAM[:3, :3]
            cam = p_map + R_map @ _rz(sk.yaw_enu) @ T_BASE_CAM[:3, 3]
            pending.append((t + 0.07, t, w.depth_image(R_cam, cam, K, W, H)))
        while pending and pending[0][0] <= t + 1e-9:
            _, stamp, img = pending.pop(0)
            mf.on_depth(stamp, img, K, T_BASE_CAM, stride=1)
        if m.phase == "follow":
            if link.start_id is None:                                        # refnav -> /blueraven/mppi/start
                msg = json.loads(json.dumps(M.start_msg(link.begin(t), pad)))
                mppi_id = msg["id"]
                mf.start(prm.route, S.Pad(*msg["pad"]))                      # mppi_node places the route from it
            if t - link.t_start < mppi_dies_after_s:                         # mppi_node's own 10 Hz step
                c0 = mf.cmd
                mf.tick(t, None)
                if mf.cmd is not c0:
                    wire.append((t + link_s, json.dumps(M.cmd_msg(mf.cmd, mppi_id, t))))
            while wire and wire[0][0] <= t + 1e-9:                           # refnav's /blueraven/mppi/cmd callback
                link.on_cmd(json.loads(wire.pop(0)[1]))
            m.ext = link.ext(t, pad)
            why = link.why_land(t)
            if why:
                m.request_land(why)
            states.append(m.ext["state"])
        out = m.step(t, v)
        tw, land = S.command(out, v, pad, descent, track, state, t_odom, t_state, t, prm.yaw_rate_dps, off, lag)
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
    mf.link, mf.traj_t = link, np.array(traj_t)                              # for the link tests
    return m, sk, np.array(traj), states, w, mf


@pytest.mark.parametrize("offset_deg, plant", [(0.0, (0.3, 0.0)), (-90.0, (0.3, 0.0)), (-90.0, (0.9, 0.6))])
def test_stage1_is_flown_through_the_1m_cp2_gap_and_landed(offset_deg, plant):
    m, sk, traj, states, w, mf = flight(offset_deg, plant=plant)                # (0.9, 0.6): the T4 sim's fitted lag
    assert m.result == "complete", (m.result, m.why, states[-5:])
    assert not sk.armed and sk.landed
    assert w.clearance(traj, min_height=0.35).min() > mf.mppi_p.r_drone      # the cone is flown over
    assert crossed(traj, tape_course()[4])
    assert {"GO", "ARRIVED"} <= set(states)


def test_depth_lost_in_flight_slows_then_lands_3_s_later():
    m, sk, traj, states, w, mf = flight(depth_until=8.0)                 # follow runs 6.3-16 s with depth
    assert m.land_reason.startswith("MPPI: depth silent"), (m.land_reason, states[-5:])
    assert "SLOW" in states and not sk.armed


@pytest.mark.parametrize("jitter_m", [0.0, 0.003, 0.006])
def test_a_wall_across_the_gap_blocks_or_sticks_and_lands_without_contact(jitter_m):
    w = start_world(TAPE / "course.yaml")
    shut = World(w.x_range, w.y_range, boxes=[*w.boxes, (-1.45, -0.80, 1.90, 2.40, WALL)], polys=w.polys, nets=w.nets)
    m, sk, traj, states, _, mf = flight(world=shut, jitter_m=jitter_m)
    assert m.land_reason.startswith(("MPPI stuck", "MPPI blocked")), (m.land_reason, states[-5:])
    assert shut.clearance(traj, min_height=0.35).min() > mf.mppi_p.r_drone and not sk.armed



# ---------------------------------------------------------------------------------------------- the MPPI process link
class _Cmd:                                             # a NavCommand without MPPI's result
    def __init__(self, vx=0.12, vy=-0.05, yaw=-0.4, state="GO", depth_age=0.06):
        self.vx, self.vy, self.yaw, self.state, self.depth_age, self.mppi = vx, vy, yaw, state, depth_age, None


@pytest.mark.parametrize("p0, yaw0", [((0.0, 0.0, 0.0), 0.0), ((1.2, -0.7, 0.1), 2.3), ((-3.0, 2.0, 0.0), -1.0)])
def test_the_link_gives_refnav_the_same_pad_frame_command_as_the_in_process_follower(p0, yaw0):
    pad, c = S.Pad(*p0, yaw0), _Cmd()
    msg = json.loads(json.dumps(M.cmd_msg(c, 5.0, 5.0)))
    e = M.to_ext(msg, pad)
    vn, ve = pad.vector(c.vx, c.vy)                     # what MppiFollow.tick puts in Mission.ext
    assert (e["vn"], e["ve"]) == pytest.approx((vn, ve)) and e["yaw"] == pytest.approx(math.remainder(yaw0 - c.yaw, math.tau))
    assert e["state"] == "GO" and e["depth_age"] == 0.06
    assert pad.to_odom(e["vn"], e["ve"]) == pytest.approx((c.vx, c.vy))      # and back to odom: unchanged


def test_the_link_flies_only_fresh_commands_for_this_follow():
    pad, link = S.Pad(0.0, 0.0, 0.0, 0.0), M.CmdLink()
    assert not link.on_cmd(M.cmd_msg(_Cmd(), None, 1.0))                     # before follow: nothing is flown
    sid = link.begin(10.0)
    assert link.ext(10.0, pad)["state"] == "NOCMD" and link.ext(10.0, pad)["vn"] == 0.0
    assert not link.on_cmd(M.cmd_msg(_Cmd(), None, 10.05))                   # the pre-flight loop
    assert not link.on_cmd(M.cmd_msg(_Cmd(), sid - 30.0, 10.05))            # an earlier follow's
    assert link.on_cmd(M.cmd_msg(_Cmd(), sid, 10.1))
    assert not link.on_cmd(M.cmd_msg(_Cmd(vx=9.0), sid, 10.05))             # out of order: the newer one stays
    assert link.ext(10.2, pad)["state"] == "GO" and link.ext(10.2, pad)["vn"] == pytest.approx(0.12)
    held = link.ext(10.1 + M.CMD_STALE_S + 0.01, pad)                        # stale: hold still
    assert held["state"] == "NOCMD" and (held["vn"], held["ve"]) == (0.0, 0.0) and held["yaw"] is None
    assert held["depth_age"] == pytest.approx(0.06 + M.CMD_STALE_S + 0.01)   # depth's own age keeps counting
    assert link.why_land(10.1 + M.CMD_LAND_S - 0.01) is None
    assert link.why_land(10.1 + M.CMD_LAND_S + 0.01).startswith("MPPI command silent")


def test_no_command_in_3_s_of_follow_lands_and_none_is_needed_before_follow():
    link = M.CmdLink()
    assert link.why_land(1e6) is None
    link.begin(10.0)
    assert link.why_land(10.0 + M.CMD_FIRST_S - 0.01) is None
    assert link.why_land(10.0 + M.CMD_FIRST_S + 0.01).startswith("MPPI sent no command")


def test_a_stale_command_reaches_the_aircraft_as_zero_velocity():
    m, v = _following(M.CmdLink().ext(0.0, S.Pad(0.0, 0.0, 0.0, 0.0)))
    assert m.step(0.05, v).velocity == (0.0, 0.0)


def test_mppi_dying_in_flight_holds_still_then_lands_without_contact():
    m, sk, traj, states, w, mf = flight(mppi_dies_after_s=8.0)
    assert "MPPI command silent" in m.land_reason, (m.land_reason, states[-5:])
    t_dead = mf.link.t_start + 8.0
    after = traj[mf.traj_t >= t_dead]                                        # every airborne sample from the death on
    assert len(after) >= round(M.CMD_LAND_S / DT) - 1                        # (no silent empty slice)
    assert np.linalg.norm(after - after[0], axis=1).max() < 0.2              # held still, then landed there
    assert w.clearance(traj, min_height=0.35).min() > mf.mppi_p.r_drone and not sk.armed


def test_a_slow_link_still_flies_the_route():
    m, sk, traj, states, w, mf = flight(link_s=0.25)                          # 2.5 MPPI steps late, under CMD_STALE_S
    assert m.result == "complete", (m.result, m.why, states[-5:])
    assert w.clearance(traj, min_height=0.35).min() > mf.mppi_p.r_drone


def test_refnav_never_runs_mppi_and_mppi_node_never_commands_the_drone():
    pkg = ROOT / "skybeta" / "blueraven_skybeta" / "blueraven_skybeta"
    refnav = (pkg / "refnav_node.py").read_text()
    assert "MppiFollow" not in refnav and "local_nav" not in refnav and "CmdLink" in refnav
    node = (pkg / "mppi_node.py").read_text()
    import ast
    pubs = [c.args[1] for c in ast.walk(ast.parse(node))
            if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "create_publisher"]
    assert len(pubs) == 1 and isinstance(pubs[0], ast.JoinedStr) and pubs[0].values[0].value.id == "PREFIX"
    assert "MppiPublisher(" in node and "TwistStamped" not in node and "ReliabilityPolicy.RELIABLE" in node
    assert "qos_profile_sensor_data" not in node + (pkg / "mppi_shadow.py").read_text()
    launch = (ROOT / "skybeta" / "blueraven_skybeta" / "launch" / "refnav.launch.py").read_text()
    assert "executable='mppi_node'" in launch and "' == 'mppi'" in launch and "' != 'mppi'" in launch


# ---------------------------------------------------------------------------------------------- VIO jump guard
def _started(v_cruise=0.15):
    mf = M.MppiFollow(v_cruise, 0.9, 0.6)
    mf.start(STAGE1["route"], S.Pad(0.0, 0.0, 0.0, 0.0))
    return mf


def _odom(mf, t, x, y=0.0):
    mf.on_odom(t, (x, y, 0.7), (0.0, 0.0, 0.0, 1.0), (0.1, 0.0, 0.0))


def test_flight_motion_and_real_odometry_gaps_never_trip_the_jump_guard():
    mf, rng = _started(), np.random.default_rng(1)
    t, x = 0.0, 0.0
    for k in range(600):                                  # 20 s at 30 Hz, 0.6 m/s (MPPI's cap at 0.15 cruise is 0.45)
        gap = 0.35 if k % 50 == 49 else 1 / 30            # skybeta-09's in-flight gaps reach 0.35 s
        t, x = t + gap, x + 0.6 * gap
        _odom(mf, t, x + rng.normal(0, 0.003))
    assert mf.lost is None


def test_a_vio_jump_latches_lost_until_the_next_follow_start():
    mf = _started()
    _odom(mf, 1.0, 0.0)
    _odom(mf, 1.033, 0.5)                                 # 0.5 m in 33 ms
    assert mf.lost and "jumped 0.50 m" in mf.lost
    _odom(mf, 1.066, 0.5)                                 # still where it jumped to: stays LOST
    assert mf.lost
    mf.tick(2.0, None)
    assert mf.cmd.state == "LOST" and (mf.cmd.vx, mf.cmd.vy) == (0.0, 0.0)
    m = json.loads(json.dumps(M.cmd_msg(mf.cmd, 5.0, 2.0)))
    link = M.CmdLink()
    link.begin(1.9)
    m["start"] = link.start_id
    assert link.on_cmd(m) and link.why_land(2.0).startswith("MPPI: VIO lost: odometry jumped")
    assert link.ext(2.0, S.Pad(0.0, 0.0, 0.0, 0.0))["vn"] == 0.0
    mf.start(STAGE1["route"], S.Pad(0.0, 0.0, 0.0, 0.0))
    assert mf.lost is None


def test_mppis_flight_speed_cap_follows_the_cruise():
    p = M.MppiFollow(0.10, 0.9, 0.6).mppi_p                     # ground 0.30, command 0.45 at 0.10 m/s cruise
    assert (p.v_max, p.v_lat, p.u_max) == (pytest.approx(0.30), pytest.approx(0.30), pytest.approx(0.45))
    p = M.MppiFollow(0.15, 0.9, 0.6).mppi_p
    assert (p.v_max, p.u_max) == (pytest.approx(0.45), pytest.approx(0.675))


def test_a_vio_jump_in_flight_lands_where_the_drone_really_is():
    m, sk, traj, states, w, mf = flight(odom_jump_after_s=6.0)
    assert "odometry jumped" in m.land_reason, (m.land_reason, states[-5:])
    after = traj[mf.traj_t >= mf.link.t_start + 6.0]                          # truth from the jump on
    assert np.linalg.norm(after - after[0], axis=1).max() < 0.2               # it did not chase the false position
    assert w.clearance(traj, min_height=0.35).min() > mf.mppi_p.r_drone and not sk.armed


def test_mppi_nodes_depth_callback_only_keeps_the_newest_frame_and_runs_single_threaded():
    """skybeta-09, full nav:=mppi launch: a lagging RELIABLE depth reader starved cuVSLAM (odometry 9.7 Hz).
    The callback must stay O(1); frames are decoded/placed in the tick at <= ~10 Hz, off the MPPI-step ticks."""
    import ast
    pkg = ROOT / "skybeta" / "blueraven_skybeta" / "blueraven_skybeta"
    tree = ast.parse((pkg / "mppi_node.py").read_text())
    cb = next(f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name == "_on_depth")
    calls = {getattr(c.func, "attr", getattr(c.func, "id", "")) for c in ast.walk(cb) if isinstance(c, ast.Call)}
    assert not calls & {"decode_depth", "on_depth", "_process_depth", "lookup_transform"}, calls
    tick = next(f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name == "_tick")
    assert "_process_depth" in ast.unparse(tick) and "DEPTH_PERIOD_S" in ast.unparse(tick)
    launch = (ROOT / "skybeta" / "blueraven_skybeta" / "launch" / "refnav.launch.py").read_text()
    assert "'OMP_NUM_THREADS': '1'" in launch and launch.count("additional_env=ONE_THREAD") == 2


def _depth_timing(policy: str, depth_lat: float, odom_lat: float, stamp_off: float = 20e-6, secs: float = 6.0) -> float:
    """mppi_node's depth path with real MppiFollow + pose buffer: odometry (stamped at capture, received odom_lat later)
    and on-camera depth (stamped stamp_off after the same capture, received depth_lat later), both 30 Hz; the node's
    20 Hz tick processes depth on non-step ticks at <= 10 Hz. policy 'newest' = before the fix (only the newest
    frame, dropped if it can't be placed), 'queue' = DepthQueue. -> frames placed per second in follow."""
    mf = M.MppiFollow(0.10, 0.9, 0.6)
    img = np.full((H, W), 2.0)                                              # a surface 2 m ahead: any finite depth
    ev = sorted([(k / 30 + odom_lat, 0, k / 30) for k in range(int(secs * 30))]
                + [(k / 30 + depth_lat, 1, k / 30 + stamp_off) for k in range(int(secs * 30))])
    q, newest, placed, t_proc, ticks, i = M.DepthQueue(), None, 0, -1e9, 0, 0
    for n in range(1, int(secs / 0.05) + 1):                                # mppi_node._tick every 50 ms
        t = n * 0.05
        while i < len(ev) and ev[i][0] <= t:
            _, kind, st = ev[i]
            i += 1
            if kind == 0:
                mf.on_odom(st, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
                if mf.nav is None:
                    mf.start([[1.0, 0.0], [3.0, 0.0]], S.Pad(0.0, 0.0, 0.0, 0.0))
            else:
                q.push(st, st)
                newest = st
        ticks += 1
        if ticks % M.STEP_EVERY == 1 % M.STEP_EVERY and t - t_proc >= 0.09:
            if policy == "newest":
                st, newest = newest, None
            else:
                got = q.pop(mf.stamp if mf.pos is not None else None)
                st = None if got is None else got[0]
            if st is not None:
                t_proc = t
                placed += bool(mf.on_depth(st, img, K, T_BASE_CAM, stride=1))
    return placed / secs


def test_depth_arriving_ahead_of_odometry_is_placed_not_dropped():
    """skybeta-10 route_mppi_full_160248 (run 1): depth ~20 us after its capture's odometry stamp, arriving before that
    odometry (odometry ~35 ms late). The newest-frame-only reader placed 1-2 frames per 5 s ('no pose' 386 -> 613),
    flapped SLOW/HOLD (the crawl) and landed on 'depth silent'. DepthQueue places it at ~10 Hz, and still does when
    depth arrives after odometry (run 2's timing)."""
    run1 = dict(depth_lat=0.010, odom_lat=0.035)
    assert _depth_timing("newest", **run1) < 0.5                            # the bug, reproduced: almost nothing placed
    assert _depth_timing("queue", **run1) >= 8.0                            # fixed: ~10 Hz
    run2 = dict(depth_lat=0.060, odom_lat=0.035)                            # depth later than odometry: worked before
    assert _depth_timing("newest", **run2) >= 8.0
    assert _depth_timing("queue", **run2) >= 8.0                            # and still does
    for lat in (0.0, 0.020, 0.034, 0.050, 0.100):                           # any depth latency vs 35 ms odometry
        assert _depth_timing("queue", depth_lat=lat, odom_lat=0.035) >= 8.0, lat


def test_depth_queue_keeps_too_new_frames_and_drops_older_ones():
    q = M.DepthQueue(keep=4)
    assert q.pop(None) is None and q.pop(1.0) is None
    for st in (1.00, 1.03, 1.07, 1.10):
        q.push(st, f"f{st}")
    assert q.pop(1.05) == (1.03, "f1.03")                                   # newest covered; 1.00 dropped with it
    assert [s for s, _ in q.q] == [1.07, 1.10]                              # the too-new ones wait
    assert q.pop(1.05) is None and len(q.q) == 2
    q.push(1.13, "f1.13")
    q.push(1.17, "f1.17")
    q.push(1.20, "f1.20")                                                   # keep=4: 1.07 falls off
    assert q.pop(9.0) == (1.20, "f1.20") and len(q.q) == 0


# ---------------------------------------------------------------------------------------------- frozen / heading / stale
def _odom_q(mf, t, x, yaw_deg=0.0, y=0.0):
    h = math.radians(yaw_deg) / 2
    mf.on_odom(t, (x, y, 0.4), (0.0, 0.0, math.sin(h), math.cos(h)), (0.1, 0.0, 0.0))


def test_hover_jitter_and_normal_turns_never_trip_the_freeze_or_heading_guards():
    mf, rng = _started(), np.random.default_rng(3)
    for k in range(900):                                  # 30 s at 30 Hz: sub-mm jitter, a 30 deg/s turn, a hold
        yaw = min(30.0 * max(k / 30 - 10, 0), 90.0)
        _odom_q(mf, k / 30, 0.0005 * rng.standard_normal(), yaw, 0.0005 * rng.standard_normal())
    assert mf.lost is None


def test_a_frozen_estimate_latches_lost_in_half_a_second():
    mf = _started()
    for k in range(10):
        _odom_q(mf, k / 30, 0.01 * k)                     # flying
    for k in range(10, 40):
        _odom_q(mf, k / 30, 0.09)                         # cuVSLAM output frozen, stamps still advancing
        if mf.lost:
            break
    assert mf.lost and "frozen" in mf.lost and k / 30 - 9 / 30 <= M.FREEZE_S + 0.05


def test_a_heading_jump_latches_lost():
    mf = _started()
    _odom_q(mf, 1.0, 0.0, 0.0)
    _odom_q(mf, 1.033, 0.003, 25.0)                       # +25 deg in one sample
    assert mf.lost and "heading jumped 25 deg" in mf.lost


def test_odometry_that_keeps_arriving_without_new_stamps_latches_lost():
    mf = _started()
    _odom_q(mf, 10.0, 0.0)
    for _ in range(M.REPEAT_MAX - 1):                     # duplicates of the same stamp: tolerated a while
        _odom_q(mf, 10.0, 0.0)
    assert mf.lost is None
    _odom_q(mf, 10.0, 0.0)
    assert mf.lost and "repeating" in mf.lost
    mf.ticks = 0
    mf.tick(10.1, None)
    assert mf.cmd.state == "LOST"


def test_cuvslam_freezing_in_flight_lands_where_the_drone_really_is():
    m, sk, traj, states, w, mf = flight(odom_freeze_after_s=6.0)
    assert "VIO lost: odometry frozen" in m.land_reason, (m.land_reason, states[-5:])
    after = traj[mf.traj_t >= mf.link.t_start + 6.0]                          # truth from the freeze on
    assert np.linalg.norm(after - after[0], axis=1).max() < 0.25              # not flown blind
    assert w.clearance(traj, min_height=0.35).min() > mf.mppi_p.r_drone and not sk.armed
