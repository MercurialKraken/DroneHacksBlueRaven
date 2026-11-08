"""refnav's mission through a fake organizers' stack (blueraven_skybeta.refnav_skybeta): a drone in PX4's NED
answering twists like the flight interface (enu_to_ned, no yaw offset applied to twists), odometry in a vision
frame rotated from PX4's by the interface's yaw offset, /skybeta/state at 5 Hz and a pilot who arms then
switches to Position. The adapter glue is the same few calls refnav_node makes."""
from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skybeta" / "blueraven_skybeta"))
from blueraven_skybeta import refnav_skybeta as S  # noqa: E402

from refnav import core as C  # noqa: E402
from refnav import depth as D  # noqa: E402
from refnav import follow as F  # noqa: E402

DT = 0.05                                     # refnav's 20 Hz tick
ROUTES = ROOT / "courses" / "start-2026-11-07" / "routes"
DOC = json.loads((ROUTES / "route_v6_start2.json").read_text())        # Chelsea's file, unchanged
LAUNCH_PARAMS = {"depth_mode": "blind", "pilot_arms": True, "wait_s": math.inf, "arm_wait_s": math.inf,
                 "land_timeout_s": math.inf, "height_m": 0.7, "path_source": "waypoints",
                 "corner_cut_m": 0.05, "xtrack_slow_m": 0.08, "xtrack_stop_m": 0.20,
                 "look_s": 0.5, "look_max_m": 0.5}                                             # refnav.launch.py's


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _enu_to_ned(v: np.ndarray) -> np.ndarray:
    return np.array([v[1], v[0], -v[2]])


class Stack:
    """PX4 + the flight interface + VIO. PX4 NED <- odom ENU: position_ned = R_down(offset) enu_to_ned(p),
    as the interface assumes for poses; a twist is only enu_to_ned'd. The vision frame starts at `start`."""

    def __init__(self, offset: float, start_yaw_enu: float, detect_mps: float = 0.275, tau: float = 0.3,
                 delay: float = 0.0, skew: float = 0.0, push_body: tuple[float, float] = (0.0, 0.0),
                 climb_push: tuple[float, float] = (0.0, 0.0)) -> None:
        self.off, self.tau, self.delay = offset, tau, delay   # PX4's xy velocity response: lag tau after a dead time
        self.skew = skew        # the T4: velocity turns ~skew x the nose's angle off it, the other way (uneven drag)
        # the REAL drone (flight 160449): PX4 delivers the twist plus a push fixed to the airframe, (forward, left) m/s,
        # in the air; climb_push adds to it while climbing (the take-off drift)
        self.push_body, self.climb_push = push_body, climb_push
        self.queue: list[tuple[float, np.ndarray]] = []
        self.detect_mps = detect_mps                  # PX4's land detector: commanded descent it needs on the ground
        self.p = np.zeros(3)                          # PX4 NED
        self.vel = np.zeros(3)
        self.yaw_enu = start_yaw_enu                  # the drone's yaw in the vision frame
        self.armed, self.nav, self.landed = False, 4, True
        self.t_twist = -math.inf
        self.cmd = np.zeros(3)
        self.wz = 0.0
        self.t_down: float | None = None

    def odom(self) -> tuple:
        p_enu = _enu_to_ned(_rz(-self.off) @ self.p)              # enu_to_ned is its own inverse
        v_enu = _enu_to_ned(_rz(-self.off) @ self.vel)
        v_flu = _rz(self.yaw_enu).T @ v_enu
        h = self.yaw_enu / 2
        return tuple(p_enu), (0.0, 0.0, math.sin(h), math.cos(h)), tuple(v_flu)

    def state(self) -> dict:
        return {"px4_link": True, "armed": self.armed, "nav_state": self.nav, "landed": self.landed,
                "offboard_active": self.nav == C.NAV_OFFBOARD}

    def twist(self, t: float, tw: tuple[float, float, float, float]) -> None:
        v = np.array(tw[:3])
        n = np.linalg.norm(v)
        self.cmd = _enu_to_ned(v * min(1.0, 1.5 / n) if n > 0 else v)   # the interface's speed clamp
        self.queue.append((t, np.r_[self.cmd, tw[3]]))
        self.wz, self.t_twist = tw[3], t

    def step(self, t: float) -> None:
        flying = self.armed and self.nav == C.NAV_OFFBOARD
        cmd = self.cmd if (flying and t - self.t_twist < 1.0) else np.zeros(3)   # stale twist: hold
        while len(self.queue) > 1 and self.queue[1][0] <= t - self.delay:
            self.queue.pop(0)
        if self.delay and self.queue and flying:
            cmd = np.r_[self.queue[0][1][:2], cmd[2]]
        wz = self.queue[0][1][3] if self.delay and self.queue else self.wz
        if self.skew and np.hypot(*cmd[:2]) > 1e-3:
            nose = math.remainder(math.pi / 2 - self.yaw_enu + self.off - math.atan2(cmd[1], cmd[0]), math.tau)
            cmd = _rz(-self.skew * math.sin(2.0 * nose) / 2.0) @ cmd
        if flying and self.p[2] < -0.02:
            pf, pl = self.push_body
            if self.vel[2] < -0.1:                                # climbing (NED: up is negative)
                pf, pl = pf + self.climb_push[0], pl + self.climb_push[1]
            if pf or pl:
                c, s = math.cos(self.yaw_enu), math.sin(self.yaw_enu)
                push = _rz(self.off) @ _enu_to_ned(np.array([pf * c - pl * s, pf * s + pl * c, 0.0]))
                cmd = cmd + np.r_[push[:2], 0.0]
        self.vel += (cmd - self.vel) * min(1.0, DT / self.tau)
        if flying and t - self.t_twist < 1.0:
            self.yaw_enu += wz * DT
        self.p += self.vel * DT
        if self.p[2] >= 0.0:                                      # the floor
            self.p[2], self.vel[2] = 0.0, min(self.vel[2], 0.0)
            pressing = flying and t - self.t_twist < 1.0 and cmd[2] >= self.detect_mps
            self.t_down = (self.t_down or t) if pressing or not self.armed else None
            self.landed = self.t_down is not None
            if self.armed and self.landed and self.t_down is not None and t - self.t_down >= 2.0:
                self.armed = False                                # PX4's auto-disarm after touchdown
        else:
            self.landed, self.t_down = False, None


def flight(offset: float = 0.0, start_yaw_enu: float = 0.0, pilot_takes_over_at: float | None = None,
           limit_s: float = 200.0, detect_mps: float = 0.275, takeover_phase: str = "follow",
           carry: tuple[float, float, float] | None = None, pilot_waits_s: float = 0.0,
           state_gap: tuple[float, float] | None = None, odom_gap: tuple[float, float] | None = None,
           doc: dict = DOC, tau: float = 0.3, lag_s: float | None = None, wall: Callable | None = None,
           trail: list | None = None, delay: float = 0.0, skew: float = 0.0, prm_over: dict | None = None,
           push_body: tuple[float, float] = (0.0, 0.0), climb_push: tuple[float, float] = (0.0, 0.0)):
    prm, _ = C.params_from_doc(doc, {**LAUNCH_PARAMS, **(prm_over or {})})
    m = C.Mission(prm)
    sk = Stack(offset, start_yaw_enu, detect_mps, tau, delay, skew, push_body, climb_push)
    v, pad, descent = C.Vehicle(), None, S.Descent()
    track, integ, push = S.Track(), S.XYIntegral(), S.PushEstimator()
    state, t_state, t_odom, t_arm = None, -math.inf, -math.inf, None
    sent: list[tuple[float, str, tuple]] = []
    t = DT
    while t < limit_s and m.phase != "done":
        if odom_gap is None or not odom_gap[0] <= t < odom_gap[1]:
            pad, t_odom = S.on_odom(v, pad, m.phase, *sk.odom()), t
            m.on_health(t, S.healthy(state, t_state, t))
        gap = state_gap is not None and state_gap[0] <= t < state_gap[1]
        if round(t / DT) % 4 == 0 and not gap:
            state, t_state = sk.state(), t
            S.on_state(v, state, m.phase)
        out = m.step(t, v)
        if carry is not None and m.phase == "arm":                # carried (NED m) and turned (rad) after "wait -> arm"
            sk.p[:2] += carry[:2]
            sk.yaw_enu += carry[2]
            carry = None
        tw, land = S.command(out, v, pad, descent, track, state, t_odom, t_state, t, prm.yaw_rate_dps, offset,
                             tau if lag_s is None else lag_s, prm.speed_mps, integ, prm.ki_xy, push, prm.push_tau_s,
                             prm.push_max_mps, (prm.push_ff_fwd_mps, prm.push_ff_right_mps))
        if land:
            m.request_land(land)
        if tw is not None:
            sk.twist(t, tw)
            sent.append((t, out.phase, tw))
        if (m.phase == "arm" and not sk.armed and t_arm is None and sent and t - sent[-1][0] < 1.0
                and t - m.t_phase >= pilot_waits_s):
            sk.armed, sk.nav, t_arm = True, 1, t                  # the pilot arms in Altitude...
        if t_arm is not None and sk.nav == 1 and t - t_arm >= 0.5 and t - sk.t_twist < 1.0:
            sk.nav = C.NAV_OFFBOARD                               # ...switches to Position: Offboard, once
        if pilot_takes_over_at is not None and m.phase == takeover_phase and t - m.t_phase >= pilot_takes_over_at:
            sk.nav, pilot_takes_over_at = 2, None
            sent.append((t, "takeover", ()))
        sk.step(t)
        if wall is not None:
            wall(sk, t)
        if trail is not None and v.pos is not None:
            trail.append((t, m.phase, v.pos))
        if m.phase == "land" and sk.armed and sk.nav == C.NAV_OFFBOARD:
            assert sent[-1][0] == t, "an armed tick in Offboard without a twist: the interface would hold"
        t += DT
    return m, sk, sent


def _in_start_body_frame(sk: Stack, start_yaw_enu: float) -> np.ndarray:
    p_enu = sk.odom()[0]
    return (_rz(start_yaw_enu).T @ np.array(p_enu))[:2]          # (forward, left) from the take-off pose


@pytest.mark.parametrize("offset_deg, start_yaw_deg", [(0.0, 0.0), (-90.0, 0.0), (-90.0, 90.0), (0.0, 35.0)])
def test_the_full_route_is_flown_and_landed_whatever_the_frames(offset_deg, start_yaw_deg):
    off, yaw0 = math.radians(offset_deg), math.radians(start_yaw_deg)
    m, sk, sent = flight(off, yaw0)
    assert m.result == "complete", (m.result, m.why)
    assert m.follower.points_passed == len(DOC["route"])
    assert not sk.armed and sk.landed                             # PX4 landed and disarmed itself
    end = _in_start_body_frame(sk, yaw0)
    assert np.hypot(*(end - np.array(DOC["route"][-1]))) < 0.2, end
    phases = [ph for _, ph, _ in sent]
    assert {"arm", "climb", "follow", "land"} <= set(phases)
    assert max(abs(tw[3]) for _, _, tw in sent) <= math.radians(45.0) + 1e-9
    assert max(math.hypot(tw[0], tw[1]) for _, ph, tw in sent if ph == "follow") <= 1.5
    land = [tw for _, ph, tw in sent if ph == "land"]                 # brake to a stop, then down at 0.35 over the point
    brake = next(i for i, tw in enumerate(land) if tw[2] != 0.0)
    assert brake * DT <= S.BRAKE_MAX_S + DT and all(tw[2] == -S.DESCEND_MPS for tw in land[brake:])
    assert max(math.hypot(tw[0], tw[1]) for tw in land) <= S.LAND_HOLD_MPS + 1e-9


def _from_polyline(path: F.Path, corners: np.ndarray) -> np.ndarray:
    a, b = corners[:-1], corners[1:]
    pts = np.stack([path.x, path.y], 1)[:, None, :]
    u = np.clip(((pts - a) * (b - a)).sum(2) / ((b - a) ** 2).sum(1), 0.0, 1.0)[..., None]
    return np.linalg.norm(pts - (a + u * (b - a)), axis=2).min(1)


def test_waypoints_fly_within_the_corner_cut_and_turn_only_while_moving():
    m, _, sent = flight()
    corners = np.array([(0.0, 0.0)] + [(f, -l) for f, l in DOC["route"]])
    assert _from_polyline(m.path, corners).max() <= LAUNCH_PARAMS["corner_cut_m"] + 1e-3
    follow = [tw for _, ph, tw in sent if ph == "follow"]
    assert not [tw for tw in follow if abs(tw[3]) > math.radians(10.0) and math.hypot(tw[0], tw[1]) < 0.05]


@pytest.mark.parametrize("name", ["route_v6_start2", "route_v6_start2_sim", "route_v6_start2_stage1",
                                  "route_v6_start2_stage1_sim"])
def test_chelseas_route_files_load_unchanged(name):
    prm, unknown = C.params_from_doc(json.loads((ROUTES / f"{name}.json").read_text()), LAUNCH_PARAMS)
    assert prm.speed_mps == 0.3 and not unknown


@pytest.mark.parametrize("route, why", [([(1.0, 0.0), (0.0, 0.0)], "turn in place"),       # a reversal
                                        ([(0.0, 1.0), (1.0, 1.0)], "off the nose"),         # first leg 90 deg left
                                        ([(9.0, 0.0)], "legs up to")])
def test_waypoints_that_would_turn_in_place_or_are_typos_are_refused(route, why):
    with pytest.raises(ValueError, match=why):
        C.params_from_doc({"route": route}, LAUNCH_PARAMS)


def test_carrying_the_drone_after_arm_moves_the_route_with_it():
    yaw0 = math.radians(30.0)
    m, sk, sent = flight(carry=(2.0, -1.0, yaw0))
    assert m.result == "complete", (m.result, m.why)
    climb = [tw for _, ph, tw in sent if ph == "climb"]
    assert max(math.hypot(tw[0], tw[1]) for tw in climb[:10]) < 0.05          # no dash back to the old spot
    start = np.array([-1.0, 2.0, 0.0])                                         # NED (2, -1) is ENU (-1, 2)
    end = _rz(yaw0).T @ (np.array(sk.odom()[0]) - start)
    assert np.hypot(*(end[:2] - np.array(DOC["route"][-1]))) < 0.2, end


def test_the_mission_waits_for_the_pilot_as_long_as_he_takes_through_a_state_gap():
    m, sk, sent = flight(pilot_waits_s=400.0, state_gap=(200.0, 202.0), limit_s=700.0)
    assert m.result == "complete", (m.result, m.why)
    assert not sk.armed and sk.landed


def test_offboard_entered_before_the_mission_is_ready_descends_until_px4_disarms():
    prm, _ = C.params_from_doc(DOC, LAUNCH_PARAMS)
    m, sk, v, pad, descent = C.Mission(prm), Stack(0.0, 0.0), C.Vehicle(), None, S.Descent()
    sk.armed, sk.nav = True, C.NAV_OFFBOARD                       # a fresh arm twist, then a requalify to "wait"
    sk.t_twist = 0.0
    t, sent = DT, []
    while t < 30.0 and m.phase != "done":
        pad = S.on_odom(v, pad, m.phase, *sk.odom())
        m.on_health(t, S.healthy(sk.state(), t, t))
        S.on_state(v, sk.state(), m.phase)
        out = m.step(t, v)
        tw, _ = S.command(out, v, pad, descent, S.Track(), sk.state(), t, t, t, 45.0, 0.0)
        if tw is not None:
            sk.twist(t, tw)
            sent.append(tw)
        sk.step(t)
        t += DT
    assert m.result.startswith("landed: armed in Offboard"), m.result
    assert not sk.armed and sent and all(tw[:2] == (0.0, 0.0) and tw[2] < 0 for tw in sent)


@pytest.mark.parametrize("gap", ["odom", "state"])
def test_a_silent_pose_or_state_in_flight_stops_and_lands_at_once(gap):
    m, sk, sent = flight(**{f"{gap}_gap": (40.0, 1e9)})
    assert [ph for t, ph, _ in sent if abs(t - 40.0) < DT / 2] == ["follow"]
    late = 40.0 + (S.ODOM_STALE_S if gap == "odom" else S.STATE_STALE_S) + 0.2      # + one 5 Hz state period
    assert all(tw == (0.0, 0.0, 0.0, 0.0) for t, ph, tw in sent if ph != "land" and t > late)
    t_land = next(t for t, ph, _ in sent if ph == "land")
    assert 40.0 < t_land <= late + DT, t_land
    assert m.land_reason.startswith("landing requested"), m.land_reason


def test_a_short_odometry_gap_in_flight_holds_still_and_the_route_continues():
    m, sk, sent = flight(odom_gap=(40.0, 40.7))                   # longer than the 0.35 s delivery bunching seen on skybeta-09
    assert [tw for t, ph, tw in sent if 40.0 + S.HOLD_S + DT < t < 40.7] and all(
        tw == (0.0, 0.0, 0.0, 0.0) for t, ph, tw in sent if 40.0 + S.HOLD_S + DT < t < 40.7)
    assert (m.result, m.land_reason) == ("complete", "route complete")


def test_the_take_off_pose_is_logged_again_only_when_it_moves():
    pad = S.Pad(1.0, 2.0, 0.0, 0.3)
    assert S.moved(pad, None) and not S.moved(S.Pad(1.01, 2.0, 0.0, 0.3 + math.radians(0.5)), pad)
    assert S.moved(S.Pad(1.03, 2.0, 0.0, 0.3), pad) and S.moved(S.Pad(1.0, 2.0, 0.0, 0.3 + math.radians(2)), pad)


def test_a_landing_request_after_the_pilot_armed_in_offboard_descends():
    prm, _ = C.params_from_doc(DOC, LAUNCH_PARAMS)
    m = C.Mission(prm)
    m.phase, m.t_phase, m.t_started = "arm", 0.0, 0.0
    m.request_land("take-off pose off")
    m.step(1.0, C.Vehicle(pos=(0, 0, 0), heading=0.0, pos_ok=True, armed=True, nav_state=C.NAV_OFFBOARD))
    assert m.phase == "land"


def test_the_descent_presses_down_on_the_ground_until_px4_disarms_and_never_holds():
    m, sk, sent = flight(detect_mps=0.5)                          # a detector that 0.35 m/s doesn't satisfy
    assert m.result == "complete", (m.result, m.why)
    assert not sk.armed and sk.landed
    land = [(t, tw) for t, ph, tw in sent if ph == "land"]
    pressed = [t for t, tw in land if tw[2] == -S.GROUND_PRESS_MPS]
    assert pressed and land[-1][0] - land[0][0] > S.GROUND_WAIT_S   # sat on the ground, then pressed, then disarm


def test_a_descent_that_stalls_above_the_pad_height_presses_down():
    d, v, st = S.Descent(), C.Vehicle(pos=(0.0, 0.0, -0.15), heading=0.0), {"offboard_active": True, "armed": True}
    tws = [d.twist(t * DT, v, st) for t in range(int(6.0 / DT))]      # estimate stuck 0.15 m high, not sinking
    assert tws[0][2] == -S.DESCEND_MPS and tws[-1][2] == -S.GROUND_PRESS_MPS


def test_offboard_ending_during_the_descent_stops_the_twists():
    m, sk, sent = flight(pilot_takes_over_at=0.5, takeover_phase="land", limit_s=120.0)
    t_over = next(t for t, ph, _ in sent if ph == "takeover")
    assert m.result.startswith("yielded") and sk.armed and sk.p[2] < -0.1    # the pilot has it, in the air
    assert [t for t, ph, _ in sent if ph == "land" and t > t_over + 4 * DT] == []


def test_offboard_ending_stops_the_mission_and_its_setpoints_for_good():
    m, _, sent = flight(pilot_takes_over_at=5.0, limit_s=120.0)
    assert m.phase == "done" and m.result.startswith("yielded"), m.result
    assert sent[-1][0] < m.t_phase                                # nothing sent once it yielded...
    out = m.step(m.t_phase + 1.0, C.Vehicle(pos=(0, 0, -1), heading=0.0, pos_ok=True, armed=True,
                                            nav_state=C.NAV_OFFBOARD))
    assert out.phase == "done" and S.twist(out, C.Vehicle(pos=(0, 0, -1), heading=0.0), S.Pad(0, 0, 0, 0),
                                           45.0, 0.0) is None    # ...even if Offboard came back


def test_nothing_is_sent_before_the_mission_asks_for_control():
    prm, _ = C.params_from_doc(DOC, LAUNCH_PARAMS)
    m, sk, v, pad = C.Mission(prm), Stack(0.0, 0.0), C.Vehicle(), None
    S.on_state(v, sk.state(), m.phase)
    for k in range(20):                                           # no health yet: it waits
        pad = S.on_odom(v, pad, m.phase, *sk.odom())
        out = m.step(DT * (k + 1), v)
        assert m.phase == "wait" and S.twist(out, v, pad, 45.0, 0.0) is None


def test_odometry_becomes_the_pad_frame():
    v = C.Vehicle()
    a = math.radians(30.0)
    q = (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))
    pad = S.on_odom(v, None, "wait", (1.0, 2.0, 0.1), q, (0.0, 0.0, 0.0))
    assert v.pos == pytest.approx((0.0, 0.0, 0.0)) and v.heading == pytest.approx(0.0)
    # in flight the pad stays: 1 m along the nose, 0.5 m up, turned 20 deg left, flying forward at 1 m/s
    b = a + math.radians(20.0)
    q2 = (0.0, 0.0, math.sin(b / 2), math.cos(b / 2))
    assert S.on_odom(v, pad, "follow", (1.0 + math.cos(a), 2.0 + math.sin(a), 0.6), q2, (1.0, 0.0, 0.0)) is pad
    assert v.pos == pytest.approx((1.0, 0.0, -0.5))
    assert v.heading == pytest.approx(math.radians(-20.0))         # NED heading: left is negative
    assert v.vel == pytest.approx((math.cos(math.radians(20)), -math.sin(math.radians(20))))
    R = D.quat_to_mat(*v.q)                                        # PX4's attitude convention: FRD -> NED
    assert D.quat_up_cos(*v.q) == pytest.approx(1.0) and R[2, 2] == pytest.approx(1.0)
    assert D.quat_yaw(*v.q) == pytest.approx(v.heading)


def test_the_twist_is_feed_forward_plus_the_position_loop_and_yaw_rate_is_capped():
    v = C.Vehicle(pos=(0.0, 0.0, -0.8), heading=0.0)
    out = C.Output("follow", setpoint=(0.1, -0.2, -0.7, math.radians(90.0)), velocity=(0.3, 0.0), yawspeed=0.0,
                   stream=True)
    vx, vy, vz, wz = S.twist(out, v, S.Pad(0, 0, 0, 0), 45.0, 0.0)
    # pad at yaw 0: north = odom +x, east = odom -y
    assert (vx, vy, vz) == pytest.approx((0.3 + S.KP_XY * 0.1, S.KP_XY * 0.2, -0.1))
    assert wz == pytest.approx(-math.radians(45.0))                # turning right (NED +) is ENU -, capped


def test_the_land_phase_is_reported_to_the_mission_as_auto_land():
    v = C.Vehicle()
    S.on_state(v, {"armed": True, "nav_state": C.NAV_OFFBOARD}, "follow")
    assert v.nav_state == C.NAV_OFFBOARD
    S.on_state(v, {"armed": True, "nav_state": C.NAV_OFFBOARD}, "land")
    assert v.nav_state == C.NAV_AUTO_LAND
    S.on_state(v, {"armed": True, "nav_state": 2}, "land")        # the pilot took it: not hidden
    assert v.nav_state == 2


def test_no_px4_publisher_exists_in_the_drone_program():
    """Every /fmu endpoint of refnav's node is made in RefNav._io; the drone program replaces _io and _send."""
    tree = ast.parse((ROOT / "refnav" / "node.py").read_text(encoding="utf-8"))
    io = next(f for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "RefNav"
              for f in c.body if isinstance(f, ast.FunctionDef) and f.name == "_io")
    inside = {id(n) for n in ast.walk(io)}
    fmu = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
           and n.value.startswith("/fmu/")]
    assert fmu and all(id(n) in inside for n in fmu)
    node = ast.parse((ROOT / "skybeta" / "blueraven_skybeta" / "blueraven_skybeta" / "refnav_node.py").read_text())
    cls = next(c for c in node.body if isinstance(c, ast.ClassDef) and c.name == "SkybetaRefNav")
    assert {"_io", "_send"} <= {f.name for f in cls.body if isinstance(f, ast.FunctionDef)}


def test_the_launch_file_sets_what_these_tests_fly():
    text = (ROOT / "skybeta" / "blueraven_skybeta" / "launch" / "refnav.launch.py").read_text()
    assert all(f"'{k}'" in text for k in LAUNCH_PARAMS) and "default_value='route_v6_start2'" in text


TAPE = json.loads((ROOT / "courses" / "start-2026-11-07-tape" / "routes" / "route_v6_start2_sim.json").read_text())


def test_the_position_correction_and_the_total_twist_are_bounded():
    v = C.Vehicle(pos=(0.0, 0.0, -0.7), heading=0.0)                  # the setpoint 2 m away: an estimate jump
    out = C.Output("follow", setpoint=(2.0, 0.0, -0.7, 0.0), velocity=(0.0, 0.3), yawspeed=0.0, stream=True)
    vx, vy, _, _ = S.twist(out, v, S.Pad(0, 0, 0, 0), 45.0, 0.0)
    assert (vx, vy) == pytest.approx((S.CORR_MAX_MPS, -0.3))
    out.velocity, out.accel = (0.3, 0.0), (1.0, 0.0)
    assert math.hypot(*S.twist(out, v, S.Pad(0, 0, 0, 0), 45.0, 0.0)[:2]) == pytest.approx(S.XY_MAX_MPS)


@pytest.mark.parametrize("tau, delay, lag", [(0.55, 0.15, 0.55),      # the drone: MPC_XY_VEL_P_ACC 1.8
                                             (0.9, 0.6, 1.5)])         # the T4 sim, fitted to run t6r2's bag
def test_the_cp4_corner_is_tracked_within_10_cm_with_the_velocity_lag_led(tau, delay, lag):
    trail: list = []
    m, _, sent = flight(doc=TAPE, tau=tau, delay=delay, lag_s=lag, trail=trail, limit_s=300.0)
    assert m.result == "complete", (m.result, m.why)
    path = np.stack([m.path.x, m.path.y], 1)
    dev = [np.hypot(*(path - p[:2]).T).min() for _, ph, p in trail if ph == "follow"]
    assert max(dev) < 0.1, max(dev)
    corner = (TAPE["route"][16][0], -TAPE["route"][16][1])          # CP4 Gate B approach's corner, where run 4 overshot
    near = [d for d, (_, ph, p) in zip(dev, [x for x in trail if x[1] == "follow"]) if math.dist(p[:2], corner) < 0.5]
    assert near and max(near) < 0.1, max(near)
    assert max(math.hypot(tw[0], tw[1]) for _, ph, tw in sent if ph == "follow") <= S.XY_MAX_MPS + 1e-9


def test_without_the_lead_the_sims_lag_overshoots_the_cp4_corner():
    trail: list = []
    m, _, _ = flight(doc=TAPE, tau=0.9, delay=0.6, lag_s=0.0, trail=trail, limit_s=300.0)
    path = np.stack([m.path.x, m.path.y], 1)
    corner = (TAPE["route"][16][0], -TAPE["route"][16][1])
    near = [np.hypot(*(path - p[:2]).T).min() for _, ph, p in trail if ph == "follow" and math.dist(p[:2], corner) < 0.5]
    assert max(near) > 0.1


def test_far_from_the_setpoint_for_half_a_second_it_stops_and_lands_there():
    track, v = S.Track(), C.Vehicle(pos=(0.0, 0.0, -0.7), heading=0.0)
    out = C.Output("follow", setpoint=(S.TRACK_LAND_M + 0.05, 0.0, -0.7, 0.0), velocity=(0.3, 0.0), yawspeed=0.0, stream=True)
    descent = S.Descent()
    sends = [S.command(out, v, S.Pad(0, 0, 0, 0), descent, track, {"px4_link": True}, t, t, t, 45.0, 0.0)
             for t in np.arange(1.0, 1.0 + S.TRACK_LAND_S + 1e-9, DT)]
    assert all(why is None for _, why in sends[:-1])                  # a 0.13 s odometry glitch (T4 t6r3) never lands
    tw, why = sends[-1]
    assert tw == (0.0, 0.0, 0.0, 0.0) and why.startswith("tracking error")
    assert descent.target == (0.0, 0.0)                             # the descent holds where it is, not the setpoint


def test_a_jammed_drone_stops_pushing_and_lands_where_it_is():
    jam: dict = {}

    def wall(sk: Stack, t: float) -> None:                            # from 30 s on it is stuck on something
        if t >= 30.0:
            jam.setdefault("p", sk.p[:2].copy())
            sk.p[:2], sk.vel[:2] = jam["p"], 0.0

    trail: list = []
    m, sk, sent = flight(doc=TAPE, tau=0.9, delay=0.6, lag_s=1.5, wall=wall, trail=trail, limit_s=120.0)
    assert m.land_reason.startswith("landing requested: stuck"), m.land_reason
    t_land = next(t for t, ph, _ in sent if ph == "land")
    assert 30.0 < t_land <= 30.0 + S.STUCK_S + 0.5, t_land
    assert max(math.hypot(tw[0], tw[1]) for _, _, tw in sent if tw) <= S.XY_MAX_MPS + 1e-9
    assert all(math.hypot(tw[0], tw[1]) < 0.02 for t, ph, tw in sent if ph == "land")    # no push into the jam


def test_a_drone_arriving_at_0_3_mps_lands_within_10_cm_of_the_last_point():
    sk = Stack(0.0, 0.0, tau=0.9, delay=0.6)                           # the T4 sim's soft velocity loop
    sk.armed, sk.nav, sk.landed, sk.t_twist = True, C.NAV_OFFBOARD, False, 0.0
    v = C.Vehicle()
    pad = S.on_odom(v, None, "wait", *sk.odom())
    sk.p[2] = -0.7
    vx, vy = pad.to_odom(0.3, 0.0)                                     # 0.3 m/s along the pad's north, as NED
    sk.vel[:] = (vy, vx, 0.0)
    target = (0.15, 0.0)
    out = C.Output("land", setpoint=(*target, -0.7, 0.0))
    descent, t = S.Descent(), DT
    while sk.armed and t < 30.0:
        pad = S.on_odom(v, pad, "land", *sk.odom())
        tw, _ = S.command(out, v, pad, descent, S.Track(), sk.state(), t, t, t, 45.0, 0.0)
        sk.twist(t, tw)
        sk.step(t)
        t += DT
    pad = S.on_odom(v, pad, "land", *sk.odom())
    assert not sk.armed and math.dist(v.pos[:2], target) < 0.1, v.pos


@pytest.mark.parametrize("tau, delay, lag, limit", [(0.9, 0.6, 1.5, 0.10), (0.55, 0.15, 0.55, 0.07)])
def test_the_window_route_tracks_with_the_t4s_nose_dependent_velocity_skew(tau, delay, lag, limit):
    """T4 t7r1: with the nose 10-20 deg right of the track the velocity ran 11-23 deg left of its setpoint (PX4
    heading = truth heading, estimate = truth: uneven drag, not a frame error). A nose kept near the track halves it."""
    doc = json.loads((ROUTES.parent.parent / "start-2026-11-07-tape" / "routes" / "route_v7_window_sim.json").read_text())
    trail: list = []
    m, _, _ = flight(doc=doc, tau=tau, delay=delay, lag_s=lag, skew=0.8, trail=trail, limit_s=300.0)
    assert m.result == "complete", (m.result, m.why)
    path = np.stack([m.path.x, m.path.y], 1)
    assert max(np.hypot(*(path - p[:2]).T).min() for _, ph, p in trail if ph == "follow") < limit


# ---- the take-off hold as a gate + the position integral (Chelsea, 2026-11-07; real flight 160449) ----------------
R1_DOC = json.loads((ROUTES.parent.parent / "start-2026-11-07-tape" / "routes" / "route_v7_cp2_simple_sim.json").read_text())
REAL_PUSH = (-0.056, 0.100)       # 160449 follow phase: PX4 delivered this much LESS than commanded, body (forward, left) m/s
REAL_CLIMB = (-0.30, 0.06)        # + this while climbing: reproduces 160449's ~0.4 m back-left take-off drift at 0.7 m
GATED = {"height_m": 0.4, "speed_mps": 0.10, "hover_s": 3.0, "hover_tol_m": 0.05, "hover_dwell_s": 1.0,
         "hover_max_s": 40.0, "hover_timeout": "land", "ki_xy": 0.08}


def _gated_flight(push=REAL_PUSH, climb=REAL_CLIMB, **over):
    trail: list = []
    m, sk, sent = flight(doc=R1_DOC, prm_over={**GATED, **over}, tau=0.55, lag_s=0.55, push_body=push,
                         climb_push=climb, trail=trail, limit_s=250.0)
    start = next(((t, pos) for t, ph, pos in trail if ph == "follow"), None)
    path = np.c_[m.path.x, m.path.y]
    off = [float(np.min(np.hypot(path[:, 0] - p[0], path[:, 1] - p[1]))) for t, ph, p in trail if ph == "follow"]
    return m, start, off


def test_the_real_push_without_the_integral_reproduces_160449s_steady_offset():
    m, start, off = _gated_flight(ki_xy=0.0, hover_tol_m=0.0, hover_s=10.0)     # route1-chelsea before this change
    assert start is not None and math.hypot(*start[1][:2]) > 0.15                # the hold did NOT get it back...
    assert 0.18 < float(np.median(off)) < 0.26                                  # ...and it flies ~0.22 m off (real: 0.222)


def test_the_gated_hold_with_the_integral_flies_back_over_the_take_off_point_against_the_real_push():
    m, start, off = _gated_flight()
    assert m.result == "complete"
    assert math.hypot(*start[1][:2]) <= 0.05                                    # the route starts at the take-off point
    assert float(np.percentile(off, 95)) < 0.04                                 # and the push no longer offsets it


def test_without_the_integral_the_gate_never_opens_against_a_steady_push_and_lands():
    m, start, off = _gated_flight(ki_xy=0.0)
    assert start is None and m.result.startswith("landed: not back within 0.05 m of the take-off point")


@pytest.mark.parametrize("deg", [0, 90, 180, 270])
@pytest.mark.parametrize("mag", [0.05, 0.15])
def test_a_push_of_any_direction_and_size_is_flown_back_before_the_route(deg, mag):
    a = math.radians(deg)
    m, start, off = _gated_flight(push=(mag * math.cos(a), mag * math.sin(a)),
                                  climb=(0.3 * math.cos(a + 0.7), 0.3 * math.sin(a + 0.7)))
    assert m.result == "complete" and math.hypot(*start[1][:2]) <= 0.05 and float(np.percentile(off, 95)) < 0.05


def test_with_no_push_the_gate_and_integral_change_nothing():
    m, start, off = _gated_flight(push=(0.0, 0.0), climb=(0.0, 0.0))
    assert m.result == "complete" and math.hypot(*start[1][:2]) < 0.01 and float(np.percentile(off, 95)) < 0.02


def test_the_hold_timeout_can_start_the_route_instead_of_landing():
    m, start, off = _gated_flight(ki_xy=0.0, hover_max_s=8.0, hover_timeout="start")
    assert start is not None and not m.result.startswith("landed: not back within")     # it flew the route from there


@pytest.mark.parametrize("bad", [{"hover_timeout": "hover"}, {"hover_tol_m": 0.05, "hover_s": 10.0, "hover_max_s": 5.0},
                                 {"ki_xy": -0.1}])
def test_bad_hold_parameters_are_refused(bad):
    with pytest.raises(ValueError):
        C.params_from_doc(R1_DOC, {**LAUNCH_PARAMS, **GATED, **bad})


def test_the_integral_is_body_frame_bounded_and_reset_outside_the_hold_and_follow():
    i = S.XYIntegral()
    for k in range(400):                                   # a 1 m error held for 20 s: the bound, not 1.6 m/s
        n, e = i.step(k * DT, 1.0, 0.0, math.pi / 2, 0.08)
    assert math.hypot(n, e) == pytest.approx(S.KI_MAX_MPS) and n > 0      # north error -> north push, any heading
    n2, e2 = i.step(400 * DT, 0.0, 0.0, 0.0, 0.08)                          # the drone turned: the learned push turns too
    assert (n2, e2) == pytest.approx((0.0, -S.KI_MAX_MPS), abs=1e-9)
    j = S.XYIntegral()
    for k in range(100):                                   # along the path (the reference's lead) is never integrated
        j.step(k * DT, 0.5, 0.0, 0.0, 0.08, tangent=(1.0, 0.0))
    assert (j.f, j.r) == (0.0, 0.0)


# ---- the push estimator (velocity shortfall, push_tau_s): replaces ki_xy, which orbited on the T4 (t9r5) -------------
PUSH_HOLD = {**GATED, "ki_xy": 0.0, "hover_s": 10.0, "push_tau_s": 3.0}


def _push_flight(push=REAL_PUSH, climb=REAL_CLIMB, tau=0.55, delay=0.0, lag=0.55, skew=0.0, offset=0.0, yaw0=0.0, **over):
    trail: list = []
    m, sk, sent = flight(offset=offset, start_yaw_enu=yaw0, doc=R1_DOC, prm_over={**PUSH_HOLD, **over}, tau=tau,
                         delay=delay, skew=skew, lag_s=lag, push_body=push, climb_push=climb, trail=trail, limit_s=260.0)
    hold = [(t, math.hypot(p[0], p[1])) for t, ph, p in trail if ph == "hover"]
    start = next(((t, p) for t, ph, p in trail if ph == "follow"), None)
    path = np.c_[m.path.x, m.path.y]
    off = [float(np.min(np.hypot(path[:, 0] - p[0], path[:, 1] - p[1]))) for t, ph, p in trail if ph == "follow"]
    return m, start, off, hold


@pytest.mark.parametrize("offset,yaw0", [(0.0, 0.0), (0.0, math.pi / 2), (-math.pi / 2, 0.0)])
def test_the_push_estimator_returns_to_the_take_off_point_against_the_real_push_on_the_drone(offset, yaw0):
    m, start, off, hold = _push_flight(offset=offset, yaw0=yaw0)       # drone 0.55 s; the T4's -90 deg; a 90 deg take-off
    assert m.result == "complete" and math.hypot(*start[1][:2]) <= 0.05
    assert float(np.percentile(off, 95)) < 0.03                          # 160449 flew 0.22 m off with P only


@pytest.mark.parametrize("tau,delay,skew", [(0.9, 0.6, 0.0), (1.5, 0.6, 0.0), (1.5, 0.6, 0.75)])
@pytest.mark.parametrize("push", [REAL_PUSH, (0.0, 0.0)])
def test_on_a_t4_like_slow_response_the_hold_never_orbits_and_the_route_completes(tau, delay, skew, push):
    climb = REAL_CLIMB if push != (0.0, 0.0) else (0.0, 0.0)
    m, start, off, hold = _push_flight(push=push, climb=climb, tau=tau, delay=delay, lag=1.5, skew=skew, offset=-math.pi / 2)
    assert m.result == "complete" and math.hypot(*start[1][:2]) <= 0.05
    d = [x for t, x in hold]
    assert max(d[len(d) // 2:]) <= max(d[: len(d) // 2]) + 1e-9         # the second half of the hold never exceeds the first
    if push == (0.0, 0.0):                                               # the T4 has no push: a 10 s hold, on the line
        assert hold[-1][0] - hold[0][0] < 10.5 and float(np.percentile(off, 95)) < 0.04


def test_the_push_estimator_learns_a_body_push_turns_with_the_drone_and_resets():
    p = S.PushEstimator()
    for k in range(200):                     # commanded 0, achieved 0.1 m/s north, heading north: a 0.1 m/s forward push
        b = p.step(k * DT, (0.1, 0.0), 0.0, 0.55, 3.0, 0.15)
        p.sent = (0.0, 0.0)
    assert b == pytest.approx((0.1, 0.0), abs=0.01)
    assert p.step(200 * DT, (0.0, 0.1), math.pi / 2, 0.55, 3.0, 0.15) == pytest.approx((0.0, 0.1), abs=0.01)   # turned
    for k in range(200):
        b = p.step(k * DT + 20, (1.0, 0.0), 0.0, 0.55, 3.0, 0.15)
        p.sent = (0.0, 0.0)
    assert math.hypot(*b) == pytest.approx(0.15)                         # bounded
    p.reset()
    assert (p.f, p.r, p.exp) == (0.0, 0.0, None)


def test_leaning_against_the_push_is_not_counted_as_a_jam_by_the_stuck_guard():
    # slow T4-like response + the real push, fixed 10 s hold: the route starts off the point while the estimate is still
    # learning; the drone leans against the push while the reference waits. Before: 'stuck: moved 0.04 m in 3 s while
    # commanded 0.66 m' -> landed. The lean is not motion asked for.
    m, start, off, hold = _push_flight(tau=1.5, delay=0.6, lag=1.5, offset=-math.pi / 2, hover_tol_m=0.0)
    assert not str(m.result).startswith("landed: landing requested: stuck")
    assert m.result == "complete"


def test_with_no_push_the_estimator_never_stops_a_clean_route():
    m, start, off, hold = _push_flight(push=(0.0, 0.0), climb=(0.0, 0.0), hover_tol_m=0.0)
    assert m.result == "complete"            # (real jams: test_a_jammed_drone_stops_pushing_and_lands_where_it_is)


# ---- the constant push feed-forward (push_ff_fwd_mps, push_ff_right_mps): 160449's push, cancelled from lift-off --------
FF = {"push_ff_fwd_mps": 0.056, "push_ff_right_mps": 0.100}
FF_FLIGHT = {"height_m": 0.4, "speed_mps": 0.15, "hover_s": 0.0}


def _ff_twist(phase: str, heading: float, ff: tuple[float, float]) -> np.ndarray:
    v = C.Vehicle(pos=(0.0, 0.0, -0.4), heading=heading)
    out = C.Output(phase, setpoint=(0.1, -0.2, -0.4, heading), velocity=(0.1, 0.05), stream=True)
    return np.array(S.twist(out, v, S.Pad(0, 0, 0, 0), 30.0, 0.0, push_ff=ff))


@pytest.mark.parametrize("heading, want_ne", [(0.0, (0.056, 0.100)), (math.pi / 2, (-0.100, 0.056))])
@pytest.mark.parametrize("phase", ["climb", "hover", "follow"])
def test_the_push_ff_adds_the_body_vector_turned_to_pad_ned_in_the_air(phase, heading, want_ne):
    d = _ff_twist(phase, heading, (0.056, 0.100)) - _ff_twist(phase, heading, (0.0, 0.0))
    assert (d[0], -d[1]) == pytest.approx(want_ne) and d[2:] == pytest.approx((0.0, 0.0))   # pad yaw 0: east = -odom y


def test_the_push_ff_is_off_by_default_and_on_the_ground():
    assert C.Params().push_ff_fwd_mps == 0.0 and C.Params().push_ff_right_mps == 0.0
    v = C.Vehicle(pos=(0.0, 0.0, 0.0), heading=0.0)
    out = C.Output("arm", setpoint=(0.0, 0.0, 0.0, 0.0), stream=True)
    assert S.twist(out, v, S.Pad(0, 0, 0, 0), 30.0, 0.0, push_ff=(0.1, 0.1)) == S.twist(out, v, S.Pad(0, 0, 0, 0), 30.0, 0.0)


@pytest.mark.parametrize("bad", [{"push_ff_fwd_mps": 0.25}, {"push_ff_right_mps": -0.21}])
def test_a_push_ff_over_0_2_mps_is_refused(bad):
    with pytest.raises(ValueError):
        C.params_from_doc(R1_DOC, {**LAUNCH_PARAMS, **bad})


def _ff_flight(ff: dict, offset: float = 0.0, yaw0: float = 0.0):
    trail: list = []
    m, sk, sent = flight(offset=offset, start_yaw_enu=yaw0, doc=R1_DOC, prm_over={**FF_FLIGHT, **ff}, tau=0.55,
                         lag_s=0.55, push_body=REAL_PUSH, trail=trail, limit_s=200.0)
    climb_end = next(p for t, ph, p in trail if ph not in ("wait", "arm", "climb"))
    path = np.c_[m.path.x, m.path.y]
    off = [float(np.min(np.hypot(path[:, 0] - p[0], path[:, 1] - p[1]))) for t, ph, p in trail if ph == "follow"]
    return m, math.hypot(*climb_end[:2]), float(np.percentile(off, 95))


@pytest.mark.parametrize("offset,yaw0", [(0.0, 0.0), (-math.pi / 2, math.pi / 2)])
def test_the_push_ff_cancels_160449s_push_from_lift_off(offset, yaw0):
    m, drift, xtrack = _ff_flight(FF, offset, yaw0)
    assert m.result == "complete" and drift < 0.05 and xtrack < 0.05, (m.result, drift, xtrack)
    m0, drift0, xtrack0 = _ff_flight({}, offset, yaw0)
    assert xtrack0 > 0.2, xtrack0                                        # P only: 160449's steady offset


def _ff_command(phase: str, mppi: bool, track: S.Track | None = None, t: float = 0.0, ff_follow: bool = True):
    v = C.Vehicle(pos=(0.0, 0.0, -0.4), heading=0.0)
    out = C.Output(phase, setpoint=(0.0, 0.0, -0.4, 0.0), velocity=(0.0, 0.0), stream=True)
    state = {"px4_link": True, "armed": True, "offboard_active": True}
    return S.command(out, v, S.Pad(0, 0, 0, 0), S.Descent(), track or S.Track(), state, t, t, t, 30.0, 0.0,
                     push_ff=(0.2, 0.2), mppi=mppi, push_ff_follow=ff_follow)


def test_leaning_on_the_push_ff_is_not_counted_as_a_jam_by_the_stuck_guard():
    track = S.Track()                    # the reference waits over a drone held still by its push
    for k in range(100):
        tw, why = _ff_command("follow", False, track, k * DT)
        assert why is None and math.hypot(tw[0], tw[1]) == pytest.approx(math.hypot(0.2, 0.2))


@pytest.mark.parametrize("phase, on", [("climb", True), ("hover", True), ("follow", False)])
def test_under_mppi_the_push_ff_leans_in_the_climb_and_hold_but_not_in_follow(phase, on):
    tw, _ = _ff_command(phase, True)
    assert math.hypot(tw[0], tw[1]) == pytest.approx(math.hypot(0.2, 0.2) if on else 0.0)


@pytest.mark.parametrize("phase, on", [("climb", True), ("hover", True), ("follow", False)])
def test_with_push_ff_follow_off_the_push_ff_leans_in_the_climb_and_hold_but_not_in_follow(phase, on):
    assert C.Params().push_ff_follow is True
    tw, _ = _ff_command(phase, False, ff_follow=False)
    assert math.hypot(tw[0], tw[1]) == pytest.approx(math.hypot(0.2, 0.2) if on else 0.0)
