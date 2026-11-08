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

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skybeta" / "blueraven_skybeta"))
from blueraven_skybeta import refnav_skybeta as S  # noqa: E402

from nav.course_map import CourseMap  # noqa: E402
from nav.frames import load_launch  # noqa: E402
from refnav import core as C  # noqa: E402
from refnav import depth as D  # noqa: E402
from refnav import map_route as M  # noqa: E402

DT = 0.05                                     # refnav's 20 Hz tick
DOC = json.loads((ROOT / "route.json").read_text())
LAUNCH_PARAMS = {"depth_mode": "blind", "pilot_arms": True, "wait_s": math.inf, "arm_wait_s": math.inf,
                 "land_timeout_s": math.inf}                                       # refnav.launch.py's


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _enu_to_ned(v: np.ndarray) -> np.ndarray:
    return np.array([v[1], v[0], -v[2]])


class Stack:
    """PX4 + the flight interface + VIO. PX4 NED <- odom ENU: position_ned = R_down(offset) enu_to_ned(p),
    as the interface assumes for poses; a twist is only enu_to_ned'd. The vision frame starts at `start`."""

    def __init__(self, offset: float, start_yaw_enu: float, detect_mps: float = 0.275) -> None:
        self.off = offset
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
        self.wz, self.t_twist = tw[3], t

    def step(self, t: float) -> None:
        flying = self.armed and self.nav == C.NAV_OFFBOARD
        cmd = self.cmd if (flying and t - self.t_twist < 1.0) else np.zeros(3)   # stale twist: hold
        self.vel += (cmd - self.vel) * min(1.0, DT / 0.3)
        if flying and t - self.t_twist < 1.0:
            self.yaw_enu += self.wz * DT
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
           state_gap: tuple[float, float] | None = None):
    prm, _ = C.params_from_doc(DOC, LAUNCH_PARAMS)
    launch = load_launch(ROOT / "maps" / "course.yaml", "launch_sim")
    cm = CourseMap.load(ROOT / "maps" / "course_sdf.npz")
    m = C.Mission(prm, M.builder(cm, launch, prm.height_m + launch["z"], M.DRONE_RADIUS_M + 0.20, 0.3))
    sk, v, pad, descent = Stack(offset, start_yaw_enu, detect_mps), C.Vehicle(), None, S.Descent()
    state, t_state, t_arm = None, -math.inf, None
    sent: list[tuple[float, str, tuple]] = []
    t = DT
    while t < limit_s and m.phase != "done":
        pad = S.on_odom(v, pad, m.phase, *sk.odom())
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
        tw = descent.twist(t, v, state) if out.phase == "land" else S.twist(out, v, pad, prm.yaw_rate_dps, offset)
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
    assert all(tw == (0.0, 0.0, -S.DESCEND_MPS, 0.0) for _, ph, tw in sent if ph == "land")   # detected at 0.35


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
        tw = descent.twist(t, v, sk.state()) if out.phase == "land" else S.twist(out, v, pad, 45.0, 0.0)
        if tw is not None:
            sk.twist(t, tw)
            sent.append(tw)
        sk.step(t)
        t += DT
    assert m.result.startswith("landed: armed in Offboard"), m.result
    assert not sk.armed and sent and all(tw[:2] == (0.0, 0.0) and tw[2] < 0 for tw in sent)


def test_the_descent_presses_down_on_the_ground_until_px4_disarms_and_never_holds():
    m, sk, sent = flight(detect_mps=0.5)                          # a detector that 0.35 m/s doesn't satisfy
    assert m.result == "complete", (m.result, m.why)
    assert not sk.armed and sk.landed
    land = [(t, tw) for t, ph, tw in sent if ph == "land"]
    pressed = [t for t, tw in land if tw[2] == -S.GROUND_PRESS_MPS]
    assert pressed and land[-1][0] - land[0][0] > S.GROUND_WAIT_S   # sat on the ground, then pressed, then disarm


def test_offboard_ending_during_the_descent_stops_the_twists():
    m, sk, sent = flight(pilot_takes_over_at=0.5, takeover_phase="land", limit_s=120.0)
    t_over = next(t for t, ph, _ in sent if ph == "takeover")
    assert m.phase == "land" and sk.armed and sk.p[2] < -0.1    # the pilot has it, in the air
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
    out = C.Output("follow", setpoint=(0.1, -0.2, -0.7, math.radians(90.0)), velocity=(0.4, 0.0), yawspeed=0.0,
                   stream=True)
    vx, vy, vz, wz = S.twist(out, v, S.Pad(0, 0, 0, 0), 45.0, 0.0)
    # pad at yaw 0: north = odom +x, east = odom -y
    assert (vx, vy, vz) == pytest.approx((0.5, 0.2, -0.1))
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
