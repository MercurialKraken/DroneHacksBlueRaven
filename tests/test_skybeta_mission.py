"""blueraven_skybeta.mission_core against a fake skybeta stack: a kinematic drone that answers pose
setpoints like PX4's position loop (position-only, 0.95/s gain) and twists directly, a 5 Hz
/skybeta/state and 30 Hz odometry. These are Task B's acceptance checks (arnav-claude's list)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skybeta" / "blueraven_skybeta"))
from blueraven_skybeta.mission_core import MissionParams, SkybetaMission  # noqa: E402
from nav.frames import pose_xyz_yaw  # noqa: E402  (mission_core put the repo on sys.path)

DT = 0.05                                    # the node's 20 Hz tick
LAUNCH = {"x": 1.0, "y": 2.0, "yaw_deg": 90.0, "z": 0.07}   # mark at (1, 2), facing course north
ROUTE = [(1.0, 2.0), (1.0, 3.5), (2.5, 3.5)]                 # north 1.5 m, then east 1.5 m (map)


class Sim:
    """The drone in skybeta odom (startup ENU). The stack started at `origin` (map pose) and was
    then carried to the launch mark, so odom and map differ by a translation and a rotation."""

    def __init__(self, start_odom=(0.0, 0.0, 0.0, 0.0)):
        self.x, self.y, self.z, self.yaw = start_odom
        self.ground = start_odom[2]
        self.armed = self.offboard = False
        self.landed = True
        self.t_landed: float | None = 0.0
        self.published: list = []

    def state(self):
        return {"px4_link": True, "armed": self.armed, "landed": self.landed,
                "offboard_active": self.offboard, "clamped": [], "frozen": False}

    def apply(self, cmd, t):
        self.published.append((t, cmd.pose, cmd.twist))
        if not (self.offboard and self.armed):
            return
        if cmd.pose is not None:                                   # PX4 position loop, no feed-forward
            tx, ty, tz = cmd.pose
            tz = max(tz, self.ground + 0.3)                        # the interface's floor
            self.x += 0.95 * (tx - self.x) * DT
            self.y += 0.95 * (ty - self.y) * DT
            self.z += 0.95 * (tz - self.z) * DT
        elif cmd.twist is not None:
            vx, vy, vz, wz = cmd.twist
            n = math.sqrt(vx * vx + vy * vy + vz * vz)
            if n > 1.5:
                vx, vy, vz = (c * 1.5 / n for c in (vx, vy, vz))
            self.x += vx * DT
            self.y += vy * DT
            self.z = max(self.ground, self.z + vz * DT)
            self.yaw += wz * DT
        if self.z <= self.ground + 1e-6:
            if not self.landed:
                self.landed, self.t_landed = True, t
            if self.t_landed is not None and t - self.t_landed >= 2.0:
                self.armed = self.offboard = False                # PX4 auto-disarm after touchdown
        else:
            self.landed, self.t_landed = False, None


def run(m: SkybetaMission, sim: Sim, t0: float, t1: float, state=True, odom=True, hook=None):
    t = t0
    cmds = []
    while t < t1 - 1e-9:
        if hook:
            hook(t, sim)
        if odom:
            m.on_odom(t, sim.x, sim.y, sim.z, sim.yaw)
        if state and round(t / DT) % 4 == 0:                       # 5 Hz state
            m.on_state(t, sim.state())
        cmd = m.step(t)
        sim.apply(cmd, t)
        cmds.append((t, cmd))
        t += DT
    return t, cmds


def mission(**kw):
    kw.setdefault("mission", "course")
    return SkybetaMission(MissionParams(**kw), LAUNCH, ROUTE)


def to_ready(m, sim):
    t, cmds = run(m, sim, 0.0, 2.5)                                # settle on the pad
    assert m.phase == "WAIT_OFFBOARD"
    return t


def switch_to_offboard(t, sim):
    sim.armed = sim.offboard = True


def test_init_publishes_nothing_until_the_drone_sits_still():
    m, sim = mission(), Sim()
    _, cmds = run(m, sim, 0.0, 1.5)
    assert all(c.pose is None and c.twist is None for _, c in cmds) and m.phase == "INIT"
    m2, sim2 = mission(), Sim()
    _, cmds = run(m2, sim2, 0.0, 4.0, hook=lambda t, s: setattr(s, "x", 0.05 * math.sin(7 * t)))
    assert m2.phase == "INIT"                                      # carried around: never ready


def test_waits_streaming_holds_and_never_moves_before_offboard():
    m, sim = mission(), Sim()
    t = to_ready(m, sim)
    sim.armed = True                                               # pilot armed, not yet Position
    _, cmds = run(m, sim, t, t + 5.0)
    assert m.phase == "WAIT_OFFBOARD"
    poses = [c.pose for _, c in cmds]
    assert all(p is not None for p in poses)                       # every tick: far more than 1 Hz
    assert all(np.allclose(p, (sim.x, sim.y, sim.z)) for p in poses) and all(c.twist is None for _, c in cmds)


def test_full_flight_follows_the_route_lands_and_goes_silent():
    m, sim = mission(), Sim()
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    t, cmds = run(m, sim, t, t + 60.0)
    assert m.phase == "DONE", m.why
    phases = [c.phase for _, c in cmds]
    for ph in ("TAKEOFF", "FOLLOW", "DESCEND"):
        assert ph in phases
    twists = [c.twist for _, c in cmds if c.twist is not None]
    assert max(math.sqrt(vx * vx + vy * vy + vz * vz) for vx, vy, vz, _ in twists) <= 1.5 + 1e-9
    end_map = (m.T_map_odom @ [sim.x, sim.y, sim.z, 1])[:2]
    assert np.hypot(*(end_map - np.array(ROUTE[-1]))) < 0.15       # landed at the end of the route
    after = [c for _, c in cmds if c.phase == "DONE"]
    assert after and all(c.pose is None and c.twist is None for c in after)
    # cruise is reached: the follow phase is not a crawl (pose-only would be ~0.1 m/s)
    speeds = [math.hypot(c.twist[0], c.twist[1]) for _, c in cmds if c.phase == "FOLLOW" and c.twist]
    assert np.percentile(speeds, 75) > 0.2


def test_offboard_drop_aborts_for_good_and_goes_silent():
    m, sim = mission(), Sim()
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    t, _ = run(m, sim, t, t + 6.0)
    assert m.phase in ("FOLLOW", "TAKEOFF")
    sim.offboard = False                                           # the pilot takes over
    t, cmds = run(m, sim, t, t + 1.0)
    assert m.phase == "ABORTED"
    assert sum(c.call_hold for _, c in cmds) == 1
    sim.offboard = True                                            # even if Offboard came back...
    _, cmds = run(m, sim, t, t + 3.0)
    assert all(c.pose is None and c.twist is None for _, c in cmds)   # ...never publishes again


@pytest.mark.parametrize("lose", ["state", "odom"])
def test_stale_inputs_abort_with_a_hold(lose):
    m, sim = mission(), Sim()
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    t, _ = run(m, sim, t, t + 6.0)
    t, cmds = run(m, sim, t, t + 1.0, state=(lose != "state"), odom=(lose != "odom"))
    assert m.phase == "ABORTED" and any(c.call_hold for _, c in cmds)
    assert ("stale" in m.why)


def test_stale_state_before_the_switch_stops_streaming_without_aborting():
    m, sim = mission(), Sim()
    t = to_ready(m, sim)
    t, cmds = run(m, sim, t, t + 1.0, state=False)
    assert m.phase == "WAIT_OFFBOARD" and cmds[-1][1].pose is None  # no handover while blind
    _, cmds = run(m, sim, t, t + 1.0)
    assert cmds[-1][1].pose is not None                             # streams again once fresh


def test_descent_continues_until_landed_and_disarmed():
    m, sim = mission(), Sim()
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    while m.phase != "DESCEND":
        t, _ = run(m, sim, t, t + DT)
    seen_landed_armed = False
    while m.phase == "DESCEND":
        t, cmds = run(m, sim, t, t + DT)
        c = cmds[-1][1]
        if m.phase == "DESCEND":
            assert c.twist is not None and c.twist[2] < 0           # refreshed every tick (twists expire)
            seen_landed_armed |= sim.landed and sim.armed
    assert seen_landed_armed and m.phase == "DONE"


def test_stack_started_elsewhere_still_flies_the_route_in_course_coordinates():
    m = mission()
    sim = Sim(start_odom=(0.8, -1.1, 0.0, math.radians(-35.0)))    # odom origin not at the mark
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    run(m, sim, t, t + 60.0)
    assert m.phase == "DONE"
    end_map = (m.T_map_odom @ [sim.x, sim.y, sim.z, 1])[:2]
    assert np.hypot(*(end_map - np.array(ROUTE[-1]))) < 0.15


def test_sharp_corners_stop_and_turn_in_place_then_carry_on():
    m, sim = mission(max_radius_m=0.0), Sim()                       # every corner sharp
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    t, cmds = run(m, sim, t, t + 60.0)
    assert m.phase == "DONE", m.why
    # at the corner the commanded speed drops to ~0 while the yaw rate turns the nose
    near = [c for _, c in cmds if c.phase == "FOLLOW" and c.twist
            and abs(c.twist[3]) > math.radians(20) and math.hypot(c.twist[0], c.twist[1]) < 0.08]
    assert near, "never stopped to turn at the sharp corner"
    end_map = (m.T_map_odom @ [sim.x, sim.y, sim.z, 1])[:2]
    assert np.hypot(*(end_map - np.array(ROUTE[-1]))) < 0.15


# ---------------------------------------------------------------- map_align's map <- odom (drift)
def drift(t: float, t0: float) -> np.ndarray:
    """odom's drift from the truth since t0: 3 cm/s sideways creep and 0.6 deg/s of yaw, about the
    odom origin. truth map pose = T_init . D(t) . odom pose. Twists act in odom, as PX4's do (its
    estimator runs on the same drifting VIO)."""
    a = max(0.0, t - t0)
    return pose_xyz_yaw(0.0, -0.03 * a, 0.0, math.radians(0.6 * a))


def fly_with_drift(correct: bool, stop_corrections_at: float | None = None):
    m, sim = mission(), Sim()
    t = to_ready(m, sim)
    T_init = m.T_map_odom.copy()
    t0 = t
    switch_to_offboard(t, sim)

    truth = {}

    def hook(tt, s):
        if correct and round(tt / DT) % 4 == 0 and (stop_corrections_at is None or tt - t0 < stop_corrections_at):
            assert m.on_map_from_odom(tt, T_init @ drift(tt, t0)) is None    # map_align at 5 Hz
        if not s.landed:                    # the model's drift moves a flying drone; measure at touchdown
            truth["xy"] = (T_init @ drift(tt, t0) @ [s.x, s.y, s.z, 1])[:2]
    t, _ = run(m, sim, t, t + 60.0, hook=hook)
    return m, float(np.hypot(*(truth["xy"] - np.array(ROUTE[-1])))), t


def test_init_hands_map_from_odom_to_map_align_exactly_once():
    m, sim = mission(), Sim(start_odom=(0.8, -1.1, 0.0, math.radians(-35.0)))
    _, cmds = run(m, sim, 0.0, 4.0)
    sent = [c.map_from_odom_init for _, c in cmds if c.map_from_odom_init is not None]
    assert len(sent) == 1 and np.allclose(sent[0], m.T_map_odom)
    assert np.allclose(sent[0] @ [0.8, -1.1, 0.0, 1], [LAUNCH["x"], LAUNCH["y"], LAUNCH["z"], 1])


def test_corrections_before_init_are_ignored():
    m, sim = mission(), Sim()
    run(m, sim, 0.0, 0.5)
    assert m.on_map_from_odom(0.5, np.eye(4)) is not None and m.T_map_odom is None


def test_drift_corrections_keep_the_flight_on_the_course():
    m, err_corrected, t = fly_with_drift(correct=True)
    assert m.phase == "DONE", m.why
    assert err_corrected < 0.15
    _, err_raw, _ = fly_with_drift(correct=False)
    assert err_raw > 0.3, "the drift model is too weak to show anything"
    st = m.status(t)
    assert st["map_from_odom"] == "live" and st["corr_n"] > 20 and st["corr_step_max_m"] < 0.05


def test_stale_corrections_keep_the_last_transform_not_init():
    m, _, t = fly_with_drift(correct=True, stop_corrections_at=6.0)
    T_last = m.T_map_odom.copy()
    assert not np.allclose(T_last, m.T_init)                    # still the last correction, not INIT's
    st = m.status(t)
    assert st["map_from_odom"] == "stale" and st["corr_age_s"] > 1.0 and st["corr_from_init_m"] > 0.05


def test_corrections_never_change_the_flight_height():
    m, sim = mission(), Sim()
    to_ready(m, sim)
    z0 = m.T_map_odom[2, 3]
    T = m.T_map_odom.copy()
    T[2, 3] += 0.5                                              # a bad z from map_align
    T[0, 3] += 0.1
    assert m.on_map_from_odom(3.0, T) is None
    assert m.T_map_odom[2, 3] == pytest.approx(z0) and m.T_map_odom[0, 3] == pytest.approx(T[0, 3])
    assert m.on_map_from_odom(3.1, np.full((4, 4), np.nan)) is not None


# ------------------------------------------- stack restarts and odometry gaps (skybeta-09 bench)
def until(m, sim, t, phase, limit=30.0):
    t_end = t + limit
    while m.phase != phase and t < t_end:
        t, _ = run(m, sim, t, t + DT)
    assert m.phase == phase, m.why
    return t


def test_a_short_odometry_gap_in_flight_does_not_abort():
    m, sim = mission(), Sim()
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    t = until(m, sim, t, "FOLLOW")
    t, _ = run(m, sim, t, t + 0.35, odom=False)                    # the bench saw gaps up to ~0.3 s
    t, _ = run(m, sim, t, t + 60.0)
    assert m.phase == "DONE", m.why


def test_stack_restart_before_takeoff_refixes_map_from_odom_and_resends_it():
    m = mission()
    sim = Sim(start_odom=(0.8, -1.1, 0.0, math.radians(-35.0)))
    t = to_ready(m, sim)
    T1 = m.T_map_odom.copy()
    sim.armed = True                                               # armed on the mark, not yet Position
    t_gap = t
    t, blind = run(m, sim, t, t + 3.0, odom=False)                 # the stack restarts: no odometry...
    sim.x, sim.y, sim.z, sim.yaw = 0.0, 0.0, 0.0, 0.0              # ...then a new origin, same spot
    t, after = run(m, sim, t, t + 3.0)
    assert m.phase == "WAIT_OFFBOARD" and m.n_init == 2
    assert all(c.pose is None for tt, c in blind if tt - t_gap > m.p.odom_stale_s + DT)   # silent while blind
    sent = [c.map_from_odom_init for _, c in blind + after if c.map_from_odom_init is not None]
    assert len(sent) == 1 and not np.allclose(sent[0], T1)        # a new INIT, re-sent to map_align
    assert np.allclose(sent[0] @ [0, 0, 0, 1], [LAUNCH["x"], LAUNCH["y"], LAUNCH["z"], 1])
    switch_to_offboard(t, sim)
    run(m, sim, t, t + 60.0)
    assert m.phase == "DONE", m.why
    end_map = (m.T_map_odom @ [sim.x, sim.y, sim.z, 1])[:2]
    assert np.hypot(*(end_map - np.array(ROUTE[-1]))) < 0.15


def test_odometry_reset_without_a_gap_before_takeoff_also_refixes():
    m = mission()
    sim = Sim(start_odom=(0.8, -1.1, 0.0, math.radians(-35.0)))
    t = to_ready(m, sim)
    sim.x, sim.y, sim.z, sim.yaw = 0.0, 0.0, 0.0, 0.0              # instant reset to a new origin
    t, _ = run(m, sim, t, t + 3.0)
    assert m.phase == "WAIT_OFFBOARD" and m.n_init == 2
    assert np.allclose(m.T_map_odom @ [0, 0, 0, 1], [LAUNCH["x"], LAUNCH["y"], LAUNCH["z"], 1])


def test_odometry_jump_in_flight_aborts_with_a_hold():
    m, sim = mission(), Sim()
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    t = until(m, sim, t, "FOLLOW")
    sim.x += 1.0                                                   # a reset mid-flight
    t, cmds = run(m, sim, t, t + 0.5)
    assert m.phase == "ABORTED" and "jumped" in m.why and sum(c.call_hold for _, c in cmds) == 1


def test_a_delayed_burst_of_odometry_is_not_a_jump():
    """Fast flight, then 0.3 s of samples delivered at once: captured apart (stamps), received together."""
    m, sim = mission(), Sim()
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    t = until(m, sim, t, "FOLLOW")
    x0, y0, z0, yaw0 = sim.x, sim.y, sim.z, sim.yaw
    for k in range(1, 10):                                         # 1.5 m/s, 9 samples at 30 Hz
        m.on_odom(t, x0 + 1.5 * k / 30.0, y0, z0, yaw0, stamp=t - 0.3 + k / 30.0)
    assert m.odom_jump is None
    m.on_odom(t + 0.01, x0 + 1.0, y0, z0, yaw0, stamp=t + 0.01)   # a real reset is still caught
    assert m.odom_jump is not None


def test_init_waits_for_px4_to_report_landed():
    m, sim = mission(), Sim()
    sim.landed = False
    run(m, sim, 0.0, 4.0)
    assert m.phase == "INIT" and "landed" in m.why


def test_corrections_stamped_before_the_latest_init_are_ignored():
    m, sim = mission(), Sim()
    to_ready(m, sim)
    assert m.on_map_from_odom(3.0, m.T_map_odom, stamp=m.t_fixed - 0.1) is not None
    assert m.on_map_from_odom(3.0, m.T_map_odom, stamp=m.t_fixed + 0.1) is None


# ------------------------------------------------------------------- HOVER (the first flights)
class OffsetSim(Sim):
    """PX4 holding a pose somewhere else than commanded: the guide's EKF-vs-odom frame offset."""
    offset = (0.6, 0.0)

    def apply(self, cmd, t):
        if cmd.pose is not None:
            cmd = type(cmd)(**{**vars(cmd), "pose": (cmd.pose[0] + self.offset[0], cmd.pose[1] + self.offset[1],
                                                     cmd.pose[2])})
        super().apply(cmd, t)


def hover(**kw):
    return SkybetaMission(MissionParams(mission="hover", **kw), LAUNCH)    # no route, no map


def test_hover_is_the_default_and_needs_no_route():
    assert MissionParams().mission == "hover"
    hover()
    with pytest.raises(ValueError):
        SkybetaMission(MissionParams(mission="course"), LAUNCH)
    with pytest.raises(ValueError):
        SkybetaMission(MissionParams(mission="Hover"), LAUNCH)


def test_hover_takes_off_holds_over_the_takeoff_point_and_lands_there():
    m, sim = hover(height_m=0.5, hover_s=5.0), Sim(start_odom=(0.8, -1.1, 0.0, math.radians(-35.0)))
    t = to_ready(m, sim)
    x0, y0 = sim.x, sim.y
    switch_to_offboard(t, sim)
    t, cmds = run(m, sim, t, t + 40.0)
    assert m.phase == "DONE", m.why
    phases = [c.phase for _, c in cmds]
    assert [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]] == \
        ["WAIT_OFFBOARD", "TAKEOFF", "HOVER", "DESCEND", "DONE"]           # state is 5 Hz: a few WAIT ticks
    held = [(tt, c) for tt, c in cmds if c.phase == "HOVER"]
    assert held[-1][0] - held[0][0] >= 5.0 - DT                         # hovered hover_s
    assert all(c.pose is not None and math.hypot(c.pose[0] - x0, c.pose[1] - y0) < 1e-9 for _, c in held)
    assert max(c.pose[2] for _, c in held) == pytest.approx(0.5, abs=1e-6)   # 0.5 m above the launch point
    assert math.hypot(sim.x - x0, sim.y - y0) < 0.05                       # landed where it took off
    assert all(c.pose is None and c.twist is None for c in [c for _, c in cmds if c.phase == "DONE"])


def test_hover_lands_straight_down_if_px4_holds_somewhere_else():
    m, sim = hover(hover_s=10.0), OffsetSim()
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    t, cmds = run(m, sim, t, t + 40.0)
    assert m.phase == "DONE", m.why
    desc = [c for _, c in cmds if c.phase == "DESCEND"]
    assert desc and "frame offset" in [e for _, c in cmds for e in c.events if "DESCEND" in e][0]
    assert all(c.twist is not None and c.twist[0] == 0.0 and c.twist[1] == 0.0 for c in desc)   # no steering
    x0, y0 = m.takeoff[:2]
    assert math.hypot(sim.x - x0, sim.y - y0) < 0.45                   # stopped before it got far


class OffsetZSim(Sim):
    """PX4 settles 12 cm below the commanded height (rangefinder vs VIO z): more than takeoff_tol_m."""

    def apply(self, cmd, t):
        if cmd.pose is not None:
            cmd = type(cmd)(**{**vars(cmd), "pose": (cmd.pose[0], cmd.pose[1], cmd.pose[2] - 0.12)})
        super().apply(cmd, t)


class NoClimbSim(Sim):
    """Props off on the bench: armed and in Offboard, but it never leaves the ground. PX4 disarms 10 s after
    arming (COM_DISARM_PRFLT)."""
    t_armed = None

    def apply(self, cmd, t):
        self.published.append((t, cmd.pose, cmd.twist))
        if self.armed and self.t_armed is None:
            self.t_armed = t
        if self.t_armed is not None and t - self.t_armed >= 10.0:
            self.armed = self.offboard = False


def test_takeoff_with_a_steady_height_offset_still_hovers_and_lands():
    m, sim = hover(height_m=0.5, hover_s=3.0), OffsetZSim()
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    t, cmds = run(m, sim, t, t + 40.0)
    assert m.phase == "DONE", m.why
    hover_whys = [c.why for _, c in cmds if c.phase == "HOVER"]
    assert hover_whys and "off the target height" in hover_whys[0]       # it went on, and said why
    t_hover = next(tt for tt, c in cmds if c.phase == "HOVER")
    assert t_hover - t < 0.5 / 0.3 + m.p.takeoff_settle_s + 3.0         # no endless TAKEOFF


def test_a_takeoff_that_never_climbs_lands_instead_of_hanging():
    m, sim = hover(height_m=0.5, hover_s=30.0), NoClimbSim()
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    t, cmds = run(m, sim, t, t + 20.0)
    first_desc = next((tt, c) for tt, c in cmds if c.phase == "DESCEND")
    assert "climbed only" in first_desc[1].why and first_desc[0] - t < 10.0   # before PX4's own disarm
    assert m.phase == "DONE", m.why                                       # PX4 disarmed on the ground: done


class RotatedTwistSim(Sim):
    """The interface maps our ENU twist into a PX4 frame rotated by `rot` from odom (the guide's frame offset,
    unmeasured and uncorrected): every twist lands rotated. Poses are held by PX4 itself (not rotated here)."""
    rot = math.radians(100.0)

    def apply(self, cmd, t):
        if cmd.twist is not None:
            vx, vy, vz, wz = cmd.twist
            c, s = math.cos(self.rot), math.sin(self.rot)
            cmd = type(cmd)(**{**vars(cmd), "twist": (c * vx - s * vy, s * vx + c * vy, vz, wz)})
        super().apply(cmd, t)


def descent_drift(**kw):
    m, sim = hover(height_m=0.7, hover_s=2.0, **kw), RotatedTwistSim()
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    t = until(m, sim, t, "DESCEND")
    sim.x += 0.05                                   # a few cm off the landing point as the descent starts
    x0, y0 = m.land_xy_odom
    worst = 0.0
    while m.phase == "DESCEND":
        t, _ = run(m, sim, t, t + DT)
        worst = max(worst, math.hypot(sim.x - x0, sim.y - y0))
    return m, worst


def test_default_descent_is_straight_down_so_a_frame_offset_cannot_spiral_it():
    m, worst = descent_drift()
    assert m.phase == "DONE", m.why
    assert worst < 0.06                             # stays where it was: no xy steering at all
    _, worst_hold = descent_drift(descend_xy_hold=True)
    assert worst_hold > 2 * worst                   # the xy P-loop through a 100 deg offset runs away


def test_hover_aborts_like_the_course_mission_when_offboard_drops():
    m, sim = hover(), Sim()
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    t = until(m, sim, t, "HOVER")
    sim.offboard = False
    _, cmds = run(m, sim, t, t + 1.0)
    assert m.phase == "ABORTED" and sum(c.call_hold for _, c in cmds) == 1


def test_the_package_never_publishes_to_px4_directly():
    """Nothing may command PX4 except through the flight interface: no /fmu/in anywhere. Reading /fmu/out
    is allowed only in the bench's read-only yaw_offset_probe."""
    pkg = ROOT / "skybeta" / "blueraven_skybeta"
    for f in list(pkg.rglob("*.py")):
        text = f.read_text(encoding="utf-8")
        assert "/fmu/in" not in text, f"{f} mentions /fmu/in"
        if f.name != "yaw_offset_probe.py":
            assert "/fmu/" not in text, f"{f} mentions /fmu/"
        else:
            assert "create_publisher" not in text, "the probe must stay read-only"
