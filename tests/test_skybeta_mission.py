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
ROUTE = [(0.0, 0.0), (1.5, 0.0), (1.5, -1.5)]   # take-off frame: 1.5 m forward, then 1.5 m to the right


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
    kw.setdefault("mission", "route")
    kw.setdefault("route_hover_s", 0.0)          # straight on to FOLLOW (the pre-route hover has its own tests)
    kw.setdefault("route_max_radius_m", 1.0)     # rounded corners (sharp ones have their own test)
    return SkybetaMission(MissionParams(**kw), ROUTE)


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


def test_stack_started_elsewhere_still_flies_the_route_from_the_takeoff_pose():
    m = mission()
    sim = Sim(start_odom=(0.8, -1.1, 0.0, math.radians(-35.0)))    # odom origin not at the mark
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    run(m, sim, t, t + 60.0)
    assert m.phase == "DONE"
    end_map = (m.T_map_odom @ [sim.x, sim.y, sim.z, 1])[:2]
    assert np.hypot(*(end_map - np.array(ROUTE[-1]))) < 0.15


def test_sharp_corners_stop_and_turn_in_place_then_carry_on():
    m, sim = mission(route_max_radius_m=0.0), Sim()                 # every corner sharp
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


def test_init_fixes_the_takeoff_frame_on_the_drones_own_pose():
    m, sim = mission(), Sim(start_odom=(0.8, -1.1, 0.0, math.radians(-35.0)))
    run(m, sim, 0.0, 4.0)
    assert np.allclose(m.T_map_odom @ [0.8, -1.1, 0.0, 1], [0, 0, 0, 1])          # the pad is the origin
    fwd = pose_xyz_yaw(0.8, -1.1, 0.0, math.radians(-35.0)) @ [1, 0, 0, 1]        # 1 m ahead of the nose
    assert np.allclose(m.T_map_odom @ fwd, [1, 0, 0, 1])                          # = +x of the take-off frame
    assert m.status(4.0)["takeoff_frame_xyz"] == [0.0, 0.0, 0.0]


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


def test_stack_restart_before_takeoff_refixes_the_takeoff_frame():
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
    assert not np.allclose(m.T_map_odom, T1)                       # a new INIT, against the new origin
    assert np.allclose(m.T_map_odom @ [0, 0, 0, 1], [0, 0, 0, 1])  # the drone, still on the pad
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
    assert np.allclose(m.T_map_odom @ [0, 0, 0, 1], [0, 0, 0, 1])


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
    return SkybetaMission(MissionParams(mission="hover", **kw))           # no route, no map


def test_hover_is_the_default_and_needs_no_route():
    assert MissionParams().mission == "hover"
    hover()
    with pytest.raises(ValueError):
        SkybetaMission(MissionParams(mission="course"))                   # removed with the map
    with pytest.raises(ValueError):
        SkybetaMission(MissionParams(mission="Hover"))
    with pytest.raises(ValueError):
        SkybetaMission(MissionParams(mission="hover"), ROUTE)             # a hover takes no route


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


class LandDetectorSim(Sim):
    """PX4's land detector (Darion's T4 finding): on the ground it reports landed, and then auto-disarms, only
    while the commanded descent is at least 1.1 x LNDMC_Z_VEL_MAX = 0.275 m/s."""

    def apply(self, cmd, t):
        was_armed = self.armed
        super().apply(cmd, t)
        pressing = cmd.twist is not None and cmd.twist[2] <= -0.275
        if was_armed and self.z <= self.ground + 1e-6 and not pressing:
            self.landed, self.t_landed = False, None               # no ground contact detected
            self.armed = self.offboard = True                      # so no auto-disarm either


def land_with_detector(**kw):
    m, sim = hover(height_m=0.5, hover_s=2.0, **kw), LandDetectorSim()
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    t = until(m, sim, t, "DESCEND")
    _, cmds = run(m, sim, t, t + 40.0)
    return m, cmds, t


def test_the_default_descent_is_firm_enough_for_px4_to_detect_the_landing():
    m, cmds, t0 = land_with_detector()
    assert MissionParams().descend_mps >= 0.3
    assert m.phase == "DONE" and m.why.startswith("landed and disarmed"), m.why
    t_done = next(tt for tt, c in cmds if c.phase == "DONE")
    assert t_done - t0 < 0.5 / 0.35 + 4.0                          # touchdown + PX4's 2 s, no long wait


def test_a_too_gentle_descent_presses_down_on_the_ground_instead_of_sitting_armed():
    m, cmds, t0 = land_with_detector(descend_mps=0.2)
    assert m.phase == "DONE", m.why
    pressing = [c for _, c in cmds if c.phase == "DESCEND" and c.twist and c.twist[2] <= -0.5]
    assert pressing and "pressing down" in pressing[0].why
    t_done = next(tt for tt, c in cmds if c.phase == "DONE")
    assert t_done - t0 < 0.5 / 0.2 + 3.0 + 4.0                     # not the 63 s the sim sat armed


def test_the_descent_never_aborts_or_calls_hold_when_odometry_goes_stale():
    m, sim = hover(height_m=0.5, hover_s=2.0), LandDetectorSim()
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    t = until(m, sim, t, "DESCEND")
    _, cmds = run(m, sim, t, t + 30.0, odom=False)                 # the cameras lose the floor's texture
    assert m.phase == "DONE", m.why
    assert not any(c.call_hold for _, c in cmds)                   # hold would climb back to the hover point
    desc = [c for _, c in cmds if c.phase == "DESCEND"]
    assert all(c.twist is not None and c.twist[2] < 0 for c in desc)
    assert any("without odometry" in c.why for c in desc)


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


# ---- arnav-claude, 2026-11-07: review fixes (frame-offset guard, take-off z cap, stale tolerance) ----

class EnuStateSim(Sim):
    """Adds the stack's odom_enu / local_enu to /skybeta/state; local = odom + offset (x/y)."""

    def __init__(self, offset=(0.0, 0.0), **kw):
        super().__init__(**kw)
        self.offset = offset

    def state(self):
        s = super().state()
        s["odom_enu"] = [self.x, self.y, self.z]
        s["local_enu"] = [self.x + self.offset[0], self.y + self.offset[1], self.z - 0.7]   # z differs by design
        return s


def test_a_frame_offset_keeps_init_waiting_and_clears_when_it_agrees():
    m, sim = hover(), EnuStateSim(offset=(0.5, 0.0))
    t, cmds = run(m, sim, 0.0, 4.0)
    assert m.phase == "INIT" and "disagree" in m.why
    assert all(c.pose is None and c.twist is None for _, c in cmds)
    sim.offset = (0.05, 0.03)                                      # restarted level: they agree again
    run(m, sim, t, t + 4.0)
    assert m.phase == "WAIT_OFFBOARD", m.why


def test_a_frame_offset_appearing_before_takeoff_stops_the_stream():
    m, sim = hover(), EnuStateSim()
    t = to_ready(m, sim)
    sim.offset = (0.0, 0.3)
    t, cmds = run(m, sim, t, t + 1.0)
    late = [c for tt, c in cmds if tt > t - 0.5]                    # after a 5 Hz state carried the offset
    assert all(c.pose is None for c in late) and "disagree" in m.why
    assert m.phase == "WAIT_OFFBOARD"                               # no Offboard is requested without setpoints


def test_takeoff_pose_never_goes_above_the_target_height():
    m, sim = hover(height_m=0.5), Sim()
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)

    def high_vision(tt, s):                                         # vision z reads 0.4 m above the target
        if m.phase == "TAKEOFF":
            s.z = max(s.z, 0.9)
    t, cmds = run(m, sim, t, t + 3.0, hook=high_vision)
    ups = [c.pose[2] for _, c in cmds if c.phase == "TAKEOFF" and c.pose is not None]
    assert ups and max(ups) <= m.takeoff[2] + 1e-9


def test_an_odometry_stall_shorter_than_odom_stale_s_does_not_abort_a_hover():
    m, sim = hover(hover_s=20.0), Sim()
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    t, _ = run(m, sim, t, t + 8.0)
    assert m.phase == "HOVER", m.why
    t, _ = run(m, sim, t, t + 0.6, odom=False)                       # a 0.6 s stall (seen: 0.39 s)
    t, _ = run(m, sim, t, t + 2.0)
    assert m.phase == "HOVER", m.why


# ---------------------------------------------------------- mission "route": route files, the take-off frame
from blueraven_skybeta.mission_core import load_route  # noqa: E402

ROUTES = ROOT / "skybeta" / "blueraven_skybeta" / "routes"


def write_route(tmp_path, data, name="r.json"):
    import json
    p = tmp_path / name
    p.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return p


def fly_route(corners, start=(0.0, 0.0, 0.0, 0.0), limit=240.0, sim=None, **kw):
    """Fly a route mission on the toy drone to DONE/ABORTED. Returns the mission, the sim, the
    trajectory [(phase, x, y, z)] in odom and the commands."""
    m = SkybetaMission(MissionParams(mission="route", **kw), corners)
    sim = sim or Sim(start_odom=start)
    t = to_ready(m, sim)
    switch_to_offboard(t, sim)
    track, cmds_all = [], []
    t_end = t + limit
    while m.phase not in ("DONE", "ABORTED") and t < t_end:
        t, cmds = run(m, sim, t, t + 1.0, hook=lambda tt, s: track.append((m.phase, s.x, s.y, s.z)))
        cmds_all += cmds
    return m, sim, track, cmds_all


def test_route_files_in_the_repo_all_load():
    for f in sorted(ROUTES.glob("*.json")):
        corners = load_route(f)
        assert corners[0] == (0.0, 0.0) and len(corners) >= 2, f.name


@pytest.mark.parametrize("data, why", [
    ("not json", "route file"),
    ({"points": [[1, 0]]}, "non-empty"),
    ({"route": []}, "non-empty"),
    ({"route": [[1.0, 0.0, 0.7]]}, "point 0"),
    ({"route": [[1.0, "0"]]}, "point 0"),
    ({"route": [[1.0, True]]}, "point 0"),
    ('{"route": [[1.0, NaN]]}', "point 0"),
    ({"route": [[1.0, 0.0], [1.0, 0.01]]}, "leg to point 1"),       # a duplicate corner
    ({"route": [[0.01, 0.0]]}, "leg to point 0"),                     # no leg at all
    ({"route": [[9.0, 0.0]]}, "leg to point 0"),                      # a typo: 9 m in one go
    ({"route": [[6.0, 0.0], [6.0, 6.0], [12.0, 6.0]]}, "over 10.0 m"),
])
def test_a_bad_route_file_is_refused_with_the_reason(tmp_path, data, why):
    with pytest.raises(ValueError, match=why):
        load_route(write_route(tmp_path, data))


def test_a_missing_route_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="route file"):
        load_route(tmp_path / "missing.json")


def test_route_mission_needs_corners_from_the_takeoff_point():
    with pytest.raises(ValueError, match="take-off point"):
        SkybetaMission(MissionParams(mission="route"))
    with pytest.raises(ValueError, match="take-off point"):
        SkybetaMission(MissionParams(mission="route"), [(1.5, 0.0), (2.0, 0.0)])
    m = SkybetaMission(MissionParams(mission="route"), [(0.0, 0.0), (1.5, 0.0), (1.5, -1.0)])
    assert all(c.radius == 0.0 for c in m.path.corners)            # stop-and-turn by default


def test_route_hovers_first_then_flies_from_wherever_it_took_off_and_lands_at_the_end(tmp_path):
    corners = load_route(write_route(tmp_path, {"route": [[1.736, 0.085]]}))
    start = (0.8, -1.1, 0.0, math.radians(-35.0))                    # the stack started elsewhere, rotated
    m, sim, track, cmds = fly_route(corners, start=start, route_hover_s=3.0)
    assert m.phase == "DONE", m.why
    T = pose_xyz_yaw(*start)                                          # the take-off pose, in odom
    end = (np.linalg.inv(T) @ [sim.x, sim.y, sim.z, 1])[:2]           # where it landed, from take-off
    assert np.hypot(*(end - [1.736, 0.085])) < 0.10
    phases = [c.phase for _, c in cmds]
    assert [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]] == \
        ["WAIT_OFFBOARD", "TAKEOFF", "HOVER", "FOLLOW", "DESCEND", "DONE"]   # state is 5 Hz: a few WAIT ticks
    held = [(tt, c) for tt, c in cmds if c.phase == "HOVER"]
    assert held[-1][0] - held[0][0] >= 3.0 - DT
    x0, y0 = m.takeoff[:2]
    assert all(c.pose is not None and math.hypot(c.pose[0] - x0, c.pose[1] - y0) < 1e-9 for _, c in held)
    assert max(z for ph, _, _, z in track) == pytest.approx(0.7, abs=0.08)        # height_m above the pad


def test_route_hover_zero_goes_straight_to_follow():
    m, sim, _, cmds = fly_route([(0.0, 0.0), (1.0, 0.0)], route_hover_s=0.0)
    assert m.phase == "DONE" and "HOVER" not in {c.phase for _, c in cmds}


def test_route_drifting_in_the_pre_route_hover_lands_instead_of_flying():
    m, sim, _, cmds = fly_route([(0.0, 0.0), (1.5, 0.0)], sim=OffsetSim(), route_hover_s=10.0)
    assert m.phase == "DONE", m.why
    assert "FOLLOW" not in {c.phase for _, c in cmds}               # never set off on the route


def test_route_stops_and_turns_in_place_at_a_sharp_corner_then_carries_on():
    corners = [(0.0, 0.0), (1.4, 0.0), (1.7, 0.0), (1.6, -1.0)]
    m, sim, track, cmds = fly_route(corners)
    assert m.phase == "DONE", m.why
    turns = [c for _, c in cmds if c.phase == "FOLLOW" and c.twist
             and abs(c.twist[3]) > math.radians(20) and math.hypot(c.twist[0], c.twist[1]) < 0.08]
    assert turns, "never stopped to turn"
    assert np.hypot(sim.x - corners[-1][0], sim.y - corners[-1][1]) < 0.15


def test_no_map_anywhere_in_the_flight_package():
    """Arnav's directive (2026-11-07): the field has no map. Nothing the drone runs may load the course map,
    the planner, map alignment or the map-based centring."""
    pkg = ROOT / "skybeta" / "blueraven_skybeta"
    banned = ("course_map", "CourseMap", "course_sdf", "course.yaml", "route_planner", "map_align",
              "wall_centring", "MAP_FROM_ODOM", "load_launch")
    for f in pkg.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        for word in banned:
            assert word not in text, f"{f.relative_to(ROOT)} mentions {word}"
