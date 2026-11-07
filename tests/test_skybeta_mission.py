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


def test_the_package_never_publishes_to_px4_directly():
    pkg = ROOT / "skybeta" / "blueraven_skybeta"
    for f in list(pkg.rglob("*.py")):
        assert "/fmu/" not in f.read_text(encoding="utf-8"), f"{f} mentions /fmu/"
