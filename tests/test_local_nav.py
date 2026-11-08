"""Map-free navigation: the live depth grid (nav/local_grid.py) and waypoint lines + MPPI (nav/local_nav.py),
closed loop against a simulated forward depth camera in small 2.5D worlds (tests/sim_world.py)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from nav.local_grid import LocalGrid, LocalGridParams
from nav.local_nav import LocalNav, LocalNavParams, waypoint_path
from sim_world import Drone, World

WALL = 1.73          # walls are 68 in tall


def cell_value(g: LocalGrid, x: float, y: float) -> float:
    i, j, ok = g.cells(np.array([[x, y]]))
    assert ok[0]
    return float(g.L[i[0], j[0]])


# ---------------------------------------------------------------------------------------------- grid
def test_a_wall_ahead_is_occupied_and_the_space_before_it_free():
    w = World((-1, 6), (-2, 2), boxes=[(3.0, 3.2, -2, 2, WALL)])
    g = LocalGrid()
    cam = np.array([0.0, 0.0, 0.7])
    for _ in range(3):
        g.integrate(w.depth_points(cam, 0.0), cam, 0.7)
    assert cell_value(g, 3.0, 0.0) >= g.p.l_occ
    assert cell_value(g, 1.5, 0.0) < 0.0
    d = g.grid2d()
    from nav.mppi import Mppi
    m = Mppi(d, np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([0.3, 0.3]))
    assert float(m.clearance(np.array([[2.0, 0.0]]))[0]) == pytest.approx(1.0, abs=0.06)


def test_floor_is_not_an_obstacle_but_a_tall_rock_is():
    w = World((-1, 6), (-5, 5), boxes=[(1.5, 1.95, -0.2, 0.25, 0.46)])   # an 18 in boulder, 0.46 m tall
    g = LocalGrid()
    cam = np.array([0.0, 0.0, 0.7])
    for _ in range(3):
        g.integrate(w.depth_points(cam, 0.0, n_v=31), cam, 0.7)
    occ = g.occupied()
    i, j, _ = g.cells(np.array([[1.5, 0.0]]))
    assert occ[i[0] - 1:i[0] + 2, j[0] - 2:j[0] + 3].any()                 # the boulder's face
    i, j, _ = g.cells(np.array([[1.0, 0.8]]))
    assert not occ[i[0], j[0]]                                            # floor beside it: not an obstacle
    assert occ.sum() < 200                                                # nothing else got marked


def test_low_returns_far_away_are_ignored():
    w = World((-1, 8), (-5, 5), boxes=[(3.5, 3.95, -0.2, 0.25, 0.42)])   # rock top 3.5 m out: beyond low_range_m
    g = LocalGrid()
    cam = np.array([0.0, 0.0, 0.7])
    g.integrate(w.depth_points(cam, 0.0, n_v=31), cam, 0.7)
    assert not g.occupied().any()


def test_a_rock_the_drone_flies_over_is_not_an_obstacle():
    w = World((-1, 6), (-5, 5), boxes=[(1.5, 1.73, -0.115, 0.115, 0.30)])  # the CP2 cone: 0.30 m, 0.4 m under us
    g = LocalGrid()
    cam = np.array([0.0, 0.0, 0.7])
    for _ in range(3):
        g.integrate(w.depth_points(cam, 0.0, n_v=31), cam, 0.7)
    assert not g.occupied().any()


def test_a_tunnel_ceiling_is_not_a_wall():
    w = World((-1, 6), (-2, 2), boxes=[(2.0, 5.0, -2, 2, 0.0)])
    g = LocalGrid()
    cam = np.array([0.0, 0.0, 0.7])
    ceiling = np.c_[np.linspace(1.0, 3.0, 50), np.zeros(50), np.full(50, 1.2)]   # 0.5 m above the drone
    g.integrate(ceiling, cam, 0.7)
    assert not g.occupied().any()


def test_walls_out_of_view_are_remembered_and_phantoms_get_cleared():
    w = World((-1, 6), (-5, 5), boxes=[(3.6, 3.8, -5, 5, WALL)])
    g = LocalGrid()
    cam = np.array([0.0, 0.0, 0.7])
    phantom = np.array([[2.5, 0.0, 0.7]])                                   # beyond the never-cleared zone
    g.integrate(phantom, cam, 0.7)
    g.integrate(phantom, cam, 0.7)
    assert cell_value(g, 2.5, 0.0) >= g.p.l_occ
    for _ in range(6):                                                    # the camera looks past it to the wall
        g.integrate(w.depth_points(cam, 0.0), cam, 0.7)
    assert cell_value(g, 2.5, 0.0) < g.p.l_occ
    assert cell_value(g, 3.6, 0.0) >= g.p.l_occ
    for _ in range(5):                                                    # now looking the other way
        g.integrate(w.depth_points(cam, math.pi), cam, 0.7)
    assert cell_value(g, 3.6, 0.0) >= g.p.l_occ                           # still remembered


def test_a_boulder_under_the_camera_is_not_erased_by_rays_passing_over_it():
    w = World((-1, 6), (-5, 5), boxes=[(1.0, 1.3, -0.15, 0.15, 0.46), (4.0, 4.2, -5, 5, WALL)])
    g = LocalGrid()
    for x in (0.0, 0.2):                                                  # far enough to see it: marked
        cam = np.array([x, 0.0, 0.7])
        for _ in range(2):
            g.integrate(w.depth_points(cam, 0.0, n_v=31), cam, 0.7)
    assert g.occupied()[g.cells(np.array([[1.0, 0.0]]))[0][0] - 1:g.cells(np.array([[1.0, 0.0]]))[0][0] + 2].any()
    cam = np.array([0.6, 0.0, 0.7])                                       # now too close: only the wall is seen
    for _ in range(8):
        g.integrate(w.depth_points(cam, 0.0, n_v=31), cam, 0.7)
    i, j, _ = g.cells(np.array([[1.0, 0.0]]))
    assert g.occupied()[i[0] - 1:i[0] + 2, j[0] - 3:j[0] + 4].any()


def test_the_window_follows_the_drone_and_keeps_world_positions():
    g = LocalGrid(LocalGridParams(recentre_m=1.0))
    cam = np.array([0.0, 0.0, 0.7])
    g.integrate(np.array([[2.0, 1.0, 0.7]] * 3), cam, 0.7)
    g.integrate(np.array([[2.0, 1.0, 0.7]] * 3), cam, 0.7)
    before = cell_value(g, 2.0, 1.0)
    g.follow((3.3, -1.2))
    assert np.max(np.abs(g.centre - (3.3, -1.2))) <= g.p.cell
    assert cell_value(g, 2.0, 1.0) == pytest.approx(before)


# ---------------------------------------------------------------------------------------------- path
def test_waypoint_trajectory_rounds_corners_slows_on_them_and_stops_at_the_end():
    p = LocalNavParams()
    P, v = waypoint_path((0, 0), [(3, 0), (3, -3)], p)
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    assert np.hypot(*np.diff(P, axis=0).T).max() <= p.step_m + 1e-9           # dense, no jumps
    corner = np.hypot(P[:, 0] - 3.0, P[:, 1]).min()
    assert corner == pytest.approx(p.corner_dev_m, abs=0.01)                  # cuts the corner by corner_dev_m only
    R = p.corner_dev_m / (1 / math.cos(math.pi / 4) - 1)
    assert v[np.argmin(np.abs(s - s[-1] / 2))] == pytest.approx(math.sqrt(p.a_lat * R), abs=0.01)
    assert v[np.argmin(np.abs(s - 1.5))] == pytest.approx(p.v_cruise)
    assert v[0] == 0.0 and v[-1] == 0.0
    dv2 = np.abs(np.diff(v ** 2)) / np.maximum(np.diff(s), 1e-9)
    assert dv2.max() <= 2 * p.a_long + 1e-6                                   # speed changes within a_long
    P2, v2 = waypoint_path((0, 0), [(1.0, 0), (1.0, 0.3)], p)                 # a short leg limits the radius
    assert np.hypot(P2[:, 0] - 1.0, P2[:, 1]).min() <= p.corner_dev_m + 1e-9


# ---------------------------------------------------------------------------------------------- closed loop
def fly(world: World, waypoints, start=(0.0, 0.0), yaw0=0.0, seconds=60.0, depth_every=1):
    nav = LocalNav(waypoints, start, yaw0)
    drone = Drone(np.array(start, float), yaw0)
    dt = nav.mppi.p.dt
    traj, states = [drone.x.copy()], []
    for k in range(int(seconds / dt)):
        if k % depth_every == 0:
            cam = drone.cam()
            nav.add_depth(world.depth_points(cam, drone.yaw), cam, drone.z, k * dt)
        cmd = nav.step(drone.x, drone.v, k * dt)
        states.append(cmd.state)
        if cmd.state == "ARRIVED":
            break
        drone.step((cmd.vx, cmd.vy), cmd.yaw, dt)
        traj.append(drone.x.copy())
    return np.array(traj), states, nav


def cp1_world() -> World:
    """Straight 4 m, a wall ahead, turn right, 4 m on (corridors 1.2 m wide)."""
    return World((-1.0, 5.5), (-5.5, 1.5), boxes=[
        (-0.7, -0.5, -0.8, 0.8, WALL),       # behind the start
        (-0.7, 4.8, 0.6, 0.8, WALL),         # left wall, then the wall ahead's top corner
        (-0.7, 3.4, -0.8, -0.6, WALL),       # right wall of the first corridor
        (4.6, 4.8, -4.7, 0.8, WALL),         # the wall ahead, continuing as the second corridor's left wall
        (3.2, 3.4, -4.7, -0.6, WALL),        # second corridor's right wall
        (3.2, 4.8, -4.9, -4.7, WALL),        # its end
    ])


def test_cp1_straight_wall_ahead_turn_right_with_rough_waypoints():
    w = cp1_world()
    traj, states, nav = fly(w, [(4.3, 0.25), (3.8, -3.6)])               # both ~0.3 m off the corridor centres
    assert states[-1] == "ARRIVED", states[-5:]
    assert w.clearance(traj).min() > nav.mppi.p.r_drone
    assert "BLOCKED" not in states


def test_a_slanted_wall_crossing_the_line_is_slid_along():
    w = World((-1.0, 7.0), (-1.5, 1.5), boxes=[(-0.7, -0.5, -1, 1, WALL), (-0.7, 6.5, 0.6, 0.8, WALL),
                                               (6.3, 6.5, -1.0, 0.8, WALL)],
              polys=[(np.array([[-0.5, -1.0], [6.3, -1.0], [6.3, -0.30], [-0.5, -0.80]]), WALL)])
    traj, states, nav = fly(w, [(5.6, 0.0)])                              # 1.4 m wide narrowing to 0.9 m
    assert states[-1] == "ARRIVED", states[-5:]
    assert w.clearance(traj).min() > nav.mppi.p.r_drone


def test_a_tall_boulder_on_the_line_is_flown_round():
    w = World((-1.0, 7.0), (-1.5, 1.5), boxes=[(-0.7, -0.5, -1.2, 1.2, WALL), (-0.7, 6.5, 1.0, 1.2, WALL),
                                               (-0.7, 6.5, -1.2, -1.0, WALL), (6.3, 6.5, -1.2, 1.2, WALL),
                                               (2.8, 3.25, -0.22, 0.23, 0.46)])
    traj, states, nav = fly(w, [(5.6, 0.0)])
    assert states[-1] == "ARRIVED", states[-5:]
    boulder = np.hypot(np.maximum(np.abs(traj[:, 0] - 3.025) - 0.225, 0), np.maximum(np.abs(traj[:, 1] - 0.005) - 0.225, 0))
    assert boulder.min() > nav.mppi.p.r_drone
    assert w.clearance(traj).min() > nav.mppi.p.r_drone


def test_a_moved_wall_jutting_across_the_line():
    w = World((-1.0, 7.0), (-1.5, 1.5), boxes=[(-0.7, -0.5, -1.2, 1.2, WALL), (-0.7, 6.5, 1.0, 1.2, WALL),
                                               (-0.7, 6.5, -1.2, -1.0, WALL), (6.3, 6.5, -1.2, 1.2, WALL),
                                               (2.9, 3.1, -1.0, 0.10, WALL)])       # juts 0.1 m past the line
    traj, states, nav = fly(w, [(5.6, 0.0)])
    assert states[-1] == "ARRIVED", states[-5:]
    assert w.clearance(traj).min() > nav.mppi.p.r_drone


def new_start_world() -> World:
    """Darion's corrected drawing (v2) of the rebuilt start (map metres, approximate): wall A, the diagonal wall B, the
    0.752 m CP2 gap between their ends (~0.10 m a side for a 0.56 m drone), the 0.30 m cone 0.19 m before the gap
    line, the barrier, and the nets (which depth can't see)."""
    a, b = np.array([-1.747, 0.762]), np.array([-1.023, 1.932])
    n = np.array([-(b - a)[1], (b - a)[0]]) / np.hypot(*(b - a)) * 0.05
    wall_b = np.array([a - n, b - n, b + n, a + n])
    return World((-3.3, 1.2), (0.2, 3.3),
                 boxes=[(-3.048, 1.0, 0.41, 0.51, WALL),                 # the barrier
                        (-2.743, -1.277, 2.59, 2.69, WALL),              # wall A
                        (-1.484, -1.254, 2.220, 2.450, 0.30)],           # the cone (as a 0.23 m box), at the gap
                 polys=[(wall_b, WALL)],
                 nets=[(-3.3, -3.048, 0.2, 3.3), (-3.3, 1.2, 3.048, 3.3)])


@pytest.mark.parametrize("noisy", [False, True])
def test_new_start_through_the_cp2_gap_over_the_cone(noisy):
    w = new_start_world()
    start, yaw0 = (-2.515, 0.914), math.pi / 2                          # the launch mark, facing north
    wps = [(-2.45, 2.05), (-1.13, 2.30), (-0.49, 2.52)]                 # ~0.02-0.05 m off a human's best guess
    nav = LocalNav(wps, start, yaw0)
    drone = Drone(np.array(start, float), yaw0)
    rng = np.random.default_rng(3)
    dt = nav.mppi.p.dt
    traj, states = [drone.x.copy()], []
    for k in range(int(60.0 / dt)):
        cam = drone.cam()
        pts = w.depth_points(cam, drone.yaw, noise_k=0.006 if noisy else 0.0, dropout=0.65 if noisy else 0.0,
                             outliers=0.01 if noisy else 0.0, rng=rng)
        nav.add_depth(pts, cam, drone.z, k * dt)
        cmd = nav.step(drone.x, drone.v, k * dt)
        states.append(cmd.state)
        if cmd.state == "ARRIVED":
            break
        drone.step((cmd.vx, cmd.vy), cmd.yaw, dt)
        traj.append(drone.x.copy())
    traj = np.array(traj)
    assert states[-1] == "ARRIVED", states[-5:]
    assert w.clearance(traj, min_height=0.35).min() > nav.mppi.p.r_drone      # walls and nets; the cone is flown over
    # it went THROUGH the gap: crossed the segment between wall A's east end and wall B's NE end
    p, q = np.array([-1.277, 2.640]), np.array([-1.023, 1.932])
    side = np.sign((q[0] - p[0]) * (traj[:, 1] - p[1]) - (q[1] - p[1]) * (traj[:, 0] - p[0]))
    k = int(np.argmax(side != side[0]))
    assert k > 0
    t = float(np.dot(traj[k] - p, q - p) / np.dot(q - p, q - p))
    assert 0.0 < t < 1.0


# ---------------------------------------------------------------------------------------------- supervisor
def corridor_world() -> World:
    return World((-1.0, 8.0), (-1.5, 1.5), boxes=[(-0.7, -0.5, -1.2, 1.2, WALL), (-0.7, 7.5, 0.6, 0.8, WALL),
                                                  (-0.7, 7.5, -0.8, -0.6, WALL), (7.3, 7.5, -1.2, 1.2, WALL)])


def test_waits_for_depth_before_moving():
    nav = LocalNav([(5.0, 0.0)], (0.0, 0.0), 0.0)
    c = nav.step((0.0, 0.0), (0.0, 0.0), 0.0)
    assert c.state == "WAIT" and (c.vx, c.vy) == (0.0, 0.0)
    w, cam = corridor_world(), np.array([0.1, 0.0, 0.7])
    for k in range(nav.p.min_depth_frames):
        nav.add_depth(w.depth_points(cam, 0.0), cam, 0.7, 0.1 * k)
    assert nav.step((0.0, 0.0), (0.0, 0.0), 0.5).state == "GO"


def test_slows_on_the_remembered_grid_when_depth_stops():
    w = corridor_world()
    nav = LocalNav([(6.5, 0.0)], (0.0, 0.0), 0.0)
    d = Drone(np.array([0.0, 0.0]), 0.0)
    states, speeds = [], []
    for k in range(500):                                                  # 0.2 m/s after 4 s: ~30 s to the end
        t = 0.1 * k
        if t < 4.0:                                                       # depth stops after 4 s (a dark tunnel)
            cam = d.cam()
            nav.add_depth(w.depth_points(cam, d.yaw), cam, d.z, t)
        c = nav.step(d.x, d.v, t)
        states.append(c.state)
        if c.state == "ARRIVED":
            break
        d.step((c.vx, c.vy), c.yaw, 0.1)
        speeds.append(float(np.hypot(*d.v)))
    assert "SLOW" in states and states[-1] == "ARRIVED"
    assert max(speeds[60:]) <= nav.p.v_no_depth + 0.05                     # 2 s after depth stopped
    assert w.clearance(np.array([d.x])).min() > nav.mppi.p.r_drone


def test_a_dead_end_stops_short_and_reports_it():
    w = World((-1.0, 8.0), (-1.5, 1.5), boxes=[(-0.7, -0.5, -1.2, 1.2, WALL), (-0.7, 7.5, 0.6, 0.8, WALL),
                                               (-0.7, 7.5, -0.8, -0.6, WALL), (3.0, 3.2, -0.8, 0.8, WALL)])
    traj, states, nav = fly(w, [(5.5, 0.0)], seconds=30.0)
    assert states[-1] in ("STUCK", "BLOCKED") or "STUCK" in states
    assert w.clearance(traj).min() > nav.mppi.p.r_drone
    assert traj[:, 0].max() < 3.0 - nav.mppi.p.r_drone



# ---------------------------------------------------------------------------------------------- go round and rejoin
def clip_world(cy: float = 0.15) -> World:
    """A 2.2 m wide corridor along +x with a wall-height 0.4 m box at (3.0, cy) on / beside the trajectory (y = 0)."""
    return World((-1.0, 8.0), (-1.5, 1.5), boxes=[(-0.7, -0.5, -1.3, 1.3, WALL), (-0.7, 7.5, 1.1, 1.3, WALL),
                                                  (-0.7, 7.5, -1.3, -1.1, WALL), (7.3, 7.5, -1.3, 1.3, WALL),
                                                  (2.8, 3.2, cy - 0.2, cy + 0.2, WALL)])


@pytest.mark.parametrize("cy", [0.15, 0.40])                                # through the line / clipping its edge
def test_goes_round_an_obstacle_on_the_trajectory_and_rejoins_it(cy):
    w = clip_world(cy)
    traj, states, nav = fly(w, [(6.5, 0.0)], seconds=45.0)
    assert states[-1] == "ARRIVED", states[-5:]
    assert w.clearance(traj).min() > nav.mppi.p.r_drone                     # never touched it
    box = np.hypot(np.maximum(np.abs(traj[:, 0] - 3.0) - 0.2, 0), np.maximum(np.abs(traj[:, 1] - cy) - 0.2, 0))
    assert box.min() > nav.mppi.p.r_drone
    assert np.abs(traj[:, 1]).max() > 0.08                                  # it did leave the trajectory
    after = traj[traj[:, 0] > 4.6]                                          # 1.4 m past the box's far side
    assert len(after) and np.abs(after[:, 1]).max() < 0.10                  # and came back onto it
