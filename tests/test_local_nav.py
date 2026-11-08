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
    w = World((-1.0, 7.0), (-1.6, 1.6), boxes=[(-0.7, -0.5, -1.3, 1.3, WALL), (-0.7, 6.5, 1.1, 1.3, WALL),
                                               (-0.7, 6.5, -1.3, -1.1, WALL), (6.3, 6.5, -1.3, 1.3, WALL),
                                               (2.8, 3.25, -0.22, 0.23, 0.46)])      # passages 0.87 m each side
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


def _start_world(wall_a, wall_b, cone) -> World:
    a, b = np.array(wall_b[0], float), np.array(wall_b[1], float)
    n = np.array([-(b - a)[1], (b - a)[0]]) / np.hypot(*(b - a)) * 0.05
    (ax0, ay), (ax1, _) = wall_a
    cx, cy = cone
    return World((-3.3, 1.2), (0.2, 3.3),
                 boxes=[(-3.048, 1.0, 0.41, 0.51, WALL),                 # the barrier
                        (ax0, ax1, ay - 0.05, ay + 0.05, WALL),          # wall A
                        (cx - 0.115, cx + 0.115, cy - 0.115, cy + 0.115, 0.30)],   # the cone (as a 0.23 m box)
                 polys=[(np.array([a - n, b - n, b + n, a + n]), WALL)],          # wall B, the diagonal
                 nets=[(-3.3, -3.048, 0.2, 3.3), (-3.3, 1.2, 3.048, 3.3)])


# The rebuilt start. FINAL = Darion's tape map (chelsea/course-map ea5071d): CP2 gap 1.00 m, centre (-1.098, 2.153).
# V2_TIGHT = his earlier drawing, a 0.752 m gap with the cone just before it: kept as a tight-gap stress test.
START_FINAL = dict(wall_a=((-2.743, 2.591), (-1.339, 2.591)), wall_b=((-1.747, 0.762), (-0.857, 1.715)),
                   cone=(-1.431, 2.286), wps=[(-2.45, 2.00), (-1.098, 2.153), (-0.485, 2.490)])
START_V2_TIGHT = dict(wall_a=((-2.743, 2.640), (-1.277, 2.640)), wall_b=((-1.747, 0.762), (-1.023, 1.932)),
                      cone=(-1.369, 2.335), wps=[(-2.45, 2.05), (-1.13, 2.30), (-0.49, 2.52)])


def new_start_world(layout: dict | None = None) -> World:
    L = layout or START_FINAL
    return _start_world(L["wall_a"], L["wall_b"], L["cone"])


@pytest.mark.parametrize("layout", ["final", "v2_tight"])
@pytest.mark.parametrize("noisy", [False, True])
def test_new_start_through_the_cp2_gap_over_the_cone(noisy, layout):
    L = START_FINAL if layout == "final" else START_V2_TIGHT
    w = new_start_world(L)
    start, yaw0 = (-2.515, 0.914), math.pi / 2                          # the launch mark, facing north
    nav = LocalNav(L["wps"], start, yaw0)
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
    assert w.clearance(traj, min_height=0.35).min() > nav.mppi.p.r_drone      # never touches (walls, nets; cone flown over)
    if layout == "v2_tight":                       # 0.752 m: with the 5 cm buffer ~1 cm to spare: stopping is right
        assert states[-1] in ("ARRIVED", "BLOCKED") or "STUCK" in states
        return
    assert states[-1] == "ARRIVED", states[-5:]
    # it went THROUGH the gap: crossed the segment between wall A's east end and wall B's NE end
    p, q = np.array(L["wall_a"][1]), np.array(L["wall_b"][1])
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
    states, speeds, cmds = [], [], []
    for k in range(500):
        t = 0.1 * k
        if not 4.0 <= t < 5.5:                                            # depth drops out for 1.5 s mid-flight
            cam = d.cam()
            nav.add_depth(w.depth_points(cam, d.yaw), cam, d.z, t)
        c = nav.step(d.x, d.v, t)
        states.append(c.state)
        cmds.append(float(np.hypot(c.vx, c.vy)))
        if c.state == "ARRIVED":
            break
        d.step((c.vx, c.vy), c.yaw, 0.1)
        speeds.append(float(np.hypot(*d.v)))
    assert "SLOW" in states and "HOLD" not in states and states[-1] == "ARRIVED"
    slow = [v for v, st in zip(cmds, states) if st == "SLOW"]
    assert max(slow[3:]) <= nav.p.v_no_depth_frac * nav.p.v_cruise + 1e-6       # commands obey the cap (the T4-like
                                                                                # drone itself slows after its dead time)
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



def test_shield_stops_short_of_a_wall_that_appears_ahead():
    """Flying at 0.45 m/s, a wall appears 1.2 m ahead across the whole corridor: it must stop clear of it."""
    from nav.mppi import Grid2D, Mppi
    free = Grid2D(np.full((200, 60), 3.0), (-1.0, -1.5), 0.05)
    m = Mppi(free, np.array([[0.0, 0.0], [8.0, 0.0]]), np.array([0.45, 0.45]))
    x, v = np.zeros(2), np.array([0.45, 0.0])
    m.step(x, v)
    X = -1.0 + 0.05 * np.arange(200)[:, None] + np.zeros((200, 60))
    m.set_grid(Grid2D(np.minimum(3.0, 1.2 - X), (-1.0, -1.5), 0.05))      # wall face at x = 1.2
    a = math.exp(-m.p.dt / m.p.tau)
    b = m.p.tau * (1 - a)
    for _ in range(60):
        r = m.step(x, v)
        u = np.array(r.u)
        x = x + u * m.p.dt + (v - u) * b
        v = u + (v - u) * a
    assert x[0] < 1.2 - m.p.r_drone


@pytest.mark.parametrize("cruise", [0.15, 0.45])
def test_a_shut_cp2_gap_is_never_flown_through_even_after_waiting(cruise):
    """Hinson's integration agent's case (hinson/mppi 218ec30, strict xfail on c29d114): the CP2 gap is SHUT and the
    drone lingers there. With time-based grid memory (12 s) it forgot wall B, out of view beside it, and flew through.
    Memory now fades per metre flown, so 60 s of waiting at 0.15 or 0.45 m/s must never get it through or let it touch."""
    from nav.local_nav import LocalNavParams
    L = START_FINAL
    a, b = np.array(L["wall_a"][1]), np.array(L["wall_b"][1])
    w0 = new_start_world(L)
    plug = np.array([a + (a - b) * 0.05, b + (b - a) * 0.05])            # a wall right across the gap
    n = np.array([-(plug[1] - plug[0])[1], (plug[1] - plug[0])[0]]) / np.hypot(*(plug[1] - plug[0])) * 0.05
    w = World(w0.x_range, w0.y_range, boxes=w0.boxes, nets=w0.nets,
              polys=w0.polys + [(np.array([plug[0] - n, plug[1] - n, plug[1] + n, plug[0] + n]), WALL)])
    start, yaw0 = (-2.515, 0.914), math.pi / 2
    nav = LocalNav(L["wps"], start, yaw0, nav=LocalNavParams(v_cruise=cruise))
    drone = Drone(np.array(start, float), yaw0)
    traj, states = [drone.x.copy()], []
    for k in range(int((45.0 / cruise ** 0.5 + 60.0) / 0.1)):           # reach the gap, then linger ~60 s
        cam = drone.cam()
        nav.add_depth(w.depth_points(cam, drone.yaw), cam, drone.z, k * 0.1)
        cmd = nav.step(drone.x, drone.v, k * 0.1)
        states.append(cmd.state)
        drone.step((cmd.vx, cmd.vy), cmd.yaw, 0.1)                       # keeps obeying, as if nobody lands it
        traj.append(drone.x.copy())
    traj = np.array(traj)
    assert w.clearance(traj, min_height=0.35).min() > nav.mppi.p.r_drone  # never touched anything
    side = np.sign((b[0] - a[0]) * (traj[:, 1] - a[1]) - (b[1] - a[1]) * (traj[:, 0] - a[0]))
    assert (side == side[0]).all()                                       # never crossed the gap line
    assert "STUCK" in states or "BLOCKED" in states                      # and it said so



def test_no_depth_for_long_holds_still():
    """Hinson's review: with the camera dead, never keep flying on memory. After depth_dead_s: HOLD, zero velocity."""
    w = corridor_world()
    nav = LocalNav([(6.5, 0.0)], (0.0, 0.0), 0.0)
    d = Drone(np.array([0.0, 0.0]), 0.0)
    last = None
    for k in range(80):
        t = 0.1 * k
        if t < 3.0:
            cam = d.cam()
            nav.add_depth(w.depth_points(cam, d.yaw), cam, d.z, t)
        last = nav.step(d.x, d.v, t)
        d.step((last.vx, last.vy), last.yaw, 0.1)
    assert last.state == "HOLD" and (last.vx, last.vy) == (0.0, 0.0)


def test_the_sent_command_is_what_the_prediction_replays():
    """Hinson's review: when LocalNav overrides MPPI's command (zero on ARRIVED/BLOCKED, WAIT), the history the pose
    prediction replays must hold what was really sent."""
    nav = LocalNav([(5.0, 0.0)], (0.0, 0.0), 0.0)
    nav.step((0.0, 0.0), (0.0, 0.0), 0.0)                                 # WAIT: a zero is recorded
    assert len(nav.mppi._inflight) == 1 and not nav.mppi._inflight[-1].any()
    nav.mppi._inflight[-1] = np.array([0.3, 0.0])
    nav.mppi.note_sent((0.0, 0.0), replace_last=True)
    assert not nav.mppi._inflight[-1].any()



def test_yaw_turns_only_while_moving_and_at_most_30_deg_s():
    """Darion: near-pure rotation diverged cuVSLAM on the T4 (no IMU on the real drone). Stopped: no yaw at all."""
    nav = LocalNav([(0.0, 3.0)], (0.0, 0.0), 0.0)                          # the line goes left: the heading wants +90 deg
    pts = np.c_[np.full(60, 2.5), np.linspace(-1, 1, 60), np.full(60, 0.7)]
    for k in range(6):
        nav.add_depth(pts, (0.0, 0.0, 0.7), 0.7, 0.1 * k)
    yaws = [nav.step((0.0, 0.0), (0.0, 0.0), 0.6 + 0.1 * k).yaw for k in range(10)]
    assert max(abs(y) for y in yaws) < 1e-9                               # not moving: never turns
    y0 = nav.yaw_cmd
    y1 = nav.step((0.0, 0.0), (0.0, 0.2), 1.6).yaw                         # moving: turns, within 30 deg/s (0.1 s)
    assert 0 < y1 - y0 <= math.radians(30) * 0.1 + 1e-9



def test_odometry_jitter_at_a_hover_does_not_fade_walls():
    """Hinson's jitter test: 6 mm odometry noise per frame at a hover must not count as distance flown."""
    w = World((-1, 6), (-5, 5), boxes=[(1.5, 1.7, -5, 5, WALL)])
    g = LocalGrid()
    rng = np.random.default_rng(0)
    cam = np.array([0.0, 0.0, 0.7])
    for _ in range(3):
        g.integrate(w.depth_points(cam, 0.0), cam, 0.7)
    for k in range(1800):                                                 # 60 s at 30 Hz, looking away from the wall
        c = cam + np.r_[rng.normal(0, 0.006, 2), 0.0]
        g.integrate(np.zeros((0, 3)), c, 0.7, 0.1 + k / 30)
    assert cell_value(g, 1.5, 0.0) >= g.p.l_occ                            # still remembered after a minute of jitter


def test_local_nav_passes_the_yaw_through_and_learns_the_push():
    """LocalNav.step(yaw=...) feeds MPPI's push estimate: in the corridor with the real drone's body-frame push it
    converges and the flight stays clear of the walls; without the yaw there is no estimate."""
    push = np.array([-0.056, 0.100])
    for use_yaw in (True, False):
        w = corridor_world()
        nav = LocalNav([(6.5, 0.0)], (0.0, 0.0), 0.0, nav=LocalNavParams(v_cruise=0.15))
        d = Drone(np.array([0.0, 0.0]), 0.0)
        dt = nav.mppi.p.dt
        ys = []
        for k in range(300):
            cam = d.cam()
            nav.add_depth(w.depth_points(cam, d.yaw), cam, d.z, k * dt)
            cmd = nav.step(d.x, d.v, k * dt, pose_stamp=k * dt, yaw=d.yaw if use_yaw else None)
            c, s = math.cos(d.yaw), math.sin(d.yaw)
            pw = np.array([c * push[0] - s * push[1], s * push[0] + c * push[1]])
            d.step((cmd.vx + pw[0], cmd.vy + pw[1]), cmd.yaw, dt)              # the push rides with the command
            ys.append(d.x[1])
        if use_yaw:
            assert np.hypot(*(nav.mppi.push.d - push)) < 0.02
            assert np.abs(ys[-100:]).max() < 0.05                             # centred despite the push
            assert max(np.abs(ys)) < 0.6 - 0.2 - nav.mppi.p.r_drone + 0.1     # never near a wall (inner face 0.6)
        else:
            assert not nav.mppi.push.d.any()
