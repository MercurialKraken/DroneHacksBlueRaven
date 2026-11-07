"""Continuous path following (refnav.follow) and the mission flying it."""
from __future__ import annotations

import math

import numpy as np
import pytest

from refnav import core as C
from refnav import follow as F
from test_refnav_core import ROUTE, fly

U_TURN = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0), (0.0, 2.0), (2.0, 2.0)]


def test_fillet_path_is_continuous_and_cuts_the_corner_by_the_arc() -> None:
    p = F.fillet_path([(0, 0), (2, 0), (2, 2)], 0.4)
    assert np.hypot(np.diff(p.x), np.diff(p.y)).max() < 0.021
    assert np.degrees(np.abs(F.wrap(np.diff(p.psi)))).max() < 3.0
    assert np.hypot(p.x - 2, p.y).min() == pytest.approx(0.4 * (math.sqrt(2) - 1), abs=0.01)
    assert p.k.max() == pytest.approx(1 / 0.4)                  # right turn in NED is +


def test_a_corner_with_little_room_is_a_tight_arc_flown_slowly() -> None:
    p = F.fillet_path([(0, 0), (1, 0), (1, 1)], 0.05)
    assert np.isfinite(p.k).all() and p.k.max() == pytest.approx(1 / 0.05)
    assert np.degrees(np.abs(F.wrap(np.diff(p.psi)))).max() < 25.0
    v = F.speed_profile(p, 0.5, math.radians(45), 0.5, 0.5)
    assert v[p.index(1.0)] <= math.radians(45) * 0.05 + 1e-9 and v[0] == 0.0 and v[-1] == 0.0


def test_the_speed_profile_keeps_every_bound() -> None:
    p = F.fillet_path(U_TURN, 0.5)
    v = F.speed_profile(p, 0.6, math.radians(45), 0.5, 0.3)
    k = np.abs(p.k)
    assert v.max() <= 0.6 + 1e-9 and v[0] == 0.0 and v[-1] == 0.0
    assert (v * k <= math.radians(45) + 1e-9).all() and (v * v * k <= 0.5 + 1e-9).all()
    assert (np.abs(np.diff(v ** 2)) / np.diff(p.s) <= 2 * 0.3 + 1e-6).all()


def test_the_speed_profile_changes_acceleration_smoothly_for_little_time() -> None:
    """Against the bang-bang profile (full acceleration, then full braking): far less jerk, little more time."""
    p = F.fillet_path(U_TURN, 0.5)
    v = F.speed_profile(p, 0.6, math.radians(45), 0.5, 0.3)
    top = np.minimum(0.6, np.sqrt(np.divide(math.radians(45) ** 2, p.k ** 2, out=np.full(len(p.k), np.inf), where=p.k != 0)))
    top = np.minimum(top, np.sqrt(np.divide(0.5, np.abs(p.k), out=np.full(len(p.k), np.inf), where=p.k != 0)))
    top[[0, -1]] = 0.0
    bang = np.sqrt(F._accel_limit(top ** 2, F.STEP_M, 0.3))

    def jerk_and_time(v):
        ds = np.diff(p.s)
        a = np.diff(v ** 2) / (2 * ds)
        vm = (v[1:] + v[:-1]) / 2
        return np.abs(np.diff(a) / ds[1:] * vm[1:]).max(), np.sum(ds / np.maximum(vm, F.MIN_SPEED))
    (j, t), (j0, t0) = jerk_and_time(v), jerk_and_time(bang)
    assert j < j0 / 4, (j, j0)
    assert t < t0 * 1.1, (t, t0)


def _follower(path):
    v = F.speed_profile(path, 0.5, math.radians(45), 0.5, 0.5)
    lim = F.Limits(accel=0.5, look_min=0.3, look_s=1.5, look_max=1.2, face_tol=math.radians(8), yaw_slow=math.radians(35),
                   xtrack_slow=0.12, xtrack_stop=0.35, arrive_m=0.15)
    return F.Follower(path, v, lim, [])


def _fly_lagging(path, fol, seconds=90.0, dt=0.02, tau=0.4, kp=1.0):
    """A vehicle like PX4's position loop: velocity = feed-forward + kp * error, reached with lag tau; yaw <= 45 deg/s."""
    pos, vel, yaw = np.array([path.x[0], path.y[0]]), np.zeros(2), float(path.psi[0])
    xt, s_seen, t = [], [], 0.0
    while t < seconds:
        c = fol.step(tuple(pos), tuple(vel), yaw, dt, math.inf)
        want = np.array(c.vel) + kp * (np.array(c.pos) - pos)
        vel += (want - vel) * dt / tau
        pos += vel * dt
        yaw += float(np.clip(F.wrap(c.yaw - yaw), -math.radians(45) * dt, math.radians(45) * dt))
        xt.append(c.xtrack)
        s_seen.append(fol.s_ref)
        t += dt
        if c.done:
            break
    return c, np.array(xt), np.array(s_seen), t


def test_a_lagging_vehicle_follows_a_u_turn_route_to_the_end() -> None:
    path = F.fillet_path(U_TURN, 0.4)
    c, xt, s, t = _fly_lagging(path, _follower(path))
    assert c.done, path.length
    assert xt.max() < 0.15
    assert (np.diff(s) >= -1e-12).all()                         # the reference never goes back
    assert t < 40.0


def test_the_projection_window_never_jumps_to_the_parallel_corridor() -> None:
    path = F.fillet_path(U_TURN, 0.4)
    fol = _follower(path)
    fol.s_hat = fol.s_ref = 1.0                                 # on the first corridor, y = 0
    fol.step((1.0, 0.95), (0.0, 0.0), 0.0, 0.02, math.inf)     # nearer the return corridor at y = 1
    assert fol.s_hat < 2.0


def test_the_reference_waits_at_a_tight_corner_until_the_nose_turns() -> None:
    path = F.fillet_path([(0, 0), (1, 0), (1, 1)], 0.05)
    fol = _follower(path)
    fol.s_ref = fol.s_hat = 0.95                                # where the arc starts
    c = fol.step((0.95, 0.0), (0.0, 0.0), 0.0, 0.1, math.inf)  # facing north, the path turns east
    held = fol.s_ref
    assert c.scale == 0.0 and held <= 0.95 + F.LEAD_M + 1e-9   # only the lead ahead of the aircraft
    assert c.yaw > math.radians(60)                             # but it is told to turn east
    c = fol.step((0.95, 0.0), (0.0, 0.0), c.yaw, 0.1, math.inf)
    assert c.scale == 1.0 and fol.s_ref > held


def test_the_reference_never_passes_the_observed_free_space() -> None:
    path = F.fillet_path([(0, 0), (3, 0)], 0.3)
    fol = _follower(path)
    for _ in range(200):
        c = fol.step(fol.path.point(fol.s_ref), (0.5, 0.0), 0.0, 0.05, 1.0)
    assert fol.s_ref <= 1.0 + 1e-9 and c.vel[0] < 0.05


def test_a_zero_length_leg_between_two_arcs_has_no_heading_jump() -> None:
    p = F.fillet_path([(0, 0), (2, 0), (2, 0.4), (4, 0.4)], 1.0)
    assert np.degrees(np.abs(F.wrap(np.diff(p.psi)))).max() < 10.0     # 0.2 m arcs turn ~6 deg per sample
    assert np.diff(p.s).max() <= F.STEP_M + 1e-9


def test_the_mission_flies_the_route_and_lands() -> None:
    m, px4, log = fly(C.Params(route=ROUTE))
    assert m.result == "complete", (m.result, m.why)
    assert "follow" in [p for _, p, *_ in log]
    assert np.hypot(px4.pos[0] - 3.0, px4.pos[1] + 3.0) < 0.3


def test_the_mission_streams_velocity_feed_forward() -> None:
    seen = {}

    def grab(t, m):
        if m.phase == "follow" and m.cmd is not None and np.hypot(*m.cmd.vel) > 0.1:
            seen["v"] = m.cmd.vel
    fly(C.Params(route=ROUTE), on_tick=grab)
    assert "v" in seen


def test_the_reference_never_falls_behind_the_aircraft() -> None:
    path = F.fillet_path([(0, 0), (3, 0)], 0.3)
    fol = _follower(path)
    fol.s_ref, fol.s_hat = 0.8, 0.9                             # the aircraft has overrun a slow reference
    fol.step((1.0, 0.0), (0.4, 0.0), math.pi / 2, 0.05, math.inf)    # facing away: the reference would not move
    assert fol.s_ref >= 1.0 + F.LEAD_M - F.STEP_M
    fol2 = _follower(path)
    fol2.s_ref, fol2.s_hat = 0.4, 0.9
    fol2.step((1.0, 0.0), (0.4, 0.0), math.pi / 2, 0.05, 0.5)  # ...but never into space depth has not seen
    assert fol2.s_ref <= 0.5 + 1e-9


def test_se2_correction_round_trips() -> None:
    c = F.SE2(0.3, -0.2, math.radians(5))
    n, e = c.inverse().point(*c.point(1.2, -0.7))
    assert (n, e) == pytest.approx((1.2, -0.7))
    assert c.vector(1.0, 0.0) == pytest.approx((math.cos(math.radians(5)), math.sin(math.radians(5))))


def test_a_drift_correction_moves_the_flown_track_not_the_path() -> None:
    """PX4 believes it is 0.3 m west of where the map says: the aircraft must fly the route 0.3 m west in PX4's frame."""
    m, px4, log = fly(C.Params(route=ROUTE), on_tick=lambda t, m: m.on_correction(t, 0.0, 0.3, 0.0))
    assert m.result == "complete", (m.result, m.why)
    assert px4.pos[1] == pytest.approx(-3.0 - 0.3, abs=0.15)


@pytest.mark.parametrize("turn_deg", [150.0, 160.0, 169.0])
def test_a_sharp_corner_still_passes_within_the_cut_of_its_point(turn_deg: float) -> None:
    a = math.radians(180.0 - turn_deg)
    p = F.fillet_path([(0, 0), (2, 0), (2 - 2 * math.cos(a), 2 * math.sin(a))], 0.3, 0.15)
    assert np.hypot(p.x - 2, p.y).min() <= 0.15 + 0.005             # the arc's sampling, not its geometry
    assert np.isfinite(p.k).all()


@pytest.mark.parametrize("back", [(0.0, 0.0), (0.0, 0.17)])         # 180 and ~175 deg
def test_a_reversal_reaches_its_point_and_stops_there(back: tuple[float, float]) -> None:
    p = F.fillet_path([(0, 0), (2, 0), back], 0.3, 0.15)
    apex = int(np.argmin(np.hypot(p.x - 2, p.y)))
    assert np.hypot(p.x[apex] - 2, p.y[apex]) < 1e-9
    assert p.length == pytest.approx(2.0 + math.hypot(2.0 - back[0], back[1]), abs=1e-6)
    v = F.speed_profile(p, 0.6, math.radians(45), 0.25, 0.5)
    assert v[apex] < 0.02 and v.max() > 0.4 and np.isfinite(v).all()


def test_polyline_path_drops_repeated_samples() -> None:
    x, y = np.array([0.0, 0.5, 1.0, 1.5]), np.array([0.0, 0.1, 0.1, 0.0])
    a = F.polyline_path(x, y)
    b = F.polyline_path(np.repeat(x, 2), np.repeat(y, 2))
    assert np.array_equal(a.s, b.s) and np.array_equal(a.k, b.k) and np.isfinite(b.k).all()
    with pytest.raises(ValueError):
        F.polyline_path(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))
