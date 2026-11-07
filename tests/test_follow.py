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


def test_no_room_for_an_arc_is_a_sharp_corner_with_zero_speed() -> None:
    p = F.fillet_path([(0, 0), (1, 0), (1, 1)], 0.0)
    assert p.corners == [pytest.approx(1.0)]
    v = F.speed_profile(p, 0.5, math.radians(45), 0.5, 0.5, 0.5)
    assert v[p.index(1.0)] == 0.0 and v[0] == 0.0 and v[-1] == 0.0


def test_speed_respects_turn_rate_and_braking() -> None:
    p = F.fillet_path(U_TURN, 0.5)
    v = F.speed_profile(p, 0.6, math.radians(45), 0.5, 0.5, 0.3)
    arc = np.abs(p.k) > 0                                          # arc samples, 0.5 m radius
    mid = arc & np.roll(arc, 6) & np.roll(arc, -6)                 # away from the ends, where speed blends in
    assert (v[mid] <= math.radians(45) * 0.5 + 1e-6).all()
    dv2 = np.diff(v ** 2) / F.STEP_M
    assert dv2.min() >= -2 * 0.3 - 1e-9 and dv2.max() <= 2 * 0.5 + 1e-9


def _follower(path):
    v = F.speed_profile(path, 0.5, math.radians(45), 0.5, 0.5, 0.5)
    lim = F.Limits(accel=0.5, look_min=0.3, look_s=1.5, look_max=1.2, face_tol=math.radians(8), yaw_slow=math.radians(35),
                   xtrack_slow=0.12, xtrack_stop=0.35, end_tol=0.15)
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


def test_the_reference_waits_at_a_sharp_corner_until_the_nose_faces_the_next_leg() -> None:
    path = F.fillet_path([(0, 0), (1, 0), (1, 1)], 0.0)
    fol = _follower(path)
    fol.s_ref = fol.s_hat = 1.0
    c = fol.step((1.0, 0.0), (0.0, 0.0), 0.0, 0.1, math.inf)   # facing north, the next leg goes east
    assert fol.s_ref == pytest.approx(1.0) and c.scale == 0.0
    assert c.yaw == pytest.approx(math.pi / 2, abs=0.05)        # but it is told to turn east
    fol.step((1.0, 0.0), (0.0, 0.0), math.pi / 2, 0.1, math.inf)
    assert fol.s_ref > 1.0


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


def test_a_small_kink_is_flown_through_slowly_not_stopped_at() -> None:
    p = F.fillet_path([(0, 0), (1, 0), (2, 0.07)], 0.0)            # a 4 deg kink with no room for an arc
    assert p.corners == []
    v = F.speed_profile(p, 0.5, math.radians(45), 0.5, 0.5, 0.5)
    assert v[p.index(1.0)] > 0.1
    q = F.fillet_path([(0, 0), (1, 0), (1, 1)], 0.0)               # a 90 deg corner still stops and turns
    assert q.corners == [pytest.approx(1.0)]
