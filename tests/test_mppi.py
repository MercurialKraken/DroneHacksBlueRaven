"""nav/mppi.py: weights, model, sampling limits, costs, and closed-loop behaviour on synthetic maps (numpy, no ROS)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from nav.course_map import CourseMap
from nav.mppi import Grid2D, Mppi, MppiParams, grid_from_course_map

Z = 0.7


def grid_map(dist_fn, x_range=(-1.0, 9.0), y_range=(-1.5, 1.5), cell=0.05):
    xs = np.arange(x_range[0], x_range[1] + 1e-9, cell)
    ys = np.arange(y_range[0], y_range[1] + 1e-9, cell)
    zs = np.arange(0.0, 1.5 + 1e-9, cell)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    F = np.repeat(dist_fn(X, Y)[:, :, None], len(zs), axis=2)
    return grid_from_course_map(CourseMap({"planning": F}, origin=(xs[0], ys[0], zs[0]), cell=cell), Z)


def corridor_map(width=1.2, pillar=None):
    def d(X, Y):
        out = width / 2.0 - np.abs(Y)                            # walls at |y| = width / 2
        if pillar is not None:
            cx, cy, r = pillar
            out = np.minimum(out, np.hypot(X - cx, Y - cy) - r)
        return out
    return grid_map(d)


def box_inside(X, Y, x0, x1, y0, y1):
    return np.minimum.reduce([X - x0, x1 - X, Y - y0, y1 - Y])


def l_corridor_map(half=0.6):
    """Free space: a corridor east along y=0 to x=4, then north along x=4 (pessimistic distance at the junction)."""
    return grid_map(lambda X, Y: np.maximum(box_inside(X, Y, -1.0, 4 + half, -half, half),
                                            box_inside(X, Y, 4 - half, 4 + half, -half, 5.0)),
                    x_range=(-1.0, 5.5), y_range=(-1.0, 5.5))


def profile(s, length, v, ramp=0.8):
    return np.minimum.reduce([np.full_like(s, v), v * np.sqrt(np.clip(s / ramp, 0, 1)) + 0.05,
                              v * np.sqrt(np.clip((length - s) / ramp, 0, 1))])


def straight_path(length=6.0, v=0.45, step=0.05):
    x = np.arange(0.0, length + 1e-9, step)
    return np.c_[x, np.zeros_like(x)], profile(x, length, v)


def l_path(v=0.45, v_corner=0.25, step=0.05):
    a = np.arange(0.0, 4.0, step)
    b = np.arange(0.0, 4.0 + 1e-9, step)
    P = np.r_[np.c_[a, np.zeros_like(a)], np.c_[np.full_like(b, 4.0), b]]
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])))]
    vref = profile(s, s[-1], v)
    vref = np.where(np.abs(s - 4.0) < 0.6, np.minimum(vref, v_corner), vref)
    return P, vref


def fly(m: Mppi, x0=(0.0, 0.0), seconds=25.0, depth=None):
    """Closed loop against the same first-order lag the controller assumes."""
    p = m.p
    a = math.exp(-p.dt / p.tau)
    b = p.tau * (1.0 - a)
    x, v = np.array(x0, float), np.zeros(2)
    if depth is not None:
        m.set_depth_points(depth)
    traj, res = [x.copy()], []
    for _ in range(int(seconds / p.dt)):
        r = m.step(x, v)
        res.append(r)
        u = np.array(r.u)
        x = x + u * p.dt + (v - u) * b
        v = u + (v - u) * a
        traj.append(x.copy())
        if r.s >= m.length - 0.05 and np.linalg.norm(v) < 0.03:
            break
    return np.array(traj), res


# ---------------------------------------------------------------------------------------------- weights
def test_weights_are_the_plain_softmax():
    P, vref = straight_path()
    m = Mppi(corridor_map(), P, vref)
    S = np.random.default_rng(1).normal(50.0, 10.0, 500)
    w, lam, ess = m.weights(S)
    ref = np.exp(-(S - S.min()) / lam)
    np.testing.assert_allclose(w, ref / ref.sum(), rtol=1e-9)
    assert w.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e4])
def test_temperature_hits_the_effective_sample_size_target_at_any_cost_scale(scale):
    P, vref = straight_path()
    m = Mppi(corridor_map(), P, vref, MppiParams(ess_target=0.1))
    S = scale * np.random.default_rng(2).gamma(2.0, 3.0, 1024)
    _, _, ess = m.weights(S)
    assert ess == pytest.approx(102.4, rel=0.1)


# ---------------------------------------------------------------------------------------------- model
def test_rollout_is_the_exact_first_order_lag_solution():
    P, vref = straight_path()
    m = Mppi(corridor_map(), P, vref)
    p = m.p
    u, v0, x0 = np.array([0.5, -0.2]), np.array([-0.1, 0.3]), np.array([1.0, 0.2])
    X, Vel = m.rollout(x0, v0, np.broadcast_to(u, (1, p.horizon, 2)).copy())
    t = p.dt * p.horizon
    v_true = u + (v0 - u) * math.exp(-t / p.tau)
    x_true = x0 + u * t + (v0 - u) * p.tau * (1.0 - math.exp(-t / p.tau))
    np.testing.assert_allclose(Vel[0, -1], v_true, atol=1e-12)
    np.testing.assert_allclose(X[0, -1], x_true, atol=1e-12)


# ---------------------------------------------------------------------------------------------- sampling
def test_samples_respect_the_change_limit_forward_only_and_speed_caps():
    P, vref = straight_path()
    m = Mppi(corridor_map(), P, vref)
    p = m.p
    m.U[:] = np.array([0.4, 0.0])
    u_last = np.array([0.3, 0.1])
    pr, vr, tg, nm = m.reference(1.0)
    targets = np.stack([m.U, np.zeros_like(m.U), vr])
    V = m.sample(u_last, targets, tg, nm, 2000)
    dV = np.diff(np.concatenate([np.broadcast_to(u_last, (len(V), 1, 2)), V], axis=1), axis=1)
    assert np.linalg.norm(dV, axis=-1).max() <= p.a_cmd * p.dt + 1e-9      # straight path: frames don't turn
    assert (V[..., 0] >= -1e-12).all()                                    # never backwards along the path
    assert np.linalg.norm(V, axis=-1).max() <= p.v_max + 1e-9
    assert np.abs(V[..., 1]).max() <= p.v_lat + 1e-9
    spread = V[3:, -1, 1].std()                                           # Brownian: grows along the horizon
    assert spread > 2.0 * V[3:, 2, 1].std()


def test_brake_candidate_ramps_down_under_the_change_limit():
    P, vref = straight_path()
    m = Mppi(corridor_map(), P, vref)
    p = m.p
    pr, vr, tg, nm = m.reference(1.0)
    V = m.sample(np.array([0.6, 0.0]), np.zeros((1, p.horizon, 2)), tg, nm, 1)
    brake = V[0, :, 0]
    np.testing.assert_allclose(brake[:4], [0.45, 0.30, 0.15, 0.0], atol=1e-9)
    assert (brake[4:] == 0).all()


def test_reference_is_re_anchored_and_follows_the_speed_profile():
    P, vref = straight_path()
    m = Mppi(corridor_map(), P, vref)
    pr, vr, tg, nm = m.reference(2.0)                                    # cruise section: 0.45 m/s
    assert pr[0, 0] == pytest.approx(2.0 + 0.45 * m.p.dt, abs=1e-6)
    np.testing.assert_allclose(vr[:10, 0], 0.45, atol=1e-6)
    pr_end, vr_end, _, _ = m.reference(m.length)                          # at the end it holds the end point
    np.testing.assert_allclose(pr_end[-1], P[-1], atol=1e-9)
    np.testing.assert_allclose(vr_end, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------------------------- costs
def test_a_crash_costs_by_how_early_and_nothing_after_it_counts():
    P, vref = straight_path()
    m = Mppi(corridor_map(width=1.2), P, vref)
    p = m.p
    H = p.horizon
    ref = tuple(np.asarray(a) for a in m.reference(1.0))
    X = np.broadcast_to(ref[0], (3, H, 2)).copy()
    X[1, 10:, 1] = 0.5                                                   # inside the wall's radius from step 10
    X[2, 20:, 1] = 0.5                                                   # ... from step 20
    Vel = np.broadcast_to(ref[1], (3, H, 2)).copy()
    V = Vel.copy()
    c = m.costs(X, Vel, V, V[0, 0], ref)
    assert not bool(c["crashed"][0]) and bool(c["crashed"][1]) and bool(c["crashed"][2])
    assert int(c["first"][1]) == 10 and int(c["first"][2]) == 20
    S = c["S"]
    assert S[1] > S[2] > S[0]
    # the crash term is exactly crash * (H - t); the steps before it are identical to the clean rollout's
    pre1 = S[1] - p.crash * (H - 10)
    pre2 = S[2] - p.crash * (H - 20)
    assert pre1 < pre2 < S[0] + 1e-9


def test_outside_the_map_counts_as_a_crash():
    P, vref = straight_path()
    m = Mppi(corridor_map(), P, vref)
    assert float(m.clearance(np.array([[50.0, 0.0]]))[0]) == -1.0


def test_clearance_is_the_map_slice_min_depth():
    P, vref = straight_path()
    m = Mppi(corridor_map(width=2.0), P, vref)
    q = np.array([[3.0, 0.0]])
    assert float(m.clearance(q)[0]) == pytest.approx(1.0, abs=1e-6)
    m.set_depth_points(np.array([[3.4, 0.0, Z]]))
    assert float(m.clearance(q)[0]) == pytest.approx(0.4, abs=0.05)
    assert float(m.clearance(q)[0]) <= 0.4 + 1e-9                         # rasterising never looks further
    m.set_depth_points(None)
    assert float(m.clearance(q)[0]) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------------------------- closed loop
def test_follows_a_straight_corridor_centred_to_the_end():
    P, vref = straight_path()
    m = Mppi(corridor_map(), P, vref)
    traj, res = fly(m)
    assert res[-1].s >= m.length - 0.15
    assert np.max(np.abs(traj[:, 1])) < 0.10
    assert not any(r.blocked for r in res)
    assert m.clearance(traj).min() > m.p.r_drone + 0.15


def test_goes_around_a_pillar_on_the_path_without_touching_it():
    P, vref = straight_path()
    m = Mppi(corridor_map(width=2.2, pillar=(3.0, 0.0, 0.12)), P, vref)   # gaps 0.98 m each side
    traj, res = fly(m, seconds=35.0)
    d_pillar = np.hypot(traj[:, 0] - 3.0, traj[:, 1]) - 0.12
    assert d_pillar.min() > m.p.r_drone
    assert np.max(np.abs(traj[:, 1])) > 0.25                              # it actually stepped aside
    assert res[-1].s >= m.length - 0.15


def test_depth_points_are_obstacles_even_where_the_map_is_free():
    P, vref = straight_path()
    blob = np.c_[3.0 + np.linspace(-0.08, 0.08, 9), np.linspace(-0.08, 0.08, 9), np.full(9, Z)]
    m = Mppi(corridor_map(width=2.2), P, vref)                         # the map has no pillar, depth sees one
    traj, res = fly(m, seconds=35.0, depth=blob)
    d = np.min(np.hypot(traj[:, None, 0] - blob[None, :, 0], traj[:, None, 1] - blob[None, :, 1]), axis=1)
    assert d.min() > m.p.r_drone
    assert res[-1].s >= m.length - 0.15


def test_turns_a_corner_without_touching_the_walls():
    P, vref = l_path()
    m = Mppi(l_corridor_map(), P, vref)
    traj, res = fly(m, seconds=40.0)
    assert res[-1].s >= m.length - 0.15
    assert m.clearance(traj).min() > m.p.r_drone
    assert not any(r.blocked for r in res)


def test_commands_obey_the_change_limit_in_closed_loop():
    P, vref = straight_path()
    m = Mppi(corridor_map(width=2.2, pillar=(3.0, 0.0, 0.12)), P, vref)
    traj, res = fly(m, seconds=12.0)
    U = np.array([r.u for r in res])
    assert np.linalg.norm(np.diff(U, axis=0), axis=1).max() <= m.p.a_cmd * m.p.dt + 1e-6
    assert np.linalg.norm(U, axis=1).max() <= m.p.v_max + 1e-9


def test_every_candidate_crashing_stops_and_says_blocked():
    P, vref = straight_path()
    m = Mppi(corridor_map(width=1.2), P, vref)
    r = m.step((1.0, 0.7), (0.0, 0.0))                                   # inside the wall: every rollout starts in it
    assert r.blocked and r.u == (0.0, 0.0)


def test_same_seed_same_commands():
    P, vref = straight_path()
    a = Mppi(corridor_map(), P, vref, MppiParams(seed=7))
    b = Mppi(corridor_map(), P, vref, MppiParams(seed=7))
    for _ in range(5):
        ra, rb = a.step((0.2, 0.05), (0.1, 0.0)), b.step((0.2, 0.05), (0.1, 0.0))
        assert ra.u == rb.u


def test_progress_never_runs_backwards():
    P, vref = straight_path()
    m = Mppi(corridor_map(), P, vref)
    m.step((2.0, 0.0), (0.0, 0.0))
    s1 = m.s_now
    m.step((1.5, 0.0), (0.0, 0.0))                                       # pushed back: progress stays
    assert m.s_now >= s1


def test_signed_lateral_error_is_positive_on_the_left():
    P, vref = straight_path()
    m = Mppi(corridor_map(), P, vref)
    assert m.step((1.0, 0.2), (0.0, 0.0)).e_lat == pytest.approx(0.2, abs=1e-9)
    m.reset()
    assert m.step((1.0, -0.2), (0.0, 0.0)).e_lat == pytest.approx(-0.2, abs=1e-9)


def test_runs_on_cupy_when_available():
    cp = pytest.importorskip("cupy")
    P, vref = straight_path()
    m = Mppi(corridor_map(), P, vref, xp=cp)
    r = m.step((0.5, 0.0), (0.2, 0.0))
    assert np.isfinite(r.u).all() and not r.blocked


def test_set_path_restarts_progress_and_keeps_the_warm_start():
    P, vref = straight_path()
    m = Mppi(corridor_map(), P, vref)
    for _ in range(5):
        m.step((1.0, 0.0), (0.3, 0.0))
    U_before, u_before = m.U.copy(), m.u_last.copy()
    Q = np.c_[np.linspace(1.0, 5.0, 81), np.full(81, 0.2)]               # a new, shifted centerline
    m.set_path(Q, np.full(81, 0.45))
    assert m.s_now is None and m.length == pytest.approx(4.0)
    np.testing.assert_array_equal(m.U, U_before)
    np.testing.assert_array_equal(m.u_last, u_before)
    r = m.step((1.0, 0.0), (0.3, 0.0))
    assert r.s == pytest.approx(0.0, abs=1e-9) and r.e_lat == pytest.approx(-0.2, abs=1e-9)


def test_grid_can_be_swapped_every_cycle():
    P, vref = straight_path()
    m = Mppi(corridor_map(width=2.2), P, vref)
    q = np.array([[3.0, 0.0]])
    assert float(m.clearance(q)[0]) == pytest.approx(1.1, abs=1e-6)
    m.set_grid(Grid2D(np.full((10, 10), 0.3), (2.5, -0.5), 0.1))
    assert float(m.clearance(q)[0]) == pytest.approx(0.3)
    assert float(m.clearance(np.array([[9.0, 0.0]]))[0]) == -1.0          # off the new grid


def test_dead_time_plans_from_the_predicted_state():
    """With dead time, MPPI rolls the commands already in flight forward first: same commands, later start."""
    P, vref = straight_path()
    m = Mppi(corridor_map(), P, vref, MppiParams(dead_time_s=0.3))
    r1 = m.step((1.0, 0.0), (0.3, 0.0))
    assert len(m._inflight) == 1                                          # this cycle's command is now remembered
    for _ in range(4):
        m.step((1.0, 0.0), (0.3, 0.0))
    assert len(m._inflight) == 5
    assert r1.plan[0, 0] > 1.0 + 0.3 * 0.1                                # the plan starts ahead of the drone
    r_old = m.step((1.0, 0.0), (0.3, 0.0), pose_age_s=0.3)                # a 0.3 s old pose: predicted further on
    r_new = m.step((1.0, 0.0), (0.3, 0.0), pose_age_s=0.0)
    assert r_old.plan[0, 0] > r_new.plan[0, 0]
    m.reset()
    assert m._inflight == []


def test_pose_age_is_capped_and_never_negative():
    """A stamp on the wrong clock must not make MPPI predict seconds ahead (capped at max_pose_age_s), and a stamp
    from the future means no age at all."""
    P, vref = straight_path()

    def plan_start(age):
        m = Mppi(corridor_map(), P, vref, MppiParams(seed=5))
        for _ in range(8):
            m.step((1.0, 0.0), (0.3, 0.0))
        return m.step((1.0, 0.0), (0.3, 0.0), pose_age_s=age).plan[0, 0]
    assert plan_start(5.0) == pytest.approx(plan_start(0.5))
    assert plan_start(-1.0) == pytest.approx(plan_start(0.0))
    assert plan_start(0.5) > plan_start(0.0)



def test_pose_age_is_reported_raw_with_a_clamp_flag():
    P, vref = straight_path()
    m = Mppi(corridor_map(), P, vref)
    r = m.step((1.0, 0.0), (0.3, 0.0), pose_age_s=2.0)                    # e.g. a stamp on the wrong clock
    assert r.pose_age == 2.0 and r.pose_age_clamped
    r = m.step((1.0, 0.0), (0.3, 0.0), pose_age_s=0.04)
    assert r.pose_age == 0.04 and not r.pose_age_clamped



def test_inside_the_crash_radius_it_escapes_instead_of_freezing():
    """Hinson: pushed inside r_drone + r_buffer, every rollout used to crash at step 0 (BLOCKED, no recovery)."""
    P, vref = straight_path()
    m = Mppi(corridor_map(width=1.2), P, vref)                             # walls at y = +-0.6
    y0 = 0.6 - (m.p.r_drone + m.p.r_buffer) + 0.02                         # 2 cm inside the crash radius of y = +0.6
    r = m.step((2.0, y0), (0.0, 0.0))
    assert not r.blocked
    x, v = np.array([2.0, y0]), np.zeros(2)
    a = math.exp(-m.p.dt / m.p.tau)
    b = m.p.tau * (1 - a)
    for _ in range(30):
        r = m.step(x, v)
        u = np.array(r.u)
        x = x + u * m.p.dt + (v - u) * b
        v = u + (v - u) * a
    assert x[1] < y0 - 0.02                                               # it moved away from the wall


# ---------------------------------------------------------------------------------------------- steady push
PUSH = (-0.056, 0.100)            # skybeta-09 route 1 (flight 160449): body frame (forward, left), m/s


def fly_pushed(m: Mppi, push=PUSH, heading=0.0, seconds=20.0, noise=0.0, gap_every=0, use_yaw=True, wall=None,
               seed=0):
    """Closed loop against the T4 plant MPPI assumes (lag + dead time) plus a steady BODY-frame push.
    noise: position noise (m); gap_every: every n-th cycle re-sends the previous pose (an odometry gap);
    wall: (t, dx): at t a wall appears dx ahead of the drone (on top of the map)."""
    p = m.p
    rng = np.random.default_rng(seed)
    a = math.exp(-p.dt / p.tau)
    b = p.tau * (1.0 - a)
    n_dead = int(round(p.dead_time_s / p.dt))
    c, s = math.cos(heading), math.sin(heading)
    pw = np.array([[c, -s], [s, c]]) @ np.asarray(push, float)
    x, v, q, meas = np.zeros(2), np.zeros(2), [], None
    rows, wall_x = [], None
    for k in range(int(seconds / p.dt)):
        t = k * p.dt
        if wall is not None and wall_x is None and t >= wall[0]:
            wall_x = float(x[0] + wall[1])
            g = m._map_grid
            X = m.g_origin[0] + m.g_cell * np.arange(g.shape[0])[:, None] + np.zeros(g.shape)
            m.set_grid(Grid2D(np.minimum(g, wall_x - X), m.g_origin, m.g_cell))
        if meas is None or not (gap_every and k % gap_every == 0):
            meas = x + rng.normal(0.0, noise, 2)
        r = m.step(meas, v, yaw=heading if use_yaw else None, t=t)
        q.append(np.array(r.u))
        ua = q[-1 - n_dead] if len(q) > n_dead else np.zeros(2)
        tgt = ua + pw
        x = x + tgt * p.dt + (v - tgt) * b
        v = tgt + (v - tgt) * a
        rows.append((t, x[0], x[1], -s * x[0] + c * x[1], *m.push.d, *r.dist))
    return np.array(rows), wall_x


def pushed_mppi(v=0.15, heading=0.0, grid=None, length=8.0):
    sp = np.arange(0.0, length + 1e-9, 0.05)
    c, s = math.cos(heading), math.sin(heading)
    P = np.c_[sp * c, sp * s]
    vref = np.minimum(v, np.minimum(sp / 0.5 + 0.05, (length - sp) / 0.5))
    grid = grid or Grid2D(np.full((300, 300), 3.0), (-6.0, -6.0), 0.05)
    return Mppi(grid, P, vref, MppiParams(tau=0.9, dead_time_s=0.6, samples=256))


@pytest.mark.parametrize("heading", [0.0, 1.0])
def test_a_steady_body_push_is_learnt_and_the_offset_it_causes_goes(heading):
    """The real drone's back-left push: the estimate converges to it within a few seconds (in the body frame, at any
    heading) and the steady cross-track offset it caused (18 cm at 0.15 m/s) all but goes."""
    on, _ = fly_pushed(pushed_mppi(heading=heading), heading=heading)
    off, _ = fly_pushed(pushed_mppi(heading=heading), heading=heading, use_yaw=False)
    err = np.hypot(on[:, 4] - PUSH[0], on[:, 5] - PUSH[1])
    assert err[on[:, 0] >= 6.0].max() < 0.03 and err[-1] < 0.01
    c, s = math.cos(heading), math.sin(heading)
    np.testing.assert_allclose(on[-1, 6:8], [c * PUSH[0] - s * PUSH[1], s * PUSH[0] + c * PUSH[1]], atol=0.01)
    steady_on = np.abs(on[on[:, 0] >= 12.0, 3]).mean()
    steady_off = np.abs(off[off[:, 0] >= 12.0, 3]).mean()
    assert steady_off > 0.10 and steady_on < 0.03 and steady_on < 0.3 * steady_off
    assert not off[:, 4:8].any()                                           # no yaw: no estimate, no bias


def test_no_push_gives_a_near_zero_estimate_with_noisy_gappy_odometry():
    rows, _ = fly_pushed(pushed_mppi(), push=(0.0, 0.0), noise=0.005, gap_every=4)
    assert np.hypot(rows[:, 4], rows[:, 5]).max() < 0.02


def test_without_yaw_the_commands_are_exactly_those_without_the_estimator():
    P, vref = straight_path()
    m1, m2 = Mppi(corridor_map(), P, vref), Mppi(corridor_map(), P, vref)
    x, v = np.zeros(2), np.zeros(2)
    a = math.exp(-m1.p.dt / m1.p.tau)
    b = m1.p.tau * (1 - a)
    for k in range(40):
        r1 = m1.step(x, v)
        r2 = m2.step(x, v, yaw=None, t=123.0 + 0.1 * k)
        assert r1.u == r2.u and not r2.dist.any()
        u = np.array(r1.u)
        x = x + u * m1.p.dt + (v - u) * b
        v = u + (v - u) * a
    assert m2.push.n_fit == 0


def test_the_push_estimate_skips_repeated_and_older_poses():
    from nav.mppi import PushEstimator
    e = PushEstimator(MppiParams())
    assert e.add_pose(1.0, (0.0, 0.0), 0.0)
    assert not e.add_pose(1.0, (0.1, 0.0), 0.0)                            # same stamp
    assert not e.add_pose(0.9, (0.1, 0.0), 0.0)                            # older
    assert not e.add_pose(1.1, (0.0, 0.0), 0.0)                            # the very same position: a re-sent pose
    assert e.add_pose(1.1, (0.001, 0.0), 0.0)
    for k in range(100):
        e.add_cmd(0.1 * k, (0.1, 0.0))
    assert e.cmds[0][0] <= 9.9 - (e.p.dist_window_s + e.p.dead_time_s + 1.0) and len(e.cmds) < 60


def test_braking_under_a_push_stops_clear_of_a_wall_and_holds_off_the_side_wall():
    """0.9 m corridor, the push towards its left wall; a wall appears 0.8 m ahead. Braking means coming to REST
    (commanding against the push), so the drone stops clear of the new wall and does not drift into the side wall."""
    xs, ys = -1.0 + 0.05 * np.arange(240), -1.0 + 0.05 * np.arange(41)
    side = 0.45 - np.abs(ys)[None, :] + np.zeros((len(xs), 1))
    m = pushed_mppi(grid=Grid2D(side, (-1.0, -1.0), 0.05), length=10.0)
    rows, wall_x = fly_pushed(m, seconds=25.0, wall=(8.0, 0.8))
    clr = np.minimum(0.45 - np.abs(rows[:, 2]), np.where(rows[:, 0] >= 8.0, wall_x - rows[:, 1], 9.0))
    assert clr.min() >= m.p.r_drone
    assert np.ptp(rows[-50:, 1]) < 0.02 and np.ptp(rows[-50:, 2]) < 0.02     # held still for the last 5 s


def test_the_push_is_learnt_during_a_hover_flown_by_someone_else():
    """observe(): refnav holds the hover (a position loop, 0.5/s) against the push; its poses and the commands it
    sent teach MPPI the push before it takes over, so MPPI starts with it."""
    m = pushed_mppi()
    p = m.p
    a = math.exp(-p.dt / p.tau)
    b = p.tau * (1.0 - a)
    pw, x, v, q = np.asarray(PUSH, float), np.zeros(2), np.zeros(2), []
    for k in range(80):                                                    # 8 s of hover, heading 0
        u = 0.5 * -x
        u = u * min(1.0, 0.25 / max(float(np.hypot(*u)), 1e-9))
        m.observe(x, 0.0, u, t=0.1 * k)
        q.append(u)
        ua = q[-7] if len(q) >= 7 else np.zeros(2)
        x = x + (ua + pw) * p.dt + (v - ua - pw) * b
        v = ua + pw + (v - ua - pw) * a
    assert np.hypot(*(m.push.d - pw)) < 0.02
    r = m.step(x, v, yaw=0.0, t=8.0)                                        # MPPI's first cycle already leans on it
    np.testing.assert_allclose(r.dist, pw, atol=0.02)
