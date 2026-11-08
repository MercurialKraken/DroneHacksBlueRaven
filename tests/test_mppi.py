"""nav/mppi.py: weights, temperature, and closed-loop behaviour on synthetic corridors (pure numpy, no ROS)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from nav.course_map import CourseMap
from nav.mppi import Mppi, MppiParams

Z = 0.7


def corridor_map(width=1.2, length=8.0, pillar=None, cell=0.05):
    xs = np.arange(-1.0, length + 1.0 + 1e-9, cell)
    ys = np.arange(-1.5, 1.5 + 1e-9, cell)
    zs = np.arange(0.0, 1.5 + 1e-9, cell)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    d = width / 2.0 - np.abs(Y)                                  # walls at |y| = width / 2
    if pillar is not None:
        cx, cy, r = pillar
        d = np.minimum(d, np.hypot(X - cx, Y - cy) - r)
    F = np.repeat(d[:, :, None], len(zs), axis=2)
    return CourseMap({"planning": F}, origin=(xs[0], ys[0], zs[0]), cell=cell)


def straight_path(length=6.0, v=0.45, step=0.05):
    x = np.arange(0.0, length + 1e-9, step)
    P = np.c_[x, np.zeros_like(x)]
    ramp = 0.8                                                    # m to accelerate / brake
    vref = np.minimum.reduce([np.full_like(x, v), v * np.sqrt(np.clip(x / ramp, 0, 1)) + 0.05,
                              v * np.sqrt(np.clip((length - x) / ramp, 0, 1))])
    return P, vref


def fly(m: Mppi, x0=(0.0, 0.0), seconds=25.0, tau=0.35, depth=None):
    p = m.p
    x, v = np.array(x0, float), np.zeros(2)
    if depth is not None:
        m.set_depth_points(depth)
    traj, res = [x.copy()], []
    for _ in range(int(seconds / p.dt)):
        r = m.step(x, v)
        res.append(r)
        u = np.array(r.u)
        v = v + (u - v) * min(p.dt / tau, 1.0)                   # the same lag the controller assumes
        x = x + v * p.dt
        traj.append(x.copy())
        if r.s >= m.length - 0.05 and np.linalg.norm(v) < 0.03:
            break
    return np.array(traj), res


def test_weights_are_the_plain_softmax_when_gamma_is_zero():
    P, vref = straight_path()
    m = Mppi(corridor_map(), P, vref, Z)
    S = np.random.default_rng(1).normal(50.0, 10.0, 500)
    w, lam, ess = m._weights(S)
    ref = np.exp(-(S - S.min()) / lam)
    np.testing.assert_allclose(w, ref / ref.sum(), rtol=1e-9)
    assert w.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e4])
def test_temperature_hits_the_effective_sample_size_target_at_any_cost_scale(scale):
    P, vref = straight_path()
    m = Mppi(corridor_map(), P, vref, Z, MppiParams(ess_target=0.1))
    S = scale * np.random.default_rng(2).gamma(2.0, 3.0, 768)
    _, _, ess = m._weights(S)
    assert ess == pytest.approx(76.8, rel=0.1)


def test_follows_a_straight_corridor_centred_to_the_end():
    P, vref = straight_path()
    cm = corridor_map()
    m = Mppi(cm, P, vref, Z)
    traj, res = fly(m)
    assert res[-1].s >= m.length - 0.15
    assert np.max(np.abs(traj[:, 1])) < 0.10                     # stays on the centre line
    assert not any(r.blocked for r in res)
    clear = m.clearance(traj[None])[0]
    assert clear.min() > m.p.r_safe - 0.02


def test_goes_around_a_pillar_on_the_path_without_touching_it():
    P, vref = straight_path()
    cm = corridor_map(width=2.2, pillar=(3.0, 0.0, 0.12))         # gaps 0.98 m: a lane clear of r_safe exists
    m = Mppi(cm, P, vref, Z)
    traj, res = fly(m, seconds=30.0)
    assert res[-1].s >= m.length - 0.15
    d_pillar = np.hypot(traj[:, 0] - 3.0, traj[:, 1]) - 0.12
    assert d_pillar.min() > m.p.r_drone                          # never inside the collision radius
    assert np.max(np.abs(traj[:, 1])) > 0.25                     # it actually stepped aside


def test_depth_points_are_obstacles_even_where_the_map_is_free():
    P, vref = straight_path()
    cm = corridor_map(width=2.2)                                 # the map has no pillar ...
    blob = np.c_[np.full(9, 3.0) + np.linspace(-0.08, 0.08, 9), np.linspace(-0.08, 0.08, 9), np.full(9, Z)]
    m = Mppi(cm, P, vref, Z)
    traj, res = fly(m, seconds=30.0, depth=blob)                  # ... but depth sees one
    d = np.min(np.hypot(traj[:, None, 0] - blob[None, :, 0], traj[:, None, 1] - blob[None, :, 1]), axis=1)
    assert d.min() > m.p.r_drone
    assert res[-1].s >= m.length - 0.15


def test_every_candidate_colliding_brakes_and_says_blocked():
    P, vref = straight_path()
    m = Mppi(corridor_map(width=1.2), P, vref, Z)
    r = m.step((1.0, 0.7), (0.0, 0.0))                           # inside the wall: every rollout starts in it
    assert r.blocked and r.u == (0.0, 0.0)


def test_same_seed_same_commands():
    P, vref = straight_path()
    a = Mppi(corridor_map(), P, vref, Z, MppiParams(seed=7))
    b = Mppi(corridor_map(), P, vref, Z, MppiParams(seed=7))
    for _ in range(5):
        ra, rb = a.step((0.2, 0.05), (0.1, 0.0)), b.step((0.2, 0.05), (0.1, 0.0))
        assert ra.u == rb.u


def test_progress_never_runs_backwards():
    P, vref = straight_path()
    m = Mppi(corridor_map(), P, vref, Z)
    m.step((2.0, 0.0), (0.0, 0.0))
    s1 = m.s_now
    m.step((1.5, 0.0), (0.0, 0.0))                               # pushed back: progress stays
    assert m.s_now >= s1


def test_gamma_adds_the_control_cost_and_still_runs():
    P, vref = straight_path()
    m = Mppi(corridor_map(), P, vref, Z, MppiParams(gamma=0.5))
    traj, res = fly(m, seconds=20.0)
    assert res[-1].s > 3.0 and not any(r.blocked for r in res)
