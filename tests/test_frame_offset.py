"""blueraven_skybeta.frame_offset against the flight interface's own conventions (skybeta 0.3.3 frames.py):
yaw_ned = pi/2 - yaw_enu (+ offset), positions rotated by R_down(+offset) after enu_to_ned, twists NOT."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skybeta" / "blueraven_skybeta"))
from blueraven_skybeta.frame_offset import (circular_stats, offset_sample, twist_for_interface,  # noqa: E402
                                            yaw_enu_from_xyzw, yaw_ned_from_wxyz)


def enu_to_ned(v):                                     # the interface's frames.enu_to_ned
    return np.array([v[1], v[0], -v[2]])


def r_down(v_ned, a):                                  # setpoint_position_enu_to_ned's offset rotation
    c, s = math.cos(a), math.sin(a)
    return np.array([v_ned[0] * c - v_ned[1] * s, v_ned[0] * s + v_ned[1] * c, v_ned[2]])


def test_quaternion_yaws():
    a = math.radians(30.0)
    assert yaw_enu_from_xyzw(0, 0, math.sin(a / 2), math.cos(a / 2)) == pytest.approx(a)
    b = math.radians(90.0)                              # PX4: 90 deg about Down = facing east
    assert yaw_ned_from_wxyz(math.cos(b / 2), 0, 0, math.sin(b / 2)) == pytest.approx(b)


@pytest.mark.parametrize("delta_deg", [0.0, 7.5, -12.0, 170.0])
def test_offset_sample_recovers_a_constant_offset_at_any_heading(delta_deg):
    d = math.radians(delta_deg)
    got = []
    for psi in np.radians([-150, -40, 0, 75, 160]):    # the drone turned by hand to several headings
        px4 = (math.pi / 2 - psi) + d                    # PX4's heading = the interface's yaw_ned + offset
        got.append(offset_sample(psi, math.atan2(math.sin(px4), math.cos(px4))))
    mean, sd, n = circular_stats(got)
    assert n == 5 and sd < 1e-9
    assert math.degrees(math.atan2(math.sin(mean - d), math.cos(mean - d))) == pytest.approx(0.0, abs=1e-9)


def test_circular_stats_across_the_wrap():
    mean, sd, _ = circular_stats([math.radians(179), math.radians(-179)])
    assert abs(abs(math.degrees(mean)) - 180.0) < 1e-9 and math.degrees(sd) < 1.5


@pytest.mark.parametrize("delta_deg", [0.0, 5.0, -20.0])
def test_twists_land_in_ned_like_positions_do(delta_deg):
    d = math.radians(delta_deg)
    rng = np.random.default_rng(2)
    for _ in range(10):
        v = rng.uniform(-1, 1, 3)
        vx, vy = twist_for_interface(v[0], v[1], d)
        as_twist = enu_to_ned([vx, vy, v[2]])           # what the interface does to a twist
        as_position = r_down(enu_to_ned(v), d)          # what it does to a position
        assert np.allclose(as_twist, as_position)
    assert twist_for_interface(0.3, -0.1, 0.0) == (0.3, -0.1)    # the default is the identity
