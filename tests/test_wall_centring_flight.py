"""The skybeta mission with wall centring, flown by the toy drone down a corridor whose REAL walls stand
12 cm left of where the map puts them. Depth frames (5 Hz) are simulated from the real walls."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "skybeta" / "blueraven_skybeta"), str(HERE)]
from blueraven_skybeta.mission_core import MissionParams, SkybetaMission  # noqa: E402
import test_skybeta_mission as T  # noqa: E402
from test_wall_centring import CM, faces_of, sense  # noqa: E402

LAUNCH = {"x": -0.5, "y": 0.0, "yaw_deg": 0.0, "z": 0.07}
ROUTE = [(-0.5, 0.0), (5.5, 0.0)]
SHIFT = 0.12                                                    # the real corridor, built 12 cm to the left


def fly(centring: bool, shift=SHIFT, right_wall=True):
    p = MissionParams(mission="course", wall_centring=centring, cruise_mps=0.3)
    m = SkybetaMission(p, LAUNCH, ROUTE, CM)
    sim = T.Sim()
    T.LAUNCH = LAUNCH
    t, _ = T.run(m, sim, 0.0, 2.5)
    assert m.phase == "WAIT_OFFBOARD"
    sim.armed = sim.offboard = True
    faces = faces_of(0.5 + shift, (-0.5 + shift) if right_wall else None)
    lateral = []
    whys = []
    pending = []                                                # frames are delivered a tick after capture

    def hook(tt, s):
        if m.T_map_odom is None:
            return
        while pending:
            r = m.on_depth_points(tt, *pending.pop())
            whys.append(m.centring.last.why if (r is None and m.centring and m.centring.last) else r)
        x, y, z = (m.T_map_odom @ [s.x, s.y, s.z, 1.0])[:3]    # no drift: believed = true
        yaw = s.yaw + m.yaw_off
        if round(tt / T.DT) % 4 == 0 and m.phase == "FOLLOW":  # depth at 5 Hz, captured at this pose
            P = sense((x, y, yaw), (x, y, yaw), faces)
            c, sn = math.cos(-yaw), math.sin(-yaw)
            rel = P[:, :2] - [x, y]
            pending.append((tt, np.c_[c * rel[:, 0] - sn * rel[:, 1], sn * rel[:, 0] + c * rel[:, 1], P[:, 2] - z]))
        if m.phase == "FOLLOW" and 1.5 <= x <= 4.5:
            lateral.append(y - shift)                           # distance from the REAL centre line
    t, _ = T.run(m, sim, t, t + 60.0, hook=hook)
    assert m.phase == "DONE", m.why
    m.whys = whys
    return np.asarray(lateral), m


def test_without_centring_the_drone_flies_the_maps_centre_line():
    lat, _ = fly(False)
    assert np.mean(lat) == pytest.approx(-SHIFT, abs=0.02)      # 12 cm off the real centre


def test_with_centring_it_flies_the_real_centre():
    lat, m = fly(True)
    assert abs(np.mean(lat)) < 0.03 and np.max(np.abs(lat)) < SHIFT   # centred between the real walls
    assert m.whys.count("both walls") > 0.6 * len(m.whys)       # most frames saw both real walls
    assert m.status(0.0)["centring_m"] is not None


def test_one_real_wall_missing_keeps_the_mapped_distance_from_the_other():
    lat, _ = fly(True, shift=0.08, right_wall=False)            # right side open in reality (a gap)
    # y - shift = 0 means the mapped 0.48 m from the real left wall: the left wall alone is honoured
    assert abs(np.mean(lat)) < 0.03


def test_centring_is_off_unless_asked_and_inert_in_hover():
    assert MissionParams().wall_centring is False
    m = SkybetaMission(MissionParams(mission="hover", wall_centring=True), LAUNCH)
    assert m.centring is None and m.on_depth_points(0.0, 0.0, np.zeros((0, 3))) == "wall centring is off"
