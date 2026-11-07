"""onboard.vio.land_freeze: the vision pose freezes only while PX4 itself is landing near the floor."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onboard.vio.land_freeze import NAV_AUTO_LAND, NAV_DESCEND, LandFreeze  # noqa: E402

NAV_POSCTL, NAV_TAKEOFF, NAV_OFFBOARD = 2, 17, 14
SAMPLE = {'position': (1.0, 2.0, 0.2), 'velocity': (0.0, 0.0, 0.0)}


def test_never_latches_on_the_pad_or_in_offboard():
    f = LandFreeze(0.25)
    for nav in (NAV_TAKEOFF, NAV_OFFBOARD, NAV_POSCTL):
        assert f.update(nav, True, 0.1, 0.05, SAMPLE) is None
    assert f.update(NAV_OFFBOARD, True, 0.15, 0.05, SAMPLE) is None
    assert f.update(NAV_AUTO_LAND, False, 0.01, 0.05, SAMPLE) is None


def test_latches_in_auto_land_and_descend_near_the_floor():
    for nav in (NAV_AUTO_LAND, NAV_DESCEND):
        f = LandFreeze(0.25)
        assert f.update(nav, True, 0.6, 0.05, SAMPLE) is None
        assert f.update(nav, True, 0.2, 0.05, SAMPLE) is SAMPLE


def test_needs_a_fresh_range_and_a_sample():
    f = LandFreeze(0.25)
    assert f.update(NAV_AUTO_LAND, True, 0.2, 0.5, SAMPLE) is None
    assert f.update(NAV_AUTO_LAND, True, None, 0.05, SAMPLE) is None
    assert f.update(NAV_AUTO_LAND, True, 0.2, None, SAMPLE) is None
    assert f.update(NAV_AUTO_LAND, True, 0.2, 0.05, None) is None


def test_stays_latched_on_stale_or_jumping_range_and_keeps_the_first_sample():
    f = LandFreeze(0.25)
    f.update(NAV_AUTO_LAND, True, 0.2, 0.05, SAMPLE)
    later = {'position': (9.0, 9.0, 0.0)}
    assert f.update(NAV_AUTO_LAND, True, 0.2, 5.0, later) is SAMPLE
    assert f.update(NAV_AUTO_LAND, True, 3.0, 0.05, later) is SAMPLE
    assert f.update(NAV_AUTO_LAND, True, None, None, later) is SAMPLE
    assert f.update(NAV_DESCEND, True, 3.0, 0.05, later) is SAMPLE


def test_releases_on_disarm_and_on_leaving_the_landing_modes():
    f = LandFreeze(0.25)
    f.update(NAV_AUTO_LAND, True, 0.2, 0.05, SAMPLE)
    assert f.update(NAV_AUTO_LAND, False, 0.01, 0.05, SAMPLE) is None
    f.update(NAV_AUTO_LAND, True, 0.2, 0.05, SAMPLE)
    assert f.update(NAV_POSCTL, True, 0.2, 0.05, SAMPLE) is None
    assert f.update(NAV_OFFBOARD, True, 0.2, 0.05, SAMPLE) is None
