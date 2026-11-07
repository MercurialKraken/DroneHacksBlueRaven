"""onboard.vio.touchdown_hold: hold the last good pose only while landing and close to the floor."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onboard.vio.touchdown_hold import TouchdownHold  # noqa: E402


def descend(h, t0, ranges, poses):
    out = []
    for k, (r, p) in enumerate(zip(ranges, poses)):
        h.on_range(t0 + k * 0.1, r)
        out.append(h.filter(t0 + k * 0.1, p))
    return out


def test_cruise_dips_never_engage_without_a_landing():
    h = TouchdownHold()
    out = descend(h, 0.0, [0.85, 0.6, 0.4, 0.6, 0.85], ["a", "b", "c", "d", "e"])
    assert [held for _, held in out] == [False] * 5
    assert h.n["engaged"] == 0


def test_landing_holds_the_last_pose_seen_above_ref():
    h = TouchdownHold()
    h.landing = True
    out = descend(h, 0.0, [0.88, 0.82, 0.75, 0.65, 0.4, 0.1, 0.0, 0.0], ["p88", "p82", "p75", "p65", "p40", "p10", "p00", "p00b"])
    pubs = [p for p, _ in out]
    assert pubs[:3] == ["p88", "p82", "p75"]          # 0.75 is above engage (0.7): still live
    assert pubs[3:] == ["p82"] * 5                     # held at the last pose with range >= 0.8
    assert h.active and h.n["engaged"] == 1


def test_take_off_releases_the_hold_after_release_s():
    h = TouchdownHold(release_s=0.3)
    h.landing = True
    descend(h, 0.0, [0.9, 0.5, 0.0], ["a", "b", "c"])
    assert h.active
    h.landing = False
    out = descend(h, 1.0, [0.5, 0.95, 1.1, 1.3, 1.5, 1.7], ["d", "e", "f", "g", "h", "i"])
    assert [held for _, held in out][:3] == [True, True, True]
    assert out[-1] == ("i", False) and h.n["released"] == 1


def test_no_reference_means_no_hold():
    h = TouchdownHold()
    h.landing = True
    out = descend(h, 0.0, [0.5, 0.2, 0.0], ["a", "b", "c"])   # never saw a pose above ref_m
    assert all(not held for _, held in out) and not h.active
