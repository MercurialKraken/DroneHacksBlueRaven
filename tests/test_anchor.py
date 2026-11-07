"""onboard.vio.anchor: a re-anchored track continues from the last published pose."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onboard.vio.anchor import Anchor, yaw_of  # noqa: E402


def qz(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def test_identity_by_default():
    a = Anchor()
    p, q = a.apply((1.0, 2.0, 0.8), qz(0.3))
    assert p == (1.0, 2.0, 0.8) and abs(yaw_of(q) - 0.3) < 1e-12


def test_rebase_makes_the_jump_invisible_and_keeps_motion():
    a = Anchor()
    last_p, last_q = (1.0, 2.0, 0.8), qz(0.5)
    raw_p, raw_q = (790.0, 11031.0, 3.0), qz(0.5 + 0.94)        # the T4 garbage jump, position and yaw
    a.rebase(last_p, last_q, raw_p, raw_q)
    p, q = a.apply(raw_p, raw_q)
    assert math.dist(p, last_p) < 1e-9 and abs(yaw_of(q) - 0.5) < 1e-9
    # moving 0.1 m forward along the new track's own heading moves 0.1 m along the published heading
    h = yaw_of(raw_q)
    p2, _ = a.apply((raw_p[0] + 0.1 * math.cos(h), raw_p[1] + 0.1 * math.sin(h), raw_p[2]), raw_q)
    assert abs(math.dist(p2, p) - 0.1) < 1e-9
    assert abs(math.atan2(p2[1] - p[1], p2[0] - p[0]) - 0.5) < 1e-9
