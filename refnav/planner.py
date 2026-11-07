"""Reactive gap picking from one levelled depth frame: position is trusted, depth is not. PURE (numpy only).
Bearings are from the body heading, positive right, as in depth.clearance; only observed bearings count."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import depth as D

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .core import Params

MEMORY_EVERY_S = 0.25


@dataclass(frozen=True)
class Gap:
    bearing: float      # radians from the body heading, positive right
    clear_m: float      # usable clearance along it, after the range discount


class PointMemory:
    """Recent depth points in world (n, e) so an obstacle still counts after it leaves the camera's view."""

    def __init__(self, keep_s: float) -> None:
        self.keep_s = keep_s
        self.frames: list[tuple[float, np.ndarray]] = []

    def add(self, t: float, points: np.ndarray, pos: Sequence[float], heading: float) -> None:
        self.frames = [f for f in self.frames if t - f[0] <= self.keep_s]
        if len(points) == 0 or (self.frames and t - self.frames[-1][0] < MEMORY_EVERY_S):
            return
        c, s = math.cos(heading), math.sin(heading)
        world = points.copy()
        world[:, 0] = pos[0] + points[:, 0] * c - points[:, 1] * s
        world[:, 1] = pos[1] + points[:, 0] * s + points[:, 1] * c
        self.frames.append((t, world))

    def merged(self, points: np.ndarray, pos: Sequence[float], heading: float, half_view: float) -> np.ndarray:
        """The live points plus remembered ones the camera cannot currently see, in the live level frame."""
        if not self.frames:
            return points
        world = np.vstack([f[1] for f in self.frames])
        c, s = math.cos(heading), math.sin(heading)
        dn, de = world[:, 0] - pos[0], world[:, 1] - pos[1]
        fwd, right = dn * c + de * s, -dn * s + de * c
        unseen = (fwd <= 0) | (np.abs(right) > math.tan(half_view) * fwd)
        return np.vstack([points, np.column_stack([fwd, right, world[:, 2]])[unseen]])


def usable_clearance(clear_m: float, p: Params) -> float:
    r = min(clear_m, p.gap_look_m)
    return r - p.gap_trust * r * r


def choose_gap(frame: D.DepthFrame, points: np.ndarray, goal: float, last: float, p: Params) -> Gap | None:
    """Best observed, clear bearing in the fan off the nose, scored toward `goal`, or None. `last` is the
    bearing flown so far: the turn penalty stops the choice flipping between two similar gaps."""
    best, best_score = None, -math.inf
    n = int(p.gap_fan_deg // p.gap_step_deg)
    for i in range(-n, n + 1):
        b = math.radians(i * p.gap_step_deg)
        coverage, _ = frame.corridor_coverage(b, p.gap_half_m, p.band_m, p.gap_stop_m + p.lead_max_m)
        if coverage < p.min_corridor_coverage:
            continue
        clear = usable_clearance(D.clearance(points, b, p.gap_half_m, p.band_m, p.gap_min_hits), p)
        if clear < p.gap_stop_m:
            continue
        # (1 + cos) / 2 keeps the score positive when the goal is behind, so longer clearance always helps.
        score = clear * (1 + math.cos(b - goal)) / 2 - p.gap_turn_cost * abs(math.atan2(math.sin(b - last), math.cos(b - last)))
        if score > best_score:
            best, best_score = Gap(b, clear), score
    return best
