"""Map-free navigation: straight lines through coarse waypoints, MPPI on a live depth grid. Pure (numpy; no ROS).

Frame: the odometry (take-off) frame, metres. Waypoints are rough by design: the organizers withhold the course's
dimensions, so a line may clip a wall that moved or slants. MPPI's obstacle cost outweighs its pull towards the line,
so it slides along such a wall and keeps making progress.

* Reference: the polyline start -> waypoint 1 -> ... -> last waypoint, densified. Speed: cruise on the straights,
  v_corner within corner_slow_m of a waypoint where the line turns by more than corner_deg, braking (a_stop) only
  into the last waypoint.
* Progress: the projection onto that polyline (nav.mppi: monotone, searched only in a window around the current
  progress, so it can't jump to a later leg that passes nearby).
* Obstacles: nav.local_grid from depth (add_depth), handed to MPPI as its clearance grid whenever it changed.
* Heading: along the line yaw_lookahead_m ahead of the drone's progress, rate-limited, so the camera looks into a
  turn before the drone gets there.
* State (the caller acts on it):
  WAIT     fewer than min_depth_frames depth frames so far: zero velocity (never move on an empty grid);
  GO       flying;
  SLOW     no depth for depth_stale_s (the dark tunnel, a dropout): the speed profile is scaled to v_no_depth and
           MPPI flies on the remembered grid and the line. Never a stop on its own: odometry loss is the shell's guard;
  ARRIVED  within accept_m of the last waypoint and slower than arrive_speed: the caller lands;
  BLOCKED  every MPPI candidate crashes: zero velocity; the caller holds, then lands;
  STUCK    progress along the line grew less than stuck_m in stuck_s: the caller holds, then lands.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from nav.local_grid import LocalGrid, LocalGridParams
from nav.mppi import Mppi, MppiParams, MppiResult


@dataclass(frozen=True)
class LocalNavParams:
    v_cruise: float = 0.45
    v_corner: float = 0.25
    corner_deg: float = 30.0
    corner_slow_m: float = 0.6
    a_stop: float = 0.4
    step_m: float = 0.05
    accept_m: float = 0.25
    arrive_speed: float = 0.10
    yaw_lookahead_m: float = 0.6
    yaw_rate_dps: float = 45.0
    min_depth_frames: int = 5
    depth_stale_s: float = 0.5
    v_no_depth: float = 0.20
    stuck_s: float = 8.0
    stuck_m: float = 0.10


def waypoint_path(start: Any, waypoints: Sequence[Sequence[float]], p: LocalNavParams) -> tuple[np.ndarray, np.ndarray]:
    """Densified polyline start -> waypoints with its speed profile."""
    V = np.r_[np.asarray(start, float)[None, :2], np.asarray(waypoints, float).reshape(-1, 2)]
    keep = np.r_[True, np.hypot(*np.diff(V, axis=0).T) > 1e-6]
    V = V[keep]
    if len(V) < 2:
        raise ValueError("need at least one waypoint away from the start")
    pts, s_corner = [V[:1]], []
    s_acc = 0.0
    for a, b in zip(V[:-1], V[1:]):
        L = float(np.hypot(*(b - a)))
        n = max(1, int(math.ceil(L / p.step_m)))
        t = np.linspace(0.0, 1.0, n + 1)[1:]
        pts.append(a + t[:, None] * (b - a))
        s_acc += L
        s_corner.append(s_acc)
    P = np.concatenate(pts)
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    v = np.full(len(P), p.v_cruise)
    for k in range(1, len(V) - 1):                                 # interior waypoints: slow if the line turns
        d1, d2 = V[k] - V[k - 1], V[k + 1] - V[k]
        turn = abs(math.degrees(math.atan2(d1[0] * d2[1] - d1[1] * d2[0], float(np.dot(d1, d2)))))
        if turn > p.corner_deg:
            v = np.where(np.abs(s - s_corner[k - 1]) <= p.corner_slow_m, np.minimum(v, p.v_corner), v)
    v = np.minimum(v, np.sqrt(2.0 * p.a_stop * np.maximum(s[-1] - s, 0.0)))
    return P, v


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class NavCommand:
    vx: float                     # odometry frame, m/s
    vy: float
    yaw: float                    # heading setpoint, rad (odometry frame)
    state: str                    # WAIT | GO | SLOW | ARRIVED | BLOCKED | STUCK
    mppi: MppiResult | None
    depth_age: float = math.inf   # s since the last depth frame


class LocalNav:
    def __init__(self, waypoints: Sequence[Sequence[float]], start_xy: Any, start_yaw: float,
                 nav: LocalNavParams | None = None, grid: LocalGridParams | None = None,
                 mppi: MppiParams | None = None, xp: Any = None) -> None:
        self.p = nav or LocalNavParams()
        self.grid = LocalGrid(grid, centre=start_xy)
        self.path, self.v_ref = waypoint_path(start_xy, waypoints, self.p)
        self.mppi = Mppi(self.grid.grid2d(), self.path, self.v_ref, mppi, xp)
        self.goal = self.path[-1].copy()
        self.yaw_cmd = float(start_yaw)
        self._dirty = False
        self.t_last: float | None = None
        self.t_depth: float | None = None
        self.depth_frames = 0
        self._scale = 1.0
        self._prog = (None, -math.inf)          # (time, progress) of the last real progress

    def add_depth(self, pts_odom: Any, sensor_xyz: Any, z_drone: float, t: float | None = None) -> dict[str, int]:
        """One depth frame, already in the odometry frame (pose at the frame's capture time)."""
        out = self.grid.integrate(pts_odom, sensor_xyz, z_drone, t)
        self._dirty = True
        self.depth_frames += 1
        self.t_depth = t
        return out

    def _path_heading(self, s: float) -> float:
        ps = self.mppi.ps
        k = int(np.clip(np.searchsorted(ps, s, side="right"), 1, len(ps) - 1))
        return math.atan2(self.mppi.py[k] - self.mppi.py[k - 1], self.mppi.px[k] - self.mppi.px[k - 1])

    def step(self, pos: Any, vel: Any, t: float) -> NavCommand:
        depth_age = math.inf if self.t_depth is None else max(t - self.t_depth, 0.0)
        if self.depth_frames < self.p.min_depth_frames:
            self.t_last = t
            return NavCommand(0.0, 0.0, self.yaw_cmd, "WAIT", None, depth_age)
        origin = self.grid.origin.copy()
        self.grid.follow(np.asarray(pos, float)[:2])                  # the window follows the drone even without depth
        if self._dirty or not np.array_equal(origin, self.grid.origin):
            self.mppi.set_grid(self.grid.grid2d())
            self._dirty = False
        stale = depth_age > self.p.depth_stale_s
        scale = min(1.0, self.p.v_no_depth / self.p.v_cruise) if stale else 1.0
        if scale != self._scale:
            self.mppi.set_speed_profile(self.v_ref * scale)
            self._scale = scale
        r = self.mppi.step(np.asarray(pos, float)[:2], np.asarray(vel, float)[:2])
        dt = self.mppi.p.dt if self.t_last is None else max(t - self.t_last, 0.0)
        self.t_last = t
        if self._prog[0] is None or r.s >= self._prog[1] + self.p.stuck_m:
            self._prog = (t, r.s)
        want = self._path_heading(min(r.s + self.p.yaw_lookahead_m, self.mppi.length))
        dmax = math.radians(self.p.yaw_rate_dps) * dt
        self.yaw_cmd = _wrap(self.yaw_cmd + float(np.clip(_wrap(want - self.yaw_cmd), -dmax, dmax)))
        speed = float(np.hypot(*np.asarray(vel, float)[:2]))
        near = float(np.hypot(*(np.asarray(pos, float)[:2] - self.goal))) <= self.p.accept_m
        if near and speed < self.p.arrive_speed:
            state = "ARRIVED"
        elif r.blocked:
            state = "BLOCKED"
        elif not near and t - self._prog[0] > self.p.stuck_s:
            state = "STUCK"
        else:
            state = "SLOW" if stale else "GO"
        u = (0.0, 0.0) if state in ("ARRIVED", "BLOCKED") else r.u
        return NavCommand(u[0], u[1], self.yaw_cmd, state, r, depth_age)
