"""Map-free navigation: straight lines through coarse waypoints, MPPI on a live depth grid. Pure (numpy; no ROS).

Frame: the odometry (take-off) frame, metres. Waypoints are rough by design: the organizers withhold the course's
dimensions, so a line may clip a wall that moved or slants. MPPI's obstacle cost outweighs its pull towards the line,
so it slides along such a wall and keeps making progress.

* Reference: a smooth trajectory through start -> waypoint 1 -> ... -> last waypoint (waypoint_path): straight legs,
  corners rounded into arcs at most corner_dev_m inside the waypoint, speed from curvature (sqrt(a_lat R)) and
  limited to a_long speeding up and slowing down; it stops at the last waypoint.
* Progress: the projection onto that polyline (nav.mppi: monotone, searched only in a window around the current
  progress, so it can't jump to a later leg that passes nearby).
* Obstacles: nav.local_grid from depth (add_depth), handed to MPPI as its clearance grid whenever it changed.
* Heading: along the line yaw_lookahead_m ahead of the drone's progress, so the camera looks into a turn before the
  drone gets there; at most yaw_rate_dps, scaled by speed (full at yaw_full_speed), so it turns while moving and
  never spins in place.
* State (the caller acts on it):
  WAIT     fewer than min_depth_frames depth frames so far: zero velocity (never move on an empty grid);
  GO       flying;
  SLOW     no depth for depth_stale_s (a dropout): speeds capped at v_no_depth_frac * cruise; MPPI flies on the
           remembered grid and the line;
  HOLD     no depth for depth_dead_s: zero velocity; the caller holds, then lands (never fly blind on memory); Never a stop on its own: odometry loss is the shell's guard;
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
    corner_dev_m: float = 0.08        # rounded corners pass at most this far inside the waypoint
    a_lat: float = 0.40               # m/s^2: speed on a corner arc of radius R is at most sqrt(a_lat R)
    a_long: float = 0.40              # m/s^2: speed-up and slow-down along the trajectory (into and out of corners)
    v_start: float = 0.0              # the trajectory starts from rest
    step_m: float = 0.05
    accept_m: float = 0.25
    arrive_speed: float = 0.10
    yaw_lookahead_m: float = 0.6
    yaw_rate_dps: float = 30.0        # cap (Darion: cuVSLAM diverged in a 45 deg/s turn on the T4; no IMU on the drone)
    yaw_full_speed: float = 0.10      # the yaw rate scales with speed, full at this: a stopped drone never spins in
                                      # place (near-pure rotation gives stereo VIO nothing to triangulate)
    min_depth_frames: int = 5
    depth_stale_s: float = 0.5
    v_no_depth_frac: float = 0.5      # SLOW: at most this fraction of cruise (0.075 m/s at 0.15 cruise)
    depth_dead_s: float = 2.0         # no depth this long: HOLD (zero), the flight program holds then lands
    stuck_s: float = 8.0
    stuck_m: float = 0.10


# The flight default: PX4's velocity response as fitted on the T4 through the real flight_interface (Hinson: lag 0.9 s +
# 0.6 s dead time). The shield's braking check and the pose prediction use it, so a flight node that passes nothing
# still gets the measured plant (Hinson's review, arnav/mppi@33506a2). Re-fit on the real drone's first MPPI flight.
FLIGHT_MPPI = MppiParams(tau=0.9, dead_time_s=0.6)


def waypoint_path(start: Any, waypoints: Sequence[Sequence[float]], p: LocalNavParams) -> tuple[np.ndarray, np.ndarray]:
    """A smooth trajectory through the waypoints: dense points (step_m apart) and a speed for each.

    Geometry: straight legs between waypoints, each interior corner replaced by a circular arc tangent to both legs.
    The arc's radius is the largest that keeps it within corner_dev_m of the waypoint and uses at most half of either
    leg: for a heading change phi, R = corner_dev / (1/cos(phi/2) - 1), tangent length R tan(phi/2).
    Speed: v_cruise, at most sqrt(a_lat R) on an arc, then limited by a_long both ways (forward pass from v_start,
    backward pass to 0 at the last waypoint). So it slows smoothly into a corner, speeds up out of it, and stops at
    the end."""
    V = np.r_[np.asarray(start, float)[None, :2], np.asarray(waypoints, float).reshape(-1, 2)]
    keep = np.r_[True, np.hypot(*np.diff(V, axis=0).T) > 1e-6]
    V = V[keep]
    if len(V) < 2:
        raise ValueError("need at least one waypoint away from the start")
    legs = np.diff(V, axis=0)
    L = np.hypot(*legs.T)
    u = legs / L[:, None]
    # per interior corner: tangent length and radius
    tl, rad, sgn = np.zeros(len(V)), np.full(len(V), np.inf), np.zeros(len(V))
    for k in range(1, len(V) - 1):
        cross = u[k - 1, 0] * u[k, 1] - u[k - 1, 1] * u[k, 0]
        phi = math.atan2(cross, float(np.dot(u[k - 1], u[k])))
        if abs(phi) < 1e-3:
            continue
        R = p.corner_dev_m / (1.0 / math.cos(abs(phi) / 2.0) - 1.0)
        R = min(R, 0.5 * min(L[k - 1], L[k]) / math.tan(abs(phi) / 2.0))
        tl[k], rad[k], sgn[k] = R * math.tan(abs(phi) / 2.0), R, math.copysign(1.0, phi)
    pts, kap = [V[:1]], [0.0]

    def line(a: np.ndarray, b: np.ndarray) -> None:
        d = float(np.hypot(*(b - a)))
        if d < 1e-9:
            return
        n = max(1, int(math.ceil(d / p.step_m)))
        t = np.linspace(0.0, 1.0, n + 1)[1:]
        pts.append(a + t[:, None] * (b - a))
        kap.extend([0.0] * n)

    cur = V[0]
    for k in range(1, len(V)):
        a_end = V[k] - u[k - 1] * tl[k] if k < len(V) - 1 else V[k]
        line(cur, a_end)
        cur = a_end
        if k < len(V) - 1 and np.isfinite(rad[k]):
            R, sg = rad[k], sgn[k]
            normal = np.array([-u[k - 1, 1], u[k - 1, 0]]) * sg           # towards the arc's centre
            c = a_end + normal * R
            th0 = math.atan2(*(a_end - c)[::-1])
            phi = 2.0 * math.atan(tl[k] / R) * sg
            n = max(2, int(math.ceil(abs(phi) * R / p.step_m)))
            th = th0 + np.linspace(0.0, phi, n + 1)[1:]
            arc = c + R * np.c_[np.cos(th), np.sin(th)]
            pts.append(arc)
            kap.extend([1.0 / R] * n)
            cur = arc[-1]
    P = np.concatenate(pts)
    kap = np.asarray(kap)
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    v = np.minimum(p.v_cruise, np.sqrt(p.a_lat / np.maximum(kap, 1e-9)))
    v[0] = min(v[0], p.v_start)
    for i in range(1, len(v)):                                           # speed up at most a_long
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * p.a_long * (s[i] - s[i - 1])))
    v[-1] = 0.0
    for i in range(len(v) - 2, -1, -1):                                  # slow down at most a_long
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * p.a_long * (s[i + 1] - s[i])))
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
                 mppi: MppiParams | None = None, xp: Any = None, fence: dict | None = None) -> None:
        """fence: {"fwd": [min, max], "left": [min, max]} metres from (start_xy, start_yaw): the arena's nets (depth
        can't see them); MPPI then never plans across them."""
        self.p = nav or LocalNavParams()
        self.grid = LocalGrid(grid, centre=start_xy)
        self.path, self.v_ref = waypoint_path(start_xy, waypoints, self.p)
        self.mppi = Mppi(self.grid.grid2d(), self.path, self.v_ref, mppi or FLIGHT_MPPI, xp)
        if fence:
            self.mppi.set_fence(start_xy, start_yaw, fence["fwd"], fence["left"])
        self.goal = self.path[-1].copy()
        self.yaw_cmd = float(start_yaw)
        self._dirty = False
        self.t_last: float | None = None
        self.t_depth: float | None = None
        self.depth_frames = 0
        self._scale = 1.0
        self._prog = (None, -math.inf)          # (time, progress) of the last real progress

    def clear_grid(self) -> None:
        """Forget every obstacle cell (the drone changed height: cells were classified with the old height band,
        e.g. a window sill seen from below still blocks after a climb). Fresh depth rebuilds it."""
        self.grid.L[:] = 0.0
        self.grid.low[:] = False
        self._dirty = True

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

    def observe(self, pos: Any, t: float, yaw: float, u_sent: Any, pose_stamp: float | None = None) -> np.ndarray:
        """A cycle flown by the flight program itself (its hover before FOLLOW): the pose and the velocity command it
        sent (odom frame) teach MPPI the drone's steady push before it takes over. -> the push (body frame)."""
        age = 0.0 if pose_stamp is None else max(t - pose_stamp, 0.0)
        return self.mppi.observe(np.asarray(pos, float)[:2], yaw, u_sent, pose_age_s=age, t=t)

    def rest_u(self, yaw: float | None) -> tuple[float, float]:
        """The command that holds the drone still: minus the steady push estimate (map frame) at this heading;
        zero without a yaw or an estimate (a hold then drifts with whatever push there is)."""
        m = self.mppi
        if yaw is None or not m.p.dist_on:
            return (0.0, 0.0)
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        d = m.push.d
        return (float(-(c * d[0] - s * d[1])), float(-(s * d[0] + c * d[1])))

    def step(self, pos: Any, vel: Any, t: float, pose_stamp: float | None = None,
             yaw: float | None = None) -> NavCommand:
        """One 10 Hz cycle. pos/vel: odometry in the ODOM frame (an Odometry twist is in base_link: convert it with
        nav.mppi_ros.body_to_odom); t: now; pose_stamp: the odometry's capture stamp (same
        clock as t). MPPI predicts through the pose's age, so stale poses (delivery gaps) don't steer it wrong.
        yaw: the odometry's heading (rad, odom frame) at pose_stamp: MPPI then learns and leans against the drone's
        steady body-frame push (nav.mppi.PushEstimator); None: no push estimate."""
        depth_age = math.inf if self.t_depth is None else max(t - self.t_depth, 0.0)
        if self.depth_frames < self.p.min_depth_frames or depth_age > self.p.depth_dead_s:
            self.t_last = t
            rest = self.rest_u(yaw)                                    # zero, or minus the steady push
            self.mppi.note_sent(rest, replace_last=False, t=t)
            state = "WAIT" if self.depth_frames < self.p.min_depth_frames else "HOLD"
            return NavCommand(rest[0], rest[1], self.yaw_cmd, state, None, depth_age)
        origin = self.grid.origin.copy()
        self.grid.follow(np.asarray(pos, float)[:2])                  # the window follows the drone even without depth
        if self._dirty or not np.array_equal(origin, self.grid.origin):
            self.mppi.set_grid(self.grid.grid2d())
            self._dirty = False
        stale = depth_age > self.p.depth_stale_s
        scale = self.p.v_no_depth_frac if stale else 1.0
        if scale != self._scale:
            self.mppi.set_speed_profile(self.v_ref * scale)
            self.mppi.set_speed_cap(self.p.v_no_depth_frac * self.p.v_cruise if stale else None)
            self._scale = scale
        age = 0.0 if pose_stamp is None else max(t - pose_stamp, 0.0)
        r = self.mppi.step(np.asarray(pos, float)[:2], np.asarray(vel, float)[:2], pose_age_s=age, yaw=yaw, t=t)
        dt = self.mppi.p.dt if self.t_last is None else max(t - self.t_last, 0.0)
        self.t_last = t
        if self._prog[0] is None or r.s >= self._prog[1] + self.p.stuck_m:
            self._prog = (t, r.s)
        want = self._path_heading(min(r.s + self.p.yaw_lookahead_m, self.mppi.length))
        moving = min(1.0, float(np.hypot(*np.asarray(vel, float)[:2])) / max(self.p.yaw_full_speed, 1e-6))
        dmax = math.radians(self.p.yaw_rate_dps) * dt * moving
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
        # stopping under a steady push means leaning against it: rest = -push (MPPI's own "brake" in its w space)
        u = (float(-r.dist[0]), float(-r.dist[1])) if state in ("ARRIVED", "BLOCKED") else r.u
        if u != r.u:
            self.mppi.note_sent(u, replace_last=True)            # the prediction must replay what was really sent
        return NavCommand(u[0], u[1], self.yaw_cmd, state, r, depth_age)
