"""ROS glue for the MPPI navigator: route files -> odom waypoints, and the logging / debug topics. ROS is imported
lazily, so the pure helpers here are tested without it.

Topics (frame = the odometry frame, e.g. 'odom'):
  /blueraven/mppi/plan        nav_msgs/Path               every cycle   the chosen trajectory (MPPI's weighted plan)
  /blueraven/mppi/status      std_msgs/String (JSON)      every cycle   state, progress, errors, ESS, crash %, ms, depth age
  /blueraven/mppi/trajectory  nav_msgs/Path (latched)     once          the smooth waypoint trajectory being followed
  debug only (debug:=true; keep OFF in flight, it costs bandwidth):
  /blueraven/mppi/reference   nav_msgs/Path               every cycle   the re-anchored reference
  /blueraven/mppi/samples     visualization_msgs/MarkerArray  debug_hz  the heaviest n_samples rollouts, colour = weight
                                                                        (viridis: dark light, yellow heavy), red = crashed
  /blueraven/mppi/grid        nav_msgs/OccupancyGrid      debug_hz      the depth grid (100 obstacle, 0 free, -1 unseen)
Record them with the usual topics and open the bag in Foxglove (3D panel: the paths, markers and grid together).
"""
from __future__ import annotations

import json
import math
from typing import Any, Sequence

import numpy as np

PREFIX = "/blueraven/mppi"


# ---------------------------------------------------------------------------------------------- pure helpers
def route_to_odom(route_fl: Sequence[Sequence[float]], takeoff_xy: Any, takeoff_yaw: float) -> np.ndarray:
    """Route waypoints [forward, left] (metres, from the take-off pose: the routes/*.json 'route' format) -> odom x, y,
    given the drone's odom position and heading at take-off (INIT)."""
    F = np.asarray(route_fl, dtype=float).reshape(-1, 2)
    c, s = math.cos(takeoff_yaw), math.sin(takeoff_yaw)
    R = np.array([[c, -s], [s, c]])
    return np.asarray(takeoff_xy, dtype=float)[:2] + F @ R.T


def body_to_odom(v_body: Any, q_wxyz: Any) -> np.ndarray:
    """Velocity from a nav_msgs/Odometry twist (expressed in child_frame_id, base_link FLU) -> the odom frame, which is
    what LocalNav.step wants: v_odom = R(q) v_body, q = the same message's pose orientation (w, x, y, z).
    (Passing twist.linear straight in is only right at yaw 0: at a 90 deg turn the pose prediction and the shield
    would roll the wrong velocity. Hinson's review.)"""
    from nav.pose_buffer import quat_wxyz_to_mat
    return quat_wxyz_to_mat(*np.asarray(q_wxyz, dtype=float)) @ np.asarray(v_body, dtype=float)[:3]


def status_dict(cmd: Any) -> dict[str, Any]:
    """The per-cycle status as plain JSON-able numbers."""
    r = cmd.mppi
    out: dict[str, Any] = {"state": cmd.state, "vx": round(cmd.vx, 3), "vy": round(cmd.vy, 3),
                           "yaw": round(cmd.yaw, 3),
                           "depth_age": None if not math.isfinite(cmd.depth_age) else round(cmd.depth_age, 3)}
    if r is not None:
        out.update(s=round(r.s, 3), e_lat=round(r.e_lat, 3), ess=round(r.ess, 1), lam=r.lam if math.isfinite(r.lam) else None,
                   crash_frac=round(r.crash_frac, 3), side=r.side, blocked=r.blocked, min_d_plan=round(r.min_d_plan, 3),
                   best_cost=round(r.best_cost, 2), ms=round(r.ms, 1), shielded=r.shielded,
                   pose_age=round(r.pose_age, 3), pose_age_clamped=r.pose_age_clamped,
                   dist=[round(float(r.dist[0]), 3), round(float(r.dist[1]), 3)])
    return out


def occupancy(grid: Any) -> tuple[np.ndarray, tuple[float, float], float, int, int]:
    """nav.local_grid.LocalGrid -> (int8 data in ROS row-major order (x fastest), origin of cell (0,0)'s CORNER, cell,
    width, height). 100 = obstacle, 0 = seen free, -1 = never seen."""
    occ, known = grid.occupied(), grid.known()
    D = np.where(occ, 100, np.where(known, 0, -1)).astype(np.int8)        # [i = x, j = y]
    cell = float(grid.p.cell)
    origin = (float(grid.origin[0]) - 0.5 * cell, float(grid.origin[1]) - 0.5 * cell)
    return D.T.ravel(), origin, cell, D.shape[0], D.shape[1]


def sample_lines(r: Any, n: int) -> list[tuple[np.ndarray, tuple[float, float, float, float]]]:
    """The n heaviest debug rollouts as (points (H, 2), rgba): viridis by weight, red if crashed."""
    if r is None or not len(r.samples):
        return []
    w = r.sample_w / max(float(r.sample_w.max()), 1e-12)
    out = []
    for k in np.argsort(-w)[:n]:
        if r.sample_crashed[k]:
            rgba = (0.9, 0.15, 0.15, 0.5)
        else:
            x = float(w[k])                                                   # cheap viridis: purple -> teal -> yellow
            rgba = (0.27 + 0.7 * x * x, 0.0 + 0.9 * x, 0.33 + 0.3 * x - 0.5 * x * x, 0.25 + 0.75 * x)
        out.append((r.samples[k], rgba))
    return out


# ---------------------------------------------------------------------------------------------- ROS publisher
class MppiPublisher:
    """Call publish_trajectory(nav) once after INIT, then publish(cmd, nav) every cycle."""

    def __init__(self, node: Any, frame_id: str = "odom", z: float = 0.7, debug: bool = False,
                 debug_hz: float = 2.0, n_samples: int = 30) -> None:
        from nav_msgs.msg import OccupancyGrid, Path
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from std_msgs.msg import String
        from visualization_msgs.msg import MarkerArray
        self.node, self.frame_id, self.z = node, frame_id, float(z)
        self.debug, self.debug_period, self.n_samples = debug, 1.0 / max(debug_hz, 1e-3), n_samples
        self._t_debug = -math.inf
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_plan = node.create_publisher(Path, f"{PREFIX}/plan", 10)
        self.pub_status = node.create_publisher(String, f"{PREFIX}/status", 10)
        self.pub_traj = node.create_publisher(Path, f"{PREFIX}/trajectory", latched)
        if debug:
            self.pub_ref = node.create_publisher(Path, f"{PREFIX}/reference", 10)
            self.pub_samples = node.create_publisher(MarkerArray, f"{PREFIX}/samples", 1)
            self.pub_grid = node.create_publisher(OccupancyGrid, f"{PREFIX}/grid", 1)

    def _path(self, pts: np.ndarray, stamp: Any) -> Any:
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Path
        msg = Path()
        msg.header.frame_id, msg.header.stamp = self.frame_id, stamp
        for x, y in np.asarray(pts, dtype=float).reshape(-1, 2):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = float(x), float(y), self.z
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        return msg

    def publish_trajectory(self, nav: Any) -> None:
        if self.debug:
            nav.mppi.debug_samples = max(nav.mppi.debug_samples, 4 * self.n_samples)
        self.pub_traj.publish(self._path(nav.path, self.node.get_clock().now().to_msg()))

    def publish(self, cmd: Any, nav: Any) -> None:
        from std_msgs.msg import String
        now = self.node.get_clock().now()
        stamp = now.to_msg()
        r = cmd.mppi
        if r is not None:
            self.pub_plan.publish(self._path(r.plan, stamp))
        self.pub_status.publish(String(data=json.dumps(status_dict(cmd))))
        if not self.debug or r is None:
            return
        if len(r.reference):
            self.pub_ref.publish(self._path(r.reference, stamp))
        t = now.nanoseconds * 1e-9
        if t - self._t_debug < self.debug_period:
            return
        self._t_debug = t
        self.pub_samples.publish(self._markers(r, stamp))
        self.pub_grid.publish(self._grid(nav.grid, stamp))

    def _markers(self, r: Any, stamp: Any) -> Any:
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
        from visualization_msgs.msg import Marker, MarkerArray
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for k, (pts, rgba) in enumerate(sample_lines(r, self.n_samples)):
            m = Marker()
            m.header.frame_id, m.header.stamp = self.frame_id, stamp
            m.ns, m.id, m.type, m.action = "mppi_samples", k, Marker.LINE_STRIP, Marker.ADD
            m.scale.x = 0.01
            m.pose.orientation.w = 1.0
            m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
            m.points = [Point(x=float(x), y=float(y), z=self.z) for x, y in pts]
            arr.markers.append(m)
        return arr

    def _grid(self, grid: Any, stamp: Any) -> Any:
        from nav_msgs.msg import OccupancyGrid
        data, origin, cell, w, h = occupancy(grid)
        msg = OccupancyGrid()
        msg.header.frame_id, msg.header.stamp = self.frame_id, stamp
        msg.info.resolution, msg.info.width, msg.info.height = cell, w, h
        msg.info.origin.position.x, msg.info.origin.position.y = origin
        msg.info.origin.position.z = self.z
        msg.info.origin.orientation.w = 1.0
        msg.data = data.tolist()
        return msg
