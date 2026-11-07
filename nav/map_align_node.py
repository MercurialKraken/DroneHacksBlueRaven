"""ROS 2 node (Humble and Jazzy): map-based drift correction for the follower (todos.md Task 2, Task D).

Fits each depth frame to the course map with the pose AT THE DEPTH FRAME'S STAMP (a 1 s buffer of pose and
attitude, interpolated; frames outside it are refused) and filters the fits with nav.map_kf (Kalman, the
default) or nav.map_align.DriftCorrector (filter:=lowpass, the version flown on 2026-11-07).

Two pose sources:

  pose_source:=px4 (the sims; default)
      /fmu/out/vehicle_local_position_v1 + /fmu/out/vehicle_attitude. Their timestamp_sample is PX4 time
      (microseconds since PX4 boot); it is mapped into ROS time by nav.pose_buffer.ClockMap from each
      message's receipt time (min-filtered offset). odom = ENU from the launch pose (launch_key).
      Publishes /blueraven/odom_correction (PoseStamped, frame 'px4_local_ned'), as before:
        corrected = R(yaw_C) * p_px4 + (x_C, y_C), x north, y east; heading_corrected = heading_px4 + yaw_C
        (clockwise from north). position.x/y = x_C/y_C; orientation w = cos(yaw_C/2), z = sin(yaw_C/2).

  pose_source:=odometry (the real drone, skybeta)
      nav_msgs/Odometry on odom_topic (default /visual_slam/tracking/odometry): ENU 'odom' = the stack's
      startup frame (x forward at start, y left), FLU body, stamped in ROS time (no mapping).
      The starting map <- odom is taken from init_topic (default /blueraven/map_from_odom_init:
      geometry_msgs/PoseStamped, frame 'map', transient local, published once by the skybeta adapter at INIT
      on the mark; pose = the odom origin in the map, yaw about +z CCW), so both use the SAME start. If
      init_topic is "", it is computed here from launch_key and the first odometry pose:
      T_map_odom = T_map_launch . inv(T_odom_now).
      Depth (Task C) and odometry carry the same capture stamp, so the buffer returns the exact sample.
      Publishes /blueraven/map_from_odom (same message and convention): the full corrected map <- odom
      (x, y, rotation yaw about z). A consumer maps odom poses into the map with it, in place of its INIT T.

Both modes publish only after `min_accepted` accepted fits, stamped with the depth frame's stamp, plus
/blueraven/odom_correction/status (JSON): counts, fit, filter 1-sigma, refused frames, timing.

  python3 -m nav.map_align_node --ros-args -p use_sim_time:=true -p launch_key:=launch_sim
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from nav.course_map import CourseMap
from nav.frames import load_launch
from nav.map_align import SE2, CorrectorParams, DriftCorrector, compose, correction_ned, inverse, wrap
from nav.map_kf import KalmanCorrector, KfParams
from nav.pose_buffer import ClockMap, PoseBuffer, enu_flu_to_ned_frd_wxyz
from refnav import depth as D

try:        # the shared names (chelsea-claude's nav/frames.py on chelsea/course-map); identical literals until merged
    from nav.frames import MAP_FRAME, MAP_FROM_ODOM_INIT_TOPIC, MAP_FROM_ODOM_TOPIC
except ImportError:
    MAP_FRAME, MAP_FROM_ODOM_INIT_TOPIC, MAP_FROM_ODOM_TOPIC = "map", "/blueraven/map_from_odom_init", "/blueraven/map_from_odom"

ROOT = Path(__file__).resolve().parents[1]
PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
LATCHED = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


def _yaw_ned(q) -> float:                                   # PX4 attitude q = (w, x, y, z), FRD -> NED
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _yaw_of_xyzw(q) -> float:
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class MapAlignNode(Node):
    def __init__(self) -> None:
        super().__init__("map_align")
        p = lambda n, v: self.declare_parameter(n, v).value
        self.depth_topic = p("depth_topic", "/camera/depth/image_raw")
        self.info_topic = p("depth_info_topic", "/camera/depth/camera_info")
        self.base_frame = p("base_frame", "base_link")
        self.rate_hz = float(p("rate_hz", 5.0))
        self.min_accepted = int(p("min_accepted", 3))
        self.source = str(p("pose_source", "px4"))
        if self.source not in ("px4", "odometry"):
            raise ValueError("pose_source must be px4 or odometry")
        self.filter = str(p("filter", "kf"))
        # drone mode: the skybeta adapter's latched INIT map <- odom (chelsea-claude's interface, agreed on the bus);
        # "" = compute it here from launch_key and the first odometry pose
        self.init_topic = str(p("init_topic", MAP_FROM_ODOM_INIT_TOPIC if self.source == "odometry" else ""))
        launch_key = str(p("launch_key", ""))
        if not launch_key and not (self.source == "odometry" and self.init_topic):
            raise ValueError("launch_key is required (launch_sim in the sims, launch on the drone)")
        self.launch = load_launch(p("course_yaml", str(ROOT / "maps" / "course.yaml")), launch_key) if launch_key else None
        self.cm = CourseMap.load(p("course_map", str(ROOT / "maps" / "course_sdf.npz")))
        self.tau_s = float(p("tau_s", 2.0))
        self.buf = PoseBuffer(float(p("buffer_s", 1.0)), float(p("max_gap_s", 0.1)))
        self.clock_map = ClockMap()
        self.dc = None
        self.start: SE2 | None = None
        self.launch_z = self.launch["z"] if self.launch else 0.0
        self.K = None; self.extrinsic = None
        self.t_last = -1e9; self.accepted = 0; self.skips: dict[str, int] = {}; self.times: list[float] = []
        self.tf_buffer = Buffer(); self.tf_listener = TransformListener(self.tf_buffer, self)
        if self.source == "px4":
            from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
            self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self._on_att, PX4_QOS)
            self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self._on_lpos, PX4_QOS)
            self.start = SE2(self.launch["x"], self.launch["y"], math.radians(self.launch["yaw_deg"] - 90.0))
            self._make_filter()
            self.pub = self.create_publisher(PoseStamped, "/blueraven/odom_correction", 10)
        else:
            from nav_msgs.msg import Odometry
            self.create_subscription(Odometry, p("odom_topic", "/visual_slam/tracking/odometry"), self._on_odom, qos_profile_sensor_data)
            if self.init_topic:
                self.create_subscription(PoseStamped, self.init_topic, self._on_init, LATCHED)
            self.pub = self.create_publisher(PoseStamped, MAP_FROM_ODOM_TOPIC, 10)
        self.create_subscription(CameraInfo, self.info_topic, self._on_info, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self._on_depth, qos_profile_sensor_data)
        self.pub_status = self.create_publisher(String, "/blueraven/odom_correction/status", 10)
        self.get_logger().info(f"map_align: pose_source {self.source}, filter {self.filter}, launch {self.launch}, "
                               f"map {self.cm.shape} cell {self.cm.cell} m, {self.rate_hz} Hz")

    def _make_filter(self) -> None:
        if self.filter == "kf":
            self.dc = KalmanCorrector(self.cm, self.start, KfParams())
        else:
            self.dc = DriftCorrector(self.cm, self.start, CorrectorParams(tau_s=self.tau_s))

    # --- poses ----------------------------------------------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_att(self, m) -> None:
        self.clock_map.observe(self._now(), int(m.timestamp))
        t = self.clock_map.to_ros(int(m.timestamp_sample))
        self.buf.add_attitude(t, tuple(float(v) for v in m.q))                       # (w, x, y, z) FRD -> NED

    def _on_lpos(self, m) -> None:
        if not (m.xy_valid and m.z_valid):
            return
        self.clock_map.observe(self._now(), int(m.timestamp))
        self.buf.add_position(self.clock_map.to_ros(int(m.timestamp_sample)), (m.x, m.y, m.z))

    def _on_odom(self, m) -> None:
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.buf.add_position(t, (p.x, p.y, p.z))
        self.buf.add_attitude(t, (q.x, q.y, q.z, q.w))                               # ENU <- FLU, x y z w
        if self.start is None and not self.init_topic and self.launch is not None:
            # T_map_odom = T_map_launch . inv(T_odom_now), on the mark at startup
            T_map_launch = SE2(self.launch["x"], self.launch["y"], math.radians(self.launch["yaw_deg"]))
            T_odom_now = SE2(p.x, p.y, _yaw_of_xyzw((q.x, q.y, q.z, q.w)))
            self.start = compose(T_map_launch, inverse(T_odom_now))
            self.launch_z = self.launch["z"] - p.z
            self._make_filter()
            self.get_logger().info(f"map <- odom at INIT (computed here): {self.start}")

    def _on_init(self, m: PoseStamped) -> None:
        if self.start is not None:
            return
        q = m.pose.orientation
        self.start = SE2(m.pose.position.x, m.pose.position.y, _yaw_of_xyzw((q.x, q.y, q.z, q.w)))
        self.launch_z = float(m.pose.position.z)
        self._make_filter()
        self.get_logger().info(f"map <- odom at INIT (from the adapter): {self.start}")

    def _pose_at(self, t: float):
        """(odom ENU pose, PX4-style q_wxyz for levelling) at t, or None."""
        at = self.buf.at(t)
        if at is None:
            return None
        pos, q = at
        if self.source == "px4":
            n, e, dn = pos
            return (e, n, -dn + self.launch_z, wrap(math.pi / 2 - _yaw_ned(q))), tuple(q)
        x, y, z = pos
        return (x, y, z + self.launch_z, _yaw_of_xyzw(q)), enu_flu_to_ned_frd_wxyz(q)

    # --- depth ----------------------------------------------------------------------------------------------
    def _on_info(self, m: CameraInfo) -> None:
        if self.K is None:
            self.K = (m.k[0], m.k[4], m.k[2], m.k[5])

    def _skip(self, why: str) -> None:
        self.skips[why] = self.skips.get(why, 0) + 1
        if self.skips[why] in (1, 10, 100, 1000):
            self.get_logger().warning(f"depth frame skipped ({self.skips[why]}x): {why}")

    def _calibration(self, frame_id: str):
        if self.extrinsic is not None and self.extrinsic[0] == frame_id:
            return self.extrinsic
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, frame_id, Time())
        except Exception as e:  # noqa: BLE001 (tf2 raises several types)
            self._skip(f"no TF {self.base_frame} <- {frame_id} ({type(e).__name__})")
            return None
        r, t = tf.transform.rotation, tf.transform.translation
        self.extrinsic = (frame_id, D.quat_to_mat(r.w, r.x, r.y, r.z), np.array([t.x, t.y, t.z]))
        self.get_logger().info(f"calibration from TF: {self.base_frame} <- {frame_id} t=({t.x:.3f},{t.y:.3f},{t.z:.3f})")
        return self.extrinsic

    def _on_depth(self, m: Image) -> None:
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if stamp - self.t_last < 1.0 / self.rate_hz:
            return
        if self.dc is None:
            return self._skip("waiting for the starting map <- odom (launch pose / INIT)")
        if self.K is None:
            return self._skip("waiting for camera_info")
        cal = self._calibration(m.header.frame_id)
        if cal is None:
            return
        at = self._pose_at(stamp)
        if at is None:
            span = self.buf.span()
            return self._skip("no pose at the depth stamp (outside the buffer or a gap)"
                              + ("" if span is None else f": stamp {stamp:.3f}, buffer {span[0]:.3f}-{span[1]:.3f}"))
        odom, q_level = at
        self.t_last = stamp
        w0 = time.perf_counter()
        try:
            d = D.decode_depth(m.encoding, m.height, m.width, m.step, bytes(m.data))
        except ValueError as e:
            return self._skip(str(e))
        f = D.DepthFrame(d, self.K, cal[1], cal[2], D.level_rotation(q_level), stride=4, border_px=24)
        T = self.dc.update(stamp, odom, f.points(0.4, 3.5))
        self.times.append((time.perf_counter() - w0) * 1e3)
        self.times = self.times[-100:]
        r = self.dc.last
        if r is not None and r.ok:
            self.accepted += 1
        valid = self.accepted >= self.min_accepted
        ps = PoseStamped()
        ps.header.stamp = m.header.stamp
        if self.source == "px4":
            xC, yC, yawC = correction_ned(self.start, T)
            ps.header.frame_id = "px4_local_ned"
            ps.pose.position.x, ps.pose.position.y = float(xC), float(yC)
            ps.pose.orientation.w, ps.pose.orientation.z = math.cos(yawC / 2), math.sin(yawC / 2)
            corr = {"frame": "px4_local_ned", "x": round(xC, 4), "y": round(yC, 4), "yaw_deg": round(math.degrees(yawC), 3)}
        else:
            ps.header.frame_id = MAP_FRAME
            ps.pose.position.x, ps.pose.position.y = float(T.x), float(T.y)
            ps.pose.orientation.w, ps.pose.orientation.z = math.cos(T.yaw / 2), math.sin(T.yaw / 2)
            corr = {"frame": "map", "map_odom": {"x": round(T.x, 4), "y": round(T.y, 4), "yaw_deg": round(math.degrees(T.yaw), 3)}}
        if valid:
            self.pub.publish(ps)
        st = {"t": round(stamp, 3), "valid": valid, "published": valid, "accepted": self.accepted, "filter": self.filter,
              **self.dc.n, "fit_ok": bool(r.ok) if r else None, "why": (r.why if r else ""),
              "inliers": int(r.inliers) if r else 0,
              "rms_before": (round(r.rms_before, 4) if r and math.isfinite(r.rms_before) else None),
              "rms_after": (round(r.rms_after, 4) if r and math.isfinite(r.rms_after) else None),
              "observable": ("".join("xyY"[i] if r.observable[i] else "-" for i in range(3)) if r else "---"),
              "correction": corr, "skips": self.skips,
              "ms_p50": round(float(np.percentile(self.times, 50)), 2), "ms_max": round(max(self.times), 2)}
        if self.filter == "kf":
            sx, sy, syaw = self.dc.sigma()
            st["sigma"] = {"x": round(sx, 4), "y": round(sy, 4), "yaw_deg": round(syaw, 3)}
            st["gate_d2"] = None if self.dc.last_d2 is None else round(self.dc.last_d2, 2)
        self.pub_status.publish(String(data=json.dumps(st)))


def main() -> None:
    rclpy.init()
    node = MapAlignNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
