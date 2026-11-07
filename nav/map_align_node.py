"""ROS 2 node: map-based drift correction (todos.md Task 2) for refnav's continuous follower.

Runs nav.map_align.DriftCorrector on depth frames and PX4's pose and publishes

  /blueraven/odom_correction         geometry_msgs/PoseStamped, frame_id 'px4_local_ned', stamped with the
                                     depth frame it came from, about 5 Hz, ONLY once the corrector has a
                                     valid estimate (at least `min_accepted` accepted fits). An SE2
                                     correction C: corrected = R(yaw_C) * p_px4 + (x_C, y_C) with x north,
                                     y east; heading_corrected = heading_px4 + yaw_C, yaw clockwise from
                                     north (PX4's sense). position.x/y = x_C/y_C; orientation = rotation by
                                     yaw_C about the NED z axis (w = cos(yaw_C/2), z = sin(yaw_C/2)).
  /blueraven/odom_correction/status  std_msgs/String JSON per processed frame: counts, fit result, the
                                     correction, last reject reason.

It never publishes to PX4. Interface agreed with hinson-claude on the bus (message ns42y6YMdjYC).

Run from the repo root:
  python3 -m nav.map_align_node --ros-args -p use_sim_time:=true -p launch_key:=launch_sim
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from nav.course_map import CourseMap
from nav.frames import load_launch
from nav.map_align import SE2, CorrectorParams, DriftCorrector, correction_ned, wrap
from refnav import depth as D

ROOT = Path(__file__).resolve().parents[1]
PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)


def _yaw_ned(q) -> float:                                   # PX4 attitude q = (w, x, y, z), FRD -> NED
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class MapAlignNode(Node):
    def __init__(self) -> None:
        super().__init__("map_align")
        from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
        p = lambda n, v: self.declare_parameter(n, v).value
        self.depth_topic = p("depth_topic", "/camera/depth/image_raw")
        self.info_topic = p("depth_info_topic", "/camera/depth/camera_info")
        self.base_frame = p("base_frame", "base_link")
        self.rate_hz = float(p("rate_hz", 5.0))
        self.min_accepted = int(p("min_accepted", 3))
        launch = load_launch(p("course_yaml", str(ROOT / "maps" / "course.yaml")), p("launch_key", "launch_sim"))
        self.launch_z = launch["z"]
        self.start = SE2(launch["x"], launch["y"], math.radians(launch["yaw_deg"] - 90.0))
        cm = CourseMap.load(p("course_map", str(ROOT / "maps" / "course_sdf.npz")))
        self.dc = DriftCorrector(cm, self.start, CorrectorParams(tau_s=float(p("tau_s", 2.0))))
        self.K = None; self.q = None; self.pos = None; self.extrinsic = None
        self.t_last = -1e9; self.accepted = 0; self.skips: dict[str, int] = {}
        self.tf_buffer = Buffer(); self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", lambda m: setattr(self, "q", tuple(m.q)), PX4_QOS)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self._on_lpos, PX4_QOS)
        self.create_subscription(CameraInfo, self.info_topic, self._on_info, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self._on_depth, qos_profile_sensor_data)
        self.pub = self.create_publisher(PoseStamped, "/blueraven/odom_correction", 10)
        self.pub_status = self.create_publisher(String, "/blueraven/odom_correction/status", 10)
        self.get_logger().info(f"map_align: launch {launch}, map {cm.shape} cell {cm.cell} m, {self.rate_hz} Hz")

    def _on_lpos(self, m) -> None:
        self.pos = (m.x, m.y, m.z) if (m.xy_valid and m.z_valid) else None

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
        from rclpy.time import Time
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, frame_id, Time())
        except Exception as e:
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
        if self.K is None or self.q is None or self.pos is None:
            return self._skip("waiting for camera_info, attitude and a valid PX4 position")
        cal = self._calibration(m.header.frame_id)
        if cal is None:
            return
        self.t_last = stamp
        try:
            d = D.decode_depth(m.encoding, m.height, m.width, m.step, bytes(m.data))
        except ValueError as e:
            return self._skip(str(e))
        f = D.DepthFrame(d, self.K, cal[1], cal[2], D.level_rotation(self.q), stride=4, border_px=24)
        n, e, dn = self.pos
        odom = (e, n, -dn + self.launch_z, wrap(math.pi / 2 - _yaw_ned(self.q)))   # PX4 NED -> odom ENU
        T = self.dc.update(stamp, odom, f.points(0.4, 3.5))
        r = self.dc.last
        if r is not None and r.ok:
            self.accepted += 1
        xC, yC, yawC = correction_ned(self.start, T)
        valid = self.accepted >= self.min_accepted
        if valid:
            ps = PoseStamped()
            ps.header.stamp = m.header.stamp
            ps.header.frame_id = "px4_local_ned"
            ps.pose.position.x, ps.pose.position.y = float(xC), float(yC)
            ps.pose.orientation.w, ps.pose.orientation.z = math.cos(yawC / 2), math.sin(yawC / 2)
            self.pub.publish(ps)
        st = {"t": round(stamp, 3), "valid": valid, "published": valid, "accepted": self.accepted, **self.dc.n,
              "fit_ok": bool(r.ok) if r else None, "why": (r.why if r else ""),
              "inliers": int(r.inliers) if r else 0,
              "rms_before": (round(r.rms_before, 4) if r and math.isfinite(r.rms_before) else None),
              "rms_after": (round(r.rms_after, 4) if r and math.isfinite(r.rms_after) else None),
              "observable": ("".join("xyY"[i] if r.observable[i] else "-" for i in range(3)) if r else "---"),
              "correction_ned": {"x": round(xC, 4), "y": round(yC, 4), "yaw_deg": round(math.degrees(yawC), 3)},
              "skips": self.skips}
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
