#!/usr/bin/env python3
"""ROS 2 wrapper around nav.live_map: depth in, live obstacles and status out. No flight commands.

Run from the repository root, with the onboard services (cuVSLAM, vision bridge, px4_odometry) up:

    python3 -m nav.live_map_node --ros-args -p use_sim_time:=true -p launch_key:=launch_sim -p depth_scale:=0.985

Inputs (all standard ROS):
  depth        sensor_msgs/Image (16UC1 mm or 32FC1 m) + CameraInfo           depth_topic, depth_info_topic
  pose         TF map <- the depth image's frame_id, at the image's stamp. The chain is
               map -> odom (this node, from maps/course.yaml, until Task 2 publishes a corrected one)
               -> base_link (onboard/adapters/px4_odom.py, PX4's estimate) -> camera (static TF)
  health       std_msgs/Bool, true = the estimator feeding PX4 is healthy        health_topic

Outputs:
  /blueraven/live_obstacles   sensor_msgs/PointCloud2 (x, y, z float32) in `map`: obstacle voxel
                              centres. Rebuild the combined map elsewhere with
                              CombinedMap(course, ObstacleCloud(course, points)).
  /blueraven/live_map/status  std_msgs/String JSON: frames used, frames skipped by reason, the last
                              frame's counts (explained, unexplained, misfit...), trusted, obstacles.
  TF map -> odom              static, from the launch pose (publish_map_to_odom:=false once the
                              drift-correction node publishes its own; two publishers would fight).

Mapping pauses, and nothing is changed, while the pose can't be trusted: estimator unhealthy or
silent, for recover_s after that, after a pose jump, and for any frame whose returns disagree with
the course map (see LiveParams.max_misfit).

map -> odom: odom is PX4's local frame as px4_odom.py publishes it (ENU: x = PX4 east, y = PX4
north), with its origin at base_link where the estimator started. PX4 north is the drone's heading
at launch, so map <- odom = rotate about z by (launch yaw - 90 deg), then translate to the launch
point at base_link height. Facing north (yaw 90) at launch makes the rotation the identity.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from nav.course_map import CourseMap  # noqa: E402
from nav.frames import load_launch, map_from_odom, quaternion_z  # noqa: E402
from nav.live_map import LiveObstacleLayer, LiveParams, PoseGate  # noqa: E402
from refnav.depth import decode_depth, quat_to_mat  # noqa: E402


def cloud(points: np.ndarray, frame: str, stamp: Any) -> PointCloud2:
    m = PointCloud2()
    m.header.frame_id, m.header.stamp = frame, stamp
    pts = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
    m.height, m.width = 1, len(pts)
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
    m.is_bigendian, m.point_step, m.row_step, m.is_dense = False, 12, 12 * len(pts), True
    m.data = pts.tobytes()
    return m


class LiveMapNode(Node):
    def __init__(self) -> None:
        super().__init__("blueraven_live_map")
        dp = self.declare_parameter
        course_path = Path(dp("course_map", str(ROOT / "maps" / "course_sdf.npz")).value)
        config_path = Path(dp("course_config", str(ROOT / "maps" / "course.yaml")).value)
        self.launch_key = dp("launch_key", "launch").value
        self.map_frame = dp("map_frame", "map").value
        self.odom_frame = dp("odom_frame", "odom").value
        depth_topic = dp("depth_topic", "/camera/depth/image_raw").value
        info_topic = dp("depth_info_topic", "").value or depth_topic.rsplit("/", 1)[0] + "/camera_info"
        health_topic = dp("health_topic", "/estimator/healthy").value
        self.min_period = 1.0 / float(dp("max_rate_hz", 5.0).value)
        self.tf_timeout = Duration(seconds=float(dp("tf_timeout_s", 0.05).value))
        params = LiveParams(depth_scale=float(dp("depth_scale", 1.0).value),
                            explain_tol_m=float(dp("explain_tol_m", LiveParams.explain_tol_m).value),
                            max_misfit=float(dp("max_misfit", LiveParams.max_misfit).value))

        self.course = CourseMap.load(course_path)
        self.live = LiveObstacleLayer(self.course, params=params)
        self.gate = PoseGate(recover_s=float(dp("recover_s", 3.0).value),
                             max_speed_mps=float(dp("max_speed_mps", 2.5).value))
        self.K: np.ndarray | None = None
        self.last_integrate = -math.inf
        self.used = 0
        self.skips: dict[str, int] = {}
        self.last_reason = "starting"
        self.last_stats: dict[str, Any] | None = None

        self.tf_buffer = Buffer()
        try:            # its own thread: a lookup that waits inside a callback must not block TF reception
            self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        except TypeError:                                     # an older tf2_ros without spin_thread
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.tf_timeout = Duration(seconds=0.0)
        if bool(dp("publish_map_to_odom", True).value):
            launch = load_launch(config_path, self.launch_key)
            t = map_from_odom(launch)[:3, 3]
            q = quaternion_z(math.radians(launch["yaw_deg"] - 90.0))
            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id, tf.child_frame_id = self.map_frame, self.odom_frame
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (float(c) for c in t)
            (tf.transform.rotation.x, tf.transform.rotation.y,
             tf.transform.rotation.z, tf.transform.rotation.w) = (float(c) for c in q)
            self.static_tf = StaticTransformBroadcaster(self)
            self.static_tf.sendTransform(tf)
            self.get_logger().info(f"map -> odom from {config_path.name}:{self.launch_key} {launch}")

        self.pub_cloud = self.create_publisher(PointCloud2, dp("obstacles_topic", "/blueraven/live_obstacles").value, 10)
        self.pub_status = self.create_publisher(String, dp("status_topic", "/blueraven/live_map/status").value, 10)
        self.create_subscription(Bool, health_topic, self._on_health, 10)
        self.create_subscription(CameraInfo, info_topic, self._on_info, qos_profile_sensor_data)
        self.create_subscription(Image, depth_topic, self._on_depth, qos_profile_sensor_data)
        self.create_timer(0.5, self._publish_status)
        self.get_logger().info(f"live map on {depth_topic}; course {course_path.name} "
                               f"({self.course.meta.get('mesh_sha256', '?')[:12]}); health {health_topic}; "
                               f"sim time {self.get_parameter('use_sim_time').value}")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _skip(self, key: str, detail: str) -> None:
        self.skips[key] = self.skips.get(key, 0) + 1
        self.last_reason = detail
        if self.skips[key] in (1, 10, 100, 1000):
            self.get_logger().warning(f"depth frame not mapped ({self.skips[key]}x {key}): {detail}")

    def _on_health(self, m: Bool) -> None:
        self.gate.on_health(self._now(), bool(m.data))

    def _on_info(self, m: CameraInfo) -> None:
        if m.k[0] > 0 and m.k[4] > 0:
            self.K = np.array(m.k, dtype=float).reshape(3, 3)

    def _on_depth(self, m: Image) -> None:
        now = self._now()
        if now <= 0.0 or now - self.last_integrate < self.min_period:
            return                                            # before /clock, or rate-limited: not a fault
        if self.K is None:
            return self._skip("no_camera_info", "no CameraInfo yet")
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, m.header.frame_id, Time.from_msg(m.header.stamp),
                                                 timeout=self.tf_timeout)
        except Exception as e:                                # not on the TF tree yet, or not at that time
            return self._skip("no_tf", f"no TF {self.map_frame} <- {m.header.frame_id} at the image time "
                                       f"({type(e).__name__})")
        r, p = tf.transform.rotation, tf.transform.translation
        T = np.eye(4)
        T[:3, :3] = quat_to_mat(r.w, r.x, r.y, r.z)
        T[:3, 3] = (p.x, p.y, p.z)
        ok, why = self.gate.check(now, T[:3, 3])
        if not ok:
            return self._skip("pose", why)
        try:
            depth = decode_depth(m.encoding, m.height, m.width, m.step, bytes(m.data))
        except ValueError as e:
            return self._skip("undecodable", str(e))
        st = self.live.integrate(depth, self.K, T)
        self.last_integrate = now
        self.last_stats = asdict(st)
        if st.rejected:
            return self._skip("misfit", st.notes[-1] if st.notes else "pose disagrees with the map")
        self.used += 1
        self.last_reason = "ok"
        self.pub_cloud.publish(cloud(self.live.obstacle_points(), self.map_frame, m.header.stamp))

    def _publish_status(self) -> None:
        status = {"frames_mapped": self.used, "skipped": self.skips, "last_reason": self.last_reason,
                  "obstacles": int(len(self.live.obstacle_voxels())), "trusted": self.live.trusted,
                  "launch_key": self.launch_key, "last_frame": self.last_stats}
        self.pub_status.publish(String(data=json.dumps(status)))


def main(argv: list[str] | None = None) -> None:
    rclpy.init(args=argv if argv is not None else sys.argv)
    node = LiveMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
