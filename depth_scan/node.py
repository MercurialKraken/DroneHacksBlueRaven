"""ROS 2 node: on-camera depth -> sensor_msgs/LaserScan horizontal slice at flight height.

    python3 -m depth_scan.node --ros-args -p depth_topic:=/oak/stereo/image_raw

Subscribes ONLY to the depth image and its camera_info (Hinson's on-camera depth product). It must
never subscribe to /oak/*/image_raw, image_mono, image_rect or compressed: extra camera readers
starve cuVSLAM. Publishes one LaserScan per depth frame, header.stamp = the depth frame's own
(capture) stamp, header.frame_id = base_frame. No map, no PX4 topic, no setpoint, no arming.

The static TF base_frame <- depth frame is looked up once (retried until found). Levelling by
attitude is not done here: at hover and cruise tilt the +-0.2 m band at 0.5-1.5 m mostly holds;
the pure function accepts R_level if a consumer needs it.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from depth_scan.slice import SliceConfig, horizontal_slice

FORBIDDEN_INPUT = ("image_raw/compressed", "image_mono", "image_rect", "/left/", "/right/")


def _quat_to_R(x, y, z, w) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def image_to_array(msg) -> np.ndarray:
    """sensor_msgs/Image (16UC1 or 32FC1, row padding and byte order handled) -> 2-D array."""
    dt = {"16UC1": np.uint16, "mono16": np.uint16, "32FC1": np.float32}.get(msg.encoding)
    if dt is None:
        raise ValueError(f"unsupported depth encoding {msg.encoding!r}")
    dt = np.dtype(dt).newbyteorder(">" if msg.is_bigendian else "<")
    item = dt.itemsize
    if msg.step < msg.width * item:
        raise ValueError("step smaller than a row")
    buf = np.frombuffer(bytes(msg.data), np.uint8)[: msg.step * msg.height]
    rows = buf.reshape(msg.height, msg.step)[:, : msg.width * item]
    return rows.copy().view(dt).reshape(msg.height, msg.width).astype(dt.newbyteorder("="))


def main(argv=None) -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.time import Time
        from sensor_msgs.msg import CameraInfo, Image, LaserScan
        from tf2_ros import Buffer, TransformListener
    except ImportError as e:  # laptop without ROS
        print(f"ROS 2 not available ({e}); the pure function is depth_scan.slice", file=sys.stderr)
        return 1

    class DepthScan(Node):
        def __init__(self):
            super().__init__("blueraven_depth_scan")
            p = self.declare_parameter
            self.depth_topic = p("depth_topic", "/oak/stereo/image_raw").value
            info_topic = p("info_topic", "/oak/stereo/camera_info").value
            out_topic = p("out_topic", "/blueraven/depth/scan").value
            self.base_frame = p("base_frame", "base_link").value
            self.cfg = SliceConfig(
                band_m=p("band_m", 0.20).value,
                range_min=p("range_min", 0.20).value,
                range_max=p("range_max", 3.0).value,
                angle_increment_deg=p("angle_increment_deg", 1.5).value,
                min_pixels=p("min_pixels", 2).value,
            )
            if any(f in self.depth_topic for f in FORBIDDEN_INPUT) or out_topic.startswith("/fmu"):
                raise SystemExit(f"refusing topics {self.depth_topic} -> {out_topic}")
            self.K = None
            self.tf = None
            self.buf = Buffer()
            self.listener = TransformListener(self.buf, self)
            self.pub = self.create_publisher(LaserScan, out_topic, 10)
            self.create_subscription(CameraInfo, info_topic, self.on_info, qos_profile_sensor_data)
            self.create_subscription(
                Image, self.depth_topic, self.on_depth, qos_profile_sensor_data
            )
            self.get_logger().info(f"{self.depth_topic} -> {out_topic} ({self.cfg})")

        def on_info(self, m):
            self.K = (m.k[0], m.k[4], m.k[2], m.k[5])

        def on_depth(self, msg):
            if self.K is None:
                return
            if self.tf is None:
                try:
                    tr = self.buf.lookup_transform(self.base_frame, msg.header.frame_id, Time())
                except Exception:  # noqa: BLE001  (tf2 raises several types)
                    self.get_logger().warn("waiting for TF", throttle_duration_sec=5.0)
                    return
                q, t = tr.transform.rotation, tr.transform.translation
                self.tf = (_quat_to_R(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z]))
                self.get_logger().info(f"TF {self.base_frame} <- {msg.header.frame_id}: {t}")
            try:
                sl = horizontal_slice(
                    image_to_array(msg),
                    *self.K,
                    R_base_cam=self.tf[0],
                    t_base_cam=self.tf[1],
                    config=self.cfg,
                )
            except ValueError as e:
                self.get_logger().warn(f"frame skipped: {e}", throttle_duration_sec=5.0)
                return
            out = LaserScan()
            out.header.stamp = msg.header.stamp  # capture time, not now
            out.header.frame_id = self.base_frame
            out.angle_min, out.angle_increment = sl.angle_min, sl.angle_increment
            out.angle_max = sl.angle_max
            out.range_min, out.range_max = sl.range_min, sl.range_max
            out.ranges = [float(r) if not math.isnan(r) else math.nan for r in sl.ranges]
            self.pub.publish(out)

    rclpy.init(args=argv)
    node = DepthScan()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
