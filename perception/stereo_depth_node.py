"""ROS 2 node: rectified stereo pair -> depth image (16UC1 mm) + CameraInfo. Observer only.

No PX4 topics at all: it never publishes /fmu/in/* or /skybeta/setpoint/*, never arms, never commands.
It only turns two camera images into a depth image other nodes can read.

  in   left_image / right_image    sensor_msgs/Image mono8, rectified (sim: /stereo/left|right/image)
       left_info / right_info      sensor_msgs/CameraInfo (P used; rectified pinhole)
       baseline: right P[3] = -fx*B when set, else TF left_frame -> right_frame
  out  out_image  (default /blueraven/stereo_depth/image_raw)   16UC1 millimetres, 0 = invalid,
                  frame_id = left image frame, stamp = left image stamp
       out_info   (default /blueraven/stereo_depth/camera_info) intrinsics at the output scale
       out_status (default /blueraven/stereo_depth/status)      std_msgs/String JSON, ~1 Hz

The default output is NOT /camera/depth/image_raw because the sim already publishes depth there and
two publishers on one topic would interleave. On the drone (which has no depth) remap the outputs to
/camera/depth/image_raw and /camera/depth/camera_info so map_align runs unchanged. If another node
already publishes on out_image, this node refuses to start unless allow_shared_output:=true.

Real-drone topic names for the stereo images are UNKNOWN here (organizers' Isaac ROS OAK driver);
set left_image/right_image/left_info/right_info parameters to whatever `ros2 topic list` shows.

Run (sim, read-only; needs OpenCV):
  python3 -m perception.stereo_depth_node --ros-args -p use_sim_time:=true
"""

from __future__ import annotations

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from .stereo_depth import StereoConfig, compute_depth, depth_to_16uc1_mm


def _mono8(m: Image) -> np.ndarray:
    if m.encoding.lower() not in ("mono8", "8uc1"):
        raise ValueError(f"need mono8, got {m.encoding}")
    if m.step < m.width or len(m.data) != m.step * m.height:
        raise ValueError("image buffer does not match step*height")
    a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)[:, : m.width]
    return np.ascontiguousarray(a)


def _stamp_ns(m) -> int:
    return m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec


class StereoDepth(Node):
    def __init__(self):
        super().__init__("blueraven_stereo_depth")
        p = self.declare_parameter
        self.t_left = p("left_image", "/stereo/left/image").value
        self.t_right = p("right_image", "/stereo/right/image").value
        self.t_linfo = p("left_info", "/stereo/left/camera_info").value
        self.t_rinfo = p("right_info", "/stereo/right/camera_info").value
        self.left_frame = p("left_frame", "oak_left").value
        self.right_frame = p("right_frame", "oak_right").value
        out_image = p("out_image", "/blueraven/stereo_depth/image_raw").value
        out_info = p("out_info", "/blueraven/stereo_depth/camera_info").value
        out_status = p("out_status", "/blueraven/stereo_depth/status").value
        allow_shared = bool(p("allow_shared_output", False).value)
        self.max_rate_hz = float(p("max_rate_hz", 5.0).value)
        self.max_pair_dt_ns = int(float(p("max_pair_dt_ms", 5.0).value) * 1e6)
        self.cfg = StereoConfig(
            scale=float(p("scale", 0.5).value),
            num_disparities=int(p("num_disparities", 64).value),
            block_size=int(p("block_size", 5).value),
            min_depth_m=float(p("min_depth_m", 0.3).value),
            max_depth_m=float(p("max_depth_m", 6.0).value),
        )
        self.depth_scale = float(p("depth_scale_correction", 1.0).value)

        existing = self.count_publishers(out_image)
        if existing and not allow_shared:
            raise RuntimeError(
                f"{out_image} already has {existing} publisher(s); refusing to interleave depth. "
                "Pick another out_image or set allow_shared_output:=true deliberately."
            )

        from tf2_ros import Buffer, TransformListener

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.linfo = None
        self.rinfo = None
        self.baseline = None
        self.baseline_source = None
        self.right_buf: list[Image] = []
        self.last_pub = 0.0
        self.stats = {"published": 0, "dropped_unpaired": 0, "dropped_rate": 0, "errors": 0}
        self.last_valid = None
        self.last_ms = None

        self.pub_img = self.create_publisher(Image, out_image, qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, out_info, qos_profile_sensor_data)
        self.pub_status = self.create_publisher(String, out_status, 10)
        q = qos_profile_sensor_data
        self.create_subscription(CameraInfo, self.t_linfo, self._on_linfo, q)
        self.create_subscription(CameraInfo, self.t_rinfo, self._on_rinfo, q)
        self.create_subscription(Image, self.t_right, self._on_right, q)
        self.create_subscription(Image, self.t_left, self._on_left, q)
        self.create_timer(1.0, self._status)
        self.get_logger().info(
            f"observer only: {self.t_left} + {self.t_right} -> {out_image} at <= {self.max_rate_hz} Hz, "
            f"scale {self.cfg.scale}, sim time {self.get_parameter('use_sim_time').value}"
        )

    def _on_linfo(self, m):
        self.linfo = m

    def _on_rinfo(self, m):
        self.rinfo = m

    def _baseline_m(self):
        if self.baseline is not None:
            return self.baseline
        if self.rinfo is not None and self.rinfo.p[0] > 0 and self.rinfo.p[3] != 0.0:
            self.baseline, self.baseline_source = -self.rinfo.p[3] / self.rinfo.p[0], "right P[3]"
        else:
            from rclpy.time import Time

            try:
                tf = self.tf_buffer.lookup_transform(self.left_frame, self.right_frame, Time())
            except Exception:
                return None
            t = tf.transform.translation
            self.baseline, self.baseline_source = float(np.linalg.norm([t.x, t.y, t.z])), "TF"
        self.get_logger().info(f"baseline {self.baseline:.4f} m from {self.baseline_source}")
        return self.baseline

    def _on_right(self, m):
        self.right_buf.append(m)
        if len(self.right_buf) > 10:
            self.right_buf.pop(0)

    def _on_left(self, m):
        now = time.monotonic()
        if self.max_rate_hz > 0 and now - self.last_pub < 1.0 / self.max_rate_hz:
            self.stats["dropped_rate"] += 1
            return
        if self.linfo is None:
            return
        baseline = self._baseline_m()
        if baseline is None:
            return
        ls = _stamp_ns(m)
        best = min(self.right_buf, key=lambda r: abs(_stamp_ns(r) - ls), default=None)
        if best is None or abs(_stamp_ns(best) - ls) > self.max_pair_dt_ns:
            self.stats["dropped_unpaired"] += 1
            return
        P = self.linfo.p if any(self.linfo.p) else None
        fx, fy, cx, cy = (
            (P[0], P[5], P[2], P[6])
            if P
            else (self.linfo.k[0], self.linfo.k[4], self.linfo.k[2], self.linfo.k[5])
        )
        try:
            r = compute_depth(_mono8(m), _mono8(best), fx, fy, cx, cy, baseline, self.cfg)
        except (ValueError, RuntimeError) as e:
            self.stats["errors"] += 1
            if self.stats["errors"] in (1, 10, 100):
                self.get_logger().warning(f"stereo pair not used ({self.stats['errors']}x): {e}")
            return
        depth = r.depth_m * self.depth_scale if self.depth_scale != 1.0 else r.depth_m
        mm = depth_to_16uc1_mm(depth)
        out = Image()
        out.header.stamp = m.header.stamp
        out.header.frame_id = m.header.frame_id or self.left_frame
        out.height, out.width = mm.shape
        out.encoding = "16UC1"
        out.is_bigendian = 0
        out.step = out.width * 2
        out.data = mm.astype("<u2").tobytes()
        info = CameraInfo()
        info.header = out.header
        info.height, info.width = out.height, out.width
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = [r.fx, 0.0, r.cx, 0.0, r.fy, r.cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [r.fx, 0.0, r.cx, 0.0, 0.0, r.fy, r.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.pub_img.publish(out)
        self.pub_info.publish(info)
        self.last_pub = now
        self.stats["published"] += 1
        self.last_valid, self.last_ms = r.valid_fraction, r.elapsed_ms

    def _status(self):
        doc = {
            "schema": "blueraven.stereo_depth/1",
            **self.stats,
            "valid_fraction": None if self.last_valid is None else round(self.last_valid, 3),
            "compute_ms": None if self.last_ms is None else round(self.last_ms, 1),
            "baseline_m": self.baseline,
            "baseline_source": self.baseline_source,
            "config": {"scale": self.cfg.scale, "num_disparities": self.cfg.num_disparities},
            "control_output": "NONE",
        }
        self.pub_status.publish(String(data=json.dumps(doc, separators=(",", ":"))))


def main():
    rclpy.init()
    node = StereoDepth()
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
