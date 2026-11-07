"""ROS 2 observer: left mono camera -> /vision/floor_line (std_msgs/String JSON).

OBSERVER ONLY: no PX4 publisher, no vehicle command, no arming/mode/parameter client.
Nothing in refnav reads this topic yet.

  in   image_topic  sensor_msgs/Image (mono8/mono16; rgb8/bgr8 accepted for hardware drivers),
                    default /stereo/left/image (the lab's declared native topic)
  out  out_topic    std_msgs/String JSON, schema "vision.floor_line/1" (see vision/README.md).
                    An unusable frame is published as {"rejected": "<reason>"} so silence never
                    means "line ahead".
  params  polarity (bright|dark), roi_top (0.35), max_age_ms (200), every_nth (1)

Requires OpenCV in the companion (python3 -c 'import cv2'). Run only after the team agrees:
  python3 -m vision.line_node --ros-args -p use_sim_time:=true
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .floor_line import SCHEMA, LineConfig, floor_line_message


class FloorLine(Node):
    def __init__(self):
        super().__init__("vision_floor_line")
        self.image_topic = self.declare_parameter("image_topic", "/stereo/left/image").value
        out_topic = self.declare_parameter("out_topic", "/vision/floor_line").value
        # DroneHacks course (confirmed from site photos 2026-11-06): the maze route is BLACK tape on
        # an orange floor, so the line is DARKER than the floor -> "dark". Override per camera if needed.
        self.config = LineConfig(
            polarity=str(self.declare_parameter("polarity", "dark").value),
            roi_top=float(self.declare_parameter("roi_top", 0.35).value),
        )
        self.max_age_ms = float(self.declare_parameter("max_age_ms", 200.0).value)
        self.every_nth = max(1, int(self.declare_parameter("every_nth", 1).value))
        self.count = 0
        self.rejects: dict[str, int] = {}
        self.pub = self.create_publisher(String, out_topic, 10)
        self.create_subscription(Image, self.image_topic, self._on_image, qos_profile_sensor_data)
        self.get_logger().info(
            f"observer only: {self.image_topic} -> {out_topic}; polarity {self.config.polarity}; "
            f"sim time {self.get_parameter('use_sim_time').value}"
        )

    def _on_image(self, m):
        self.count += 1
        if self.count % self.every_nth:
            return
        stamp_ns = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
        try:
            doc = floor_line_message(
                m.encoding,
                m.height,
                m.width,
                m.step,
                bytes(m.data),
                stamp_ns,
                self.get_clock().now().nanoseconds,
                m.header.frame_id,
                self.config,
                bool(m.is_bigendian),
                self.max_age_ms,
            )
        except (ValueError, RuntimeError) as e:
            key = type(e).__name__
            self.rejects[key] = self.rejects.get(key, 0) + 1
            if self.rejects[key] in (1, 10, 100, 1000):
                self.get_logger().warning(f"image not used ({self.rejects[key]}x): {e}")
            doc = {
                "schema": SCHEMA,
                "frame_id": m.header.frame_id,
                "stamp_ns": stamp_ns,
                "rejected": str(e),
                "detected": False,
                "control_output": "DISABLED",
            }
        self.pub.publish(String(data=json.dumps(doc, separators=(",", ":"))))


def main():
    rclpy.init()
    node = FloorLine()
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
