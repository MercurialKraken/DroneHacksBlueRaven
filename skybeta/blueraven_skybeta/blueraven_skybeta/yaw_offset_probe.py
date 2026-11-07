#!/usr/bin/env python3
"""Measure odom_to_ned_yaw_offset_rad on the bench (competitor_guide.md section 1). READ-ONLY: subscribes to
the vision odometry and PX4's attitude, publishes nothing, never touches a camera topic.

    ros2 run blueraven_skybeta yaw_offset_probe --ros-args -p seconds:=40.0

Keep the drone on the ground, props off. Turn it slowly by hand to 3-4 headings during the run, pausing a
few seconds at each. A constant offset (small spread) is the value to pass as
odom_to_ned_yaw_offset_rad; a spread that grows with heading means something else is wrong (stop and ask).
"""
from __future__ import annotations

import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from blueraven_skybeta.frame_offset import circular_stats, offset_sample, yaw_enu_from_xyzw, yaw_ned_from_wxyz


class YawOffsetProbe(Node):
    def __init__(self) -> None:
        super().__init__("blueraven_yaw_offset_probe")
        from px4_msgs.msg import VehicleAttitude
        self.seconds = float(self.declare_parameter("seconds", 40.0).value)
        att_topic = self.declare_parameter("attitude_topic", "/fmu/out/vehicle_attitude").value
        odom_topic = self.declare_parameter("odometry_topic", "/visual_slam/tracking/odometry").value
        self.yaw_vision: float | None = None
        self.samples: list[float] = []
        self.headings: list[float] = []
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.create_subscription(VehicleAttitude, att_topic, self._on_att, qos_profile_sensor_data)
        self.t0 = self.get_clock().now()
        self.create_timer(1.0, self._tick)
        self.get_logger().info(f"measuring for {self.seconds:.0f} s: turn the drone slowly to a few headings")

    def _on_odom(self, m: Odometry) -> None:
        q = m.pose.pose.orientation
        self.yaw_vision = yaw_enu_from_xyzw(q.x, q.y, q.z, q.w)

    def _on_att(self, m) -> None:
        if self.yaw_vision is None:
            return
        w, x, y, z = (float(v) for v in m.q)
        yaw_px4 = yaw_ned_from_wxyz(w, x, y, z)
        self.samples.append(offset_sample(self.yaw_vision, yaw_px4))
        self.headings.append(yaw_px4)

    def _tick(self) -> None:
        el = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        mean, sd, n = circular_stats(self.samples[-50:])
        if n:
            self.get_logger().info(f"{el:4.0f} s  offset now {math.degrees(mean):+6.2f} deg  (spread {math.degrees(sd):.2f})"
                                   f"  heading {math.degrees(self.headings[-1]):+6.1f} deg")
        if el >= self.seconds:
            mean, sd, n = circular_stats(self.samples)
            span = math.degrees(max(self.headings) - min(self.headings)) if self.headings else 0.0
            self.get_logger().info(
                f"RESULT over {n} samples, headings spanning {span:.0f} deg: offset {math.degrees(mean):+.2f} deg "
                f"= {mean:+.4f} rad, spread {math.degrees(sd):.2f} deg. "
                + ("Small: fly with 0 first." if abs(math.degrees(mean)) < 3.0 else
                   f"Pass odom_to_ned_yaw_offset_rad:={mean:.4f} (after the team agrees)."))
            raise SystemExit(0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YawOffsetProbe()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
