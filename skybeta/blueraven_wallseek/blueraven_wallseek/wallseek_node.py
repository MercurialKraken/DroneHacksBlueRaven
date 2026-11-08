#!/usr/bin/env python3
"""Wall-seek mission node: odometry, /skybeta/state and depth in, /skybeta/setpoint/twist out (the organizers'
flight interface turns it into PX4 setpoints). Never sends PX4 commands, never arms: the safety pilot arms and the
interface enters Offboard on arm. Run: ros2 launch blueraven_wallseek wallseek.launch.py"""
from __future__ import annotations

import json
import math

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from blueraven_wallseek import wallseek_core as W
from blueraven_wallseek import wallseek_depth as D


class WallSeekNode(Node):
    def __init__(self) -> None:
        super().__init__("wallseek")
        p = W.Params()
        for k, v in vars(p).items():                                    # every Params field is a ROS parameter
            setattr(p, k, type(v)(self.declare_parameter(k, v).value))
        self.mission = W.WallSeek(p)
        self.yaw_offset = float(self.declare_parameter("odom_to_ned_yaw_offset_rad", 0.0).value)
        self.mount = D.CameraMount(*(float(self.declare_parameter(k, d).value) for k, d in
                                     (("camera_forward_m", 0.1053), ("camera_left_m", 0.0376), ("camera_up_m", 0.0304))))
        self.corridor = D.CorridorParams()
        self.pad: W.Pad | None = None
        self.pos = self.vel = self.yaw = None
        self.t_odom = self.t_state = self.t_depth = -1e9
        self.state: dict | None = None
        self.K: tuple[float, float, float, float] | None = None
        self.ahead: D.Ahead | None = None
        self.depth_seq = 0
        self.last_log = None
        self.pub = self.create_publisher(TwistStamped, "/skybeta/setpoint/twist", 10)
        self.pub_status = self.create_publisher(String, "/blueraven/wallseek/status", 10)
        self.create_subscription(Odometry, self.declare_parameter("odometry_topic", "/visual_slam/tracking/odometry").value,
                                 self._on_odom, 10)
        self.create_subscription(String, "/skybeta/state", self._on_state, 10)
        self.create_subscription(Image, self.declare_parameter("depth_topic", "/oak/stereo/image_raw").value,
                                 self._on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.declare_parameter("depth_info_topic", "/oak/stereo/camera_info").value,
                                 self._on_info, qos_profile_sensor_data)
        self.vio_var, self.vo_state, self.t_vo = None, None, -1e9
        try:    # skybeta-09 publishes cuVSLAM's tracker state (vo_state 2 = FAILED); the T4 sim doesn't
            from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
            self.create_subscription(VisualSlamStatus, "/visual_slam/status", self._on_vslam_status, 10)
        except ImportError:
            self.get_logger().info("no isaac_ros_visual_slam_interfaces: landing on cuVSLAM's variance + jumps only")
        self.create_timer(0.05, self._tick)
        self.n_tick = 0
        self.get_logger().info(f"wallseek: height {p.height_m} m, standoff {p.standoff_m} m, max forward "
                               f"{p.max_forward_m} m, max right {p.max_right_m} m, yaw offset {self.yaw_offset}")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_odom(self, m: Odometry) -> None:
        p, q, v = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear
        yaw = W.yaw_enu_from_xyzw(q.x, q.y, q.z, q.w)
        if not all(math.isfinite(c) for c in (p.x, p.y, p.z, yaw, v.x, v.y)):
            return
        if self.mission.phase in ("wait", "ready") or self.pad is None:   # the pad follows the drone until Offboard
            self.pad = W.Pad(p.x, p.y, p.z, yaw)
        self.pos = self.pad.pose(p.x, p.y, p.z)
        c, s = math.cos(yaw), math.sin(yaw)                               # body FLU velocity -> ENU -> pad
        self.vel = self.pad.vector(c * v.x - s * v.y, s * v.x + c * v.y)
        self.yaw = math.remainder(self.pad.yaw0 - yaw, math.tau)
        cov = m.pose.covariance
        self.vio_var = max(cov[0], cov[7]) if math.isfinite(cov[0]) and math.isfinite(cov[7]) else None
        self.t_odom = self._now()

    def _on_vslam_status(self, m) -> None:
        self.vo_state, self.t_vo = int(m.vo_state), self._now()

    def _on_state(self, m: String) -> None:
        try:
            self.state = json.loads(m.data)
            self.t_state = self._now()
        except ValueError:
            pass

    def _on_info(self, m: CameraInfo) -> None:
        self.K = (m.k[0], m.k[4], m.k[2], m.k[5])

    def _on_depth(self, m: Image) -> None:
        if self.K is None:
            return
        try:
            d = D.decode(m.encoding, m.height, m.width, m.step, bytes(m.data))
        except ValueError as e:
            self.get_logger().warn(str(e), throttle_duration_sec=5.0)
            return
        self.ahead = D.ahead(d, self.K, self.mount, self.corridor)
        self.t_depth = self._now()
        self.depth_seq += 1

    def _tick(self) -> None:
        now = self._now()
        i = W.Inputs(now=now, pos=self.pos, vel=self.vel, yaw_err=self.yaw, t_odom=self.t_odom, state=self.state,
                     t_state=self.t_state, ahead_m=self.ahead.range_m if self.ahead else None, t_depth=self.t_depth,
                     depth_seq=self.depth_seq, vio_var=self.vio_var,
                     vo_failed=self.vo_state == 2 and now - self.t_vo < 0.5)
        out = self.mission.step(i)
        while self.mission.log:
            self.get_logger().info(self.mission.log.pop(0))
        if out.twist is not None and self.pad is not None:
            vn, ve, vz, wz = out.twist
            vx, vy = W.twist_for_interface(*self.pad.to_odom(vn, ve), self.yaw_offset)
            msg = TwistStamped()
            msg.header.stamp, msg.header.frame_id = self.get_clock().now().to_msg(), "odom"
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z, msg.twist.angular.z = vx, vy, vz, wz
            self.pub.publish(msg)
        self.n_tick += 1
        if self.n_tick % 4 == 0:                                         # 5 Hz status for the bag / a watcher
            a = self.ahead
            f = self.mission.ahead_f
            st = {"phase": out.phase, "why": out.why, "pos": self.pos, "vel": self.vel,
                  "ahead_m": None if a is None or not math.isfinite(a.range_m) else round(a.range_m, 3),
                  "ahead_filtered_m": None if f is None or not math.isfinite(f) else round(f, 3),
                  "vio_var": self.vio_var, "vo_state": self.vo_state,
                  "ahead_points": a.points if a else None, "depth_valid": round(a.valid_frac, 2) if a else None,
                  "depth_age": round(now - self.t_depth, 3), "odom_age": round(now - self.t_odom, 3),
                  "twist_pad": out.twist}
            self.pub_status.publish(String(data=json.dumps(st)))


def main() -> None:
    rclpy.init()
    node = WallSeekNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
