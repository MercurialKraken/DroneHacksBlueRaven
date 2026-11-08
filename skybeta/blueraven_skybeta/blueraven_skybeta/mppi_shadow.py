#!/usr/bin/env python3
"""MPPI in SHADOW: its own process, on refnav's live inputs (odometry, on-camera depth, the route from refnav's
take-off pose), publishing only /blueraven/mppi/*. It never commands the drone: it has no /skybeta publisher.
Run: ros2 launch blueraven_skybeta refnav.launch.py mppi_shadow:=true"""
from __future__ import annotations

import json

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from blueraven_skybeta import refnav_skybeta as S  # first: it puts the team repo (refnav/, nav/) on sys.path
from blueraven_skybeta.frame_offset import yaw_enu_from_xyzw
from blueraven_skybeta.mppi_follow import MppiFollow
from nav.mppi_ros import MppiPublisher
from refnav import depth as D

# depth frames are ~512 KB: best-effort drops the fragmented ones (skybeta-09: 2 frames in 60 s)
RELIABLE = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=2)


class MppiShadow(Node):
    def __init__(self) -> None:
        super().__init__("mppi_shadow")
        prm = lambda k, v: self.declare_parameter(k, v).value  # noqa: E731
        with open(prm("route_file", "")) as f:
            self.route = json.load(f)["route"]
        self.mppi = MppiFollow(float(prm("mppi_v_cruise_mps", 0.15)), float(prm("mppi_tau_s", 0.9)),
                               float(prm("mppi_dead_time_s", 0.6)))
        self.pub = MppiPublisher(self, z=float(prm("height_m", 0.7)), debug=True, debug_hz=2.0)
        self.base_frame = prm("base_frame", "base_link")
        self.phase, self.pad, self.t_odom, self.K, self.T, self.flying = "wait", None, -1e9, None, None, False
        from tf2_ros import Buffer, TransformListener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Odometry, prm("odometry_topic", "/visual_slam/tracking/odometry"), self._on_odom, 10)
        self.create_subscription(String, prm("status_topic", "/refnav/status"), self._on_status, 10)
        self.create_subscription(CameraInfo, prm("depth_info_topic", "/oak/stereo/camera_info"), self._on_info,
                                 RELIABLE)
        self.create_subscription(Image, prm("depth_topic", "/oak/stereo/image_raw"), self._on_depth,
                                 RELIABLE)
        self.create_timer(0.05, self._tick)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_status(self, m: String) -> None:
        self.phase = json.loads(m.data).get("phase", self.phase)

    def _on_odom(self, m: Odometry) -> None:
        p, q, v, st = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear, m.header.stamp
        if self.phase in ("wait", "arm") or self.pad is None:              # refnav's take-off pose rule (on_odom)
            self.pad = S.Pad(p.x, p.y, p.z, yaw_enu_from_xyzw(q.x, q.y, q.z, q.w))
        self.mppi.on_odom(st.sec + st.nanosec * 1e-9, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w), (v.x, v.y, v.z))
        self.t_odom = self._now()

    def _on_info(self, m: CameraInfo) -> None:
        if m.k[0] > 0 and m.k[4] > 0:
            self.K = np.array(m.k, float).reshape(3, 3)

    def _on_depth(self, m: Image) -> None:
        if self.mppi.nav is None or self.K is None:
            return
        if self.T is None:
            from rclpy.time import Time
            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, m.header.frame_id, Time())
            except Exception:                       # not (yet) on the TF tree
                return
            r, t = tf.transform.rotation, tf.transform.translation
            self.T = np.eye(4)
            self.T[:3, :3], self.T[:3, 3] = D.quat_to_mat(r.w, r.x, r.y, r.z), (t.x, t.y, t.z)
        d = D.decode_depth(m.encoding, m.height, m.width, m.step, bytes(m.data))
        self.mppi.on_depth(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, d, self.K, self.T)

    def _tick(self) -> None:
        # loops from the first pose (the pre-flight load gate sees its full cost); restarts once at follow start
        now = self._now()
        if self.pad is None or now - self.t_odom > S.HOLD_S:
            return
        if self.mppi.nav is None or (self.phase == "follow" and not self.flying):
            self.flying = self.phase == "follow"
            self.mppi.start(self.route, self.pad)
            self.pub.publish_trajectory(self.mppi.nav)
        cmd = self.mppi.cmd
        self.mppi.tick(now, self.pad)
        if self.mppi.cmd is not cmd:
            self.pub.publish(self.mppi.cmd, self.mppi.nav)


def main() -> None:
    rclpy.init()
    try:
        rclpy.spin(MppiShadow())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
