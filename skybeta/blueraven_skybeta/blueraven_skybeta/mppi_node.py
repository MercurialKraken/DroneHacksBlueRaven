#!/usr/bin/env python3
"""MPPI in ITS OWN PROCESS for nav:=mppi: refnav's control thread never runs it.

In:  odometry, on-camera depth (RELIABLE: best-effort drops the ~512 KB frames), /blueraven/mppi/start (refnav's
     follow start and take-off pose).
Out: /blueraven/mppi/cmd (std_msgs/String JSON, mppi_follow.cmd_msg) and the /blueraven/mppi/* logs.
It publishes nothing under /skybeta or /fmu: refnav decides what reaches the drone (mppi_follow.CmdLink), and
refnav alone takes off, hovers and lands.

Before follow it loops MPPI from the first pose on a dummy route (start None: refnav ignores those), so the
pre-flight odometry gate already carries MPPI's full load. Run: ros2 launch blueraven_skybeta refnav.launch.py nav:=mppi
"""
from __future__ import annotations

import json
import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from blueraven_skybeta import refnav_skybeta as S  # first: it puts the team repo (refnav/, nav/) on sys.path
from blueraven_skybeta.mppi_follow import MppiFollow, cmd_msg, yaw_enu
from nav.mppi_ros import PREFIX, MppiPublisher
from refnav import depth as D

RELIABLE = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=2)
LATCHED = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
READY_FRAMES = 5            # "mppi_node ready" once this many depth frames were placed (fly_route.sh waits for it)


class MppiNode(Node):
    def __init__(self) -> None:
        super().__init__("mppi_node")
        prm = lambda k, v: self.declare_parameter(k, v).value  # noqa: E731
        with open(prm("route_file", "")) as f:
            self.route = json.load(f)["route"]
        self.args = (float(prm("mppi_v_cruise_mps", 0.10)), float(prm("mppi_tau_s", 0.9)),
                     float(prm("mppi_dead_time_s", 0.6)))
        self.mppi = MppiFollow(*self.args)
        self.pub = MppiPublisher(self, z=float(prm("height_m", 0.7)), debug=bool(prm("debug", False)), debug_hz=2.0)
        self.pub_cmd = self.create_publisher(String, f"{PREFIX}/cmd", 10)
        self.base_frame = prm("base_frame", "base_link")
        self.start_id: float | None = None
        self.t_odom, self.K, self.T, self.n_depth, self.ready = -1e9, None, None, 0, False
        self.n_no_pose = 0
        self.logged_lost: str | None = None
        from tf2_ros import Buffer, TransformListener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Odometry, prm("odometry_topic", "/visual_slam/tracking/odometry"), self._on_odom, 10)
        self.create_subscription(String, f"{PREFIX}/start", self._on_start, LATCHED)
        self.create_subscription(CameraInfo, prm("depth_info_topic", "/oak/stereo/camera_info"), self._on_info, RELIABLE)
        self.create_subscription(Image, prm("depth_topic", "/oak/stereo/image_raw"), self._on_depth, RELIABLE)
        self.create_timer(0.05, self._tick)
        self.create_timer(5.0, self._log)
        self.get_logger().info(f"mppi_node: cruise {self.args[0]:.2f} m/s, tau {self.args[1]} s, dead time "
                               f"{self.args[2]} s; waiting for odometry, depth and refnav's follow start")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_start(self, m: String) -> None:
        d = json.loads(m.data)
        if d["id"] == self.start_id:
            return                                   # refnav repeats it; MPPI starts once per follow
        self.start_id = d["id"]
        x, y, z, yaw = d["pad"]
        self.mppi.start(self.route, S.Pad(x, y, z, yaw))
        self.pub.publish_trajectory(self.mppi.nav)
        self.get_logger().info(f"FOLLOW START {self.start_id:.3f}: route from refnav's take-off pose "
                               f"({x:.3f}, {y:.3f}) yaw {math.degrees(yaw):.1f} deg")

    def _on_odom(self, m: Odometry) -> None:
        p, q, v, st = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear, m.header.stamp
        self.mppi.on_odom(st.sec + st.nanosec * 1e-9, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w), (v.x, v.y, v.z))
        self.t_odom = self._now()
        if self.mppi.nav is None:                    # pre-flight loop from the first pose: refnav ignores it
            self.mppi.start(self.route, S.Pad(p.x, p.y, p.z, yaw_enu((q.x, q.y, q.z, q.w))))

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
            except Exception:                        # not (yet) on the TF tree
                return
            r, t = tf.transform.rotation, tf.transform.translation
            self.T = np.eye(4)
            self.T[:3, :3], self.T[:3, 3] = D.quat_to_mat(r.w, r.x, r.y, r.z), (t.x, t.y, t.z)
        d = D.decode_depth(m.encoding, m.height, m.width, m.step, bytes(m.data))
        if self.mppi.on_depth(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, d, self.K, self.T):
            self.n_depth += 1
        else:
            self.n_no_pose += 1

    def _tick(self) -> None:
        now = self._now()
        if self.mppi.nav is None or now - self.t_odom > S.HOLD_S:
            return                                   # no fresh odometry: no command (refnav holds, then lands)
        cmd = self.mppi.cmd
        self.mppi.tick(now, None)
        c = self.mppi.cmd
        if c is cmd or c is None:
            return
        if c.state == "LOST" and self.logged_lost != self.mppi.lost:
            self.logged_lost = self.mppi.lost
            self.get_logger().error(f"MPPI LOST: {self.mppi.lost}: zero velocity, refnav lands")
        self.pub_cmd.publish(String(data=json.dumps(cmd_msg(c, self.start_id, now))))
        self.pub.publish(c, self.mppi.nav)
        if not self.ready and self.n_depth >= READY_FRAMES:
            self.ready = True
            self.get_logger().info(f"mppi_node ready: {self.n_depth} depth frames placed, MPPI stepping "
                                   f"({c.mppi.ms if c.mppi else float('nan'):.0f} ms)")

    def _log(self) -> None:
        c = self.mppi.cmd
        self.get_logger().info(f"depth frames {self.n_depth} (no pose {self.n_no_pose}), start "
                               f"{self.start_id}, state {None if c is None else c.state}, ms "
                               f"{None if c is None or c.mppi is None else round(c.mppi.ms, 1)}")


def main() -> None:
    rclpy.init()
    node = MppiNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
