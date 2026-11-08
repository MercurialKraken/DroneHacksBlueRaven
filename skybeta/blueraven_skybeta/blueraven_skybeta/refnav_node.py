#!/usr/bin/env python3
"""refnav's node with its PX4 I/O replaced by the flight interface (refnav_skybeta): odometry and
/skybeta/state in, /skybeta/setpoint/twist out. Run: ros2 launch blueraven_skybeta refnav.launch.py"""
from __future__ import annotations

import json
import math

import numpy as np
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from blueraven_skybeta import refnav_skybeta as S  # first: it puts the team repo (refnav/) on sys.path
from refnav import core as C
from refnav import depth as D
from refnav.node import LOOSE, RefNav
from refnav.node import main as refnav_main


class SkybetaRefNav(RefNav):
    def _io(self) -> None:
        # the SAME value the flight interface gets: it rotates poses by it, not twists, so we do
        self.yaw_offset = float(self.declare_parameter("odom_to_ned_yaw_offset_rad", 0.0, LOOSE).value)
        self.state: dict | None = None
        self.t_state = self.t_odom = -1e9
        self.pad: S.Pad | None = None
        self.logged_pad: S.Pad | None = None
        # off by default: odometry starts where the stack started, which is the mark only if it was started there
        self.start_max_m = float(self.declare_parameter("start_max_m", float("inf"), LOOSE).value)
        self.start_max_deg = float(self.declare_parameter("start_max_deg", float("inf"), LOOSE).value)
        self.descent, self.track = S.Descent(), S.Track()
        # twist -> velocity lag: 1 / MPC_XY_VEL_P_ACC 1.8 on the drone by default; 1.5 s fits the T4 sim
        self.vel_lag_s = float(self.declare_parameter("vel_lag_s", S.VEL_LAG_S, LOOSE).value)
        self.pub_twist = self.create_publisher(TwistStamped, "/skybeta/setpoint/twist", 10)
        self.create_subscription(Odometry, self.declare_parameter(
            "odometry_topic", "/visual_slam/tracking/odometry").value, self._on_odom, 10)
        self.create_subscription(String, "/skybeta/state", self._on_state, 10)
        self.mppi = None
        if self.mission.p.nav == "mppi":
            from blueraven_skybeta.mppi_follow import MppiFollow
            v_cruise = float(self.declare_parameter("mppi_v_cruise_mps", 0.15, LOOSE).value)
            self.mppi, self.mppi_state = MppiFollow(v_cruise, self.vel_lag_s), None
            if self.mission.p.depth_mode != "assisted":      # assisted: RefNav subscribes to the same topics
                from tf2_ros import Buffer, TransformListener
                self.tf_buffer = Buffer()
                self.tf_listener = TransformListener(self.tf_buffer, self)
                self.create_subscription(CameraInfo, self.info_topic, self._on_info, qos_profile_sensor_data)
                self.create_subscription(Image, self.depth_topic, self._on_depth, qos_profile_sensor_data)

    def _on_odom(self, m: Odometry) -> None:
        p, q, v = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear
        self.pad = S.on_odom(self.v, self.pad, self.mission.phase, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w),
                             (v.x, v.y, v.z))
        now = self.t_odom = self._now()
        if self.mppi is not None:
            st = m.header.stamp
            self.mppi.on_odom(st.sec + st.nanosec * 1e-9, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w), (v.x, v.y, v.z))
        self.mission.on_health(now, S.healthy(self.state, self.t_state, now))
        if self.mission.phase == "arm" and self.pad is not self.logged_pad and S.moved(self.pad, self.logged_pad):
            self._log_start()

    def _log_start(self) -> None:
        pad, self.logged_pad = self.pad, self.pad
        off_m, off_deg = math.hypot(*pad.p0[:2]), math.degrees(pad.yaw0)
        self.get_logger().info(f"take-off pose: odom ({pad.p0[0]:.3f}, {pad.p0[1]:.3f}, {pad.p0[2]:.3f}) yaw "
                               f"{off_deg:.1f} deg; the route is flown from here, along this heading")
        if off_m > self.start_max_m or abs(off_deg) > self.start_max_deg:
            self.mission.request_land(f"take-off pose {off_m:.2f} m, {off_deg:.1f} deg from where odometry started "
                                      f"(start_max_m {self.start_max_m:g}, start_max_deg {self.start_max_deg:g})")

    def _on_state(self, m: String) -> None:
        try:
            self.state = json.loads(m.data)
        except ValueError:
            return
        self.t_state = self._now()
        S.on_state(self.v, self.state, self.mission.phase)

    def _on_depth(self, m: Image) -> None:
        if self.mppi is not None and self.mppi.nav is not None and self.K is not None:
            cal = self._calibration(m.header.frame_id)
            if cal is not None:
                try:
                    d = D.decode_depth(m.encoding, m.height, m.width, m.step, bytes(m.data))
                except ValueError as e:
                    return self._reject("undecodable", str(e), record=False)
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = cal[1], cal[2]
                fx, fy, cx, cy = self.K
                stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                if not self.mppi.on_depth(stamp, d, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), T):
                    self._reject("mppi_no_pose", "MPPI: no odometry pose at the depth frame's stamp", record=False)
        if self.mission.p.depth_mode == "assisted":
            super()._on_depth(m)
        return None

    def _tick(self) -> None:
        # MPPI steps on fresh odometry only; a silence holds through refnav_skybeta.command (HOLD_S), then lands
        now = self._now()
        if self.mppi is not None and self.mission.phase == "follow" and now - self.t_odom <= S.HOLD_S:
            if self.mppi.nav is None:
                self.mppi.start(self.mission.p.route, self.pad)
            self.mission.ext = self.mppi.tick(now, self.pad)
            state = (self.mission.ext or {}).get("state")
            if state != self.mppi_state:
                self.get_logger().info(f"MPPI {self.mppi_state} -> {state}: {self.mission.ext}")
                self.mppi_state = state
        super()._tick()

    def _send(self, out: C.Output) -> None:
        tw, land = S.command(out, self.v, self.pad, self.descent, self.track, self.state, self.t_odom, self.t_state,
                             self._now(), self.mission.p.yaw_rate_dps, self.yaw_offset, self.vel_lag_s,
                             self.mission.p.speed_mps)
        if land:
            self.mission.request_land(land)
        if tw is None:
            return
        msg = TwistStamped()
        msg.header.stamp, msg.header.frame_id = self.get_clock().now().to_msg(), "odom"
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z, msg.twist.angular.z = (float(c) for c in tw)
        self.pub_twist.publish(msg)


def main() -> None:
    refnav_main(node_type=SkybetaRefNav)


if __name__ == "__main__":
    main()
