#!/usr/bin/env python3
"""refnav's node with its PX4 I/O replaced by the flight interface (refnav_skybeta): odometry and
/skybeta/state in, /skybeta/setpoint/twist out. Run: ros2 launch blueraven_skybeta refnav.launch.py"""
from __future__ import annotations

import json

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from blueraven_skybeta import refnav_skybeta as S  # first: it puts the team repo (refnav/) on sys.path
from refnav import core as C
from refnav.node import RefNav
from refnav.node import main as refnav_main


class SkybetaRefNav(RefNav):
    def _io(self) -> None:
        # the SAME value the flight interface gets: it rotates poses by it, not twists, so we do
        self.yaw_offset = float(self.declare_parameter("odom_to_ned_yaw_offset_rad", 0.0).value)
        self.state: dict | None = None
        self.t_state = -1e9
        self.pad: S.Pad | None = None
        self.descent = S.Descent()
        self.pub_twist = self.create_publisher(TwistStamped, "/skybeta/setpoint/twist", 10)
        self.create_subscription(Odometry, self.declare_parameter(
            "odometry_topic", "/visual_slam/tracking/odometry").value, self._on_odom, 10)
        self.create_subscription(String, "/skybeta/state", self._on_state, 10)

    def _on_odom(self, m: Odometry) -> None:
        p, q, v = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear
        self.pad = S.on_odom(self.v, self.pad, self.mission.phase, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w),
                             (v.x, v.y, v.z))
        now = self._now()
        self.mission.on_health(now, S.healthy(self.state, self.t_state, now))

    def _on_state(self, m: String) -> None:
        try:
            self.state = json.loads(m.data)
        except ValueError:
            return
        self.t_state = self._now()
        S.on_state(self.v, self.state, self.mission.phase)

    def _send(self, out: C.Output) -> None:
        if out.phase == "land":
            tw = self.descent.twist(self._now(), self.v, self.state)
        else:
            tw = None if self.pad is None else S.twist(out, self.v, self.pad, self.mission.p.yaw_rate_dps,
                                                       self.yaw_offset)
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
