#!/usr/bin/env python3
"""ROS 2 (Humble) wrapper of mission_core for the organizers' skybeta stack. Publishes only
/skybeta/setpoint/pose, /skybeta/setpoint/twist and its own status; calls /skybeta/hold on abort.

    ros2 launch blueraven_skybeta blueraven.launch.py launch_key:=launch height_m:=0.7 cruise_mps:=0.3

Route: built at start-up from the course map by nav.route_planner (Arnav's startup planner), the
corners in the map frame (.map_xy). A plan with gate problems is refused: the node then never
publishes, so the interface never enters Offboard. Never arms, never requests a mode.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from blueraven_skybeta.mission_core import MissionParams, SkybetaMission, _repo_root

ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from nav.course_map import CourseMap  # noqa: E402
from nav.frames import load_launch, quaternion_z, yaw_from_quaternion  # noqa: E402


class SkybetaMissionNode(Node):
    def __init__(self) -> None:
        super().__init__("blueraven_mission")
        dp = self.declare_parameter
        sdf = Path(dp("course_map", str(ROOT / "maps" / "course_sdf.npz")).value)
        cfg = Path(dp("course_config", str(ROOT / "maps" / "course.yaml")).value)
        launch_key = dp("launch_key", "launch").value                  # the real mark on the drone
        cp4 = dp("cp4", "gate_b").value                                # "gate_b" | "window"
        rate = float(dp("rate_hz", 20.0).value)
        params = MissionParams(**{k: type(v)(dp(k, v).value) for k, v in vars(MissionParams()).items()})

        course = CourseMap.load(sdf)
        launch = load_launch(cfg, launch_key)
        try:
            from nav.route_planner import gate_problems, plan_from_files
        except ImportError as e:
            raise SystemExit(f"nav.route_planner is missing ({e}): merge arnav/startup-planner") from None
        plan = plan_from_files(sdf, cfg, launch_key, params.height_m, cp4)
        problems = gate_problems(plan)
        if problems:
            self.get_logger().error(f"route refused, nothing will be published: {problems}")
            self.mission = None
        else:
            self.mission = SkybetaMission(params, launch, plan.map_xy, course)
            self.get_logger().info(f"route {len(plan.map_xy)} corners, {self.mission.path.length:.2f} m "
                                   f"(digest {getattr(plan, 'digest', '?')}); launch {launch_key} {launch}")

        self.pub_pose = self.create_publisher(PoseStamped, "/skybeta/setpoint/pose", 10)
        self.pub_twist = self.create_publisher(TwistStamped, "/skybeta/setpoint/twist", 10)
        self.pub_status = self.create_publisher(String, dp("status_topic", "/blueraven/mission/status").value, 10)
        self.hold = self.create_client(Trigger, "/skybeta/hold")
        self.create_subscription(Odometry, "/visual_slam/tracking/odometry", self._on_odom, 10)
        self.create_subscription(String, "/skybeta/state", self._on_state, 10)
        self.create_timer(1.0 / rate, self._tick)
        self.create_timer(0.5, self._status)
        self.last = None

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_odom(self, m: Odometry) -> None:
        if self.mission is None:
            return
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.mission.on_odom(self._now(), p.x, p.y, p.z, yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def _on_state(self, m: String) -> None:
        if self.mission is None:
            return
        try:
            self.mission.on_state(self._now(), json.loads(m.data))
        except ValueError:
            self.get_logger().warning("unparseable /skybeta/state ignored", throttle_duration_sec=5.0)

    def _tick(self) -> None:
        if self.mission is None:
            return
        cmd = self.mission.step(self._now())
        self.last = cmd
        for e in cmd.events:
            self.get_logger().info(e)
        stamp = self.get_clock().now().to_msg()
        if cmd.pose is not None:
            msg = PoseStamped()
            msg.header.stamp, msg.header.frame_id = stamp, "odom"
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(c) for c in cmd.pose)
            if cmd.pose_yaw is not None:                                # all-zero quaternion = hold heading
                (msg.pose.orientation.x, msg.pose.orientation.y,
                 msg.pose.orientation.z, msg.pose.orientation.w) = quaternion_z(cmd.pose_yaw)
            else:
                msg.pose.orientation.w = 0.0
            self.pub_pose.publish(msg)
        elif cmd.twist is not None:
            msg = TwistStamped()
            msg.header.stamp, msg.header.frame_id = stamp, "odom"
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z, msg.twist.angular.z = (
                float(c) for c in cmd.twist)
            self.pub_twist.publish(msg)
        if cmd.call_hold:
            if self.hold.service_is_ready():
                self.hold.call_async(Trigger.Request())
                self.get_logger().warning(f"called /skybeta/hold: {cmd.why}")
            else:
                self.get_logger().error(f"/skybeta/hold not available: {cmd.why}")

    def _status(self) -> None:
        m = self.mission
        st = {"phase": None if m is None else m.phase, "why": "route refused" if m is None else m.why,
              "s_ref": None if m is None else round(m.s_ref, 2),
              "length": None if m is None else round(m.path.length, 2)}
        self.pub_status.publish(String(data=json.dumps(st)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SkybetaMissionNode()
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
