#!/usr/bin/env python3
"""ROS 2 (Humble) wrapper of mission_core for the organizers' skybeta stack. Publishes only
/skybeta/setpoint/pose, /skybeta/setpoint/twist and its own status; calls /skybeta/hold on abort.

    ros2 launch blueraven_skybeta blueraven.launch.py                       # mission:=hover (default)
    ros2 launch blueraven_skybeta blueraven.launch.py mission:=route route_file:=<name or path>.json

mission "hover" (the default, for first flights): take off, hold hover_s over the take-off point,
land. mission "route": the corners come from route_file, (forward, left) metres from the take-off
pose. Flight position is raw odometry in the take-off frame: no course map, no planner, no map
alignment (the field was rebuilt and has no map; Arnav's directive, 2026-11-07). A bad route file is
refused: the node then never publishes, so the interface never enters Offboard. Never arms, never
requests a mode.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from blueraven_skybeta.frame_offset import twist_for_interface
from blueraven_skybeta.mission_core import MISSIONS, MissionParams, SkybetaMission, _repo_root, load_route

ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from nav.frames import quaternion_z, yaw_from_quaternion  # noqa: E402  (pure maths, no map)

ROUTES_DIR = ROOT / "routes"           # the team's route files (mission "route"), shared with refnav


class SkybetaMissionNode(Node):
    def __init__(self) -> None:
        super().__init__("blueraven_mission")
        dp = self.declare_parameter
        # dynamic typing for numbers: hover_s:=10 and hover_s:=10.0 both work (a plain declare types the parameter
        # from its default and rejects an int for a double at start-up); casts below restore the intended types
        from rcl_interfaces.msg import ParameterDescriptor
        loose = ParameterDescriptor(dynamic_typing=True)
        # the SAME value the flight interface gets: it corrects poses for it, but not twists, so we do
        self.yaw_offset = float(dp("odom_to_ned_yaw_offset_rad", 0.0, loose).value)
        rate = float(dp("rate_hz", 20.0, loose).value)
        route_file = dp("route_file", "").value                        # mission "route": the route
        params = MissionParams(**{k: type(v)(dp(k, v, loose).value) for k, v in vars(MissionParams()).items()})
        if params.mission not in MISSIONS:
            raise SystemExit(f"mission must be one of {MISSIONS}, not {params.mission!r} "
                             f"(the map-based 'course' mission was removed: the field has no map)")

        self.mission = None
        if params.mission == "hover":
            self.mission = SkybetaMission(params)
            self.get_logger().info(f"mission HOVER: take off to {params.height_m:.2f} m, hold "
                                   f"{params.hover_s:.0f} s over the take-off point, land. No route.")
        else:
            rf = Path(route_file).expanduser()
            if route_file and not rf.exists() and rf.parent == Path("."):   # a bare name: the repo's routes/
                rf = ROUTES_DIR / rf.name                                # (the one copy, edited in place)
            route_file = str(rf.resolve()) if route_file else ""
            try:
                corners = load_route(route_file) if route_file else None
                problems = None if corners else ["mission 'route' needs route_file:=<path to a route .json>"]
            except ValueError as e:
                problems = [str(e)]
            if problems:
                self.get_logger().error(f"route refused, nothing will be published: {problems}")
            else:
                self.mission = SkybetaMission(params, corners)
                self.get_logger().info(
                    f"mission ROUTE (take-off frame: x forward, y left) from {route_file}: "
                    f"{len(corners) - 1} corners {corners[1:]}, path {self.mission.path.length:.2f} m at "
                    f"{params.height_m:.2f} m, corner arcs {params.route_max_radius_m:.2f} m (0 = stop and turn); "
                    f"lands at the last corner. Raw odometry only: no map, no planner, no map alignment")

        self.pub_pose = self.create_publisher(PoseStamped, "/skybeta/setpoint/pose", 10)
        self.pub_twist = self.create_publisher(TwistStamped, "/skybeta/setpoint/twist", 10)
        self.pub_status = self.create_publisher(String, dp("status_topic", "/blueraven/mission/status").value, 10)
        self.hold = self.create_client(Trigger, "/skybeta/hold")
        self.create_subscription(Odometry, dp("odometry_topic", "/visual_slam/tracking/odometry").value,
                                 self._on_odom, 10)
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
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.mission.on_odom(self._now(), p.x, p.y, p.z, yaw_from_quaternion(q.x, q.y, q.z, q.w),
                             stamp=stamp if stamp > 0.0 else None)

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
            vx, vy = twist_for_interface(cmd.twist[0], cmd.twist[1], self.yaw_offset)
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z, msg.twist.angular.z = (
                float(vx), float(vy), float(cmd.twist[2]), float(cmd.twist[3]))
            self.pub_twist.publish(msg)
        if cmd.call_hold:
            if self.hold.service_is_ready():
                self.hold.call_async(Trigger.Request())
                self.get_logger().warning(f"called /skybeta/hold: {cmd.why}")
            else:
                self.get_logger().error(f"/skybeta/hold not available: {cmd.why}")

    def _status(self) -> None:
        m = self.mission
        st = {"phase": None, "why": "route refused"} if m is None else m.status(self._now())
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
