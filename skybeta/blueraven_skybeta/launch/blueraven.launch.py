"""The organizers' flight interface plus Team Blue Raven's mission node.

Copied from competitor_example/launch/competitor.launch.py (skybeta 0.3.3): the flight_interface
node and its arguments are unchanged, auto_offboard_on_arm defaults to true as the guide requires.
Only the competitor node is swapped for ours. No map, planner or map alignment can be started from
here: flight position is raw odometry in the take-off frame (Arnav's directive, 2026-11-07).

    ros2 launch blueraven_skybeta blueraven.launch.py                      # a HOVER (the default)
    ros2 launch blueraven_skybeta blueraven.launch.py mission:=route route_file:=<name>.json
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

INTERFACE_ARGS = [
    ('max_altitude_m', '3.0', 'ceiling above the odometry origin, metres'),
    ('min_altitude_m', '0.3', 'floor above the odometry origin, metres'),
    ('max_speed_mps', '1.5', 'cap on the norm of a twist setpoint, m/s'),
    ('setpoint_rate_hz', '10.0', 'OffboardControlMode heartbeat rate -- PX4 needs > 2 Hz'),
    ('allow_mode_request', 'false', 'enable /skybeta/request_offboard. SIM ONLY'),
    ('auto_offboard_on_arm', 'true', 'enter Offboard once when the pilot arms, then switches to Position'),
    ('odometry_topic', '/visual_slam/tracking/odometry', 'source of the ENU pose'),
    # the guide's frame caveat (competitor_guide.md section 1): set only once measured on the bench
    ('anchor_setpoints', 'false', 'express setpoints as a displacement from the vision origin (fixes translation)'),
    ('odom_to_ned_yaw_offset_rad', '0.0', 'measured constant yaw between vision odom and PX4 (fixes rotation)'),
]
MISSION_ARGS = [
    ('mission', 'hover', "'hover' (take off, hold hover_s, land) or 'route' (fly route_file's corners, land)"),
    ('hover_s', '5.0', 'hover: seconds to hold at height_m before landing'),
    ('route_file', '', "route: {\"route\": [[forward_m, left_m], ...]} from take-off; a bare name is in routes/"),
    ('route_max_radius_m', '0.0', 'route: corner arc radius; 0 = stop and turn at every corner'),
    ('route_hover_s', '3.0', 'route: seconds to hover over the take-off point before flying'),
    ('max_yaw_rate_dps', '45.0', 'cap on the turn rate, deg/s (an in-place 90 deg turn at 45 slipped VIO on the T4)'),
    ('height_m', '0.7', 'flight height above the launch point, metres'),
    ('cruise_mps', '0.3', 'cruise speed, m/s'),
    ('competitor', 'true', 'also start the mission node'),
]


def generate_launch_description():
    declared = [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in INTERFACE_ARGS + MISSION_ARGS]
    flight_interface = Node(
        package='skybeta_flight_interface', executable='flight_interface', name='flight_interface',
        output='screen', emulate_tty=True,
        parameters=[{n: LaunchConfiguration(n) for n, _, _ in INTERFACE_ARGS}],
    )
    mission = Node(
        package='blueraven_skybeta', executable='mission', name='blueraven_mission',
        output='screen', emulate_tty=True, condition=IfCondition(LaunchConfiguration('competitor')),
        parameters=[{n: LaunchConfiguration(n) for n, _, _ in MISSION_ARGS if n != 'competitor'} | {
                    'odometry_topic': LaunchConfiguration('odometry_topic'),     # the same odometry as the interface
                    'odom_to_ned_yaw_offset_rad': LaunchConfiguration('odom_to_ned_yaw_offset_rad')}],   # twists too
    )
    return LaunchDescription(declared + [flight_interface, mission])
