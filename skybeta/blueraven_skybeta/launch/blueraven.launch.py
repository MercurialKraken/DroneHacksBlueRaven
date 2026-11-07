"""The organizers' flight interface plus Team Blue Raven's mission node.

Copied from competitor_example/launch/competitor.launch.py (skybeta 0.3.3): the flight_interface
node and its arguments are unchanged, auto_offboard_on_arm defaults to true as the guide requires.
Only the competitor node is swapped for ours.

    ros2 launch blueraven_skybeta blueraven.launch.py                      # a HOVER (the default)
    ros2 launch blueraven_skybeta blueraven.launch.py mission:=course height_m:=0.7 cruise_mps:=0.3 cp4:=gate_b
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
]
MISSION_ARGS = [
    ('mission', 'hover', "'hover' (take off, hold hover_s, land: first flights) or 'course'"),
    ('hover_s', '5.0', 'hover: seconds to hold at height_m before landing'),
    ('launch_key', 'launch', "course.yaml entry: 'launch' (the real mark) or 'launch_sim'"),
    ('height_m', '0.7', 'flight height above the launch point, metres'),
    ('cruise_mps', '0.3', 'cruise speed, m/s'),
    ('cp4', 'gate_b', "checkpoint 4 route: 'gate_b' or 'window'"),
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
        parameters=[{n: LaunchConfiguration(n)
                     for n in ('mission', 'hover_s', 'launch_key', 'height_m', 'cruise_mps', 'cp4')} | {
                    'odometry_topic': LaunchConfiguration('odometry_topic')}],   # the same odometry as the interface
    )
    return LaunchDescription(declared + [flight_interface, mission])
