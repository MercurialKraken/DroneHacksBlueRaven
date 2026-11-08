"""The organizers' flight interface (as blueraven_skybeta/launch/blueraven.launch.py: skybeta 0.3.3's
competitor.launch.py arguments, unchanged) plus the wall-seek mission.

    ros2 launch blueraven_wallseek wallseek.launch.py                       # skybeta-09 (offset 0.0, OAK depth)
    ros2 launch blueraven_wallseek wallseek.launch.py use_sim_time:=true odom_to_ned_yaw_offset_rad:=-1.5708 \
        depth_topic:=/camera/depth/image_raw depth_info_topic:=/camera/depth/camera_info lag_s:=1.5 \
        max_forward_m:=1.10                                                 # the T4 (spawn is 0.2 m further north)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

INTERFACE_ARGS = [
    ('max_altitude_m', '3.0', 'ceiling above the odometry origin, metres'),
    ('min_altitude_m', '0.3', 'floor above the odometry origin, metres'),
    ('max_speed_mps', '1.5', 'cap on the norm of a twist setpoint, m/s'),
    ('setpoint_rate_hz', '10.0', 'OffboardControlMode heartbeat rate -- PX4 needs > 2 Hz'),
    ('allow_mode_request', 'false', 'enable /skybeta/request_offboard. SIM ONLY'),
    ('auto_offboard_on_arm', 'true', 'enter Offboard once when the pilot arms, then switches to Position'),
    ('odometry_topic', '/visual_slam/tracking/odometry', 'source of the ENU pose'),
    ('anchor_setpoints', 'false', 'express setpoints as a displacement from the vision origin (fixes translation)'),
    ('odom_to_ned_yaw_offset_rad', '0.0', 'measured constant yaw between vision odom and PX4 (fixes rotation)'),
]
# the mission's tunables (wallseek_core.Params): Darion's numbers are the defaults
MISSION_ARGS = [
    ('height_m', '0.40'), ('standoff_m', '0.80'),
    ('turn_e_m', '1.10'), ('gap_n_m', '1.239'), ('gap_e_m', '1.417'), ('gap_past_m', '0.50'),
    # route mode (default, v3): P1 by depth (0.80 m), face P2 (past the CP2 gap centre), straight to it, face P4, land
    ('p2_past_m', '0.45'), ('p4_e_m', '2.80'), ('turn_rate_dps', '15.0'), ('leg_mps', '0.12'),
    ('leg_climb_to_m', '0.55'),     # climb during the gap leg to this height (Darion: 0.55; 0 = stay at height_m)
    ('approach_mps', '0.15'), ('translate_mps', '0.10'),
    ('past_end_m', '0.35'), ('max_forward_m', '1.30'), ('max_right_m', '2.0'), ('lag_s', '0.55'),
    ('camera_forward_m', '0.1053'), ('camera_left_m', '0.0376'), ('camera_up_m', '0.0304'),
    ('takeoff_kick_mps', '0.06'), ('var_check_after_m', '1.0'),
]


def generate_launch_description():
    f = lambda n: PythonExpression(['float(', LaunchConfiguration(n), ')'])     # noqa: E731  (:=0 would be an int)
    declared = ([DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in INTERFACE_ARGS]
                + [DeclareLaunchArgument(n, default_value=d) for n, d in MISSION_ARGS]
                + [DeclareLaunchArgument('slide_immediately', default_value='false',
                                         description='variant: no hover stop at the wall, slide right at once'),
                   DeclareLaunchArgument('use_sim_time', default_value='false'),
                   DeclareLaunchArgument('depth_topic', default_value='/oak/stereo/image_raw'),
                   DeclareLaunchArgument('depth_info_topic', default_value='/oak/stereo/camera_info')])
    flight_interface = Node(
        package='skybeta_flight_interface', executable='flight_interface', name='flight_interface',
        output='screen', emulate_tty=True,
        parameters=[{n: LaunchConfiguration(n) for n, _, _ in INTERFACE_ARGS}
                    | {'odom_to_ned_yaw_offset_rad': f('odom_to_ned_yaw_offset_rad'),
                       'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )
    mission = Node(
        package='blueraven_wallseek', executable='wallseek', name='wallseek', output='screen', emulate_tty=True,
        parameters=[{n: f(n) for n, _ in MISSION_ARGS}
                    | {'odom_to_ned_yaw_offset_rad': f('odom_to_ned_yaw_offset_rad'),
                       'slide_immediately': PythonExpression(["'", LaunchConfiguration('slide_immediately'),
                                                              "'.lower() == 'true'"]),
                       'odometry_topic': LaunchConfiguration('odometry_topic'),
                       'depth_topic': LaunchConfiguration('depth_topic'),
                       'depth_info_topic': LaunchConfiguration('depth_info_topic'),
                       'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )
    return LaunchDescription(declared + [flight_interface, mission])
