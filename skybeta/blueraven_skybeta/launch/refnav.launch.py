"""The organizers' flight interface (blueraven.launch.py's, unchanged) plus refnav flying route.json.

    ros2 launch blueraven_skybeta refnav.launch.py                           # the drone: blind, 0.7 m
    ros2 launch blueraven_skybeta refnav.launch.py depth_mode:=assisted      # + depth stops (needs a depth topic)
    ros2 launch blueraven_skybeta refnav.launch.py use_sim_time:=true depth_topic:=/camera/depth/image_raw \
        depth_info_topic:=/camera/depth/camera_info launch_key:=launch_sim odom_to_ned_yaw_offset_rad:=-1.5708  # T4
"""
from pathlib import Path

from blueraven_skybeta.mission_core import _repo_root
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

ROOT = _repo_root()


def generate_launch_description():
    # a float for both nodes: the interface declares a double, and :=0 would be an integer
    offset = PythonExpression(['float(', LaunchConfiguration('odom_to_ned_yaw_offset_rad'), ')'])
    interface = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(Path(__file__).resolve().with_name('blueraven.launch.py'))),
        launch_arguments={'competitor': 'false', 'odom_to_ned_yaw_offset_rad': offset}.items(),
    )
    refnav = Node(
        package='blueraven_skybeta', executable='refnav_mission', name='refnav', output='screen', emulate_tty=True,
        parameters=[str(ROOT / 'flight.yaml'), {
            'route_file': str(ROOT / 'route.json'),
            'launch_key': LaunchConfiguration('launch_key'),
            'height_m': LaunchConfiguration('height_m'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'depth_mode': LaunchConfiguration('depth_mode'),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'depth_info_topic': LaunchConfiguration('depth_info_topic'),
            # the pilot arms and switches whenever he is ready after "wait -> arm"; the descent ends only when PX4
            # disarms or Offboard ends, never in a hold
            'pilot_arms': True, 'wait_s': float('inf'), 'arm_wait_s': float('inf'), 'land_timeout_s': float('inf'),
            'odometry_topic': LaunchConfiguration('odometry_topic'),     # the same odometry as the interface
            'odom_to_ned_yaw_offset_rad': offset}],
    )
    return LaunchDescription([
        # blind on the drone: the README forbids reading /oak/* images in flight (it delays the VIO PX4 flies on)
        DeclareLaunchArgument('depth_mode', default_value='blind', description="'blind' or 'assisted'"),
        DeclareLaunchArgument('depth_topic', default_value='/oak/stereo/image_raw'),
        DeclareLaunchArgument('depth_info_topic', default_value='/oak/stereo/camera_info'),
        DeclareLaunchArgument('height_m', default_value='0.7', description='flight height above the pad, metres'),
        DeclareLaunchArgument('use_sim_time', default_value='false', description="'true' on the T4 simulator"),
        DeclareLaunchArgument('launch_key', default_value='launch',
                              description="course.yaml entry: 'launch' (the real mark) or 'launch_sim'"),
        # blueraven.launch.py's own defaults, declared here too so this file can read them for refnav
        DeclareLaunchArgument('odometry_topic', default_value='/visual_slam/tracking/odometry'),
        DeclareLaunchArgument('odom_to_ned_yaw_offset_rad', default_value='0.0'),
        interface, refnav])
