"""The organizers' flight interface (blueraven.launch.py's, unchanged) plus refnav flying route.json.

    ros2 launch blueraven_skybeta refnav.launch.py                       # depth assisted, the real launch mark
    ros2 launch blueraven_skybeta refnav.launch.py depth_mode:=blind     # waypoints and map only
"""
from pathlib import Path

from blueraven_skybeta.mission_core import _repo_root
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ROOT = _repo_root()


def generate_launch_description():
    interface = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(Path(__file__).resolve().with_name('blueraven.launch.py'))),
        launch_arguments={'competitor': 'false'}.items(),
    )
    refnav = Node(
        package='blueraven_skybeta', executable='refnav_mission', name='refnav', output='screen', emulate_tty=True,
        parameters=[str(ROOT / 'flight.yaml'), {
            'route_file': str(ROOT / 'route.json'),
            'launch_key': LaunchConfiguration('launch_key'),
            'depth_mode': LaunchConfiguration('depth_mode'),
            'depth_topic': '/oak/stereo/image_raw',
            'depth_info_topic': '/oak/stereo/camera_info',
            # the pilot arms and switches whenever he is ready after "wait -> arm"; the descent ends only when PX4
            # disarms or Offboard ends, never in a hold
            'pilot_arms': True, 'wait_s': float('inf'), 'arm_wait_s': float('inf'), 'land_timeout_s': float('inf'),
            'odometry_topic': LaunchConfiguration('odometry_topic'),     # the same odometry as the interface
            'odom_to_ned_yaw_offset_rad': LaunchConfiguration('odom_to_ned_yaw_offset_rad')}],
    )
    return LaunchDescription([
        DeclareLaunchArgument('depth_mode', default_value='assisted', description="'assisted' or 'blind'"),
        DeclareLaunchArgument('launch_key', default_value='launch',
                              description="course.yaml entry: 'launch' (the real mark) or 'launch_sim'"),
        # blueraven.launch.py's own defaults, declared here too so this file can read them for refnav
        DeclareLaunchArgument('odometry_topic', default_value='/visual_slam/tracking/odometry'),
        DeclareLaunchArgument('odom_to_ned_yaw_offset_rad', default_value='0.0'),
        interface, refnav])
