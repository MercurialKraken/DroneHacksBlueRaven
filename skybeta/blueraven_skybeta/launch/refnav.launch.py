"""The organizers' flight interface (blueraven.launch.py's, unchanged) plus refnav flying a route file's
waypoints (courses/<course>/routes/<route>.json, forward/left from the take-off pose). Nothing reads the map in flight.

    ros2 launch blueraven_skybeta refnav.launch.py                                 # the drone: full course, courses/ACTIVE
    ros2 launch blueraven_skybeta refnav.launch.py course:=v3-published route:=route_v5
    ros2 launch blueraven_skybeta refnav.launch.py route:=route_v6_start2_stage1   # through CP2 and land
    ros2 launch blueraven_skybeta refnav.launch.py nav:=mppi                       # Arnav's MPPI follows (reads depth)
    ros2 launch blueraven_skybeta refnav.launch.py route:=route_v6_start2_sim use_sim_time:=true \
        depth_topic:=/camera/depth/image_raw depth_info_topic:=/camera/depth/camera_info \
        odom_to_ned_yaw_offset_rad:=-1.5708                                          # the T4 simulator
"""
from pathlib import Path

from blueraven_skybeta.mission_core import _repo_root
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
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
    route_file = [str(ROOT / 'courses') + '/', LaunchConfiguration('course'), '/routes/', LaunchConfiguration('route'), '.json']
    refnav = Node(
        package='blueraven_skybeta', executable='refnav_mission', name='refnav', output='screen', emulate_tty=True,
        parameters=[str(ROOT / 'flight.yaml'), {
            'route_file': route_file,
            'path_source': 'waypoints', 'corner_cut_m': 0.05,
            'xtrack_slow_m': 0.08, 'xtrack_stop_m': 0.20,       # 0.20 m off the path the reference waits for it
            # nose within ~0.5 m of the path ahead: on the T4 the velocity drifts ~0.75 x the nose's angle off it, the other way
            'look_s': 0.5, 'look_max_m': 0.5,
            'vel_lag_s': LaunchConfiguration('vel_lag_s'),
            'height_m': LaunchConfiguration('height_m'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'depth_mode': LaunchConfiguration('depth_mode'),
            'nav': LaunchConfiguration('nav'),
            **{k: PythonExpression(['float(', LaunchConfiguration(k), ')'])
               for k in ('mppi_v_cruise_mps', 'mppi_tau_s', 'mppi_dead_time_s')},
            'depth_topic': LaunchConfiguration('depth_topic'),
            'depth_info_topic': LaunchConfiguration('depth_info_topic'),
            # the pilot arms and switches whenever he is ready after "wait -> arm"; the descent ends only when PX4
            # disarms or Offboard ends, never in a hold
            'pilot_arms': True, 'wait_s': float('inf'), 'arm_wait_s': float('inf'), 'land_timeout_s': float('inf'),
            'odometry_topic': LaunchConfiguration('odometry_topic'),     # the same odometry as the interface
            'odom_to_ned_yaw_offset_rad': offset}],
    )
    mppi = {k: PythonExpression(['float(', LaunchConfiguration(k), ')'])
            for k in ('mppi_v_cruise_mps', 'mppi_tau_s', 'mppi_dead_time_s', 'height_m')}
    # MPPI in shadow: its own process at low priority, publishing only /blueraven/mppi/*; never commands the drone
    shadow = Node(
        package='blueraven_skybeta', executable='mppi_shadow', name='mppi_shadow', output='screen', prefix='nice -n 10',
        condition=IfCondition(LaunchConfiguration('mppi_shadow')),
        parameters=[{**mppi, 'route_file': route_file,
                     'use_sim_time': LaunchConfiguration('use_sim_time'),
                     'odometry_topic': LaunchConfiguration('odometry_topic'),
                     'depth_topic': LaunchConfiguration('depth_topic'),
                     'depth_info_topic': LaunchConfiguration('depth_info_topic')}],
    )
    return LaunchDescription([
        DeclareLaunchArgument('mppi_shadow', default_value='false',
                              description='true: MPPI runs alongside refnav on the same inputs, publishing /blueraven/mppi/* only'),
        # blind: refnav's own depth checks off. nav:=mppi reads /oak/stereo/image_raw (on-camera depth) whatever this is
        DeclareLaunchArgument('depth_mode', default_value='blind', description="'blind' or 'assisted'"),
        DeclareLaunchArgument('nav', default_value='refnav', description="follower: 'refnav' or 'mppi'"),
        DeclareLaunchArgument('mppi_v_cruise_mps', default_value='0.15', description="MPPI's cruise speed, m/s"),
        # MPPI's model of PX4's velocity response: the T4 fit (0.9 s lag + 0.6 s dead time, Arnav's FLIGHT_MPPI) until the drone's is fitted
        DeclareLaunchArgument('mppi_tau_s', default_value='0.9'),
        DeclareLaunchArgument('mppi_dead_time_s', default_value='0.6'),
        DeclareLaunchArgument('depth_topic', default_value='/oak/stereo/image_raw'),
        DeclareLaunchArgument('depth_info_topic', default_value='/oak/stereo/camera_info'),
        DeclareLaunchArgument('height_m', default_value='0.7', description='flight height above the pad, metres'),
        DeclareLaunchArgument('vel_lag_s', default_value='0.55',
                              description='twist -> velocity lag: 0.55 on the drone (1 / MPC_XY_VEL_P_ACC 1.8), 1.5 on the T4 sim (fitted)'),
        DeclareLaunchArgument('use_sim_time', default_value='false', description="'true' on the T4 simulator"),
        DeclareLaunchArgument('course', default_value=(ROOT / 'courses' / 'ACTIVE').read_text().strip(),
                              description="courses/<course>/: the course the route belongs to"),
        DeclareLaunchArgument('route', default_value='route_v6_start2',
                              description="courses/<course>/routes/<route>.json: *_sim files start at the sim spawn"),
        # blueraven.launch.py's own defaults, declared here too so this file can read them for refnav
        DeclareLaunchArgument('odometry_topic', default_value='/visual_slam/tracking/odometry'),
        DeclareLaunchArgument('odom_to_ned_yaw_offset_rad', default_value='0.0'),
        interface, refnav, shadow])
