"""blueraven.launch.py with mission:=route: the hard-coded route, for launchers that use the name `mission` for
something else (hinson/real-drone's tools/drone_sim/sim.launch.py takes mission:=PACKAGE/LAUNCH_FILE).

    ros2 launch blueraven_skybeta route.launch.py route_file:=<path or name>.json [odom_to_ned_yaw_offset_rad:=...]

Every other argument is blueraven.launch.py's own.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetLaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    main = os.path.join(get_package_share_directory('blueraven_skybeta'), 'launch', 'blueraven.launch.py')
    return LaunchDescription([
        SetLaunchConfiguration('mission', 'route'),     # before the include: its declared default doesn't override it
        IncludeLaunchDescription(PythonLaunchDescriptionSource(main)),
    ])
