"""SIM-ONLY: the drone's own launch file, unchanged, with every node on simulation time.

    ros2 launch tools/drone_sim/sim.launch.py mission:=blueraven_skybeta/blueraven.launch.py [ARG:=VALUE ...]

`mission` is PACKAGE/LAUNCH_FILE; every other ARG:=VALUE reaches the included launch file as on the drone.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import SetParameter
from launch_ros.substitutions import FindPackageShare


def include_mission(context):
    package, launch_file = context.launch_configurations['mission'].split('/')
    path = FindPackageShare(package).perform(context) + '/launch/' + launch_file
    return [IncludeLaunchDescription(PythonLaunchDescriptionSource(path))]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mission', description='PACKAGE/LAUNCH_FILE of the drone program'),
        SetParameter(name='use_sim_time', value=True),
        OpaqueFunction(function=include_mission),
    ])
