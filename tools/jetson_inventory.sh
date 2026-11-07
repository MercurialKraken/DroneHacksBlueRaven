#!/usr/bin/env bash
# Read-only inventory of the drone's Jetson: changes nothing, publishes nothing, arms nothing.
# usage: bash tools/jetson_inventory.sh 2>&1 | tee jetson_inventory.txt
set -u
h() { printf '\n===== %s\n' "$1"; }
c() { printf '\n$ %s\n' "$1"; timeout 25 bash -c "$1" 2>&1 | head -n "${2:-30}" || true; }

if [ -z "${ROS_DISTRO:-}" ]; then
  for f in /opt/ros/*/setup.bash; do [ -f "$f" ] && . "$f" && echo "(sourced $f)" && break; done
fi

h "System"
c "cat /etc/nv_tegra_release"
c "lsb_release -ds; uname -m; python3 --version"
c "sudo -n nvpmodel -q 2>/dev/null || nvpmodel -q 2>/dev/null || echo 'nvpmodel: not readable'"
c "df -h / | tail -1; free -h | head -2"

h "ROS 2"
c 'echo "ROS_DISTRO=${ROS_DISTRO:-unset} ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset} RMW=${RMW_IMPLEMENTATION:-default}"'
c "ls /opt/ros"
c "ros2 pkg list | grep -E 'depthai|px4|isaac|visual_slam|launch_ros|tf2_ros$|robot_state'"
c "find /home /opt /root -maxdepth 6 -type d -name px4_msgs -path '*install*' 2>/dev/null | head"

h "cuVSLAM"
c "python3 -c 'import cuvslam; print(\"python cuvslam\", cuvslam.get_version())'"
c "pip3 list 2>/dev/null | grep -i -E 'cuvslam|numpy|scipy|pytest|depthai'"
c "docker ps --format '{{.Image}}  {{.Names}}' 2>/dev/null"
c "docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -i -E 'isaac|vslam|slam|ros'"

h "PX4 link (uXRCE-DDS agent)"
c "pgrep -af -i microxrce || echo 'no MicroXRCEAgent process'"
c "command -v MicroXRCEAgent micro-xrce-dds-agent; snap list 2>/dev/null | grep -i xrce"
c "systemctl list-units --all --no-pager 2>/dev/null | grep -i -E 'xrce|micro|px4|ros|depthai|oak'"
c "ls -l /dev/ttyTHS* /dev/ttyACM* /dev/ttyUSB* 2>/dev/null"
c "ros2 topic list | grep -c '^/fmu/' ; ros2 topic list | grep -E '^/fmu/out/(vehicle_status|vehicle_local_position|vehicle_odometry|distance_sensor|estimator_status_flags)'"
c "python3 -c 'from px4_msgs.msg import VehicleStatus as S, VehicleLocalPosition as P, VehicleOdometry as O; print({k.__name__: getattr(k, \"MESSAGE_VERSION\", 0) for k in (S, P, O)})'"

h "OAK camera"
c "lsusb | grep -i -E '03e7|movidius|luxonis' || echo 'no OAK on USB'"
c "python3 -c 'import depthai; print(\"depthai\", depthai.__version__)'"
c "ros2 pkg prefix depthai_ros_driver && ls \$(ros2 pkg prefix depthai_ros_driver)/share/depthai_ros_driver/launch"

h "This repo"
c "cd \"\$(dirname \"$0\")/..\" && pwd && git log --oneline -1 2>/dev/null"
c "python3 -c 'import numpy, sys; print(sys.version.split()[0], \"numpy\", numpy.__version__)'"
echo; echo "done: paste this output back."
