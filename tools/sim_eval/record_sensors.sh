#!/usr/bin/env bash
# Raw sensor bag, used to re-run cuVSLAM offline with different settings (replay_vio.sh).
# About 8 MB/s. Saved OUTSIDE the workspace, because the workspace archive is capped at
# 200 MB / 2000 files.
# usage: tools/sim_eval/record_sensors.sh <run_name>        (Ctrl+C to stop)
set -euo pipefail
name=${1:?usage: record_sensors.sh <run_name>}
dir=/home/flyx/bags
mkdir -p "$dir"
exec ros2 bag record -s mcap -o "$dir/$name" \
  /tf_static \
  /stereo/left/image /stereo/right/image \
  /stereo/left/camera_info /stereo/right/camera_info \
  /stereo/imu
