#!/usr/bin/env bash
# Re-run cuVSLAM on a recorded sensor bag in an ISOLATED ROS domain (not 42, the live sim)
# and log its odometry with record_eval.py, for comparison with analyze.py.
#
# usage: tools/sim_eval/replay_vio.sh <bag_dir> <out_dir> [extra cuVSLAM ros-args...]
#   e.g. tools/sim_eval/replay_vio.sh /home/flyx/bags/flight1 runs/flight1/replay_1x \
#          -p gyro_noise_density:=0.00034 -p accel_noise_density:=0.0025
set -euo pipefail
bag=${1:?usage: replay_vio.sh <bag_dir> <out_dir> [ros-args...]}
out=${2:?usage: replay_vio.sh <bag_dir> <out_dir> [ros-args...]}
shift 2
export ROS_DOMAIN_ID=${REPLAY_DOMAIN_ID:-43}
if [ "$ROS_DOMAIN_ID" = "42" ]; then echo "refusing: domain 42 is the live sim" >&2; exit 1; fi
cd "$(dirname "$0")/../.."
mkdir -p "$out"

# imu_frequency 480 matches the live deployment override in /etc/flyx/onboard.yaml.
python3 -m onboard.slam.cuvslam_lib_node --ros-args \
  --params-file onboard/config/vio.yaml -p use_sim_time:=true -p imu_frequency:=480.0 "$@" \
  > "$out/cuvslam.log" 2>&1 &
vio=$!
python3 tools/sim_eval/record_eval.py --out "$out" --no-truth > "$out/recorder.log" 2>&1 &
rec=$!
trap 'kill -INT $rec $vio 2>/dev/null || true' EXIT

sleep 5                                   # DDS discovery
ros2 bag play "$bag" --clock
sleep 2                                   # let the last frames drain
kill -INT "$rec" "$vio" 2>/dev/null || true
wait "$rec" "$vio" 2>/dev/null || true
trap - EXIT
echo "replay done: $out  (cuVSLAM log: $out/cuvslam.log)"
tail -n 2 "$out/cuvslam.log" || true
