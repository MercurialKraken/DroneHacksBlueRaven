#!/usr/bin/env bash
# Task C acceptance 3: Jetson utilization with the organizers' stack running, idle vs with our depth node,
# plus proof the organizers' VIO still holds 30 Hz with vo_state 1. Run ON THE JETSON HOST (tegrastats is
# a host tool), props off, on USB power. Nothing here installs anything or touches /fmu/in.
#
#   tools/jetson_budget.sh LABEL SECONDS [ROS_CONTAINER]
#     LABEL          e.g. idle, or depth   (start the depth node yourself before the 'depth' run)
#     SECONDS        capture length, e.g. 60
#     ROS_CONTAINER  container that has ros2 and sees the organizers' topics (default: the first running one)
#
# docker exec uses bash -ic: in skybeta:0.3.2 ros2 is set up by the interactive bashrc, not a login shell (arnav-claude).
# Writes budget_LABEL_tegrastats.log, budget_LABEL_vio_hz.txt, budget_LABEL_vo_state.txt; then
#   python3 tools/jetson_budget.py budget_idle_tegrastats.log budget_depth_tegrastats.log
set -euo pipefail
LABEL=${1:?label}; SECS=${2:?seconds}; CT=${3:-$(docker ps --format '{{.Names}}' | head -1)}
echo "power mode: $(sudo nvpmodel -q 2>/dev/null | head -1 || echo unknown)   container: $CT"
rm -f "budget_${LABEL}_tegrastats.log"
sudo tegrastats --interval 500 --logfile "budget_${LABEL}_tegrastats.log" &
TP=$!
docker exec "$CT" bash -ic "timeout ${SECS} ros2 topic hz /visual_slam/tracking/odometry --window 300" > "budget_${LABEL}_vio_hz.txt" 2>&1 &
HP=$!
docker exec "$CT" bash -ic "timeout ${SECS} ros2 topic echo --field vo_state /visual_slam/status" > "budget_${LABEL}_vo_state.txt" 2>&1 &
VP=$!
if [ "$LABEL" != idle ]; then
  docker exec "$CT" bash -ic "timeout ${SECS} ros2 topic echo --field data /stereo_depth/status" > "budget_${LABEL}_depth_status.txt" 2>&1 &
fi
sleep "$SECS"
sudo tegrastats --stop || kill "$TP" 2>/dev/null || true
wait "$HP" "$VP" 2>/dev/null || true
echo "VIO rate (last lines):"; tail -3 "budget_${LABEL}_vio_hz.txt"
echo "vo_state counts:"; grep -E '^[0-9]+$' "budget_${LABEL}_vo_state.txt" | sort | uniq -c
echo "samples: $(wc -l < "budget_${LABEL}_tegrastats.log")"
