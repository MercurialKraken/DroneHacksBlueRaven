#!/bin/bash
# SIM-ONLY, inside the dronehacks_gpu container: one flight of the drone program as it runs on the drone.
# Usage: run.sh [--course NAME] [--depth none|sim] [--latency-ms MS] [--drop P] [--route FILE|none] [--timeout S] [--name NAME] [ARG:=VALUE ...]
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
MISSION=${MISSION:-blueraven_skybeta/blueraven.launch.py}    # PACKAGE/LAUNCH_FILE of the drone program
YAW_OFF=-1.5708                                               # T4: cuVSLAM odom is -90 deg from PX4 NED
SKYBETA_GIT=https://github.com/Artash-N/skybeta.git; SKYBETA_SHA=3c42933    # flight interface 0.3.3, as on skybeta-09
WS=/tmp/drone_sim_ws; DASH=http://127.0.0.1:8080
WORLD_COURSE=$(readlink /opt/devenv/courses/active)          # the course the T4 world was built from (use_course.sh)
COURSE=$WORLD_COURSE; DEPTH=none; LATENCY_MS=0; DROP=0.01; ROUTE=; TIMEOUT=300; NAME=run
while [[ $# -gt 0 && $1 == --* ]]; do
  case $1 in
    --course) COURSE=$2 ;; --depth) DEPTH=$2 ;; --latency-ms) LATENCY_MS=$2 ;; --drop) DROP=$2 ;;
    --route) ROUTE=$2 ;; --timeout) TIMEOUT=$2 ;; --name) NAME=$2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift 2
done
[[ $DEPTH == none || $DEPTH == sim ]] || { echo "--depth none|sim" >&2; exit 2; }
[[ $COURSE == "$WORLD_COURSE" ]] || { echo "refused: the T4 world is course '$WORLD_COURSE': /opt/devenv/use_course.sh $COURSE first" >&2; exit 1; }
[[ -f $REPO/courses/$COURSE/course.yaml ]] || { echo "refused: no courses/$COURSE in $REPO" >&2; exit 1; }
ROUTE=${ROUTE:-$REPO/courses/$COURSE/route.json}
source /etc/devenv.sh
export BLUERAVEN_REPO=$REPO
OUT=${DEVENV_HOST_HOME:-/home/flatypus}/bags/drone_sim_${NAME}_$(date -u +%Y%m%dT%H%M%SZ)
LOGS=$WS/logs/$(basename "$OUT"); mkdir -p "$LOGS"
state() { curl -sf "$DASH/state" | python3 -c "import json,sys; s=json.load(sys.stdin); print($1)"; }

OURS='skybeta_flight_interface|blueraven_skybeta|refnav|drone_sim/(pilot|score|oak_depth|sim\.launch)|bags/drone_sim_'
pgrep -f "$OURS" > /dev/null && { echo "refused: a flight program or another drone_sim run is running" >&2; exit 1; }
[[ $(state "s['armed']") == False ]] || { echo "refused: PX4 is armed" >&2; exit 1; }

# The organizers' flight interface at the drone's version, then the repo's skybeta/ packages, every run.
if [[ ! -d $WS/src/skybeta ]]; then
  git clone -q "$SKYBETA_GIT" "$WS/src/skybeta" && git -C "$WS/src/skybeta" checkout -q "$SKYBETA_SHA" || exit 1
fi
(cd "$WS" && colcon build --base-paths src/skybeta/src/skybeta_flight_interface "$REPO/skybeta" > "$LOGS/build.log" 2>&1) \
  || { tail -20 "$LOGS/build.log"; exit 1; }
source "$WS/install/setup.bash"

PIDS=()
start() { local name=$1; shift; setsid "$@" > "$LOGS/$name.log" 2>&1 < /dev/null & PIDS+=($!); }
cleanup() {
  for p in "${PIDS[@]}"; do kill -TERM -- -"$p" 2>/dev/null; done; sleep 5
  for p in "${PIDS[@]}"; do kill -KILL -- -"$p" 2>/dev/null; done
  ros2 param set /oak_stereo latency_ms 0.0 > /dev/null; ros2 param set /oak_stereo drop_p 0.01 > /dev/null
  echo "left running: $(pgrep -fc "$OURS" || true) (expect 0)"
}
trap cleanup EXIT

echo "reset world"
curl -sf -X POST "$DASH/action/reset?by=drone_sim" > /dev/null || { echo "reset refused" >&2; exit 1; }
sleep 5
for _ in $(seq 180); do
  [[ $(state "s['launch']['status'] == 'done' and s['healthy'] and s['landed']" 2>/dev/null) == True ]] && break; sleep 1
done
[[ $(state "s['launch']['status'] == 'done' and s['healthy']") == True ]] || { echo "world not healthy after reset" >&2; exit 1; }

ros2 param set /oak_stereo latency_ms "$(python3 -c "print(float($LATENCY_MS))")" > /dev/null
ros2 param set /oak_stereo drop_p "$(python3 -c "print(float($DROP))")" > /dev/null
[[ $DEPTH == sim ]] && start oak_depth python3 "$HERE/oak_depth.py" --ros-args -p use_sim_time:=true

# Frame probe: the measured odom -> NED yaw must be the offset the interface is given.
probe=$(timeout 30 ros2 run blueraven_skybeta yaw_offset_probe --ros-args -p seconds:=6.0 2>&1 | grep RESULT)
echo "frame probe: ${probe#*RESULT }"
python3 -c "import re,sys,math; m=re.search(r'offset ([-+0-9.]+) deg', sys.argv[1]); \
sys.exit(0 if m and abs((float(m.group(1)) - math.degrees($YAW_OFF) + 180) % 360 - 180) <= 5 else 1)" "$probe" \
  || { echo "refused: frame probe is more than 5 deg from $YAW_OFF rad" >&2; exit 1; }

start bag ros2 bag record -s mcap --use-sim-time -o "$OUT" -e \
  '^/(skybeta|refnav|blueraven)/|^/visual_slam/tracking/odometry$|^/fmu/out/(vehicle_local_position_v1|vehicle_status_v1|vehicle_land_detected|estimator_status_flags|vehicle_attitude)$|^/fmu/in/(vehicle_visual_odometry|trajectory_setpoint|offboard_control_mode|vehicle_command)$|^/sim/truth/|^/oak/stereo/camera_info$|^/tf(_static)?$|^/rosout$'
start score python3 "$HERE/score.py" "$LOGS/score.json" "$ROUTE" "$REPO/courses/$COURSE/course.yaml"
for _ in $(seq 60); do grep -q "Subscribed to topic" "$LOGS/bag.log" && grep -q ready "$LOGS/score.log" && break; sleep 0.5; done

echo "mission: $MISSION course:=$COURSE odom_to_ned_yaw_offset_rad:=$YAW_OFF launch_key:=launch_sim $*  (repo $(git -C "$REPO" rev-parse --short HEAD 2> /dev/null))"
start mission ros2 launch "$HERE/sim.launch.py" mission:="$MISSION" course:="$COURSE" odom_to_ned_yaw_offset_rad:=$YAW_OFF launch_key:=launch_sim "$@"
python3 "$HERE/pilot.py" "$TIMEOUT" --ros-args -p use_sim_time:=true | tee "$LOGS/pilot.log"
verdict=${PIPESTATUS[0]}
sleep 2
cleanup; trap - EXIT
cp -r "$LOGS" "$OUT/logs" 2>/dev/null
echo "== score ($( [[ $verdict == 0 ]] && echo 'completed: self-disarm, no takeover' || echo 'NOT completed: pilot took over'))"
cat "$LOGS/score.json"
echo "bag and logs: $OUT"
