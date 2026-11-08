#!/usr/bin/env bash
# fly_route.sh -- fly a route file on skybeta-09: STAGE 1 (the first checkpoint: through the CP2 gap, land) by
# default, or the full course. Run ON THE DRONE HOST (ssh dronehacks@skybeta-09.local), in its own SSH window.
#
#   OFFSET=<rad> ./fly_route.sh go                       # our shell (blueraven_skybeta mission:=route), stage 1
#   OFFSET=<rad> ROUTE=route_v6_start2 ./fly_route.sh go # ... the full course
#   PROGRAM=refnav OFFSET=<rad> ./fly_route.sh go        # Hinson's refnav (hinson/real-drone) instead
#   ./fly_route.sh status    # another window: nodes, mission phase, PX4 state (one shot)
#   ./fly_route.sh stop      # stop the program and the recording (also what Ctrl-C in 'go' does)
#   ./fly_route.sh check     # only the pre-flight checks
#   ./fly_route.sh build     # rebuild blueraven_skybeta after copying new code (never restarts the stack)
#
# The hover's go.sh/stop.sh waited a fixed 4 s (the example exiting), 15 s (reaching WAIT_OFFBOARD) and 3+2 s (the bag
# closing). Here each of those is a poll on the event itself. Unlike that stop.sh, NTP is never turned back on.
#
# OFFSET is odom_to_ned_yaw_offset_rad (Darion's hover-bag read-out). It has no default ON PURPOSE: the route is flown
# with twists, which the interface does not rotate, so a wrong value flies the drone off at that angle.
# There are no fixed waits: each step polls (every 0.5 s) for the thing it needs and stops with the reason if that
# doesn't come within its timeout. Never restarts the stack (a restart needs an FC replug); never arms.
set -uo pipefail

C=skybeta-stack                                   # the organizers' stack container
REPO=/root/team_ws/src/blueraven                  # this repository, inside the container
PROGRAM=${PROGRAM:-shell}                         # shell | refnav
ROUTE=${ROUTE:-route_v6_start2_stage1}            # a route file name, without .json
COURSE=${COURSE:-start-2026-11-07-tape}           # hinson/real-drone's course folder (courses/<COURSE>/routes/)
HEIGHT=${HEIGHT:-0.7}
RADIUS=${RADIUS:-0.3}                             # shell: corner arcs (m): it turns while moving, never in place
BAGS=/root/team_ws/bags
TOPICS="/visual_slam/tracking/odometry /visual_slam/status /skybeta/state /skybeta/setpoint/pose \
/skybeta/setpoint/twist /blueraven/mission/status /fmu/in/vehicle_visual_odometry /fmu/out/vehicle_local_position \
/fmu/out/vehicle_status_v1 /fmu/out/estimator_status_flags /fmu/out/vehicle_attitude /fmu/out/vehicle_land_detected \
/fmu/out/distance_sensor /tf /tf_static"          # no camera images: an outside image reader starved cuVSLAM

in_c() { docker exec "$C" bash -ic "$1" 2>/dev/null; }      # inside the stack container, ROS sourced
die() { echo "STOP: $*" >&2; exit 1; }
say() { echo "[$(date +%H:%M:%S)] $*"; }
poll() {                                          # poll <timeout s> <what> <command...>: retry every 0.5 s
  local limit=$1 what=$2; shift 2
  local deadline=$((SECONDS + limit))
  until "$@"; do
    (( SECONDS >= deadline )) && die "$what: not within $limit s"
    sleep 0.5                                     # the poll interval, not a wait: it goes on the moment it's true
  done
}
nodes() { in_c "ros2 node list" | tr -d '\r'; }
count() { nodes | grep -cx "$1"; }
state() { in_c "timeout 3 ros2 topic echo --once /skybeta/state" | tr -d '\r'; }
phase() { in_c "timeout 3 ros2 topic echo --once /blueraven/mission/status" | grep -o '"phase": "[A-Z_]*"' | cut -d'"' -f4; }
route_file() {                                    # the route file inside the container, either repo layout
  in_c "for f in $REPO/courses/$COURSE/routes/$ROUTE.json $REPO/routes/$ROUTE.json; do [ -f \$f ] && echo \$f && break; done" | tr -d '\r'
}

check() {
  [ "$(hostname)" = skybeta-09 ] || die "this is $(hostname), not skybeta-09 (another team's drone is a disqualification)"
  [ "$(timedatectl show -p NTP --value)" = no ] || die "NTP is on: sudo timedatectl set-ntp false (a clock step restarts the stack)"
  docker ps --format '{{.Names}}' | grep -qx "$C" || die "the $C container isn't running"
  say "code on the drone: $(in_c "cat $REPO/VERSION_FLOWN 2>/dev/null || git -C $REPO rev-parse --short HEAD" | tr -d '\r')"
  RF=$(route_file); [ -n "$RF" ] || die "no $ROUTE.json under $REPO/courses/$COURSE/routes or $REPO/routes"
  say "route: $RF"
  in_c "timeout 3 ros2 topic echo --once /visual_slam/tracking/odometry" | grep -q position || die "no odometry (cuVSLAM)"
  local s; s=$(state)
  echo "$s" | grep -q '"px4_link": true' || die "/skybeta/state: no PX4 link: $s"
  echo "$s" | grep -q '"armed": false' || die "PX4 is ARMED already"
  echo "$s" | grep -q '"landed": true' || die "PX4 doesn't report landed"
  if (( $(count /competitor_example) > 0 )); then   # the stack starts the organizers' example (a 5 m square on arm)
    say "stopping the organizers' example (Ctrl-C in its tmux window)"
    docker exec "$C" tmux send-keys -t skybeta:competitor C-c
    poll 20 "the organizers' example (/competitor_example) exiting" \
      bash -c "[ \$(docker exec $C bash -ic 'ros2 node list' 2>/dev/null | tr -d '\r' | grep -cx /competitor_example) = 0 ]"
  fi
  (( $(count /flight_interface) == 0 )) || die "a flight_interface is already running (two must never run): ./fly_route.sh stop, or find who started it"
  (( $(count /oak) == 1 )) || die "expected exactly one /oak camera node, found $(count /oak)"
  say "checks passed (PROGRAM=$PROGRAM ROUTE=$ROUTE HEIGHT=$HEIGHT)"
}

stop() {                                          # SIGINT, then wait until they have exited (a bag closes cleanly)
  docker exec "$C" pkill -INT -f "ros2 launch blueraven_skybeta" 2>/dev/null && say "stopping the program"
  docker exec "$C" pkill -INT -f "ros2 bag record" 2>/dev/null && say "stopping the recording"
  local deadline=$((SECONDS + 20))
  while docker exec "$C" pgrep -f "ros2 launch blueraven_skybeta|ros2 bag record|flight_interface" >/dev/null 2>&1; do
    (( SECONDS >= deadline )) && { say "still running after 20 s: docker exec $C pkill -f 'ros2 bag record|flight_interface'"; return 1; }
    sleep 0.5
  done
  say "stopped (NTP untouched: keep it off on flight days)"
}

build() {                                         # after copying new code to the drone; the stack is not restarted
  in_c "cd /root/team_ws && team_build.sh --packages-select blueraven_skybeta" | tail -3
}

status() {
  say "nodes: flight_interface $(count /flight_interface), mission $(( $(count /blueraven_mission) + $(count /refnav) )), oak $(count /oak), competitor_example $(count /competitor_example)"
  say "mission: $(in_c "timeout 3 ros2 topic echo --once /blueraven/mission/status" | grep -o "'{.*}'" | tr -d "'")"
  say "state: $(state | grep -o "'{.*}'" | tr -d "'")"
}

go() {
  [ -n "${OFFSET:-}" ] || die "set OFFSET=<odom_to_ned_yaw_offset_rad>: the hover-bag value from Darion (0.0 only if that is what was measured)"
  check
  local run; run=${ROUTE}_$(date +%H%M%S)
  trap 'echo; say "Ctrl-C: stopping"; stop; exit 130' INT
  # 1. record, and only go on once the recorder has actually subscribed (no fixed wait)
  docker exec -d "$C" bash -ic "mkdir -p $BAGS && cd $BAGS && exec ros2 bag record -s mcap -o $run $TOPICS > $BAGS/$run.record.log 2>&1"
  poll 15 "the recorder subscribing to /skybeta/state" \
    bash -c "docker exec $C bash -ic 'ros2 topic info -v /skybeta/state' 2>/dev/null | grep -q rosbag2"
  say "recording $BAGS/$run"
  # 2. the program, in the background with its log; Ctrl-C here stops it
  local launch
  if [ "$PROGRAM" = shell ]; then
    launch="ros2 launch blueraven_skybeta blueraven.launch.py mission:=route route_file:=$RF height_m:=$HEIGHT route_max_radius_m:=$RADIUS odom_to_ned_yaw_offset_rad:=$OFFSET"
  elif [ "$PROGRAM" = refnav ]; then
    launch="ros2 launch blueraven_skybeta refnav.launch.py course:=$COURSE route:=$ROUTE height_m:=$HEIGHT odom_to_ned_yaw_offset_rad:=$OFFSET"
  else
    die "PROGRAM must be shell or refnav"
  fi
  say "launching: $launch"
  docker exec -d "$C" bash -ic "cd /root/team_ws && source install/setup.bash && exec $launch > $BAGS/$run.launch.log 2>&1"
  poll 30 "one /flight_interface" bash -c "[ \$(docker exec $C bash -ic 'ros2 node list' 2>/dev/null | tr -d '\r' | grep -cx /flight_interface) = 1 ]"
  # 3. ready: the shell says so itself (INIT -> WAIT_OFFBOARD once the drone has sat still 2 s on the mark)
  if [ "$PROGRAM" = shell ]; then
    poll 60 "the shell reaching WAIT_OFFBOARD (drone still on the mark, landed, fresh odometry)" \
      bash -c "docker exec $C bash -ic 'timeout 3 ros2 topic echo --once /blueraven/mission/status' 2>/dev/null | grep -q WAIT_OFFBOARD"
  else
    poll 30 "the /refnav node" bash -c "docker exec $C bash -ic 'ros2 node list' 2>/dev/null | grep -qx /refnav"
    say "refnav is up: watch its log for its ready line: docker exec $C tail -f $BAGS/$run.launch.log"
  fi
  local nfi nm; nfi=$(count /flight_interface); nm=$(( $(count /blueraven_mission) + $(count /refnav) ))
  [ "$nfi" = 1 ] && [ "$nm" = 1 ] && [ "$(count /competitor_example)" = 0 ] || { stop; die "process count: flight_interface $nfi, mission $nm, competitor_example $(count /competitor_example) (need 1, 1, 0)"; }
  echo
  say "READY. Team approves, then the PILOT: arm with the mode switch on ALTITUDE, throttle low, then switch to POSITION within a few seconds."
  say "Abort: the pilot takes over (Position/Altitude). If it then sits ARMED on the ground, Ctrl-C here (the interface stops publishing), then disarm."
  # 4. follow it to the end: phases as they change, then PX4 disarming
  local last="" p
  while :; do
    if [ "$PROGRAM" = shell ]; then
      p=$(phase); [ -n "$p" ] && [ "$p" != "$last" ] && { say "phase: $p"; last=$p; }
      [ "$p" = DONE ] || [ "$p" = ABORTED ] && break
    else
      state | grep -q '"armed": true' && last=armed
      [ "$last" = armed ] && state | grep -q '"armed": false' && break
    fi
    sleep 0.5
  done
  poll 30 "PX4 disarming" bash -c "docker exec $C bash -ic 'timeout 3 ros2 topic echo --once /skybeta/state' 2>/dev/null | grep -q '\"armed\": false'"
  say "disarmed"
  stop
  say "done. Bag: $BAGS/$run (log: $BAGS/$run.launch.log). Off the drone: scp -r dronehacks@skybeta-09.local:~/team_ws/bags/$run ."
}

case "${1:-go}" in
  go) go ;;
  status) status ;;
  stop) stop ;;
  check) check ;;
  build) build ;;
  *) echo "usage: $0 [go|status|stop|check|build]"; exit 2 ;;
esac
