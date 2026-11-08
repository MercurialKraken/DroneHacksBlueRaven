#!/usr/bin/env bash
# fly_route.sh -- fly a route file on skybeta-09: ROUTE 1 (take-off -> through the CP2 gap -> land) by default.
# Run ON THE DRONE HOST (ssh dronehacks@skybeta-09.local). Arnav's run routine: two terminals, two commands.
#
#   Terminal 1:  cd ~/team_ws && PROGRAM=refnav COURSE=start-2026-11-07-tape ROUTE=route_v7_cp2_simple OFFSET=0.0 ./fly_route.sh go
#   Terminal 2:  cd ~/team_ws && ./fly_route.sh status       # one line per second until Ctrl-C
#   Afterwards:  ./fly_route.sh stop                          # stops the program + recording, prints the bag path
#   ./fly_route.sh check     # only the pre-flight checks (go runs them first)
#   ./fly_route.sh build     # rebuild blueraven_skybeta after copying new code (never restarts the stack)
#
# 'go' does everything in one shot: NTP off, the image, the organizers' example stopped, process counts (ps, never
# 'ros2 node list': a cold daemon printed zeros), odometry ~30 Hz, the PX4 link, the route file refnav will read (its
# speed and points printed); then the recording, the launch, and GO once refnav logs 'take-off pose'. It REFUSES if
# a flight_interface, mission, launch or recording already runs (stop first). No fixed sleeps: every wait polls
# every 0.5 s with a timeout and names the failing check. It never arms and never restarts the stack (a restart
# needs an FC replug). OFFSET is odom_to_ned_yaw_offset_rad and has no default ON PURPOSE (a wrong value flies the
# drone off at that angle); skybeta-09's measured value is 0.0 (Darion's hover_160526 read-out: -0.11 deg).
# ABORT: the pilot takes over -> Ctrl-C terminal 1 (the interface stops publishing) -> the pilot disarms.
set -uo pipefail

C=skybeta-stack                                   # the organizers' stack container
PROGRAM=${PROGRAM:-refnav}                        # refnav | shell
ROUTE=${ROUTE:-route_v7_cp2_simple}               # a route file name, without .json (route 1, Darion's method)
COURSE=${COURSE:-start-2026-11-07-tape}           # the course folder (courses/<COURSE>/routes/)
HEIGHT=${HEIGHT:-0.7}
RADIUS=${RADIUS:-0.3}                             # shell only: corner arcs (m)
EXPECT_IMAGE=${EXPECT_IMAGE:-skybeta-team:0.3.3}  # the stack image Arnav expects (on-camera depth)
MAX_SPEED=${MAX_SPEED:-0.15}                      # the route file's speed_mps may not exceed this (Chelsea: all 0.15)
BAGS=/root/team_ws/bags                           # = ~/team_ws/bags on the drone host
WS=/root/team_ws
MON=/tmp/fly_status.py                            # the status monitor, copied into the container on use
TOPICS="/visual_slam/tracking/odometry /visual_slam/status /skybeta/state /skybeta/setpoint/pose \
/skybeta/setpoint/twist /blueraven/mission/status /refnav/status /fmu/in/vehicle_visual_odometry \
/fmu/out/vehicle_local_position /fmu/out/vehicle_status_v1 /fmu/out/estimator_status_flags /fmu/out/vehicle_attitude \
/fmu/out/vehicle_land_detected /fmu/out/distance_sensor /tf /tf_static \
/blueraven/mppi/plan /blueraven/mppi/status /blueraven/mppi/trajectory"   # absent topics are harmless (MPPI off)
DEPTH_BAG=${DEPTH_BAG:-1}                         # 1: also record on-camera depth (Arnav's MPPI replay); ~15 MB/s at
                                                  # 640x400 16UC1 30 Hz. An outside depth reader cost ~1 ms EV age
                                                  # (Hinson); the per-second status shows odometry Hz if it starves
(( DEPTH_BAG )) && TOPICS="$TOPICS /oak/stereo/image_raw /oak/stereo/camera_info"
MIN_FREE_GB=${MIN_FREE_GB:-5}                     # free space the bag disk must have (depth fills ~1 GB a minute)
FI_PROC="lib/skybeta_flight_interface/flight_interface"
MISSION_PROC='lib/blueraven_skybeta/(mission|refnav_mission)( |$)'
EXAMPLE_PROC='ros2 launch competitor_example|/lib/competitor_example/'
LAUNCH_PROC="ros2 launch blueraven_skybeta"
BAG_PROC='ros2 bag record -s mcap -o [A-Za-z0-9_]+_[0-9]{6} '   # OUR recorder only (its -o is <ROUTE>_HHMMSS)
OURS_PROC="$LAUNCH_PROC|$BAG_PROC|$FI_PROC|$MISSION_PROC"

in_c() { docker exec "$C" bash -ic "$1" 2>/dev/null; }      # inside the stack container, ROS sourced
in_ws() { in_c "source $WS/install/setup.bash >/dev/null 2>&1; $1"; }   # ...with the team workspace too
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
# processes in the container matching an ERE, not counting shells/tmux that merely mention it (exactly one number)
procs() { docker exec "$C" pgrep -af "$1" 2>/dev/null | grep -vE '^[0-9]+ (([^ ]*/)?(ba)?sh|tmux|pgrep) ' | grep -c . ; }
state() { in_c "timeout 3 ros2 topic echo --once /skybeta/state" | tr -d '\r'; }
# the monitor, with the team workspace sourced too (px4_msgs); stderr kept, bash's no-tty job-control notes dropped
monitor() { docker exec "$C" bash -ic "source $WS/install/setup.bash >/dev/null 2>&1; $1 python3 $MON $2" 2>&1 \
              | tr -d '\r' | grep -vE '^bash: (cannot set terminal process group|no job control)'; }

put_monitor() {                                   # the status monitor: rclpy, small topics only (no images)
  docker exec -i "$C" bash -c "cat > $MON" <<'PY'
import glob, json, os, re, statistics, sys, time
import rclpy
from rclpy.qos import qos_profile_sensor_data as QOS   # best effort: matches reliable and best-effort publishers
from nav_msgs.msg import Odometry
from std_msgs.msg import String
try:
    from px4_msgs.msg import EstimatorStatusFlags, VehicleOdometry
except ImportError:
    EstimatorStatusFlags = VehicleOdometry = None
PATS = {"interface": r"lib/skybeta_flight_interface/flight_interface",
        "mission": r"lib/blueraven_skybeta/(mission|refnav_mission)( |$)",
        "example": r"ros2 launch competitor_example|/lib/competitor_example/",
        "recorder": r"ros2 bag record"}
SHELL = re.compile(r"^(([^ ]*/)?(ba)?sh|tmux|pgrep) ")


def procs():
    n = dict.fromkeys(PATS, 0)
    for f in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            cmd = open(f, "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        if not cmd or SHELL.match(cmd) or "fly_status.py" in cmd:
            continue
        for k, p in PATS.items():
            n[k] += bool(re.search(p, cmd))
    return n


def yn(v):
    return "?" if v is None else ("YES" if v else "no")


def main():
    until = "--until-disarmed" in sys.argv
    summary = "--summary" in sys.argv
    dur = float(os.environ.get("DURATION") or 0)
    rclpy.init()
    node = rclpy.create_node("fly_status")
    st = {"odom": 0, "ev": 0, "ev_age": None, "flags": None, "state": None, "rphase": None, "mphase": None}

    def js(m):
        try:
            return json.loads(m.data)
        except ValueError:
            return None

    def on_odom(m):
        st["odom"] += 1

    def on_ev(m):
        st["ev"] += 1
        st["ev_age"] = (m.timestamp - m.timestamp_sample) / 1000.0

    def on_flags(m):
        st["flags"] = (bool(getattr(m, "cs_ev_pos", False)), bool(getattr(m, "cs_ev_yaw", False)))

    def on_state(m):
        st["state"] = js(m)

    def on_r(m):
        st["rphase"] = (js(m) or {}).get("phase")

    def on_m(m):
        st["mphase"] = (js(m) or {}).get("phase")

    node.create_subscription(Odometry, "/visual_slam/tracking/odometry", on_odom, QOS)
    node.create_subscription(String, "/skybeta/state", on_state, QOS)
    node.create_subscription(String, "/refnav/status", on_r, QOS)
    node.create_subscription(String, "/blueraven/mission/status", on_m, QOS)
    if VehicleOdometry is not None:
        node.create_subscription(VehicleOdometry, "/fmu/in/vehicle_visual_odometry", on_ev, QOS)
        node.create_subscription(EstimatorStatusFlags, "/fmu/out/estimator_status_flags", on_flags, QOS)
    t0 = last = time.monotonic()
    rates, armed_seen, line = [], False, ""
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()
            if now - last < 1.0:
                continue
            dt, last = now - last, now
            s = st["state"] or {}
            armed = s.get("armed")
            n = procs()
            hz, ev_hz = st["odom"] / dt, st["ev"] / dt
            rates.append(hz)
            st["odom"] = st["ev"] = 0
            age = "" if st["ev_age"] is None else f" age {st['ev_age']:.0f} ms"
            f = st["flags"]
            fused = "fusion ?" if f is None else ("fused " + "+".join(k for k, v in zip(("pos", "yaw"), f) if v)
                                                   if any(f) else "NOT FUSED")
            ev = f"EV {ev_hz:4.1f} Hz{age} {fused}" if VehicleOdometry is not None else "EV n/a (no px4_msgs)"
            line = (f"[{time.strftime('%H:%M:%S')}] phase {st['rphase'] or st['mphase'] or '-'} | odom {hz:4.1f} Hz | "
                    f"{ev} | PX4 link {yn(s.get('px4_link'))} armed {yn(armed)} "
                    f"landed {yn(s.get('landed'))} | procs interface {n['interface']} mission {n['mission']} "
                    f"example {n['example']} recorder {n['recorder']}")
            print(line, flush=True)
            armed_seen = armed_seen or armed is True
            if until and armed_seen and armed is False:
                print("DISARMED", flush=True)
                break
            if dur and now - t0 >= dur:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if summary:
            steady = rates[1:] or rates            # the first second includes discovery
            print(f"SUMMARY odom_hz={statistics.median(steady) if steady else 0:.1f}", flush=True)
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


main()
PY
}

check() {
  [ "$(hostname)" = skybeta-09 ] || die "this is $(hostname), not skybeta-09 (another team's drone is a disqualification)"
  if [ "$(timedatectl show -p NTP --value)" != no ]; then          # turning it OFF never steps the clock
    say "NTP is on: turning it off (sudo may ask for the drone password)"
    sudo timedatectl set-ntp false || die "couldn't turn NTP off: sudo timedatectl set-ntp false"
  fi
  docker ps --format '{{.Names}}' | grep -qx "$C" || die "the $C container isn't running"
  local free; free=$(docker exec "$C" bash -c "mkdir -p $BAGS && df -Pk $BAGS | awk 'NR==2 {print int(\$4 / 1048576)}'")
  (( ${free:-0} >= MIN_FREE_GB )) || die "only ${free:-?} GB free for bags in $BAGS (need $MIN_FREE_GB; DEPTH_BAG=$DEPTH_BAG): delete old bags or DEPTH_BAG=0"
  say "bag disk: $free GB free (depth in the bag: $DEPTH_BAG)"
  local img; img=$(docker inspect -f '{{.Config.Image}}' "$C")
  say "image: $img"
  [[ $img == *"$EXPECT_IMAGE"* ]] || die "the stack runs $img, expected $EXPECT_IMAGE (EXPECT_IMAGE=$img to accept it)"
  [[ $ROUTE == *_sim ]] && die "ROUTE=$ROUTE is a SIM file (it starts at the simulator's spawn, 0.2 m north of the real mark): use ${ROUTE%_sim}"
  local n                                         # a second launch must never start: nobody re-runs go without stop
  n=$(procs "$LAUNCH_PROC|$BAG_PROC|$MISSION_PROC")
  [ "$n" = 0 ] || die "our launch/mission/recording is already running ($n processes): ./fly_route.sh stop first"
  # the repository refnav/the shell actually read (blueraven_skybeta.mission_core._repo_root), not a guess
  REPO=$(in_ws "python3 -c 'from blueraven_skybeta.mission_core import _repo_root; print(_repo_root())'" | tr -d '\r' | tail -1)
  [ -n "$REPO" ] || die "can't import blueraven_skybeta in $WS: build it (./fly_route.sh build)"
  say "repository: $REPO, code $(in_c "cat $REPO/VERSION_FLOWN 2>/dev/null || git -C $REPO rev-parse --short HEAD" | tr -d '\r')"
  if [ "$PROGRAM" = refnav ]; then
    in_ws "ros2 pkg executables blueraven_skybeta" | grep -q refnav_mission || die "refnav isn't built in $WS (needs hinson/real-drone): ./fly_route.sh build"
    in_ws "test -f \$(ros2 pkg prefix blueraven_skybeta)/share/blueraven_skybeta/launch/refnav.launch.py" || die "refnav.launch.py isn't installed"
    RF=$REPO/courses/$COURSE/routes/$ROUTE.json   # exactly the file refnav.launch.py reads
  elif [ "$PROGRAM" = shell ]; then
    in_c "grep -q 'def load_route' $REPO/skybeta/blueraven_skybeta/blueraven_skybeta/mission_core.py" \
      || die "this blueraven_skybeta has no route mission (hinson/real-drone's shell is hover-only): use PROGRAM=refnav"
    RF=$(in_c "for f in $REPO/courses/$COURSE/routes/$ROUTE.json $REPO/routes/$ROUTE.json; do [ -f \$f ] && echo \$f && break; done" | tr -d '\r')
  else
    die "PROGRAM must be refnav or shell"
  fi
  [ -n "$RF" ] && in_c "test -f $RF" || die "no route file $REPO/courses/$COURSE/routes/$ROUTE.json"
  local info speed
  info=$(in_c "python3 -c \"import json,hashlib; b=open('$RF','rb').read(); d=json.loads(b); p=d.get('params',{}); print(p.get('speed_mps'), 'm/s, fillet_m', p.get('fillet_m'), ', points', d['route'], ', sha1sum', hashlib.sha1(b).hexdigest()[:7])\"" | tr -d '\r' | tail -1)
  [ -n "$info" ] || die "route file $RF doesn't parse"
  say "route: $RF"
  say "route: speed $info"
  speed=${info%% *}
  awk -v s="$speed" -v m="$MAX_SPEED" 'BEGIN { exit !(s + 0 > 0 && s + 0 <= m + 1e-9) }' \
    || die "the route's speed_mps is $speed, over MAX_SPEED=$MAX_SPEED (team rule: 0.15 m/s; MAX_SPEED=$speed to accept it)"
  local s; s=$(state)
  echo "$s" | grep -q '"px4_link": true' || die "/skybeta/state: no PX4 link (FC plugged? replugged after a stack restart?): $s"
  echo "$s" | grep -q '"armed": false' || die "PX4 is ARMED already"
  echo "$s" | grep -q '"landed": true' || die "PX4 doesn't report landed"
  if [ "$(procs "$EXAMPLE_PROC")" != 0 ]; then    # the stack starts the organizers' example (a 5 m square on arm)
    say "stopping the organizers' example (Ctrl-C in its tmux window)"
    docker exec "$C" tmux send-keys -t skybeta:competitor C-c
    poll 30 "the organizers' example AND its own flight_interface exiting" \
      bash -c "[ \"\$(docker exec $C pgrep -af '$EXAMPLE_PROC|$FI_PROC' 2>/dev/null | grep -vE '^[0-9]+ (([^ ]*/)?(ba)?sh|tmux|pgrep) ' | grep -c .)\" = 0 ]"
  fi
  n=$(procs "$FI_PROC")
  [ "$n" = 0 ] || die "a flight_interface is already running ($n; two must never run): ./fly_route.sh stop, or find who started it"
  put_monitor || die "couldn't copy the status monitor into $C"
  local sum hz
  sum=$(monitor DURATION=4 --summary)
  echo "$sum" | grep -q '^SUMMARY' || die "the status monitor failed: $(echo "$sum" | tail -3)"
  echo "$sum" | grep -v SUMMARY | tail -1
  hz=$(echo "$sum" | grep -o 'odom_hz=[0-9.]*' | cut -d= -f2)
  awk -v h="${hz:-0}" 'BEGIN { exit !(h >= 20) }' || die "odometry ${hz:-none} Hz over 4 s (need ~30: cuVSLAM)"
  say "checks passed: odometry $hz Hz (PROGRAM=$PROGRAM ROUTE=$ROUTE HEIGHT=$HEIGHT)"
}

stop() {                                          # SIGINT, then wait until they have exited (a bag closes cleanly)
  docker exec "$C" pkill -INT -f "$LAUNCH_PROC" 2>/dev/null && say "stopping the program"
  docker exec "$C" pkill -INT -f "$BAG_PROC" 2>/dev/null && say "stopping the recording"
  docker exec "$C" pkill -INT -f "$MON --until-disarmed" 2>/dev/null
  local deadline=$((SECONDS + 20)) ok=0
  while [ "$(procs "$OURS_PROC")" != 0 ]; do
    (( SECONDS >= deadline )) && { say "still running after 20 s: docker exec $C pkill -f '$OURS_PROC'"; ok=1; break; }
    sleep 0.5
  done
  local run; run=$(docker exec "$C" cat "$BAGS/LAST_RUN" 2>/dev/null | tr -d '\r')
  if [ -n "$run" ]; then
    if docker exec "$C" test -f "$BAGS/$run/metadata.yaml"; then
      say "bag: ~/team_ws/bags/$run (closed cleanly; log ~/team_ws/bags/$run.launch.log)"
    else
      say "bag: ~/team_ws/bags/$run (NOT closed: no metadata.yaml yet; log ~/team_ws/bags/$run.launch.log)"
    fi
    say "copy it off: scp -r dronehacks@skybeta-09.local:~/team_ws/bags/$run ."
  fi
  (( ok == 0 )) && say "stopped (NTP untouched: keep it off between flights)"
  return $ok
}

build() {                                         # after copying new code to the drone; the stack is not restarted
  in_c "cd $WS && team_build.sh --packages-select blueraven_skybeta" | tail -3
}

status() {                                        # one line per second: phase, odometry, EV fusion, PX4, processes
  put_monitor || die "couldn't copy the status monitor into $C"
  if [ -t 0 ] && [ -t 1 ]; then                  # Ctrl-C reaches the monitor through the tty
    docker exec -it "$C" bash -ic "source $WS/install/setup.bash >/dev/null 2>&1; python3 $MON"
  else
    monitor "DURATION=${DURATION:-10}" ""
  fi
}

go() {
  [ -n "${OFFSET:-}" ] || die "set OFFSET=<odom_to_ned_yaw_offset_rad> (skybeta-09: 0.0, Darion's hover read-out)"
  [[ $OFFSET =~ ^-?[0-9]+(\.[0-9]+)?$ ]] || die "OFFSET=$OFFSET is not a number (radians)"
  check
  local run; run=${ROUTE}_$(date +%H%M%S)
  trap 'stop' EXIT                                # any failure from here on stops the recording and the program
  trap 'echo; say "Ctrl-C: stopping"; exit 130' INT
  # 1. record, and only go on once the recorder has actually subscribed (no fixed wait)
  docker exec "$C" bash -c "mkdir -p $BAGS && echo $run > $BAGS/LAST_RUN"
  docker exec -d "$C" bash -ic "mkdir -p $BAGS && cd $BAGS && exec ros2 bag record -s mcap -o $run $TOPICS > $BAGS/$run.record.log 2>&1"
  poll 30 "the recorder subscribing to /skybeta/state" \
    bash -c "docker exec $C bash -ic 'ros2 topic info -v /skybeta/state' 2>/dev/null | grep -q rosbag2"
  say "recording ~/team_ws/bags/$run"
  # 2. the program, in the background with its log
  local launch
  if [ "$PROGRAM" = shell ]; then
    launch="ros2 launch blueraven_skybeta blueraven.launch.py mission:=route route_file:=$RF height_m:=$HEIGHT route_max_radius_m:=$RADIUS odom_to_ned_yaw_offset_rad:=$OFFSET"
  else
    launch="ros2 launch blueraven_skybeta refnav.launch.py course:=$COURSE route:=$ROUTE height_m:=$HEIGHT odom_to_ned_yaw_offset_rad:=$OFFSET"
  fi
  say "launching: $launch"
  docker exec -d "$C" bash -ic "cd $WS && source install/setup.bash && exec $launch > $BAGS/$run.launch.log 2>&1"
  poll 30 "one flight_interface process" bash -c "[ \"\$(docker exec $C pgrep -fc '$FI_PROC' 2>/dev/null)\" = 1 ]"
  # 3. ready, by the program's own word
  if [ "$PROGRAM" = shell ]; then
    poll 60 "the shell reaching WAIT_OFFBOARD (drone still on the mark, landed, fresh odometry)" \
      bash -c "docker exec $C bash -ic 'timeout 3 ros2 topic echo --once /blueraven/mission/status' 2>/dev/null | grep -q WAIT_OFFBOARD"
  else
    poll 60 "refnav's 'take-off pose' log line (it has the pad frame; see ~/team_ws/bags/$run.launch.log)" \
      bash -c "docker exec $C grep -q 'take-off pose' $BAGS/$run.launch.log"
    docker exec "$C" grep -m1 'take-off pose' "$BAGS/$run.launch.log" | tr -d '\r'
  fi
  local nfi nm nex
  nfi=$(procs "$FI_PROC"); nm=$(procs "$MISSION_PROC"); nex=$(procs "$EXAMPLE_PROC")
  [ "$nfi" = 1 ] && [ "$nm" = 1 ] && [ "$nex" = 0 ] \
    || die "process count: flight_interface $nfi, mission $nm, competitor_example $nex (need 1, 1, 0)"
  echo
  say "GO. Team approves, then the PILOT: arm with the mode switch on ALTITUDE, throttle low, then switch to POSITION within a few seconds."
  say "Abort: the pilot takes over (Position/Altitude). If it then sits ARMED on the ground, Ctrl-C here (the interface stops publishing), then disarm."
  # 4. follow it to the disarm: one status line per second (the same as 'status')
  local log; log=$(mktemp)
  monitor "" --until-disarmed | tee "$log"
  if ! grep -q '^DISARMED' "$log"; then           # the monitor ended early: never stop the program before a flight
    say "the status monitor ended before a disarm: following PX4's armed state directly"
    local last="" s
    while :; do
      s=$(state)
      echo "$s" | grep -q '"armed": true' && [ "$last" != armed ] && { say "armed"; last=armed; }
      [ "$last" = armed ] && echo "$s" | grep -q '"armed": false' && break
      sleep 0.5
    done
  fi
  rm -f "$log"
  poll 30 "PX4 disarming" bash -c "docker exec $C bash -ic 'timeout 3 ros2 topic echo --once /skybeta/state' 2>/dev/null | grep -q '\"armed\": false'"
  say "disarmed"
  trap - EXIT
  stop
}

case "${1:-go}" in
  go) go ;;
  status) status ;;
  stop) stop ;;
  check) check ;;
  build) build ;;
  *) echo "usage: $0 [go|status|stop|check|build]"; exit 2 ;;
esac
