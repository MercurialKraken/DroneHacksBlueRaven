#!/usr/bin/env bash
# fly_route.sh -- fly a route on skybeta-09. Run ON THE DRONE HOST: ssh dronehacks@skybeta-09.local, cd ~/team_ws
#
#   OFFSET=0.0 ROUTE=route_v7_cp2_simple ./fly_route.sh go          # full checks, then GO
#   OFFSET=0.0 ROUTE=route_v7_cp2_simple ./fly_route.sh go --skip   # the minimum to reach GO
#   NAV=mppi OFFSET=0.0 ROUTE=route_v7_cp2_simple ./fly_route.sh go   # MPPI follows (default NAV=refnav)
#   NAV=mppi MPPI_SPEED=0.10 ...    # MPPI's cruise m/s (default 0.10); MPPI runs as its own process, mppi_node
#   PROGRAM=wallseek OFFSET=0.0 HEIGHT=0.4 ./fly_route.sh go   # Darion's wall-seek (blueraven_wallseek): no route file
#   ./fly_route.sh status     # (second terminal) one line per second until Ctrl-C
#   ./fly_route.sh stop       # stops the program + recording, prints the bag path
#   ./fly_route.sh build      # rebuild blueraven_skybeta after copying new code (never restarts the stack)
#
# Every check prints '[n/N] <what> ... PASS' or '... FAIL: <reason and fix>' and a FAIL stops everything. After GO it
# follows the flight until PX4 disarms, then stops the program and the recording. It never arms and never restarts
# the stack. No 'ros2' CLI queries: they hang on a cold daemon after a reboot (processes via pgrep, rates via rclpy).
# OFFSET is odom_to_ned_yaw_offset_rad and has no default ON PURPOSE; skybeta-09's measured value is 0.0.
# ABORT: the pilot takes over -> Ctrl-C this terminal (the interface stops publishing) -> the pilot disarms.
set -uo pipefail

C=skybeta-stack                                   # the organizers' stack container
PROGRAM=${PROGRAM:-refnav}                        # refnav (a route file) or wallseek (Darion's depth wall-seek)
NAV=${NAV:-refnav}                                # follower: refnav (blind) or mppi (MPPI on live depth)
MPPI_SPEED=${MPPI_SPEED:-0.10}                    # NAV=mppi: MPPI's cruise speed, m/s (<= MAX_SPEED)
ROUTE=${ROUTE:-route_v7_cp2_simple}               # a route file name, without .json
COURSE=${COURSE:-start-2026-11-07-tape}           # the course folder (courses/<COURSE>/routes/)
if [ "$PROGRAM" = wallseek ]; then HEIGHT=${HEIGHT:-0.40}; else HEIGHT=${HEIGHT:-0.7}; fi
MAX_SPEED=${MAX_SPEED:-0.15}                      # the route file's speed_mps may not exceed this (team rule)
BAGS=/root/team_ws/bags                           # = ~/team_ws/bags on the drone host
WS=/root/team_ws
MON=/tmp/fly_status.py                            # the status monitor, copied into the container on use
TOPICS="/visual_slam/tracking/odometry /visual_slam/status /skybeta/state /skybeta/setpoint/pose \
/skybeta/setpoint/twist /blueraven/mission/status /refnav/status /fmu/in/vehicle_visual_odometry \
/fmu/out/vehicle_local_position /fmu/out/vehicle_status_v1 /fmu/out/estimator_status_flags /fmu/out/vehicle_attitude \
/fmu/out/vehicle_land_detected /fmu/out/distance_sensor /fmu/out/battery_status /fmu/out/failsafe_flags /fmu/out/health_report /tf /tf_static \
/blueraven/mppi/plan /blueraven/mppi/status /blueraven/mppi/trajectory /blueraven/mppi/cmd /blueraven/mppi/start \
/fmu/in/trajectory_setpoint /fmu/in/offboard_control_mode /fmu/out/vehicle_local_position_setpoint /fmu/out/vehicle_attitude_setpoint"   # absent topics are harmless (MPPI off)
DEPTH_BAG=${DEPTH_BAG:-1}                         # 1: also record on-camera depth (~15 MB/s)
[ "$PROGRAM" = wallseek ] && TOPICS="$TOPICS /blueraven/wallseek/status"
(( DEPTH_BAG )) && TOPICS="$TOPICS /oak/stereo/image_raw /oak/stereo/camera_info"
MIN_FREE_GB=${MIN_FREE_GB:-5}                     # free space the bag disk must have (depth fills ~1 GB a minute)
ODOM_MIN_GO=${ODOM_MIN_GO:-28}                    # odometry Hz needed with the recorder running (hover baseline 29.6)
FI_PROC="lib/skybeta_flight_interface/flight_interface"
MISSION_PROC='lib/blueraven_skybeta/(mission|refnav_mission)( |$)|lib/blueraven_wallseek/wallseek( |$)'
EXAMPLE_PROC='ros2 launch competitor_example|/lib/competitor_example/'
LAUNCH_PROC="ros2 launch blueraven_(skybeta|wallseek)"
MPPI_PROC='lib/blueraven_skybeta/mppi_node( |$)'
BAG_PROC='ros2 bag record -s mcap -o [A-Za-z0-9_-]+_[0-9]{6} '   # OUR recorder only (its -o is <ROUTE>_HHMMSS)
OURS_PROC="$LAUNCH_PROC|$BAG_PROC|$FI_PROC|$MISSION_PROC|$MPPI_PROC"

usage() {
  cat <<EOF
usage (on skybeta-09, in ~/team_ws):
  OFFSET=0.0 ROUTE=route_v7_cp2_simple ./fly_route.sh go          # full checks, then GO
  OFFSET=0.0 ROUTE=route_v7_cp2_simple ./fly_route.sh go --skip   # the minimum to reach GO
  NAV=mppi OFFSET=0.0 ROUTE=route_v7_cp2_simple ./fly_route.sh go   # MPPI follows (default NAV=refnav)
  PROGRAM=wallseek OFFSET=0.0 HEIGHT=0.4 ./fly_route.sh go         # Darion's wall-seek (no route file)
  ./fly_route.sh status    # one line per second (second terminal)
  ./fly_route.sh stop      # stop the program + recording
  ./fly_route.sh build     # rebuild blueraven_skybeta
EOF
}

in_c() { docker exec "$C" bash -ic "$1" 2>/dev/null; }      # inside the stack container, ROS sourced
die() { echo "STOP: $*" >&2; exit 1; }
say() { echo "[$(date +%H:%M:%S)] $*"; }
I=0 N=0
step() { I=$((I + 1)); printf '[%d/%d] %s ... ' "$I" "$N" "$1"; }
pass() { echo "PASS${1:+ ($1)}"; }
fail() { echo "FAIL: $*"; exit 1; }
poll() {                                          # poll <timeout s> <fail reason> <command...>: retry every 0.5 s
  local limit=$1 why=$2; shift 2
  local deadline=$((SECONDS + limit))
  until "$@"; do
    (( SECONDS >= deadline )) && fail "$why (waited $limit s)"
    sleep 0.5
  done
}
# processes in the container matching an ERE, not counting shells/tmux that merely mention it (exactly one number)
procs() { docker exec "$C" pgrep -af "$1" 2>/dev/null | grep -vE '^[0-9]+ (([^ ]*/)?(ba)?sh|tmux|pgrep) ' | grep -c . ; }
nprocs_is() { [ "$(procs "$1")" = "$2" ]; }
# the monitor, with the team workspace sourced too (px4_msgs); bash's no-tty job-control notes dropped
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
        "mission": r"lib/blueraven_skybeta/(mission|refnav_mission)( |$)|lib/blueraven_wallseek/wallseek( |$)",
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
    rates, line = [], ""
    armed_seen = "--was-armed" in sys.argv          # a restarted follow that already saw the arm
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

route_lookup() {                                  # RF and SHA: the route file refnav.launch.py reads (repo = dir with nav/course_map.py)
  local out
  out=$(docker exec "$C" bash -c "r=\${BLUERAVEN_REPO:-\$(ls -d $WS/src/*/nav/course_map.py $WS/src/*/*/nav/course_map.py 2>/dev/null | head -1)}; r=\${r%/nav/course_map.py}; f=\$r/courses/$COURSE/routes/$ROUTE.json; echo \$f; sha1sum \$f 2>/dev/null | cut -c1-7" 2>/dev/null)
  RF=$(sed -n 1p <<<"$out"); SHA=$(sed -n 2p <<<"$out")
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
  local pkgs=blueraven_skybeta; [ "$PROGRAM" = wallseek ] && pkgs="$pkgs blueraven_wallseek"
  in_c "cd $WS && team_build.sh --packages-select $pkgs" | tail -3
}

status() {                                        # one line per second: phase, odometry, EV fusion, PX4, processes
  put_monitor || die "couldn't copy the status monitor into $C"
  if [ -t 0 ] && [ -t 1 ]; then                  # Ctrl-C reaches the monitor through the tty
    docker exec -it "$C" bash -ic "source $WS/install/setup.bash >/dev/null 2>&1; python3 $MON"
  else
    monitor "DURATION=${DURATION:-10}" ""
  fi
}

follow() {                                        # one status line per second until PX4 reports the disarm
  local log flag=""; log=$(mktemp)
  until monitor "" "--until-disarmed $flag" | tee -a "$log"; grep -q '^DISARMED' "$log"; do
    grep -q 'armed YES' "$log" && flag=--was-armed
    say "the status monitor ended before a disarm: restarting it (Ctrl-C here only after the pilot disarms)"
    sleep 1
  done
  rm -f "$log"
}

go() {
  local full=1 mode="go (full checks)"
  [ "${1:-}" = --skip ] && { full=0; mode="go --skip (minimum to reach GO)"; }
  local ws=0; [ "$PROGRAM" = wallseek ] && ws=1
  [ "$PROGRAM" = refnav ] || (( ws )) || die "PROGRAM=$PROGRAM: use refnav or wallseek"
  if (( full )); then N=$((16 - ws)); else N=9; fi
  echo "==================== fly_route.sh $mode ===================="
  if (( ws )); then
    echo "PROGRAM wallseek | OFFSET ${OFFSET:-UNSET} | HEIGHT $HEIGHT | DEPTH_BAG $DEPTH_BAG (ROUTE/COURSE/NAV unused)"
    echo "      no route file: forward on OAK depth until Wall A is 0.80 m ahead, stop, turn in place to face the"
    echo "      CP2 gap, straight through it to P2 (0.45 m past the gap centre), land there (blueraven_wallseek defaults)"
  else
    route_lookup
    echo "PROGRAM $PROGRAM | NAV $NAV | ROUTE $ROUTE (sha1 ${SHA:-?}, course $COURSE) | OFFSET ${OFFSET:-UNSET} | HEIGHT $HEIGHT | DEPTH_BAG $DEPTH_BAG"
  fi
  echo

  step "OFFSET set and numeric"
  [ -n "${OFFSET:-}" ] || fail "OFFSET is unset: put OFFSET=0.0 in front (skybeta-09's measured odom_to_ned_yaw_offset_rad)"
  [[ $OFFSET =~ ^-?[0-9]+(\.[0-9]+)?$ ]] || fail "OFFSET=$OFFSET is not a number (radians): use OFFSET=0.0"
  pass "$OFFSET rad"

  step "hostname is skybeta-09"
  case "$(hostname)" in skybeta-09|skybeta-10) true ;; *) false ;; esac || fail "this is $(hostname): flying another team's drone is a disqualification; ssh dronehacks@skybeta-09.local"
  pass

  if (( full )); then
    step "NTP off"
    if [ "$(timedatectl show -p NTP --value)" = no ]; then pass; else  # turning it OFF never steps the clock
      sudo timedatectl set-ntp false || fail "NTP is on and couldn't turn it off: run sudo timedatectl set-ntp false"
      pass "it was on: turned off"
    fi
  fi

  step "stack container $C up"
  docker ps --format '{{.Names}}' | grep -qx "$C" || fail "$C isn't running: ask the organizers (a restart needs an FC replug)"
  put_monitor || fail "couldn't copy the status monitor into $C"
  pass

  if (( full )); then
    step "bag disk >= $MIN_FREE_GB GB free"
    local free; free=$(docker exec "$C" bash -c "mkdir -p $BAGS && df -Pk $BAGS | awk 'NR==2 {print int(\$4 / 1048576)}'")
    (( ${free:-0} >= MIN_FREE_GB )) || fail "only ${free:-?} GB free in ~/team_ws/bags: delete old bags, or DEPTH_BAG=0"
    pass "$free GB"
  fi

  if (( full && ws )); then
    step "blueraven_wallseek built"
    docker exec "$C" test -x "$WS/install/blueraven_wallseek/lib/blueraven_wallseek/wallseek" \
      || fail "no ~/team_ws/install/blueraven_wallseek: PROGRAM=wallseek ./fly_route.sh build"
    pass
  elif (( full )); then
    step "route is not a *_sim file"
    [[ $ROUTE == *_sim ]] && fail "ROUTE=$ROUTE starts at the simulator's spawn, not the real mark: use ROUTE=${ROUTE%_sim}"
    pass

    step "route file exists, parses, speed <= $MAX_SPEED m/s"
    [ -n "$RF" ] && docker exec "$C" test -f "$RF" || fail "no route file ${RF:-courses/$COURSE/routes/$ROUTE.json}: check ROUTE/COURSE, copy the repo to ~/team_ws/src"
    local info speed
    info=$(docker exec "$C" python3 -c "import json,hashlib; b=open('$RF','rb').read(); d=json.loads(b); p=d.get('params',{}); print(p.get('speed_mps'), p.get('fillet_m'), len(d['route']), hashlib.sha1(b).hexdigest()[:7], d['route'])" 2>/dev/null | tail -1)
    [ -n "$info" ] || fail "$RF doesn't parse as a route (needs params and route): fix the JSON"
    read -r speed fillet npts sha points <<<"$info"
    awk -v s="$speed" -v m="$MAX_SPEED" 'BEGIN { exit !(s + 0 > 0 && s + 0 <= m + 1e-9) }' \
      || fail "speed_mps is $speed, over MAX_SPEED=$MAX_SPEED (team rule; MAX_SPEED=$speed to accept it)"
    pass "speed $speed m/s, fillet $fillet m, $npts points, sha1 $sha"
    echo "      $RF"
    echo "      points: $points"
  fi

  step "none of our launch/mission/recorder running"
  local n; n=$(procs "$LAUNCH_PROC|$BAG_PROC|$MISSION_PROC|$MPPI_PROC")
  [ "$n" = 0 ] || fail "$n of our processes already run: ./fly_route.sh stop first"
  pass

  step "organizers' example stopped (and its flight_interface exited)"
  if nprocs_is "$EXAMPLE_PROC" 0; then pass "not running"; else
    docker exec "$C" tmux send-keys -t skybeta:competitor C-c
    poll 30 "the example or its flight_interface still runs: docker exec $C tmux send-keys -t skybeta:competitor C-c" \
      nprocs_is "$EXAMPLE_PROC|$FI_PROC" 0
    pass "Ctrl-C sent; both exited"
  fi

  if (( full )); then
    step "no flight_interface left"
    n=$(procs "$FI_PROC")
    [ "$n" = 0 ] || fail "$n flight_interface already running (two must never run): find who started it, stop it"
    pass
  fi

  local run; run=$( (( ws )) && echo wallseek || echo "$ROUTE")_$(date +%H%M%S)
  trap 'stop' EXIT                                # any failure from here on stops the recording and the program
  trap 'echo; say "Ctrl-C: stopping"; exit 130' INT

  step "recorder started"
  docker exec "$C" bash -c "mkdir -p $BAGS && echo $run > $BAGS/LAST_RUN" \
    && docker exec -d "$C" bash -ic "cd $BAGS && exec ros2 bag record -s mcap -o $run $TOPICS > $BAGS/$run.record.log 2>&1" \
    || fail "docker exec failed writing LAST_RUN or starting ros2 bag record: see ~/team_ws/bags/$run.record.log"
  pass "recording ~/team_ws/bags/$run"

  step "$PROGRAM launched"
  local launch="ros2 launch blueraven_skybeta refnav.launch.py course:=$COURSE route:=$ROUTE height_m:=$HEIGHT odom_to_ned_yaw_offset_rad:=$OFFSET nav:=$NAV mppi_v_cruise_mps:=$MPPI_SPEED"
  (( ws )) && launch="ros2 launch blueraven_wallseek wallseek.launch.py height_m:=$HEIGHT odom_to_ned_yaw_offset_rad:=$OFFSET"
  docker exec -d "$C" bash -ic "cd $WS && source install/setup.bash && exec $launch > $BAGS/$run.launch.log 2>&1" \
    || fail "docker exec failed starting: $launch"
  pass "log ~/team_ws/bags/$run.launch.log"

  step "exactly one flight_interface"
  poll 30 "flight_interface count never reached exactly 1 (0: launch failed, 2: another one runs): see ~/team_ws/bags/$run.launch.log" nprocs_is "$FI_PROC" 1
  pass

  local ready='take-off pose' what="refnav logged 'take-off pose' (odometry + refnav connected)"
  (( ws )) && ready='wait -> ready' what="wallseek logged 'wait -> ready' (odometry + /skybeta/state, disarmed)"
  step "$what"
  local deadline=$((SECONDS + 60))
  until docker exec "$C" grep -q "$ready" "$BAGS/$run.launch.log"; do
    nprocs_is "$LAUNCH_PROC" 0 && fail "the launch exited: see ~/team_ws/bags/$run.launch.log"
    (( SECONDS >= deadline )) && fail "no '$ready' in 60 s (no odometry? cuVSLAM down?): ./fly_route.sh status, see ~/team_ws/bags/$run.launch.log"
    sleep 0.5
  done
  pass
  echo "      $(docker exec "$C" grep -m1 "$ready" "$BAGS/$run.launch.log" | tr -d '\r')"

  if [ "$NAV" = mppi ] && (( ! ws )); then
    step "mppi_node: its own process, live depth placed, MPPI stepping"
    awk -v s="$MPPI_SPEED" -v m="$MAX_SPEED" 'BEGIN { exit !(s + 0 > 0 && s + 0 <= m + 1e-9) }' \
      || fail "MPPI_SPEED=$MPPI_SPEED is not in (0, MAX_SPEED=$MAX_SPEED]"
    deadline=$((SECONDS + 30))
    until docker exec "$C" grep -q 'mppi_node ready' "$BAGS/$run.launch.log"; do
      nprocs_is "$MPPI_PROC" 1 || fail "mppi_node is not running: see ~/team_ws/bags/$run.launch.log (./fly_route.sh build after new code?)"
      (( SECONDS >= deadline )) && fail "no 'mppi_node ready' in 30 s (no depth? /oak/stereo/image_raw): see ~/team_ws/bags/$run.launch.log"
      sleep 0.5
    done
    pass "cruise $MPPI_SPEED m/s"
    echo "      $(docker exec "$C" grep -m1 'mppi_node ready' "$BAGS/$run.launch.log" | tr -d '\r')"
  fi

  if (( full )); then
    step "process count interface/mission/example = 1/1/0"
    local nfi nm nex
    nfi=$(procs "$FI_PROC"); nm=$(procs "$MISSION_PROC"); nex=$(procs "$EXAMPLE_PROC")
    [ "$nfi/$nm/$nex" = 1/1/0 ] || fail "got $nfi/$nm/$nex: ./fly_route.sh stop, then find the extra process"
    pass

    step "odometry >= $ODOM_MIN_GO Hz over 4 s with the recorder running"
    local sum hz
    sum=$(monitor DURATION=4 --summary)
    hz=$(echo "$sum" | grep -o 'odom_hz=[0-9.]*' | cut -d= -f2)
    [ -n "$hz" ] || fail "the status monitor failed: $(echo "$sum" | tail -2 | tr '\n' ' ')"
    awk -v h="$hz" -v m="$ODOM_MIN_GO" 'BEGIN { exit !(h >= m) }' \
      || fail "odometry $hz Hz (hover baseline 29.6): run go again with DEPTH_BAG=0"
    pass "$hz Hz"
    echo "      $(echo "$sum" | grep -v SUMMARY | tail -1)"
  fi

  echo
  echo "=========================================================================="
  echo "  GO - PILOT MAY ARM: ALTITUDE, throttle low, then POSITION"
  echo "  ABORT: pilot takes over (POSITION/ALTITUDE). If it sits ARMED on the ground,"
  echo "         Ctrl-C here (the interface stops publishing), then the pilot disarms."
  echo "=========================================================================="
  follow
  say "PX4 disarmed: stopping the program and the recording"
  trap - EXIT
  stop
}

case "${1:-}" in
  go) shift; [ $# -eq 0 ] || [ "$1" = --skip ] || { usage; exit 2; }; go "$@" ;;
  status) status ;;
  stop) stop ;;
  build) build ;;
  *) usage; exit 2 ;;
esac
