# blueraven_skybeta — our mission on the real drone

Team Blue Raven's mission node for the organizers' skybeta stack (skybeta-09). It reads the stack's
odometry and `/skybeta/state`, and flies through the organizers' flight interface: it publishes only
`/skybeta/setpoint/pose` or `/skybeta/setpoint/twist`. It **never** publishes `/fmu/*`, never arms and
never requests a mode. The safety pilot arms and switches to Position; that is the go signal.

| Mission | What it does | Needs |
|---|---|---|
| `hover` (**default**) | take off to `height_m`, hold `hover_s` over the take-off point, descend until landed and disarmed | odometry only: no map, no route |
| `course` | plan the route from `maps/` at start-up (`nav/route_planner.py`), follow it, land at its end | the course map; optionally depth (`wall_centring:=true`) |

Everything below is **written from the docs and the unit tests; none of it has run on the drone yet.**
Check each step's expected output and stop at the first surprise.

## 0. Rules for every session on the drone

- **NTP off** before the bench or a flight: `sudo timedatectl set-ntp false`. A clock step over 0.5 s
  restarts the whole stack. Turn it back on (`set-ntp true`) when done, before power-off.
- **Nothing may subscribe to `/oak/*/image_raw/compressed`**: no image bags, no Foxglove camera panels,
  no `ros2 topic hz` on them. It makes the camera driver JPEG-encode 2 x 30 Hz and delays the vision
  PX4 flies on past its 200 ms limit (Arnav's load test, 2026-11-07).
- **Nobody in front of the camera** during a run.
- Props off for the first run of any new code.

## 1. Put the code on the drone

The team workspace is `~/team_ws/src` on the drone host (`/root/team_ws/src` inside containers). The stack
builds it with `colcon build --symlink-install` when it starts, so the package finds `nav/` and `maps/`
in the repo it sits in. Copy the **reviewed** commit (no credentials on the drone; a USB stick or `scp`
from a laptop is fine):

```bash
# on a laptop: an archive of the reviewed commit
git archive --format=tar.gz -o blueraven.tar.gz <reviewed-sha>
scp blueraven.tar.gz <drone>:~/
# on the drone host
mkdir -p ~/team_ws/src/blueraven && tar -xzf ~/blueraven.tar.gz -C ~/team_ws/src/blueraven
echo <reviewed-sha> > ~/team_ws/src/blueraven/VERSION_FLOWN
```

Then point the stack at it, in `/etc/skybeta/drone.env`:

```
COMPETITOR_PKG=blueraven_skybeta
COMPETITOR_LAUNCH=blueraven.launch.py
```

and restart the stack **with the drone still on the launch mark**: `docker rm -f skybeta-stack` (back in
about 10 s). The default launch is a 5 s hover at 0.70 m. To change it for a session, run by hand instead
(next block), or edit the defaults in `launch/blueraven.launch.py` and note it on the log sheet.

By hand: attach (`docker exec -it skybeta-stack tmux attach -t skybeta`), select the `competitor`
window, Ctrl-C it (two flight interfaces must never run together), then in that window:

```bash
cd /root/team_ws && team_build.sh --packages-select blueraven_skybeta && source install/setup.bash
ros2 launch blueraven_skybeta blueraven.launch.py mission:=hover height_m:=0.5 hover_s:=30.0
```

Pass numbers with a decimal point (`30.0`, not `30`): the parameters are typed from their defaults.
If the node can't find `nav/`, add `export BLUERAVEN_REPO=/root/team_ws/src/blueraven` before launching.

**Expect** in the log: `mission HOVER: take off to 0.50 m, hold 30 s over the take-off point, land. No
route.`, then `INIT -> WAIT_OFFBOARD: map <- odom fixed; ...` once the drone has sat still for 2 s.

## 2. Watch it

```bash
ros2 topic echo /blueraven/mission/status          # phase, why, inits, map_from_odom, ...
ros2 topic echo /skybeta/state --once              # px4_link true, local_enu not null, landed true
ros2 topic hz /visual_slam/tracking/odometry       # ~30 Hz (NOT the camera topics)
```

Phases: `INIT` (silent: waiting for fresh state + odometry, PX4 `landed`, 2 s still) -> `WAIT_OFFBOARD`
(streams a hold at the current position) -> `TAKEOFF` -> `HOVER` -> `DESCEND` -> `DONE`, or `ABORTED`.
`why` says what it is waiting for or why it stopped.

## 3. Record (every run)

From a second shell on the drone (`host/run.sh shell`), before arming; Ctrl-C after it has disarmed:

```bash
ros2 bag record -s mcap -o /root/team_ws/bags/flight_NN \
  /visual_slam/tracking/odometry /visual_slam/status /skybeta/state \
  /skybeta/setpoint/pose /skybeta/setpoint/twist /blueraven/mission/status \
  /fmu/in/vehicle_visual_odometry /fmu/out/vehicle_local_position /fmu/out/vehicle_status_v1 \
  /fmu/out/estimator_status_flags /fmu/out/vehicle_attitude /tf /tf_static
```

No images (rule 0). Afterwards, on a laptop, sw-claude's analyzer (branch `shiori/floor-line-and-flight-check`):
`python3 tools/flight_check/analyze_flight.py bags/flight_NN --out flight_NN`. Read: VIO jumps or freezes,
the data age at PX4 (anything over 200 ms means PX4 stopped trusting vision), dropouts, and pose tracking.

## 4. Bench, props OFF (before any props-on flight)

1. **Start-up.** Node up, `WAIT_OFFBOARD` after 2 s still on the mark. Pick the drone up: status should
   stay `WAIT_OFFBOARD` (it streams holds wherever it is). Put it back.
2. **Frame offset (the guide's section 1 caveat).** On the ground, compare the two headings:
   ```bash
   ros2 topic echo /visual_slam/tracking/odometry --field pose.pose.orientation
   ros2 topic echo /fmu/out/vehicle_attitude --qos-reliability best_effort --qos-durability transient_local
   ```
   Turn the drone by hand to a few headings. If the yaws differ by a constant, that constant is
   `odom_to_ned_yaw_offset_rad` (a launch argument here). Fly a hover with both at zero first unless the
   difference is large. Write it down either way.
3. **Handover.** Safety pilot: arm with the mode switch on **Altitude**, throttle low, then switch to
   **Position**. Expect `offboard_active` true in `/skybeta/state` and `WAIT_OFFBOARD -> TAKEOFF` in the
   log; motors spin up. With no props it can't climb: PX4 disarms about 10 s after arming
   (`COM_DISARM_PRFLT`), and the node logs `TAKEOFF -> ABORTED: Offboard ended ...`, calls `/skybeta/hold`
   once and goes silent. **That is the expected bench result.**
4. **Take-back practice.** Restart the node (it never re-engages after an abort, by design). Arm, switch
   to Position, then raise the throttle towards centre: Offboard must end, the node must go `ABORTED`.
5. **Stack restart (optional).** In `WAIT_OFFBOARD`, restart the stack: the node goes back to `INIT`
   ("odometry lost before take-off: re-fixing map <- odom") and returns to `WAIT_OFFBOARD`.

## 5. Hover, props ON (tethered, low)

The team approves the mission before it arms. The pilot holds the transmitter, has tested the kill switch,
and takes over at **any** unexpected motion.

1. Drone still on the launch mark; status `WAIT_OFFBOARD`; recording started.
2. Pilot arms (Altitude, throttle low), then switches to Position within a few seconds.
3. Expect: climb to `height_m` at 0.3 m/s; `HOVER` for `hover_s`; `DESCEND` at 0.2 m/s; PX4 detects the
   touchdown and disarms (about 2 s); `DONE`. If it is on the floor and still armed after about 5 s, the
   pilot disarms it (the node keeps commanding a gentle descent until PX4 reports landed and disarmed).
4. **If it drifts more than 0.4 m from the take-off point** while climbing or hovering, the node descends
   straight down on its own (`... (PX4 frame offset?): landing straight down`). That points at the frame
   offset of step 4.2: measure it before the next flight.
5. **Abort** (one `/skybeta/hold`, then silence): `/skybeta/state` older than 0.6 s, odometry older than
   0.5 s, an odometry jump, or Offboard ending. The pilot then lands it: in Position, low throttle descends.

Pass (sw-claude's checklist step 1): no VIO jumps; horizontal drift p95 under 0.15 m; lands and disarms
by itself.

## 6. Course mission (later)

`mission:=course launch_key:=launch cp4:=gate_b` (or `window`). The route is planned at start-up and
refused (nothing published) if a checkpoint can't be crossed. `keep_centre_m` (0.02) keeps the route
centred in corridors. `wall_centring:=true` also steers toward the real corridor centre seen in Darion's
depth (`/blueraven/depth/image_raw`; run his depth node too): **off until it has been checked on real
depth**. With map_align running, the node publishes `/blueraven/map_from_odom_init` once and follows
`/blueraven/map_from_odom`.

## Parameters

| Name | Default | |
|---|---|---|
| `mission` | `hover` | `hover` or `course` |
| `height_m` | 0.70 | flight height above the launch point |
| `hover_s` | 5.0 | hover: seconds at height |
| `launch_key` | `launch` | `maps/course.yaml` entry: `launch` (real mark) or `launch_sim` |
| `cruise_mps` | 0.30 | course speed |
| `cp4` | `gate_b` | `gate_b` or `window` |
| `keep_centre_m` | 0.02 | planner centring (< 0: the tight default) |
| `wall_centring` | false | course: depth-based corridor centring |
| `odom_to_ned_yaw_offset_rad`, `anchor_setpoints` | 0.0, false | the flight interface's frame-offset fixes |

## Not tested

Never run under ROS or on the drone. Tested only as unit tests against a kinematic toy drone
(`tests/test_skybeta_mission.py`, `tests/test_wall_centring*.py`): the phase logic, the aborts, restarts,
the hover, the drift interface and centring. Unknown until the bench: real `/skybeta/state` contents, the
interface's behaviour at the handover, PX4's tracking, the frame offset, and whether a twist descent
lands and disarms cleanly.
