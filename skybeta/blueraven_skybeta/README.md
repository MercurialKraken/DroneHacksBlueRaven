# blueraven_skybeta — our mission on the real drone

Team Blue Raven's mission node for the organizers' skybeta stack (skybeta-09). It reads the stack's
odometry and `/skybeta/state`, and flies through the organizers' flight interface: it publishes only
`/skybeta/setpoint/pose` or `/skybeta/setpoint/twist`. It **never** publishes `/fmu/*`, never arms and
never requests a mode. The safety pilot arms and switches to Position; that is the go signal.

| Mission | What it does | Needs |
|---|---|---|
| `hover` (**default**) | take off to `height_m`, hold `hover_s` over the take-off point, descend until landed and disarmed | odometry only: no map, no route |
| `route` | take off, hover `route_hover_s`, fly a route file's corners (forward, left from take-off), land at the last | odometry and a file in `routes/`; no map |

Everything below is **written from the docs and the unit tests; none of it has run on the drone yet.**
Check each step's expected output and stop at the first surprise.

## Hover day: command sheet

Connect over a **USB-C data cable into the Jetson's port**: `ssh dronehacks@skybeta-09.local` from PowerShell
(password from Arnav, in person). If the laptop's USB adapter sits on 169.254.x.x: in an admin PowerShell,
`New-NetIPAddress -InterfaceAlias '<UsbNcm adapter from Get-NetAdapter>' -IPAddress 192.168.55.100 -PrefixLength 24`,
then `ssh dronehacks@192.168.55.1`. **First command every time: `hostname`; it must print `skybeta-09`**
(logging in to another team's drone is a disqualification).

Four SSH windows. **HOST** = the drone host. **NODE**, **WATCH**, **RECORD** = `~/skybeta/host/run.sh shell`
each (a `root@` prompt; paste container commands only there). Don't restart the stack. `<sha>` = the
reviewed commit. Sections 1-5 explain each step.

```bash
# --- laptop: package the reviewed commit, copy it over (or use a USB stick) ---
git archive --format=tar.gz -o blueraven-<sha>.tar.gz <sha>
scp blueraven-<sha>.tar.gz dronehacks@skybeta-09.local:~/

# --- HOST ---
hostname                                                         # skybeta-09, or log out
sudo timedatectl set-ntp false                                   # rule 0
ls ~/team_ws/src                                                 # expect nothing else that would build
mkdir -p ~/team_ws/src/blueraven && tar -xzf ~/blueraven-<sha>.tar.gz -C ~/team_ws/src/blueraven
echo <sha> > ~/team_ws/src/blueraven/VERSION_FLOWN
# stop the organizers' example (it flies a 5 m square on arm + Position); two flight interfaces must never run
docker exec -it skybeta-stack tmux attach -t skybeta            # Ctrl-b w -> competitor, Ctrl-C, then Ctrl-b d

# --- NODE: build ours ---
cd /root/team_ws && team_build.sh --packages-select blueraven_skybeta && source install/setup.bash

# --- WATCH: health, then the frame offset (props OFF, drone on the ground, turn it by hand) ---
source /root/team_ws/install/setup.bash
ros2 topic echo /skybeta/state --once                           # px4_link: true, local_enu not null
timeout 5 ros2 topic hz /visual_slam/tracking/odometry          # ~30 Hz
ros2 run blueraven_skybeta yaw_offset_probe --ros-args -p seconds:=40.0   # prints RESULT: offset, spread

# --- RECORD: record (no images), before arming; Ctrl-C after it has disarmed ---
ros2 bag record -s mcap -o /root/team_ws/bags/hover_NN \
  /visual_slam/tracking/odometry /visual_slam/status /skybeta/state \
  /skybeta/setpoint/pose /skybeta/setpoint/twist /blueraven/mission/status \
  /fmu/in/vehicle_visual_odometry /fmu/out/vehicle_local_position /fmu/out/vehicle_status_v1 \
  /fmu/out/estimator_status_flags /fmu/out/vehicle_attitude /tf /tf_static

# --- NODE: our node (drone still on the launch mark). Add odom_to_ned_yaw_offset_rad:=<rad> only if agreed ---
ros2 launch blueraven_skybeta blueraven.launch.py mission:=hover height_m:=0.5 hover_s:=30.0

# --- WATCH: watch it ---
ros2 topic echo /blueraven/mission/status                       # INIT -> WAIT_OFFBOARD -> ...

# --- after: Ctrl-C the node (NODE) and the recording (RECORD); then on HOST ---
sudo timedatectl set-ntp true                                    # before power-off
# --- laptop: fetch the bags ---
scp -r dronehacks@skybeta-09.local:~/team_ws/bags/hover_NN .
```

**Battery swap = a reboot:** redo `hostname`, NTP off, and stop the competitor example again (the build in
`~/team_ws` survives).

**Before walking away from the drone:** after a stack restart the `competitor` window runs whatever
`COMPETITOR_PKG` names in `/etc/skybeta/drone.env`, by default the organizers' example, which **takes off
on arm + Position and flies a 5 m square**. Power down, or set `COMPETITOR_PKG=blueraven_skybeta` (a hover
by default), so nobody arms into the square by accident.

## 0. Rules for every session on the drone

- **NTP off** before the bench or a flight: `sudo timedatectl set-ntp false`. A clock step over 0.5 s
  restarts the whole stack. Turn it back on (`set-ntp true`) when done, before power-off.
- **Nothing may subscribe to `/oak/*/image_raw` or `/oak/*/image_raw/compressed`**: no image bags, no
  Foxglove camera panels, no `ros2 topic hz` on them. Delivering those out of the driver (JPEG encoding,
  or copying the raw colour frames out of its NITROS path) delays the vision PX4 flies on past its 200 ms
  limit (Arnav's tests, 2026-11-07). **No depth node in flight**, not even on `/oak/*/image_mono`: any
  out-of-process image reader competes with cuVSLAM's feed (Arnav's repeat test). Nothing reads camera images.
- **Nobody in front of the camera** during a run.
- Props off for the first run of any new code.

## 1. Put the code on the drone

The team workspace is `~/team_ws/src` on the drone host (`/root/team_ws/src` inside containers). The stack
builds it with `colcon build --symlink-install` when it starts, so the package finds `nav/`
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

**Run it by hand** (the way to test; Arnav's rule: don't restart the stack). Attach
(`docker exec -it skybeta-stack tmux attach -t skybeta`), select the `competitor` window (Ctrl-b w), Ctrl-C it
(two flight interfaces must never run together), detach (Ctrl-b d). Then, in your own
`~/skybeta/host/run.sh shell`:

```bash
cd /root/team_ws && team_build.sh --packages-select blueraven_skybeta && source install/setup.bash
ros2 launch blueraven_skybeta blueraven.launch.py mission:=hover height_m:=0.5 hover_s:=30.0
```

For scored runs (later, the team decides): point the stack at our package in `/etc/skybeta/drone.env`
(`COMPETITOR_PKG=blueraven_skybeta`, `COMPETITOR_LAUNCH=blueraven.launch.py`); it takes effect at the next
stack start, so set it before a power-up, never by restarting a running stack.

Our node takes `30` or `30.0`; write **decimals for the flight interface's own arguments**
(`odom_to_ned_yaw_offset_rad:=0.0`, `max_speed_mps:=1.5`): it types them from their defaults and refuses
an integer at start-up.
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
2. **Frame offset (the guide's section 1 caveat).** On the ground, props off:
   `ros2 run blueraven_skybeta yaw_offset_probe --ros-args -p seconds:=40.0`, turning the drone slowly by hand
   to 3-4 headings. It reads the vision odometry and PX4's attitude (read-only), converts both with the
   flight interface's own conventions and prints `RESULT: offset ... spread ...`. A small spread means a
   constant offset: that is `odom_to_ned_yaw_offset_rad` (a launch argument; our node then applies it to its
   twists too, which the interface itself does not). Under 3 deg: fly the hover at 0 first. A spread that
   grows with heading is not a constant offset: stop and ask. Write the result down either way.
3. **Handover.** Safety pilot: arm with the mode switch on **Altitude**, throttle low, then switch to
   **Position**. Expect `offboard_active` true in `/skybeta/state` and `WAIT_OFFBOARD -> TAKEOFF` in the
   log; motors spin up (the interface's 0.3 m floor asks for 0.3 m at once). With no props it can't climb:
   about 7 s later the node logs `TAKEOFF -> DESCEND: climbed only 0.00 of 0.50 m ...: landing`, and when
   PX4 disarms (about 10 s after arming, `COM_DISARM_PRFLT`) it logs `DESCEND -> DONE`. If PX4 disarms
   first, it logs `TAKEOFF -> ABORTED: Offboard ended ...` instead and calls `/skybeta/hold` once. **Either
   is the expected bench result**; anything else (no TAKEOFF, an error, a crash) is not.
4. **Take-back practice.** Restart the node (it never re-engages after an abort, by design). Arm, switch
   to Position, then raise the throttle towards centre: Offboard must end, the node must go `ABORTED`.
5. **Stack restart (optional).** In `WAIT_OFFBOARD`, restart the stack: the node goes back to `INIT`
   ("odometry lost before take-off: re-fixing map <- odom") and returns to `WAIT_OFFBOARD`.

## 5. Hover, props ON (tethered, low)

The team approves the mission before it arms. The pilot holds the transmitter, has tested the kill switch,
and takes over at **any** unexpected motion.

1. Drone still on the launch mark; status `WAIT_OFFBOARD`; recording started.
2. Pilot arms (Altitude, throttle low), then switches to Position within a few seconds.
3. Expect: a jump to the interface's 0.3 m floor as Offboard starts, then the climb to `height_m` at 0.3 m/s.
   If PX4 settles more than 8 cm off the height, the node waits up to 5 s, then hovers anyway and says by
   how much (`holding -0.12 m off the target height`); if it climbed less than half way, it lands. Then
   `HOVER` for `hover_s`; `DESCEND` straight down at 0.35 m/s (no sideways correction: a twist goes through
   the unmeasured frame offset, and a correction rotated ~90 deg circles, >90 deg runs away;
   `descend_xy_hold:=true` turns it on once the offset is measured and set); PX4 detects the
   touchdown and disarms (about 2 s); `DONE`. The descent is 0.35 m/s because PX4's land detector only
   sees a descent at >= 0.275 m/s commanded: at 0.2 m/s the T4 sim sat armed on the ground for 63 s. If the
   drone is on the floor 3 s without PX4 reporting landed, the node presses down at 0.6 m/s and says so; if
   it still stays armed, the pilot disarms it.
   The descent never aborts and never calls `/skybeta/hold`, even if odometry is lost near the floor: the
   interface's hold returns to the last safe position, which would be the hover point in the air.
4. **If it drifts more than 0.4 m from the take-off point** while climbing or hovering, the node descends
   straight down on its own (`... (PX4 frame offset?): landing straight down`). That points at the frame
   offset of step 4.2: measure it before the next flight.
5. **Abort** (one `/skybeta/hold`, then silence): `/skybeta/state` older than 0.6 s, odometry older than
   0.5 s, an odometry jump, or Offboard ending. The pilot then lands it: in Position, low throttle descends.

Pass (sw-claude's checklist step 1): no VIO jumps; horizontal drift p95 under 0.15 m; lands and disarms
by itself.

## 6. Route mission: the hard-coded course

**The run script:** `skybeta/blueraven_skybeta/drone/fly_route.sh` (copy it to the drone host). In its own SSH window:
`OFFSET=0.0 ./fly_route.sh go` flies ROUTE 1 with refnav, the defaults: `route_v7_cp2_simple`, Darion's method to
CP2 (straight forward, 90 deg right, straight through the gap, land past it; from `tools/make_route_cp2_simple.py`).
`ROUTE=route_v7_window` is Hinson's full course. skybeta-09's measured offset is 0.0; `_sim` files are refused on the
drone; `PROGRAM=shell` only where the package has the route mission. It checks the host, NTP, the camera, odometry,
PX4 and that the program is built. It stops the organizers' example, starts the recording, launches, waits until the
program is ready (refnav's take-off pose, or the shell's WAIT_OFFBOARD) and checks one flight_interface plus one
mission. Then it says READY for the pilot, follows the phases
to the disarm, and stops everything. `status`, `stop`, `check` and `build` do what they say. It has no fixed waits:
each step polls for its event.

```bash
# NODE, drone on the mark facing course north
ros2 launch blueraven_skybeta blueraven.launch.py mission:=route route_file:=<name>.json height_m:=0.7 odom_to_ned_yaw_offset_rad:=<MEASURED>
# then count processes before arming: one flight_interface, one blueraven_mission (Shiori's checklist)
```

Take off, hover `route_hover_s` (3 s) over the take-off point, fly the file's corners at `cruise_mps`, land at
the last corner. A route file is `{"route": [[forward_m, left_m], ...]}`, metres from the take-off pose (the
take-off point itself is implied). A bare file name is looked up in `routes/` (the one copy: edit it there with
nano and relaunch, no rebuild). A bad file is refused at start-up and nothing is published. **Expect** in the
log: `mission ROUTE (take-off frame: x forward, y left) from .../routes/<file>: N corners [...]`.

**No map at runtime.** The field was rebuilt and there is no map of it (Arnav's directive, 2026-11-07): flight
position is raw odometry in the take-off frame. Course mode, the start-up planner, map alignment and the
map-based wall centring were removed from this package; nothing here can start them.

**Before a route flight:**
- **The frame offset must be known.** The route is flown with twists, which the interface does NOT rotate by
  the frame offset (poses it does). Our node rotates them by `odom_to_ned_yaw_offset_rad`; with the wrong value
  the drone flies off at that angle (T4: -90 deg). Measure it on the bench (`yaw_offset_probe`, section 4.2)
  or from the hover bag, and pass it.
- **Point the drone exactly along course north** on the mark: the route is relative to the take-off heading.
  3 deg off moves a point 2 m away by about 10 cm.
- **Corners:** `route_max_radius_m` 0 stops and turns in place at every corner. An in-place 90 deg turn at
  45 deg/s slipped cuVSLAM 0.4 m on the T4: lower `max_yaw_rate_dps`, or use arcs (`route_max_radius_m` > 0,
  turning while moving), only as tested in the sim.
- **Staged:** first take-off -> through the CP2 gap -> land, then extend the file.

## Parameters

| Name | Default | |
|---|---|---|
| `mission` | `hover` | `hover` or `route` |
| `route_file` | (none) | route: a `.json` path, or a bare name in `routes/` |
| `route_hover_s` | 3.0 | route: hover before flying (0 = go at once) |
| `route_max_radius_m` | 0.0 | route: corner arcs (0 = stop and turn) |
| `max_yaw_rate_dps` | 45.0 | cap on the turn rate |
| `height_m` | 0.70 | flight height above the launch point |
| `hover_s` | 5.0 | hover: seconds at height |
| `cruise_mps` | 0.30 | route speed |
| `odom_to_ned_yaw_offset_rad` | 0.0 | measured frame offset: passed to the interface (poses) and our node (twists) |
| `anchor_setpoints` | false | the interface's translation fix |

## Not tested

Never run under ROS or on the drone. Tested only as unit tests against a kinematic toy drone
(`tests/test_skybeta_mission.py`): the phase logic, the aborts, restarts, the hover and the route.
The hover flew on skybeta-09 on 2026-11-07; the route has not flown. Unknown until the bench: real `/skybeta/state` contents, the
interface's behaviour at the handover, PX4's tracking, the frame offset, and whether a twist descent
lands and disarms cleanly.
