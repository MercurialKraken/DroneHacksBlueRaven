# skybeta-10 run commands (final run)

Drone: **skybeta-10**. Deployed and verified: `arnav/skybeta10` 1797480, plus `route_full_hold_v25`.
Final run: **MPPI, full course, 1.1 m, 0.25 m/s**. It passed 2/2 in T4 Gazebo with the real window (sill 0.914 m).

---

## 1. Reset (by hand, before EVERY flight)
1. Put the drone **on the mark**: Pixhawk 110 cm south of the top red tape, **nose north**, level.
2. **Pixhawk USB unplugged.** RC on. **Kill switch (ch5) in RUN.**
3. **Both batteries out 10 s, then back in** (a full power cycle; it also resets the VIO).
4. **Nobody in the camera's view or moving behind the nets**, from power-up to landing. Hands off.

---

## 2. Connect

DroneHacks Wifi:
```
ssh dronehacks@skybeta-10.local
```
Or over USB-C:
```
ssh dronehacks@192.168.55.1
```
Clear any old tmux session:
```
tmux kill-session -t fly 2>/dev/null
```
Wait for the stack:
```
until docker ps --format '{{.Names}}' | grep -qx skybeta-stack; do sleep 2; done; echo STACK UP
```
Check it's ready:
```
cd ~/team_ws && ./fly_route.sh status
```
Wait for **PX4 link up, position not null, odom ~30 Hz**, then press **Ctrl-C**. If it says "couldn't copy the status monitor", the stack is still starting: run it again.

---

## 3. Fly: run ONE of these

**FINAL RUN: MPPI full course** (1.1 m, 0.25 m/s, net fence, window band):
```
cd ~/team_ws && tmux new -s fly "MAX_SPEED=0.6 DEPTH_BAG=0 NAV=mppi OFFSET=0.0 HEIGHT=1.1 MPPI_SPEED=0.25 ROUTE=route_mppi_full ./fly_route.sh go; bash"
```

**BACKUP: refnav dead-reckoning full course** (1.1 m, 0.25 m/s, 8 s hover after the climb):
```
cd ~/team_ws && tmux new -s fly "MAX_SPEED=0.6 DEPTH_BAG=0 OFFSET=0.0 HEIGHT=1.1 ROUTE=route_full_hold_v25 ./fly_route.sh go; bash"
```

**BACKUP if 0.25 lands "stuck" right at the start: refnav at 0.15 m/s:**
```
cd ~/team_ws && tmux new -s fly "MAX_SPEED=0.6 DEPTH_BAG=0 OFFSET=0.0 HEIGHT=1.1 ROUTE=route_full_hold ./fly_route.sh go; bash"
```

**Route 1 only, MPPI** (start → CP2 gap → land):
```
cd ~/team_ws && tmux new -s fly "MAX_SPEED=0.6 DEPTH_BAG=0 NAV=mppi OFFSET=0.0 HEIGHT=1.1 MPPI_SPEED=0.25 ROUTE=route_mppi_r1 ./fly_route.sh go; bash"
```

**Route 1 only, refnav** (8 s hover):
```
cd ~/team_ws && tmux new -s fly "MAX_SPEED=0.6 DEPTH_BAG=0 OFFSET=0.0 HEIGHT=1.1 ROUTE=route1-hold ./fly_route.sh go; bash"
```

Every check prints PASS or FAIL. A FAIL stops everything before arming, and prints the fix.

---

## 4. At GO
Wait for **`GO - PILOT MAY ARM`**, then detach: press **`Ctrl-b`**, release, then press **`d`**. If you're on USB-C, unplug it now.

Pilot:
1. ALTITUDE, throttle low.
2. Arm.
3. Switch to **POSITION**.

The code flies and lands by itself.

**Pilot watches:**
- the **north net** on the long eastward leg (VIO drift). Take over if it gets within about 0.5 m;
- the **window**;
- **MPPI swaying in place for about 5 s without progress:** take over and land.

---

## 5. After landing
Reattach:
```
tmux attach -t fly
```
Wait for the bag path line (`~/team_ws/bags/<route>_<time>`), then close the session:
```
exit
```
**If no bag path printed, or you aborted,** close the recording cleanly before powering off:
```
cd ~/team_ws && ./fly_route.sh stop
```
Then go back to **1. Reset**.

---

## Abort
1. **The pilot takes over:** mode switch out of POSITION, or throttle. **Kill switch only if a crash is imminent.**
2. Reattach:
```
tmux attach -t fly
```
3. Press **Ctrl-C**, then:
```
cd ~/team_ws && ./fly_route.sh stop
```
4. The pilot lands and disarms.

---

## If something fails
`FAIL: odometry … Hz`: wait 1 min and rerun `go`. If it fails again, run this, then rerun:
```
sudo systemctl stop cron
```
Won't arm: in QGC → Analyze → MAVLink Console:
```
listener health_report
```
Duplicate tmux session:
```
tmux kill-session -t fly
```
SSH times out: put the laptop on **DroneHacks Wifi** (not eduroam), or reseat the USB-C in the Jetson's data port.

Status any time, in a second terminal:
```
cd ~/team_ws && ./fly_route.sh status
```

---

## Notes
- `HEIGHT` on the command line overrides the route's height. `MAX_SPEED=0.6` is only a launch check: `go` refuses if `MPPI_SPEED` or the route's speed is above it, which catches typos. It does NOT set the speed: that's `MPPI_SPEED` (MPPI) or the route's `speed_mps` (refnav).
- **Window at 1.1 m:** `route_mppi_full` sets MPPI's depth band bottom to 0.15 m below the drone, so the 0.914 m sill is out of the band. If the real sill is about 1.0 m, MPPI stops before the window and lands.
- **VIO drift:** cuVSLAM is camera-only (IMU fusion off). Moving people in view, especially through the see-through nets, are the most likely trigger, so keep the view clear.
- Bags: `~/team_ws/bags/` on the drone. Copies are on the T4 at `/srv/team/real_drone/flights_2026-11-08/skybeta10/`.
