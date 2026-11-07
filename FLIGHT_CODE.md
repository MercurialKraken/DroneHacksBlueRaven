# This workspace contains the flight code that runs

`main.py` is the entrypoint, not the controller. The `refnav/` folder next to it
is ordinary editable Python source. **This local package flies the mission**;
it is not a display copy of a hidden managed controller.

| File | Responsibility | Start reading here |
|---|---|---|
| `main.py` | Select route, height and hover duration; start the local node | `HEIGHT_M`, `HOVER_S` |
| `refnav/core.py` | Arming readiness, climb, hover, navigation, depth-loss and takeover decisions, landing request | `Params`, `Mission.step` |
| `refnav/node.py` | Subscribe to real ROS 2 sensor/PX4 messages; publish Offboard setpoints and vehicle commands | `RefNav.__init__`, `_tick`, `_command` |
| `refnav/depth.py` | Decode depth and calculate observed clearance from camera calibration | `DepthFrame` |
| `refnav/launch.py`, `flight.yaml`, `route.json` | Standard ROS deployment arguments/settings shared by hardware and sim | `arguments` |
| `refnav/gps.py` | Read native GPS before the GPS-only lesson; no publishers | `wait_for_gps` |
| `tests/test_refnav_core.py` | Unit tests for the actual local flight/depth package | `python3 -m pytest -q tests` |
| `flight-source.json` | Hashes of the initial baseline copied into this project | Initial provenance, not your current run manifest |
| `onboard/slam/`, `onboard/vio/` | Actual cuVSLAM wrapper, vision conversion, timing and health checks | `onboard/README.md` |
| `onboard/adapters/`, `onboard/config/`, `onboard/services.py` | Estimated odometry/TF, GNSS health, service argv and visible process ownership | `python3 onboard/services.py status` |

## What actually makes it go up and down

In `refnav/core.py`, `Mission.step` handles the `climb` phase. PX4 positions use
NED coordinates, so up means decreasing z:

```python
target = self.home[2] - p.height_m
self.sp[2] = max(target, self.sp[2] - p.climb_rate_mps * dt)
```

The same function holds that setpoint during `hover`, then requests `land`.
`refnav/node.py::_tick` publishes the position as a real
`px4_msgs/TrajectorySetpoint` on `/fmu/in/trajectory_setpoint`, alongside
`OffboardControlMode`. It sends a normal `VEHICLE_CMD_NAV_LAND`; once PX4 reports
AUTO_LAND, this mission stops streaming setpoints and lets PX4 finish its own
landing. It never force-disarms. No browser HTTP request supplies a control tick.

The **PX4 autopilot is separate firmware**, just as it is on a real aircraft.
Its position, attitude and rate controllers run in SITL, not in this Python
workspace. Physics and the simulated sensors are also separate. The configured
VIO/estimator and its bridge now execute the editable `onboard/` source inside
this companion when the deployment declares companion ownership. Physics, PX4,
virtual sensor delivery and the simulation bridge stay managed. A legacy image
requires an explicit image/workspace migration; a source copy alone is not enough.

## Edit and verify your actual program

1. Open `refnav/core.py` in the Explorer. For example, change `Params.climb_rate_mps`
   to alter the climb ramp, or edit the `climb` logic itself. Keep the existing
   takeover, sensor-health and landing safeguards while developing a new behavior.
2. Run `python3 -m pytest -q tests` in the companion terminal. These are unit tests
   with a kinematic fake PX4: they test decisions, **not real flight dynamics**.
   They publish no commands to the running simulator. If pytest is absent in your
   current companion image, install it into your own environment and declare the
   dependency; the simulator's Python environment is separate.
3. Use Reset simulation in Session after a previous flight, then execute
   **`python3 main.py`**. It loads your saved local source. Changing a helper file
   does not make that helper the executable entrypoint. Finish the mission before
   editing flight code, just as on hardware.
4. For a managed recording, the optional workbench Run button snapshots the whole
   project. Inspect its `project/`, `manifest.json` and program log to identify
   what executed. Direct Python is not automatically a managed recording.

You can check source resolution without starting ROS or commanding a vehicle:

```sh
cd /var/lib/flyx-lab/workspace
python3 -c 'from refnav import core; print(core.__file__)'
```

In the editor terminal this should name this workspace's `refnav/core.py`.
During a managed Run it names the run's immutable `project/refnav/core.py` copy.
The supplied main refuses a missing or external package instead of silently
falling back to the toolkit's reference mission.

`python3 main.py` starts ordinary ROS directly. The optional workbench Run button
adds managed recording/grading; direct terminal execution is
not automatically a managed Run. Native PX4 examples are in `onboard/examples/`.

## Sim-to-real boundary

The flight core uses numbers and timestamps; the ROS node uses actual camera,
TF, health and PX4 messages. Neither reads simulator truth to navigate. On an
aircraft the same package needs the declared native camera/PX4 interface and
reviewed deployment parameters, including calibrated TF, metric depth, health
and clock settings. No managed runtime library or cloud CLI is required there.
The reference hardware launch is a template and has **not flown**;
simulation success is not hardware flight qualification. The cloud companion is
x86 Ubuntu/Jazzy, not ARM JetPack or a Jetson performance emulator.

The maintained baseline is `reference/refnav/` in the toolkit. New projects copy
its full source once. Your workspace is yours thereafter; worker restarts and
world switches must not reset your algorithm to the baseline. Keep the full
project in version control. Existing saved projects are never silently migrated.
