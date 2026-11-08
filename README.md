# takeoff_land

This is a complete editable companion flight project. New lab workspaces receive
these ordinary Python files and their configuration automatically. Copy the full
project and declare its dependencies when preparing hardware deployment.

Open **FLIGHT_CODE.md** for the source map. `main.py` starts the local `refnav/`
package: edit `refnav/core.py` for the climb, hover, navigation and landing
decisions, `refnav/node.py` for the ROS 2/PX4 interface, and `refnav/depth.py` for
depth processing. Run `python3 -m pytest -q tests` for the local decision tests.
Run **main.py** after editing helpers; reset after a previous run:

```sh
python3 main.py
```

Plain Python directly starts the local ROS node, like on a real companion; it
does not submit a lab operation. The optional workbench Run button
snapshot your project and provide admission, recording and independent grading.
Direct terminal programs need their own recorder, such as `ros2 bag record`.
Use Reset simulation in the Session UI when you need a fresh simulated vehicle.
The cloud companion also provides `python` as Python 3 when no interpreter of
that name has already been installed. Outside the cloud, use the reviewed
local/hardware deployment configuration. This source imports no runtime library:
`flight.yaml`, `route.json` and `refnav/launch.py` are ordinary editable files.
`/etc/flyx/mission.yaml` (or `DRONE_PARAMS_FILE`) may supply the deployment's ROS
parameters. The hardware default clock is system time; simulation deployment
selects simulated time. Both use the native topic/frame contract documented in
`docs/NATIVE_DRONE_INTERFACE.md` in the toolkit. Physical launch is not qualified.

The initial algorithm is copied unchanged from the maintained reference package.
Its earlier baseline flew on the rig on 2026-10-18 (docs/CURRENT_STATE.md); the
workspace packaging change is separate from moving-flight qualification.

`onboard/` contains the actual editable cuVSLAM, vision bridge, health/odometry
nodes, launch configuration and process manager. See `onboard/README.md`.
`python3 onboard/services.py status` names the running source and log files;
stop/start explicitly after editing perception code. Startup never arms.

## Drone-faithful sim runs on the T4 (`tools/drone_sim/`)

SIM-ONLY. `tools/drone_sim/run.sh` flies the drone program on the T4 the way it runs on skybeta-09: the
organizers' `skybeta_flight_interface` 0.3.3 (cloned and built in `/tmp/drone_sim_ws`, rebuilt after a container
restart) started by the drone's own launch file, with `odom_to_ned_yaw_offset_rad:=-1.5708` (the T4's frame) and
`launch_key:=launch_sim`. The repo's `skybeta/` packages are rebuilt every run. Each run resets the world, checks
the frame with `yaw_offset_probe` (refuses beyond 5 deg), records a light bag (no images), starts the program,
lets `pilot.py` arm and switch to Position like the safety pilot, and scores against Gazebo truth (`score.py`).
It leaves nothing running. Run inside the `dronehacks_gpu` container, one flight at a time:

```sh
MISSION=blueraven_skybeta/blueraven.launch.py tools/drone_sim/run.sh --route none --name hover hover_s:=10.0
```

- `MISSION`: PACKAGE/LAUNCH_FILE of the drone program; every `ARG:=VALUE` goes to that launch file.
- `--depth none|sim`: `sim` publishes the sim's stereo depth as `/oak/stereo/image_raw` + `camera_info` (640x400 16UC1 mm).
- `--latency-ms MS`, `--drop P`: added vision latency and frame drop probability on the cuVSLAM stereo input (default 0 and 0.01, the sim's own).
- `--route FILE|none`: route scored for completion (default `route.json`); `--timeout S`: wall seconds before the pilot lands it.

The score (`score.json` in the bag folder `~/bags/drone_sim_NAME_<UTC>/`) gives route points reached, flight time,
minimum truth clearance, estimate error p95 (PX4 xy against truth), landing distance and whether PX4 disarmed
itself; "completed" means the pilot never took over.
