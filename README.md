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
