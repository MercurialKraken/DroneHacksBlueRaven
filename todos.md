# TODOs

**Goal:** fly checkpoints 1–2 autonomously in sim, then the full course, then on the drone. Within each section, top to bottom is the order to work in.

## State estimation

- [ ] **Fix IMU initialization.** Fly in sim and check that cuVSLAM's gravity estimate appears after takeoff. If it doesn't, fix the IMU rate (500 vs 480), noise values, timestamps and IMU-to-camera extrinsic in `vio.yaml`.
- [ ] **Drift evaluator (sim only).** Log cuVSLAM's pose and PX4's estimate against `dronehacks_sim` truth. Report drift per metre.
- [ ] **Measure variance.** From the evaluator, measure the real pose and velocity error. Set `position_floor`, `orientation_floor` and `velocity_floor` in `vision_bridge.yaml`.
- [ ] **Velocity covariance.** Fill in cuVSLAM's twist covariance, or stop sending velocity to PX4.
- [ ] **Flag IMU-only frames.** Inflate their covariance instead of publishing them as normal.
- [ ] **Read the EKF2 parameters in sim** (`EV_CTRL`, `HGT_REF`, `RNG_CTRL`, `EV_DELAY`). Set `EV_DELAY` to the measured latency.
- [ ] **Rename cuVSLAM's frame** to `vio_odom`. TF tree: `map → odom (PX4) → base_link`.
- [ ] **`map→odom` localization.** Match depth against the map's distance field over (x, y, yaw) and publish the correction. Never feed it into PX4's vision input.

## Perception

- [ ] **Course map.** Turn the published OBJ into a 3D occupancy grid (5 cm) plus a distance field.
- [ ] **Local map.** Build a rolling grid from depth, turn it into a distance field, and combine it with the course map by taking the minimum. Unknown is not free.
- [ ] **Side ToF for the tunnel.** Prototype in sim with `dronehacks_sim`, as in `SIM_SENSOR_EXAMPLE.py`, then on hardware.

## Planning

- [ ] **Course config.** One file holding the launch pose, checkpoint goal regions and height band. It's the only place course-specific data lives.
- [ ] **Global path.** A* on the distance field through the goal regions.
- [ ] **MPPI on the GPU.** Only after the simple tracker below flies.

## Control

- [ ] **Path tracker.** Send `TrajectorySetpoint` with position plus velocity feedforward in xy, and position in z.

## Mission

- [ ] **Supervisor to replace refnav.**
  - Flow: start command → takeoff → maze → tunnel → chicane → land → disarm.
  - Each stage gets its own timeout and health rules.
  - Abort on a land or kill command.
- [ ] **Tunnel mode.** Center with ToF and dead-reckon for a bounded time. Don't land on bad vision health.
- [ ] **Pad landing.** Approach, descend inside the outer ring, confirm landed and disarmed.

## Hardware

- [ ] **OAK launch.** Rectified mono L/R, IMU and depth, remapped to the sim's topic names.
- [ ] **Static TF** from `base_link` to the `oak_*` frames, from the drone CAD and the OAK calibration.
- [ ] **Bench test cuVSLAM and the EV bridge on the Jetson, props off.** PX4 should report a valid position.
- [ ] **Real OAK IMU noise and rate** in `vio.yaml`.
- [ ] **Resolve the `_vN` PX4 topic names** from `px4_msgs`, the way `takeoff_land.py` does.
- [ ] **Log a warning** when cuVSLAM receives an image that isn't `mono8`.
