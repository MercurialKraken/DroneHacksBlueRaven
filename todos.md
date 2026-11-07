# TODOs

**Goal:** fly checkpoints 1–2 autonomously in sim, then the full course, then on the drone. Within each section, top to bottom is the order to work in.

## State estimation

- [ ] **First data flight** with `tools/sim_eval/` (written, not yet run on the VM). It gives drift, inertial-init timing, latency and tunnel behaviour against truth.
- [ ] **IMU noise to the sim model.** Set `gyro_noise_density` 3.4e-4 and `accel_noise_density` 2.5e-3, plus the random walks 1.82e-6 / 3.7e-5 once their units are confirmed. A/B against today's 10× values by bag replay (`tools/sim_eval/replay_vio.sh`).
- [ ] **Measure variance.** If the measured error is above PX4's 5 cm floor, set `position_floor` in `vision_bridge.yaml`.
- [ ] **Flag IMU-only frames.** Inflate their covariance instead of publishing them as normal.
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
- [ ] **Tunnel mode.** Keep vision flowing (PX4 descends after 1 s without it). Center with ToF and dead-reckon for a bounded time. Don't land on bad vision health.
- [ ] **Pad landing.** Approach, descend inside the outer ring, confirm landed and disarmed.

## Hardware

- [ ] **OAK launch.** Rectified mono L/R, IMU and depth, remapped to the sim's topic names.
- [ ] **Static TF** from `base_link` to the `oak_*` frames, from the drone CAD and the OAK calibration.
- [ ] **Bench test cuVSLAM and the EV bridge on the Jetson, props off.** PX4 should report a valid position.
- [ ] **Real OAK IMU noise and rate** in `vio.yaml`.
- [ ] **Resolve the `_vN` PX4 topic names** from `px4_msgs`, the way `takeoff_land.py` does.
- [ ] **Log a warning** when cuVSLAM receives an image that isn't `mono8`.
