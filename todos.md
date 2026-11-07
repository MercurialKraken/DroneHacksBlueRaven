# TODOs

The IDs in brackets point to [issues.md](issues.md). Put your initials after a task when you take it, and tick it off when it's done.

**Deadlines (PST)**

| When | What |
|---|---|
| Sat 7:30 PM | Run blocks and presentation slots posted |
| Sat 10 PM – Sun 2 AM | First scored block |
| **Sun 12:30 PM** | Repo link due; presentations start |
| **Sun 4:00 PM** | Everything flown must be pushed; scored runs close |

**Priorities**

- **P0**: needed for checkpoints 1–2 (maze exit).
- **P1**: needed for the full course.
- **P2**: points polish.

**Ground rules**

- Ask the team before any flight that arms, in sim or on hardware.
- Reset world after keyboard RC.
- Never restart onboard services while armed [B2].

---

## Phase 0: measure before building (Fri night) · P0

- [ ] **T01** Confirm the sim course matches the latest published layout (the sim says "CAD v3") [G3]
- [ ] **T02** Add the published course files (OBJ, STEP, GLB) to `maps/`, with the source URL, download time and version [D1]
- [ ] **T03** Snapshot the effective deployment config into `config/sim_deployment/` (no secrets): `/etc/flyx/onboard.yaml`, `/etc/flyx/mission.yaml`, `/etc/flyx/onboard.json` [H3, B3]
- [ ] **T04** Record the sim PX4 parameters: `EKF2_EV_CTRL`, `EKF2_HGT_REF`, `EKF2_RNG_CTRL`, `EKF2_EV_DELAY`, `EKF2_EVP_NOISE`, `EKF2_EVV_NOISE`, `EKF2_NOAID_TOUT` [B4, A5]
- [ ] **T05** Write a sim-only evaluator in `tools/sim_eval/` that logs PX4's estimate and cuVSLAM's pose against `dronehacks_sim` truth and reports drift per metre. Flight code must never import it [A1, D11, G5]
- [ ] **T06** Approved sim flight down corridor 1 with the evaluator running. Record drift, features per frame, and whether gravity initializes after takeoff [A1, A2]
- [ ] **T07** Tunnel probe in sim: fly to and into the tunnel. Record features per frame, depth valid fraction, `/estimator/healthy` and position error [A3]
- [ ] **T08** From the CAD, record the course dimensions in `maps/README.md`: corridor widths, Gate A and B sizes, window size and height, tunnel cross-section and length, pad rings [C3, D6]
- [ ] **T09** Ask the organizers:
  - Does our start command include arming, or does the team arm first (rule 7.4)?
  - What abort channel is expected?
  - Which cuVSLAM build is on the Jetson? [C4, F1, E3]
- [ ] **T10** Add a `.gitignore` (`__pycache__/`, `.onboard/`, `.pytest_cache/`, bags, logs) and `git rm --cached` the committed `.pyc` files and `.onboard/` [H1]

## Phase 1: real-drone bring-up (Sat morning, in parallel) · P0

- [ ] **T11** Inventory the Jetson [E3, E4, E7, C7]:
  - ROS distro and Python version
  - px4_msgs version and the `/fmu/out/*` names (with or without `_v1`)
  - Topics exported by the DDS bridge (is `distance_sensor` there?)
  - Python cuVSLAM bindings or the Isaac ROS container
  - depthai-ros version
- [ ] **T12** OAK bring-up [E1, A9, A10]:
  - depthai-ros publishing rectified mono left and right, IMU and depth
  - Remap to `/stereo/left/image`, `/stereo/right/image`, `/stereo/{left,right}/camera_info`, `/stereo/imu` and `/camera/depth/image_raw`
  - Pick the camera rate
- [ ] **T13** Static TF from `base_link` to `oak_left`, `oak_right`, `oak_imu` and the depth optical frame, from the drone CAD plus the OAK factory calibration [E2]
- [ ] **T14** Bench test, props off: cuVSLAM and the EV bridge on the Jetson. Check that `/estimator/healthy` goes true, PX4 reports `xy_valid` and heading, and note the tracking time [E3, E5]
- [ ] **T15** Set the real OAK IMU's noise and rate in `vio.yaml` [A12]
- [ ] **T16** Confirm the rangefinder reaches PX4 and read the EKF2 height setup on the real drone [B4, E7]
- [ ] **T17** Resolve the `_vN` topic suffixes from px4_msgs, the way `takeoff_land.py` does, everywhere we subscribe [C7]
- [ ] **T18** Compute budget on the Orin: cuVSLAM, depth, localization and planner together. Set the loop rates [E5, A7, A8]
- [ ] **T19** Check the clock and timesync on hardware. Use system time for `/fmu/in/*` stamps [E6, B6]

## Phase 2: core autonomy (Fri night → Sat) · P0

- [ ] **T20** Frames:
  - Rename cuVSLAM's output frame to `vio_odom`.
  - Use the TF tree `map → odom (PX4 estimate) → base_link → oak_*`.
  - The planner works in `map`; setpoints are converted to `odom` before going to PX4. [A6, B1]
- [ ] **T21** Map builder: OBJ to a 3D occupancy grid (5 cm) plus an ESDF, cropped to the net. Validate it against the sim voxel grid in a sim-only check [D1, G3]
- [ ] **T22** Course config, the only course-specific file:
  - Launch pose and heading
  - Checkpoint goal regions (regions, not waypoints)
  - Height band (stay under 68 in)
  - Chicane option [F2, F4, B3]
- [ ] **T23** Global planner: A* or fast marching on the ESDF through the goal regions. Replan if the map changes [D3, F2]
- [ ] **T24** Mission supervisor (replaces refnav's arm-on-ready):
  - Stages: idle → wait for START → takeoff → maze → tunnel → chicane → land → disarm → optional return
  - Per-stage timeout and health policy
  - An abort path
  - Never restart services in the air [C1, C4, C8, D5, B2]
- [ ] **T25** Start trigger: exactly one start command (ROS service or topic). Ignore everything after it except abort [F1]
- [ ] **T26** Fallback controller: pure pursuit along the path, sending `TrajectorySetpoint` with xy position plus velocity feedforward and z on position [C5, C6]
- [ ] **T27** Milestone: checkpoints 1–2 in sim with the fallback controller. Record evidence.

## Phase 3: full course · P1

- [ ] **T28** `map→odom` localization [A1, B1, D2]:
  - Take depth points in `odom` and align them to the map ESDF over (x, y, yaw), starting from the current estimate.
  - Use a robust loss, ignore points above 68 in and on the floor.
  - Only correct directions that are observable (degeneracy check).
  - Smooth and publish at 5–10 Hz. Never into the vision stream.
- [ ] **T29** Local perception: depth (plus ToF) into a rolling local grid and ESDF, combined with the map ESDF by taking the minimum. Unknown is not free [D4, C2, C3]
- [ ] **T30** MPPI on the GPU [D3, C2, E5]:
  - Costs: ESDF margin, path progress, height band, smoothness, speed.
  - Output position plus velocity setpoints.
  - Benchmark on the Orin; fall back to T26 if it misses its deadline.
- [ ] **T31** Tunnel mode [A3, A4, C8]:
  - Decide `imu_only_policy` or a quality flag.
  - Keep the health logic from landing us inside.
  - Center with ToF and dead-reckon for a bounded time.
  - Detect exit and require vision to reacquire.
- [ ] **T32** Auxiliary sensors: pick side and front multizone ToF (each under $100 and 50 g, counting its microcontroller). Mount, write the driver, and get inspection [D9, F6]
- [ ] **T33** CBF safety filter on raw ranges (OAK near-field plus ToF) between the planner and PX4 [D10]
- [ ] **T34** Chicane through Gate B first [D6]
- [ ] **T35** Landing: approach the pad, descend, land inside the outer ring, confirm landed and disarmed [D7]
- [ ] **T36** Vision quality: give velocity a real covariance (or stop fusing it), and mark IMU-only frames with inflated covariance or `quality` [A4, A5]
- [ ] **T37** Log a throttled warning when cuVSLAM receives an unsupported image encoding [A9]

## Phase 4: points polish · P2

- [ ] **T38** Speed tuning. Mind the 5 m/s vision health gate [B5]
- [ ] **T39** Window route (+5): 3D approach through the opening [D6]
- [ ] **T40** Return bonus: re-arm, take off, fly over the barrier (allowed on the return), land within 2 ft of the launch mark. Handle the rangefinder height jump over the barrier [D8, B4]
- [ ] **T41** Optional: enable cuVSLAM SLAM within a run for loop closure on the return. Never save or reload its map [A11, F3]
- [ ] **T42** Optional: follow the floor line for generality (mono in sim, RGB on hardware) [G1]

## Phase 5: submission

- [ ] **T43** README: architecture, how to run in sim and on hardware, deployment steps, credits (PX4, cuVSLAM, depthai-ros, ROS 2, the organizers' starter) [F5, E8]
- [ ] **T44** Submit the repo link by **Sun 12:30 PM**. Push everything flown by **Sun 4:00 PM**.
- [ ] **T45** Presentation: architecture diagram, drift plots (T05), tunnel tests (T07), failure log, design decisions
- [ ] **T46** Auxiliary sensor declarations: part number, price link, mass [F6]
- [ ] **T47** Self-review for hard-coding before the AI review: no literal coordinates, routes or timings outside the course config [F2]

## Done

- [x] Type hints on every Python module (all signatures, plus instance attributes where the type wasn't obvious). Verified as a runtime no-op: the AST is identical to `HEAD` once annotations are stripped, all 22 files compile on Python 3.10, and 30/30 unit tests pass before and after. Not covered: `main.py` (no functions) and the code embedded in `learn/*.ipynb`.
