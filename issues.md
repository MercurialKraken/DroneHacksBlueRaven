# Known issues

Review of the starter code as of 2026-11-06, baseline commit `473800c`. Type hints were added after that commit with no runtime change. Each fix links to a task in [todos.md](todos.md).

**Severity**

- **BLOCKER**: we can't score that part of the course, or can't fly the real drone, until it's fixed.
- **HIGH**: likely to cost a checkpoint or end a run.
- **MEDIUM**: costs time, points or robustness, or will surprise us on hardware.
- **LOW**: hygiene or clarity.

## Top five

1. **There's no navigation stack at all** (C1, D1, D3, D9). The starter only takes off, hovers and lands.
2. **The dark tunnel blinds our only position source** (A3). Checkpoint 3 is worth 10 points, the most of any.
3. **Odometry drifts and nothing corrects it** (A1, D2). The estimator only measures motion from takeoff, there's no map, and drift accumulates.
4. **Real-drone bring-up doesn't exist** (E1, E2, E3). The sim provides the camera topics and TF; on the Jetson we have to build them ourselves.
5. **The mission would break the rules as written** (C4, F1, F2). It arms itself, accepts no start command, and is built on hard-coded legs.

---

## A. State estimation: the cuVSLAM wrapper (`onboard/slam/`)

### A1 · Odometry only: no SLAM, no map, no loop closure · HIGH
- **Where:** [cuvslam_lib_node.py:173-187](onboard/slam/cuvslam_lib_node.py#L173-L187), [:254](onboard/slam/cuvslam_lib_node.py#L254), [:292](onboard/slam/cuvslam_lib_node.py#L292)
- **Problem:** The tracker is built from an `OdometryConfig` only. The SLAM pose that `track()` returns is discarded. The output frame is simply the drone's pose when tracking started.
- **Impact:** Position error grows with distance flown and is never corrected. PX4 fuses this as truth for x, y and yaw; the rangefinder only fixes z.
- **Fix:** Correct drift with map-based localization against the published layout (`map→odom`). See T05, T28.

### A2 · The IMU part of VIO has never been shown to initialize · HIGH
- **Where:** `.onboard/*/cuvslam.log`, which logs `gravity none yet` for about 200 s.
- **Problem:** In Inertial mode, cuVSLAM needs motion before it can estimate gravity. The only logs are from a drone sitting still. [onboard/README.md](onboard/README.md) itself says initialized-inertial operation is "not qualified".
- **Impact:** We don't know whether the IMU actually helps tracking in flight. That matters most in the tunnel.
- **Fix:** Log gravity and the tracking result during a sim flight (T06).

### A3 · The dark tunnel blinds passive stereo · BLOCKER (checkpoint 3)
- **Where:** [onboard/config/vio.yaml:16](onboard/config/vio.yaml#L16) (`imu_only_policy: publish`)
- **Problem:** The OAK-D S2 is passive stereo with no IR projector or flood light. In the dark, cuVSLAM loses its features and stereo depth goes blank. We can't add lights: rule 3.4 allows nothing else that draws power.
- **Impact:**
  - With `publish`, IMU-only poses reach PX4 as normal. They drift, maybe badly (see A2).
  - With `withhold`, the vision input stops. PX4 loses horizontal aiding, and refnav-style logic lands after 1 s of bad health, inside the tunnel. Touching the ground ends the run (rule 7.9).
- **Fix:** A deliberate tunnel mode: active ToF ranging for centering plus short, time-boxed dead reckoning (T07, T31, T32).

### A4 · Downstream code can't tell IMU-only frames from good ones · MEDIUM
- **Where:** [cuvslam_lib_node.py:278-283](onboard/slam/cuvslam_lib_node.py#L278-L283)
- **Problem:** With `publish`, a pose with too few visual features is published as ordinary odometry. Only the side topics (`/visual_slam/status`, `/visual_slam/vis/observations_count`) reveal it. The EV bridge's health check can't see it.
- **Fix:** Inflate the covariance or set `VehicleOdometry.quality` on IMU-only frames, and let the mission supervisor read tracking state (T36).

### A5 · Velocity is a raw finite difference with no covariance · MEDIUM
- **Where:** [cuvslam_lib_node.py:360-367](onboard/slam/cuvslam_lib_node.py#L360-L367)
- **Problem:** Velocity is the difference of consecutive poses at 15 Hz. The twist covariance is never filled, so the bridge sends velocity variance 0 and PX4 falls back to its `EKF2_EVV_NOISE` floor.
- **Impact:** If `EKF2_EV_CTRL` fuses EV velocity, PX4 may over-trust a noisy signal.
- **Fix:** Check the parameter (T04). Then either supply a real covariance or stop fusing velocity (T36).

### A6 · Two different frames are both called `odom` · MEDIUM
- **Where:** [cuvslam_lib_node.py:332](onboard/slam/cuvslam_lib_node.py#L332) and [px4_odom.py:65-69](onboard/adapters/px4_odom.py#L65-L69)
- **Problem:** cuVSLAM stamps its output in `odom`, meaning its startup body frame. `px4_odom` broadcasts the TF `odom→base_link` from PX4's local frame (converted to ENU). Same name, different frames.
- **Impact:** A `map→odom` transform would silently mix the two.
- **Fix:** Rename cuVSLAM's frame to `vio_odom`. Use PX4's estimate as `odom` (T20).

### A7 · Tracking runs inside the image callback on a single thread · MEDIUM
- **Where:** [cuvslam_lib_node.py:201-209](onboard/slam/cuvslam_lib_node.py#L201-L209)
- **Problem:** It takes about 1–3 ms per frame on the sim's x86 GPU, which hides the issue. On an Orin Nano, shared with depth, localization and the planner, slow tracking delays IMU callbacks and the next frames.
- **Fix:** Measure on the Jetson (T18). Use a multi-threaded executor or move tracking to a worker if needed.

### A8 · The sim camera runs at 15 Hz · LOW
- **Problem:** At higher speeds, low frame rates hurt VIO. The real OAK can run mono faster.
- **Fix:** Pick the hardware rate during bring-up (T12, T18).

### A9 · Non-mono8 images are dropped silently · MEDIUM (hardware trap)
- **Where:** [cuvslam_lib_node.py:202-203](onboard/slam/cuvslam_lib_node.py#L202-L203)
- **Problem:** Any encoding other than `mono8` or `8UC1` hits a bare `return` with no log.
- **Impact:** If the OAK driver publishes another format, the tracker never initializes and nothing says why.
- **Fix:** Add a throttled warning (T37).

### A10 · Assumes rectified pinhole images with no distortion · MEDIUM (hardware)
- **Where:** [cuvslam_lib_node.py:168-174](onboard/slam/cuvslam_lib_node.py#L168-L174)
- **Problem:** `rectified_stereo_camera = True` with a Pinhole model. The OAK must publish rectified left and right images whose CameraInfo matches.
- **Fix:** Configure depthai-ros for rectified mono (T12).

### A11 · No loop closure for the return flight · LOW
- **Problem:** The return to the launch mark revisits the start area, where SLAM loop closure would help.
- **Fix:** Optionally enable cuVSLAM SLAM within a single run. Never persist its map between runs (rule 6.4). See T41.

### A12 · IMU noise and rate are sim values · MEDIUM (hardware)
- **Where:** [vio.yaml:10-14](onboard/config/vio.yaml#L10-L14)
- **Problem:** The noise values are 10× the code defaults, tuned for the sim. The rate is overridden to 480 Hz by the sim deployment.
- **Fix:** Set them for the real OAK IMU (T15).

## B. PX4 fusion and the EV bridge (`onboard/vio/`)

### B1 · Map corrections must never go into the vision stream · HIGH (design constraint)
- **Where:** [ev_bridge.py:105-106](onboard/vio/ev_bridge.py#L105-L106), [vio_health.py](onboard/vio/vio_health.py)
- **Problem:** The bridge treats a jump over 1 m as a reset. VioHealth allows one bad sample per 10 s and then needs 5 s of quiet.
- **Impact:** Feeding map-corrected poses into the vision input would cause jumps, mark health false, and the mission lands.
- **Fix:** Keep corrections as `map→odom` on the Jetson and convert setpoints into `odom` (T20, T28).

### B2 · A bridge restart means 25 s with no vision input, and a second publisher kills it · HIGH (operational)
- **Where:** [ownership.py:22-43](onboard/vio/ownership.py#L22-L43)
- **Problem:** On startup the bridge spends 25 s proving it's the only publisher on `/fmu/in/vehicle_visual_odometry`. If it ever sees a count other than one, it raises `RuntimeError` and the node dies.
- **Impact:** A second publisher, such as `ros2 bag play` replaying that topic, or a restart in the air, means EKF2 loses aiding.
- **Fix:** Never restart onboard services or replay that topic while armed. Write this into the checklist (T24).

### B3 · Heading reference: NED in sim, FRD on hardware · MEDIUM
- **Where:** [vision_bridge.yaml](onboard/config/vision_bridge.yaml) and `/etc/flyx/onboard.yaml` (not in the repo)
- **Problem:** The sim deployment injects `heading_enu_rad` with the reference *"Proposed launch mark aligned with the source guideline +Y (CAD v3 assumption; confirm official heading before use)"*, so vision reaches PX4 as NED. The repo config has no heading, so on hardware vision goes in as FRD (startup heading, arbitrary).
- **Fix:** Anchor `map` to the known launch pose and heading. Either measure and configure the heading, or stay in FRD and absorb it in `map→odom` (T03, T22).

### B4 · EKF2 configuration is unknown, and rangefinder height jumps over obstacles · MEDIUM (HIGH for the return)
- **Problem:** We haven't read `EKF2_EV_CTRL`, `EKF2_HGT_REF`, `EKF2_RNG_CTRL`, `EKF2_EV_DELAY` or the noise and timeout parameters.
- **Impact:** If height comes primarily from the rangefinder, flying over the 68 in barrier on the return looks like the floor rising 1.73 m, and the drone climbs. Boulders (18 in) cause smaller bumps.
- **Fix:** Read the parameters (T04, T16). Handle the barrier crossing on the return (T40). Don't touch the organizer safety settings (rule 8.5).

### B5 · The vision health gate rejects speeds over 5 m/s · LOW
- **Where:** `max_speed_mps` in [ev_bridge.py](onboard/vio/ev_bridge.py)
- **Problem:** Fine for a 20 ft net, but a fast run or a VIO glitch above that speed marks vision unhealthy.
- **Fix:** Keep in mind while tuning speed (T38).

### B6 · The two examples timestamp PX4 messages differently · LOW
- **Where:** [refnav/node.py:94](refnav/node.py#L94) uses the system clock; [takeoff_land.py:142](onboard/examples/takeoff_land.py#L142) uses the node clock (sim time under `use_sim_time`).
- **Fix:** Our own nodes follow refnav, using system time for `/fmu/in/*` stamps. Verify on hardware (T19).

## C. The reference mission (`refnav/`)

### C1 · It doesn't navigate · BLOCKER
- **Problem:** It flies straight (forward, left) legs relative to the takeoff pose, read from `route.json`. That file is empty, so today it only takes off, hovers and lands. It can't solve a maze, and hand-written legs are exactly what the rule 9.5 review flags.
- **Fix:** Replace it with a supervisor, planner and controller (T23–T26). Its safety ideas are worth keeping: fail-closed health, unknown depth is not free space, never fight a takeover.

### C2 · Any obstacle ends the run with a landing · HIGH
- **Where:** [core.py:420](refnav/core.py#L420), [core.py:503](refnav/core.py#L503)
- **Problem:** A confirmed obstacle closer than `stop_m` (0.9 m) means "route blocked" and a landing. There's no avoidance and no replanning.
- **Fix:** Local perception plus planning (T29, T30).

### C3 · Narrow corridors read as blocked · HIGH
- **Where:** [core.py:71](refnav/core.py#L71), [depth.py:141](refnav/depth.py#L141)
- **Problem:** Clearance counts every point within ±`corridor_half_m` (0.6 m) of the flight line as "ahead". If corridor walls are closer than 0.6 m to the centerline (under about 1.2 m, or 4 ft, wall to wall), the side walls trip the 0.9 m stop. The only test uses a 2 m corridor.
- **Fix:** Measure real corridor widths (T08). The replacement uses a map ESDF instead (T29).

### C4 · It arms itself, refuses if already armed, and has no start command · HIGH (rules)
- **Where:** [core.py:311](refnav/core.py#L311)
- **Problem:** It arms as soon as its checks pass. If it sees the drone already armed, it gives up with "yielded".
- **Conflict:** Rule 7.4 says the team arms the drone in front of the referee, then starts it on signal. Rule 5.2 allows exactly one start command.
- **Fix:** A supervisor gated on the start command (T24, T25). Ask the organizers how arming should work (T09).

### C5 · It's slow · MEDIUM
- **Where:** [core.py:404](refnav/core.py#L404) and onward
- **Problem:** Each leg turns in place at 45°/s, waits for two confirming depth frames, flies at 0.6 m/s with the target at most 1 m ahead, then holds for 1 s. Time is worth 20 points.
- **Fix:** Fly continuous paths with velocity feedforward (T26, T30).

### C6 · Position-only setpoints · MEDIUM
- **Problem:** No velocity or acceleration feedforward, so tracking lags.
- **Fix:** Send position plus velocity in `TrajectorySetpoint`, and position in z (T26).

### C7 · Hard-coded `_v1` topic names · MEDIUM (hardware)
- **Where:** [refnav/node.py:110-112](refnav/node.py#L110-L112), [px4_health.py:46](onboard/adapters/px4_health.py#L46)
- **Problem:** Correct for the sim's px4_msgs. The Jetson's PX4 and px4_msgs versions may differ. [takeoff_land.py:12](onboard/examples/takeoff_land.py#L12) shows how to resolve the suffix from `MESSAGE_VERSION` instead.
- **Fix:** T17.

### C8 · One second of bad estimator health anywhere triggers a landing · HIGH
- **Where:** [core.py:341](refnav/core.py#L341)
- **Impact:** In the tunnel (A3) this means landing in the tunnel, which ends the run.
- **Fix:** Different health policy per stage in the supervisor (T24, T31).

### C9 · It only accepts a forward-facing depth camera · LOW
- **Where:** [refnav/node.py:205](refnav/node.py#L205)
- **Problem:** Anything more than 25° off forward is rejected, so we can't tilt the OAK down to see the floor line with this package.

### C10 · It ignores the downward rangefinder · LOW
- **Problem:** Height comes only from PX4's estimate. `/fmu/out/distance_sensor` is never read.

## D. Missing capabilities

### D1 · No course map · BLOCKER
- **Problem:** The published layout (OBJ, STEP, GLB on the event site) isn't in the repo. There's no loader, occupancy grid or ESDF.
- **Fix:** T02, T21.

### D2 · No map-based localization · HIGH
- **Problem:** Nothing turns the layout plus live depth into a `map→odom` correction. It's needed for the gates and the window even if drift is small.
- **Fix:** T28.

### D3 · No planner or trajectory controller · BLOCKER
- **Problem:** No global path and no local planner (MPPI) or tracker.
- **Fix:** T23, T26, T30.

### D4 · No local perception · HIGH
- **Problem:** No rolling grid or ESDF from depth or ToF. Unknown space must stay unknown, not free.
- **Fix:** T29.

### D5 · No mission supervisor or start trigger · BLOCKER
- **Problem:** Nothing sequences takeoff, maze, tunnel, chicane, landing, disarm and the optional return, with per-stage timeouts, health policies and abort handling.
- **Fix:** T24, T25.

### D6 · No chicane logic · HIGH
- **Problem:** Nothing chooses or flies Gate B versus the window. The window (+5) needs 3D localization and height control through an opening.
- **Fix:** T34, T39.

### D7 · No landing-pad logic · HIGH
- **Problem:** Checkpoint 5 needs the whole drone inside the pad's outer ring, landed and disarmed.
- **Fix:** T35.

### D8 · No return-bonus logic · MEDIUM
- **Problem:** Nothing re-arms, takes off, flies over the barrier and lands within 2 ft of the launch mark.
- **Fix:** T40.

### D9 · No auxiliary sensors · MEDIUM
- **Problem:** Only forward stereo and the downward ToF exist. The sides, back and the inside of the tunnel are blind.
- **Fix:** T32.

### D10 · No safety filter · MEDIUM
- **Fix:** CBF on raw ranges (T33).

### D11 · No evaluation tooling · MEDIUM
- **Problem:** No drift measurement, no recording scripts, no metrics. We can't tune or show evidence in the presentation.
- **Fix:** T05.

## E. Moving to the real drone

### E1 · No OAK bring-up · BLOCKER
- **Problem:** The sim provides `/stereo/left/image`, `/stereo/right/image`, `/stereo/*/camera_info`, `/stereo/imu` and `/camera/depth/image_raw`. On hardware, the depthai-ros launch and topic remaps must be written.
- **Fix:** T12.

### E2 · No static TF or calibration for the real airframe · BLOCKER
- **Problem:** cuVSLAM and refnav both look up `base_link→oak_*` from TF, which the sim provides.
- **Fix:** Use the drone CAD and OAK calibration (T13).

### E3 · cuVSLAM availability on the Jetson is unknown · HIGH
- **Problem:** The wrapper imports the Python `cuvslam` bindings, an x86 build in the sim. The README says the aircraft uses an Isaac ROS Visual SLAM container, which is a different interface.
- **Fix:** T11, T14.

### E4 · ROS 2 Jazzy in the sim versus Humble on the Jetson · MEDIUM
- **Problem:** Sim is Jazzy on Python 3.12; the Jetson is Humble on Python 3.10. There are API and px4_msgs differences. The type-hinted code was compile-checked on 3.10.
- **Fix:** T11.

### E5 · Orin Nano compute budget · MEDIUM
- **Problem:** cuVSLAM, depth, localization and MPPI share one 8 GB Orin Nano. Sim performance (x86 desktop GPU) says nothing about it.
- **Fix:** T18.

### E6 · Clock and timesync on hardware · LOW
- **Problem:** Hardware uses system time and needs a working uXRCE-DDS timesync.
- **Fix:** T19.

### E7 · Which PX4 topics the hardware bridge exports is unknown · MEDIUM
- **Example:** Is `/fmu/out/distance_sensor` exported?
- **Fix:** T11, T16.

### E8 · No deployment packaging · MEDIUM
- **Problem:** No dependency list, launch file or start-up script for the Jetson.
- **Fix:** T43.

## F. Rules and scoring

### F1 · One start command, then abort only · HIGH
- **Problem:** Rules 5.2, 5.3 and 7.11 require exactly one start command; afterwards only land or kill is allowed. Nothing implements this.
- **Fix:** T25.

### F2 · Hard-coding costs generality points · HIGH
- **Problem:** Fixed waypoints, routes or timings get flagged by the rule 9.5 AI review. Generality is worth 15 points.
- **Fix:** Keep course-specific facts in one config file and compute paths (T22, T23, T47).

### F3 · No run data reuse · MEDIUM
- **Problem:** Rule 6.4 forbids feeding data from one run into a later one: cuVSLAM maps, recorded bags, captured point clouds.
- **Fix:** Tune from what we learn, never load it.

### F4 · Stay below the walls · MEDIUM
- **Problem:** Some part of the frame must stay under 68 in except on the return (rule 7.7). The planner needs a height band.

### F5 · The README must explain our stack and credit open source · MEDIUM
- **Problem:** Rules 9.1 and 9.4. The current README is the starter's.
- **Fix:** T43.

### F6 · Auxiliary sensor limits and declaration · LOW
- **Problem:** Each sensor must be under $100 and 50 g, counting any microcontroller that reads it. Each needs a part number, price link and mass (rules 4.1–4.5).
- **Fix:** T46.

## G. Simulator gotchas

### G1 · No RGB on ROS in the sim · MEDIUM
- **Problem:** Only mono left/right and depth are available. The RGB and follow views are browser previews. Line following or pad detection must work on mono in the sim.

### G2 · The sim runs at about 0.5× real time while moving · MEDIUM
- **Problem:** The README's moving RTF is 0.53.
- **Fix:** Use ROS time everywhere. Wall-clock timing in the sim is not flight timing.

### G3 · The sim course version may not match the published layout · MEDIUM
- **Problem:** The sim deployment mentions "CAD v3", and the published layout is preliminary.
- **Fix:** Confirm before trusting either (T01).

### G4 · Reset the world after keyboard RC · LOW
- **Problem:** Run automated programs only after Reset world (see CLAUDE.md). Terminal Python doesn't enforce this.

### G5 · `dronehacks_sim` truth is sim-only · LOW
- **Problem:** Ground-truth pose and voxels are fine for evaluation (T05) but must never be imported by flight code.

## H. Repo hygiene

### H1 · No `.gitignore` · LOW
- **Problem:** `__pycache__/*.pyc` and the `.onboard/` runtime logs and state are committed.
- **Fix:** T10.

### H2 · Docs reference files that aren't here · LOW
- **Problem:** `docs/NATIVE_DRONE_INTERFACE.md`, `docs/CURRENT_STATE.md` and `docs/IMU_FUSION_FIXES_2026-10-27.md` live in the organizers' toolkit.

### H3 · The effective configuration isn't in the repo · MEDIUM
- **Problem:** `/etc/flyx/onboard.yaml`, `mission.yaml` and `onboard.json` override the repo config (heading, clock, IMU rate).
- **Fix:** Snapshot them (T03).

### H4 · Tests need pytest, and there's no CI · LOW
- **Status:** The 30 unit tests pass locally with pytest; they only cover refnav.

### H5 · The baseline hash manifests no longer match · LOW (expected)
- **Problem:** `flight-source.json` and `onboard-source.json` record the starter's hashes, so any edit makes them differ. Running services report `source_matches: false` until restarted.

### H6 · Compressed code style · LOW
- **Problem:** `ev_bridge.py`, `validation.py` and `ownership.py` use semicolon-packed lines. Type hints are added; formatting was left alone.
