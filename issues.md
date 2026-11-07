# Issues

## Where the code stands

- **Works (sim only):** stereo + IMU → cuVSLAM → EV bridge → PX4 EKF2 gives a valid position. The reference mission (`refnav/`) can take off, hover and land.
- **Missing:** everything needed to fly the course (map, planner, controller, mission logic), and everything needed to run on the real drone.

**[blocker]** = the drone can't fly the course until it's fixed.

## State estimation

- **The IMU only initializes once the drone moves.** Gravity stays "none yet" on the pad and appears after the first takeoff; tracking time then rises from about 1 to 5 ms. The first seconds of flight run without it. The noise values in [vio.yaml:11-14](onboard/config/vio.yaml#L11-L14) are 10× the sim's IMU model (gyro 2.5–3.4e-4, accel 1.9–2.5e-3). `imu_frequency: 500` is overridden to 480 by the sim deployment.
- **[blocker] The dark tunnel blinds the cameras.** The OAK-D S2 has no IR, so it gets no features and no depth.
  - With `imu_only_policy: publish`, drifting IMU-only poses go to PX4.
  - With `withhold`, PX4 marks position invalid after 1 s without vision (`EKF2_NOAID_TOUT=1`) and descends.
- **Drift is never corrected.** cuVSLAM runs in odometry-only mode ([cuvslam_lib_node.py:173-187](onboard/slam/cuvslam_lib_node.py#L173-L187)). Nothing matches against the course layout.
- **cuVSLAM is overconfident.** It reports σ ≈ 2–3 mm. PX4 floors that at 5 cm (`EKF2_EVP_NOISE`), but slow drift still passes straight into PX4's position.
- **Velocity goes to PX4 with variance 0.** It's a finite difference of poses at 15 Hz with no twist covariance ([cuvslam_lib_node.py:360-367](onboard/slam/cuvslam_lib_node.py#L360-L367)). PX4 doesn't fuse vision velocity today (`EKF2_EV_CTRL=9`: position and yaw only), so this only matters if that changes.
- **IMU-only frames look like good frames.** They're published as normal odometry ([cuvslam_lib_node.py:278-283](onboard/slam/cuvslam_lib_node.py#L278-L283)), and neither the bridge nor PX4 can tell.
- **Height comes from the rangefinder** (`EKF2_HGT_REF=2`). On the return, flying over the 68 in barrier will look like the floor rising 1.73 m.
- **The published camera calibration is deliberately off** (focal +0.5%, baseline +1%, principal point 2 px). cuVSLAM reads its intrinsics from it, so expect a small scale bias. Depth reads about 1.5% long.
- **Two frames are both called `odom`.** One is cuVSLAM's startup frame ([cuvslam_lib_node.py:332](onboard/slam/cuvslam_lib_node.py#L332)), the other PX4's estimate ([px4_odom.py:65-69](onboard/adapters/px4_odom.py#L65-L69)).
- **Constraint:** map or SLAM corrections must never go into PX4's vision input. The bridge's 1 m jump check marks health false and the mission lands ([ev_bridge.py:105-106](onboard/vio/ev_bridge.py#L105-L106)).
- **Constraint:** restarting the bridge means 25 s with no vision input ([ownership.py:22-43](onboard/vio/ownership.py#L22-L43)). Never restart it while armed.

## Perception

- **[blocker] No course map.** The published layout isn't loaded and there's no occupancy grid or distance field.
- **[blocker] No local obstacle map.** The only check is refnav's "anything within 0.9 m ahead".
- **Narrow corridors look blocked.** refnav counts anything within ±0.6 m of the flight line as ahead ([core.py:71](refnav/core.py#L71), [depth.py:141](refnav/depth.py#L141)), so corridors under about 1.2 m trip the stop.
- **No side sensing.** Stereo depth can't see closer than 0.40 m, so tunnel walls may be invisible. It's also coarse beyond about 2 m (0.56 m steps at 3 m), and about 64% of pixels are invalid on the pad.

## Planning

- **[blocker] No planner.** refnav flies hard-coded straight legs from `route.json`, which is empty. Hard-coded routes also cost generality points.
- **Any obstacle ends the run with a landing** ([core.py:420](refnav/core.py#L420), [core.py:503](refnav/core.py#L503)). There's no avoidance.
- **No logic for the height band, Gate B or window, or the landing pad.**

## Control

- **Slow.**
  - Position-only targets at 0.6 m/s, at most 1 m ahead of the drone.
  - Turns in place before each leg, then a 1 s hold at the end of it.
  - No velocity feedforward.

- **PX4 caps horizontal speed at 1.5 m/s and acceleration at 1 m/s²** in the sim (`MPC_XY_VEL_MAX`, `MPC_ACC_HOR`). Faster setpoints get clipped.

## Mission

- **Cramped start:** 0.31 m to the left wall and 0.50 m behind. Climb before moving sideways.

- **[blocker] Wrong start flow.** refnav arms itself as soon as it's ready, has no start command, and gives up if it sees the drone already armed ([core.py:311](refnav/core.py#L311)).
- **Lands after 1 s of bad estimator health anywhere** ([core.py:341](refnav/core.py#L341)). In the tunnel that ends the run.

## Hardware

- **[blocker] No OAK driver launch.** The sim provides `/stereo/*`, `/stereo/imu` and `/camera/depth/*`; on the drone we have to remap the driver's topics to those names.
- **[blocker] No static TF** from `base_link` to the `oak_*` frames for the real airframe.
- **cuVSLAM is unverified on the Jetson.** It's ARM, ROS Humble on Python 3.10; the sim is x86, Jazzy on 3.12.
- **Hard-coded `_v1` PX4 topic names** ([refnav/node.py:110-112](refnav/node.py#L110-L112)). They may differ on the drone's firmware.
- **Unsupported images are dropped silently.** cuVSLAM ignores anything that isn't `mono8` without a word ([cuvslam_lib_node.py:202-203](onboard/slam/cuvslam_lib_node.py#L202-L203)), and it assumes rectified images.
