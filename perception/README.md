# perception/ — stereo depth for the real drone

**Why.** The real drone publishes no depth image: the organizers' camera config disables it to save
USB bandwidth (`issues.md` → Hardware). Without depth, drift correction (`map_align`) and any live
obstacle check cannot run on hardware. `todos.md` → "Also needed": *compute it on the Jetson from
the two camera images (OpenCV SGBM, low resolution, a few Hz)*. This is that.

Observer only: no `/fmu/in/*`, no `/skybeta/setpoint/*`, no arming, no commands.

```
/stereo/left/image  + /stereo/left/camera_info  ─┐
/stereo/right/image + /stereo/right/camera_info ─┼─> stereo_depth_node ─> /blueraven/stereo_depth/image_raw   (16UC1 mm)
baseline: right P[3] or TF oak_left->oak_right ─┘                       /blueraven/stereo_depth/camera_info
                                                                         /blueraven/stereo_depth/status      (JSON, 1 Hz)
```

## Pieces

- `stereo_depth.py` — pure: `compute_depth(left, right, fx, fy, cx, cy, baseline_m, StereoConfig)`
  → `DepthResult` (float32 metres, NaN = no estimate, plus output-scale intrinsics, valid fraction,
  compute time, representable range). `depth_to_16uc1_mm()` matches the sim's depth encoding
  (millimetres, 0 = invalid).
- `stereo_depth_node.py` — thin ROS 2 wrapper: pairs left/right by stamp (≤ 5 ms), rate-capped
  (default 5 Hz), publishes depth in the left optical frame with the left stamp.

## Using it

Sim (read-only; the sim already has depth, so this publishes to its own topic for comparison):

```bash
python3 -m perception.stereo_depth_node --ros-args -p use_sim_time:=true
```

Real drone: remap the outputs onto the names `map_align` already reads, and point the inputs at the
drone's stereo topics (names UNKNOWN here — read them from `ros2 topic list` on the drone):

```bash
python3 -m perception.stereo_depth_node --ros-args \
  -p left_image:=<drone left> -p right_image:=<drone right> \
  -p left_info:=<drone left info> -p right_info:=<drone right info> \
  -p out_image:=/camera/depth/image_raw -p out_info:=/camera/depth/camera_info
```

The node refuses to start if something already publishes on `out_image` (two depth sources would
interleave), unless `allow_shared_output:=true`.

Parameters: `scale` (0.5), `num_disparities` (64, at output scale), `block_size` (5),
`min_depth_m` (0.3), `max_depth_m` (6.0), `max_rate_hz` (5), `max_pair_dt_ms` (5),
`depth_scale_correction` (1.0).

## What is verified (offline, synthetic)

`tests/test_stereo_depth.py`, 24 tests: textured planes of known disparity recover depth within
2 % at scale 0.5 and 1.0; a near/far split is separated; a textureless scene returns **no** depth
instead of a guess; the left border without a full search range is never trusted; out-of-range depth
is dropped; output intrinsics/range limits are exact; 16UC1 encoding matches the sim; bad inputs and
configs raise; a missing OpenCV is an explicit error.

Timing on a 2-core x86 laptop CPU (median per pair): 640×400 at scale 0.5 → 3 ms;
1024×640 at scale 0.5 → 14 ms. **Not measured on the Jetson Orin Nano yet.**

## Not verified yet — needs the team

1. **Is OpenCV in the drone's skybeta container?** (`python3 -c "import cv2; print(cv2.__version__)"`
   inside the container.) The team rule is numpy-only at runtime with no new pip packages. Without
   cv2 this module cannot run on the drone.
2. Drone stereo topic names, encodings, and whether the images are already rectified.
3. Accuracy on real sim frames vs the sim's own depth: `tools/depth_eval/` (in progress) will
   measure error by range, valid fraction and bias on recorded bags. The sim's published
   calibration is deliberately off (focal +0.5 %, baseline +1 %), so a ~1–1.5 % bias is expected.
4. Jetson compute time and the effect on cuVSLAM, which shares the CPU/GPU.
5. Dark tunnel: stereo cannot match without texture or light; expect no depth there (by design it
   reports none rather than a wrong value).
