# TODOs

The three core tasks are below. Task 1 comes first; tasks 2 and 3 both need it, and once it exists they can run in parallel. MPPI (Arnav) consumes all three.

```
                  ┌─► 2. Drift correction ─► corrected pose (map frame) ─┐
1. Course map ────┤                                                        ├─► MPPI ─► setpoints
   + 3D SDF       └─► 3. Global path ─────► reference path (map frame) ──┘
```

**Read first: the two frames everyone has to agree on.**

- **`map`** is the published course file's coordinates: metres, Z up, fixed forever. Walls, gates and the pad live here.
- **`odom`** is the position tracker's frame. Its origin is where the drone was at power-up, and it's rotated by whatever way the drone faced then: x = the drone's forward at power-up, y = left, z = up.
  - Sim: `/visual_slam/tracking/odometry` (our `onboard/` cuVSLAM, about 15 Hz).
  - Drone: the same topic name, from the organizers' cuVSLAM (about 30 Hz).
  - Same idea in both.
- **`T_map_odom`** converts between them: `p_map = T_map_odom · p_odom`.
  - At startup it's known: it's where the launch mark is in `map`, plus which way the drone faces.
  - Task 2's whole job is to keep it correct as `odom` drifts.
- Planning happens in `map`. Setpoints get converted back to `odom` (the inverse transform) right before they're sent to the drone.

**Shared rules for all three tasks.**

- **Plain Python, numpy only at runtime.** No new pip packages on the drone, since the container can't keep them. scipy, trimesh and similar are fine **offline** on a laptop, for building files we then commit.
- **Course-specific numbers live only in `maps/course.yaml`**, never in code. The judges' AI review flags hard-coded routes and waypoints (rule 9.5).
- **Unit tests for every pure function** go in `tests/`, run with `python3 -m pytest -q tests`.

---

## 1. Course map and 3D signed distance field (SDF)  ·  do this first

**Goal.** Give everything else one fast question to ask about the course: "for this point, how far is the nearest surface, and in which direction?"

### Steps

1. **Get the course files.**
   - Download the OBJ zip from the event page ("Download the course: OBJ + MTL (zip)"). It's in metres, Z up.
   - Put it in `maps/course/`, with a `README.md` recording the URL and download time.
   - The layout is still preliminary, so **re-download before the scored runs** and rebuild.
2. **Inspect the model.**
   - List the OBJ's object/group names (`o` / `g` lines): walls, tunnel, boulders, gates, window, pad, launch mark, net.
   - Find where the origin is and which way x and y point.
   - Check the sizes against the event page:

     | Item | Size |
     |---|---|
     | Net | 20 × 20 ft = **6.096 m**, 10 ft = 3.048 m tall |
     | Walls | **68 in = 1.727 m** tall, 4 in = 0.102 m thick |
     | Boulders | 18 in = 0.457 m across |
3. **Voxelize offline** in `tools/build_course_sdf.py` (trimesh is easiest, laptop only).
   - Use a grid that covers the net plus about 0.2 m of margin, at **5 cm** cells. That's about 130 × 130 × 64 ≈ 1.1 M cells.
   - Mark a cell occupied if it intersects a surface, and fill the solid interiors (walls, boulders).
4. **Build two signed distance fields** from the occupancy (the builder can use `scipy.ndimage.distance_transform_edt`):
   - Signed distance = (distance to the nearest occupied cell) − (distance to the nearest free cell), times the cell size. Positive in free space, negative inside objects.
   - **Planning field:** every object, plus the floor and the net boundary.
   - **Localization field:** only solid vertical surfaces (walls, tunnel sides, gate posts, boulders). Leave out the floor, the net (cameras see through it), the pad and the floor line. Task 2 uses this one.
   - Save both, with grid origin and cell size, to **`maps/course_sdf.npz`** (a few MB, committed).
5. **Runtime loader `nav/course_map.py`** (numpy only):
   ```python
   cm = CourseMap.load("maps/course_sdf.npz")
   cm.distance(points, field="planning")   # (N,3) map-frame metres -> (N,) signed metres, trilinear
   cm.gradient(points, field="planning")   # (N,3) -> (N,3) unit vectors pointing AWAY from the nearest surface
   ```
   Points outside the grid return `nan`, and each caller decides what that means: the planner treats it as blocked, localization ignores it.
6. **`maps/course.yaml`** holds every course-specific number, in `map` frame:
   - **Launch:** the launch mark's (x, y), the drone's yaw when placed on it, and base_link's height above the floor when sitting there. In the sim that height is about 0.07 m; measure it on the real drone.
   - **Checkpoint regions**, each as a box or line plus a direction of travel:

     | Checkpoint | Region |
     |---|---|
     | CP1 | end of the first corridor |
     | CP2 | Gate A |
     | CP3 | tunnel south exit |
     | CP4 | Gate B (and the window, as an option) |
     | CP5 | the pad: centre and outer-ring radius |
   - **Limits:** wall top 1.727 m, flight height band (for example 0.6–1.4 m), drone radius **0.28 m** (rotor arm plus prop).

### Check it

- **Unit tests:**
  - distance is about 0 on a wall face;
  - in a corridor's centre it's about half the corridor width;
  - the gradient points away from walls;
  - inside a wall it's negative.
- **A top-down picture** (`tools/plot_sdf.py`, matplotlib on a laptop): slices of the planning field at 0.3 m, 1.0 m and 1.5 m, saved as PNGs. Eyeball the walls, the tunnel and the window.
- **Against the sim (sim-only check, never used in flight code).**
  1. Read the sim's own 5 cm occupancy grid with `dronehacks_sim`.
  2. Find the transform between the course file and the sim world. The sim spawns the drone at ENU (-2.515, 1.114), facing yaw 90°, which should land on the launch mark. Possibly the transform is the identity.
  3. Compare the two at flight height. The sim has extra training props (`Board_*`, `Crate_*`) that won't be in the course file, which is expected. Anything else that differs means a layout mismatch, so report it.

**Done when:** `course_sdf.npz`, `course_map.py`, `course.yaml`, the PNG slices and passing tests are committed, and the sim comparison is written up in one paragraph.

---

## 2. Drift correction (`map→odom` from depth vs. the map)  ·  after task 1

**Goal.** The tracker's position drifts slowly. Keep `T_map_odom` correct by checking where the walls *appear* in live depth against where the map says they *are*.

**Inputs.**
- The localization field from task 1.
- `odom` poses from `/visual_slam/tracking/odometry`.
- Depth images: in the sim, `/camera/depth/image_raw` (16UC1, millimetres, 0 = invalid, frame `oak_left`, intrinsics in `/camera/depth/camera_info`, about 15 Hz). The real drone has **no depth yet**, which is a separate task, so build and test in the sim.
- TF `base_link → oak_left`.

### The algorithm, at 5–10 Hz

1. **Start** with `T_map_odom` taken from `course.yaml`'s launch pose (x, y, yaw, z = base_link height).
2. **Get points.**
   - Take the newest depth frame and sample every 8th–16th pixel.
   - Drop zeros, and keep 0.4–3.5 m. Beyond about 2 m sim depth gets coarse (0.5 m steps at 3 m), so you can weight far points down.
   - Sim depth reads about **1.5 % long** because the sim's published calibration is deliberately off, so multiply by 0.985.
3. **Move the points into `map`.** pixel → camera frame (K) → `base_link` (TF) → `odom`, using the **odometry pose at the depth frame's timestamp** (interpolate; don't take the latest pose) → `map` using the current `T_map_odom`.
4. **Filter.** Keep points with map height 0.15–1.65 m (above the floor, below the wall tops) and inside the course. Drop points far from any mapped surface (|distance| > 0.4 m): they're people, props, or something new. Keep about 300 points.
5. **Solve for a small correction δ = (dx, dy, dθ),** with z, roll and pitch left alone.
   - The residual for point *i* is the field's distance at the corrected point, which should be 0 if the point sits on a wall.
   - Use the drone's map position *c* as the rotation centre. With the gradient (gx, gy) at the point, the Jacobian row is `[gx, gy, gx·(−(y_i − c_y)) + gy·(x_i − c_x)]`.
   - Run 5 Gauss-Newton iterations with **Huber weights** (k ≈ 5 cm) so outliers can't drag the answer.
6. **Only correct what's observable.**
   - H = JᵀWJ is a 3×3 matrix. Take its eigenvectors, and apply δ only along those whose eigenvalue clears a threshold.
   - In a straight corridor, sideways is well determined and along-the-corridor isn't, so you correct sideways only.
7. **Sanity gates.** Reject the update if |δ| > 0.3 m or 10°, if the post-fit residual RMS is > 0.08 m, or if fewer than 50 points survive. Count every rejection.
8. **Smooth.** Apply a fraction of δ per update (α ≈ 0.2–0.3), so the planner never sees a jump.
9. **Publish**, all in `map`:
   - TF `map → odom`;
   - `/blueraven/pose_map` (PoseStamped): the corrected drone pose, for MPPI;
   - `/blueraven/localization` (String JSON): δ, the point count, residual RMS, which directions were observable, and the rejection count.

   **Never** feed any of this into PX4 or the vision bridge.

### Code layout

- `nav/localizer_core.py`: pure numpy (points + field + pose in, δ + diagnostics out), so it's unit-testable without ROS.
- `nav/localizer_node.py`: the thin ROS wrapper (subscriptions, TF, timestamps, publishing).

### Check it

- **Synthetic unit tests.**
  - Sample points on the field's walls from a known pose, offset the pose by (0.2 m, −0.1 m, 4°), and check it recovers to < 2 cm and < 0.5°.
  - In a straight-corridor case, check that the along-corridor component stays unchanged.
  - Feed 30 % random outlier points and check the result still holds.
- **In the sim against truth.** Extend `tools/sim_eval/record_eval.py` to log `/blueraven/pose_map`, then fly the corridor and compare uncorrected `odom` vs corrected `map` pose against `dronehacks_sim` truth with `tools/sim_eval/analyze.py`.
- **Recovery test.** Start with a deliberately wrong launch pose (+0.3 m, +5°). It should pull back to < 5 cm within a few seconds of seeing walls.

**Done when:** the unit tests pass, and in the sim the corrected pose's error vs truth is clearly below raw `odom`'s, with a plot to show it.

---

## 3. Global path through the checkpoints  ·  after task 1, in parallel with 2

**Goal.** A smooth reference path in `map`, from the launch mark through every checkpoint region to the pad, **computed from the map**. MPPI uses it for its "progress along the path" cost.

**Inputs.** The planning field (task 1) and `course.yaml` (launch pose, checkpoint regions in order, drone radius, height band).

### Steps

1. **Search grid.** 3D, **10 cm** cells, restricted to the height band (for example 0.4–1.5 m), which is about 61 × 61 × 11 ≈ 41k nodes, so A* runs in well under a second. A cell is allowed only if field distance ≥ **drone radius + margin** (0.28 + 0.10 m).
2. **Plan stage by stage:** launch → CP1 → CP2 (Gate A) → CP3 (tunnel exit) → CP4 (Gate B; window as an option) → CP5 (above the pad). Run **A*** from the end of one stage to any cell in the next region, then join the stages.
3. **Cost per step** = length, plus a penalty for hugging walls (for example λ / distance, or λ · max(0, d_safe − distance)²). This keeps the path in the middle of corridors, plus a small cost for leaving the nominal height. Gates and the window force the height where they need to.
4. **Smooth it.**
   - Shortcut: drop nodes while the straight line between their neighbours stays clear.
   - Resample to points 0.1 m apart, storing arc length *s* and heading (face along the path, so the camera sees ahead for tracking and depth).
   - Re-check the clearance of every point after smoothing.
5. **Last stage: the landing.** End above the pad's centre. The landing itself is a slow vertical descent done by the mission node, not part of this path.
6. **Output.**
   - `nav/global_planner.py`: pure numpy, `plan(course_map, course_cfg) -> Path`, where Path has `points (N,3)`, `s (N,)` and `yaw (N,)`, plus helpers `closest_s(p)` and `point_at(s)` for MPPI.
   - Save the result to `maps/global_path.npz`, and publish it as `/blueraven/global_path` (nav_msgs/Path, frame `map`) for viewing.
   - Plan once at startup; it's fast enough to re-plan if the map or config changes.

**Generality.** No hand-placed waypoints anywhere. The only inputs are the map and the checkpoint regions in `course.yaml`, so a different course means a different yaml, not different code.

### Check it

- **Unit test** on a toy map (two rooms joined by a door): the path goes through the door and keeps ≥ 0.38 m from every wall.
- **A plot** of the real course: the path over the 1.0 m field slice, with the checkpoint regions drawn on. Its minimum clearance should be ≥ 0.38 m, it should cross every checkpoint line in the right direction, and its total length and expected time at 1 m/s should be printed.

**Done when:** `global_planner.py` and its tests are committed, plus the plot, and MPPI can load `global_path.npz`.

---

## Also needed (owner TBD)

- **Mission node and sim glue.** Write our ROS 2 package for the drone's `~/team_ws/src` (start from the organizers' `competitor_example`): it waits for `offboard_active`, takes off, follows MPPI, then lands. Also make the cloud sim run the organizers' `skybeta_flight_interface` (with a sim-only arm helper), so **the same node flies in both**. Without this, nothing flies end to end.
- **Hardware tests** (battery, safety pilot, in the net):
  - a hover on VIO;
  - **landing:** position targets can't go below 0.3 m, so descend with a slow downward twist (about 0.2 m/s) and confirm PX4 detects the touchdown and disarms itself (`COM_DISARM_LAND = 2 s`). Checkpoint 5 depends on this.
- **Depth on the drone:** compute it on the Jetson from the two camera images (OpenCV SGBM, low resolution, a few Hz). Task 2 can't run on the real drone without it.
- **Measure drift:** a sim data flight (`tools/sim_eval/`) and a 2 m out-and-back tape-measure test on the drone.
- **Tunnel:** decide on side ToF sensors early, since parts take time to get.
- **MPPI:** Arnav. It consumes the planning field, `/blueraven/pose_map` and the global path, and outputs setpoints converted to `odom`.

Known problems and constraints are in [issues.md](issues.md).
