# Course map

Task 1 in [todos.md](../todos.md): one fast question for everything downstream: *how far is the nearest surface from this point, and in which direction?*

```python
from nav.course_map import CourseMap                  # numpy only, safe on the drone
cm = CourseMap.load("maps/course_sdf.npz")
d = cm.distance(points, field="planning")             # (N,3) map-frame metres -> (N,) signed metres
g = cm.gradient(points, field="planning")             # (N,3) unit vectors pointing AWAY from the nearest surface
d, g = cm.query(points, field="localization")         # both, one lookup
```

Distances are positive in free space, negative inside objects and about 0 on a surface. Points outside the grid return `nan`. The planner should treat `nan` as blocked; localization should ignore those points.

## Files

| File | What it is |
|---|---|
| `course/` | The organizers' OBJ, MTL and README, unchanged. Provenance and hashes are in [course/SOURCE.md](course/SOURCE.md). |
| `course.yaml` | **The only course-specific file:** launch pose, checkpoints, limits, which mesh pieces go into which field, and sanity points. |
| `course_sdf.npz` | The built fields: `planning` and `localization`, float32, 131 × 131 × 68 voxels of 5 cm, plus `origin` (centre of voxel [0,0,0]), `cell` and `meta_json` (mesh hash, build settings, pieces per field). |
| `plots/sdf_z*.png` | Top-down slices for checking by eye. |

## Frame `map`

The course file's own frame: metres, x = east, y = north, z = up, with the origin at the centre of the floor. The sim's world frame is also ENU. The x/y offset between the two is still unmeasured (see open issues below).

## The two fields

| Field | Contains | Used by |
|---|---|---|
| `planning` | Every solid piece: walls, barrier, maze dividers, tunnel and cradles, boulders, the floor and the net (walls and roof) | Global path (Task 3), MPPI |
| `localization` | Only solid surfaces a depth camera sees reliably: walls, barrier, dividers, tunnel, cradles, boulders. **No floor or net:** the camera sees through the net, and the floor is excluded by design | Drift correction (Task 2) |

Floor markings (pad, rings, cross, launch mark, guideline, approach ring) are at most 3 mm thick and are in neither field. Which mesh pieces go where is set by glob patterns in `course.yaml`. An include pattern that matches nothing stops the build, so a piece renamed in a re-download can't silently disappear.

## Rebuilding

Offline, on a laptop. The builder needs numpy, scipy and pyyaml; the plots need matplotlib.

```sh
python tools/build_course_sdf.py maps/course.yaml     # about 40 s; runs the sanity checks
python tools/plot_sdf.py maps/course.yaml             # maps/plots/sdf_z*.png
python -m pytest -q tests
```

The build runs every `sanity` point in `course.yaml` through the runtime loader. If any check fails, the result goes to `course_sdf.failed.npz` and the good file is left alone.

**How it's built.** Each piece is voxelized at a third of the output cell (1.67 cm): its triangles are sampled densely and the enclosed interior is filled. An open tube or a window stays open because it connects to the outside. Two distance transforms then give the signed distance. Zero sits at the centre of the surface voxels, which is within 0.8 cm of the true surface. The result is downsampled exactly to 5 cm.

**Accuracy.** Expect about ±1 cm near a surface. Midway between two surfaces (a corridor's centre line, the middle of the window) the field reads **2–3 cm low**, because it's interpolated linearly between 5 cm voxels across the ridge. That error is conservative: clearance looks smaller than it really is.

## Numbers that matter downstream

All at 1.0 m height unless noted. These are planning-field values from the build's sanity output.

| Place | Clearance (m) | Drone needs (0.28 radius + 0.10 margin) |
|---|---|---|
| Launch | 0.41 (barrier behind) | 0.38 |
| First corridor (1.07 m wide) | 0.51 | 0.38 |
| Maze corridor M1–M2 (1.02 m) | 0.48 | 0.38 |
| Gate A | 0.45 | 0.38 |
| Tunnel centre (1.46 m inner width) | 0.71 | 0.38 |
| Gate B (0.95 m, from C2b's north end to the barrier) | 0.44 | 0.38 |
| **Window centre** (0.914 m square, z 0.305–1.219) | **0.42** | 0.38 |

**Window: hand-off note for Task 3.** The window leaves about **4 cm of slack** in this field (about 7.7 cm in reality). Vertically, the drone's centre must be within z 0.685–0.839 m. A 10 cm A* grid can easily miss that gap, so plan the window approach on the exact centre line, or on a finer grid locally. Get Gate B (CP4) working first; the window is worth +5 only on a finished run.

## Live obstacles: `nav/live_map.py`

The static map can't know about anything that isn't in the published layout: a wall moved before judging, a prop, a person inside the net. `LiveObstacleLayer` adds those from depth, and `CombinedMap` puts both behind the same calls as `CourseMap`:

```python
from nav.live_map import LiveObstacleLayer, CombinedMap
live = LiveObstacleLayer(cm)                      # same 5 cm grid as the course map
st = live.integrate(depth_m, K, T_map_cam)        # each depth frame; T maps the OPTICAL frame to map
comb = CombinedMap(cm, live)                      # planner drop-in: planning = min(static, live)
d, g = comb.query(pts)                            # one lookup; 'localization' passes through unchanged
ok = comb.clearance_ok(setpoints, need=0.38)      # off the map is never ok
seen = live.observed(pts)                         # has any ray reached this space?
```

- **Only unexplained returns are recorded.** A depth point within 12 cm of a mapped surface (walls, net, floor) adds nothing. That keeps the layer small, and it doubles as a check that the map and the pose agree.
- **Phantoms are filtered:** a voxel needs hits in 2 separate frames. Rays that pass through it count against it, so a one-frame phantom never counts and a person who walks away is cleared.
- **A 5 px min filter on depth** joins scattered net-cord returns into a surface. Pixels with no return anywhere nearby stay unknown.
- **No return is unknown, never free.** Distance doesn't treat unseen space as blocked, because the static map is the prior there. A planner that must not fly into unseen space also checks `observed()`.
- **Misalignment fails safe to the static map.** Over 4000 obstacle voxels means the map or the pose is probably off. `live.trusted` goes false, the live layer is withheld, and a supervisor should treat that as degraded perception.
- **Cost on this laptop,** for a 640 × 400 frame: about 35 ms per frame to integrate, and about 30 ms per 41k-point query. numpy only. `FrameStats` reports explained, unexplained and off-map returns for every frame.

## Open issues

- **Launch vs sim spawn.** The launch mark's centre is (−2.515, 0.914), but `todos.md` gives the sim spawn as (−2.515, 1.114), 0.2 m further north. It could be how the sim places `base_link`, or a layout difference between the sim's CAD and v3. `tools/sim_eval/compare_course_voxels.py` measures both the spawn offset and the best-fit x/y shift between this map and the sim's own voxels. Run it in the companion once the VM is up, then write the result here.
- **Checkpoint lines match the event page's "Checkpoints from above" diagram** (checked 2026-11-07; map metres = feet × 0.3048 − 3.048). CP1 sits at the floor line's turn (y 2.585). Gate B is the line from C2b's north end straight up to the barrier (x 0.157), crossed heading west.
- **CP1 can't be fully crossed as drawn.** Whole drone across needs the centre at y ≥ 2.865, but the net at 3.048 leaves only 0.18 m, less than the 0.28 m radius. The drone's centre can safely reach about y 2.67. Ask the organizers how CP1 is judged.
- **"Barrier 3 (cut to 55 in)"** is still 1.727 m tall in the mesh. It only matters for the return over the barrier, which `issues.md` says is probably impossible.
- **Launch yaw (90°, facing north) and `base_link` height (0.07 m)** are sim values. Confirm them at the net.

## Sim comparison

*Not run yet: the VM is off.* In the companion terminal:

```sh
python3 tools/sim_eval/compare_course_voxels.py --out runs/course_compare
```

Write a one-paragraph summary of `runs/course_compare/report.txt` here, covering the spawn offset, IoU per height, and the best shift. The sim's training props (`Board_*`, `Crate_*`) aren't in the course file, so those differences are expected.
