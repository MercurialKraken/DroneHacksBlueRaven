# vision/ — floor guide line (observer only, optional)

Finds the course's floor guide line in the left camera and publishes where it is, as JSON.
**Nothing reads it yet.** It has no PX4 or skybeta publisher, sends no setpoint, and arms nothing.
It is offered as an optional, independent cue (for example to check sideways drift in a corridor);
whether and how to use it is the planner/controller owners' call.

```
/stereo/left/image   (sim, mono8 640x400)      ─┐
/oak/left/image_raw  (drone, bgr8 1024x640)    ─┴─> vision.line_node ─> /vision/floor_line
                                                                       (std_msgs/String JSON)
```

## Output (`schema: vision.floor_line/1`)

- `detected`, `status` (`supported` / `ambiguous` / `unknown`), `confidence`, `rejection_reasons`
- `bottom_offset`: where the line meets the bottom image row, −1 left … 0 centre … +1 right
- `image_lean_deg`: lean of the line going up the image, + = right. **Image-space only**, not a
  ground heading (that needs CameraInfo, TF and height).
- `points_normalized`: sampled line points, image fractions
- `low_light`: dark-frame flag; a CLAHE retry is kept only when it clearly helps
- `clutter_pruned`: true when stray blobs touching the line (dots, speckled floor) were trimmed off
- `fresh`, `age_ms`; an unusable frame is published as `{"rejected": "<reason>"}`, so silence never
  means "line ahead"
- `control_output` is always `"DISABLED"`

Branched, cropped, competing and wall-edge stripes are **not** reported as a line.

## Run

```bash
python3 -m pytest -q tests/test_vision_floor_line.py           # offline, no ROS (needs OpenCV)

# sim (defaults)
python3 -m vision.line_node --ros-args -p use_sim_time:=true -p polarity:=dark

# drone: inside the skybeta container, only with the team's agreement, and NOT in flight until a
# load test shows it leaves the vision age at PX4 unchanged (camera subscribers can slow cuVSLAM)
python3 -m vision.line_node --ros-args -p image_topic:=/oak/left/image_raw \
    -p polarity:=dark -p every_nth:=3
timeout 30s ros2 topic echo /vision/floor_line
```

Parameters: `image_topic` (`/stereo/left/image`), `out_topic` (`/vision/floor_line`),
`polarity` (`dark`: black tape on a lighter floor), `roi_top` (0.35: only the image below this
fraction is searched), `max_age_ms` (200), `every_nth` (1).

## Measured / not yet measured

- 24 offline tests on synthetic frames: clean line, offset and lean, blank, competing, wrong
  polarity, black frame, low light, bad input, plus clutter cases.
- Clutter fix (synthetic benchmark): detection on dotted floors 30 % → 85 %, speckled (terrazzo)
  floors 13 % → 100 %, no false lines on line-free frames, median offset error 0.003.
- Time per frame, x86 laptop CPU: 640×400 about 6 ms, 1024×640 bgr8 about 15 ms. **Jetson time is
  not measured**; expect it to be slower, hence `every_nth:=3` (about 10 Hz) on the drone.
- **Not yet run against the simulator's camera, the drone's camera or the real course floor.**
  `polarity` and `roi_top` must be checked on real frames (`tools/snap_camera.py` saves one).
- **Needs OpenCV** (`python3 -c 'import cv2'`). Whether the drone container has it is not
  confirmed here; never pip/apt into the running container (use the team image if it is missing).

## Credits

`vision/floor_line.py` adapts `_line` / `_line_geometry` from the author's pre-event repository
[Cyberchopin/dronehacks-autonomy-2026](https://github.com/Cyberchopin/dronehacks-autonomy-2026)
(`src/aegisland/course_perception.py`, commit `01d9ded`), credited per rule 9.1. Changes made
here: the wall-edge check compares a stripe with its own row neighbours (the original compared with
the floor median and rejected every real line when walls were brighter than the floor), and a
clutter-trimming step for dotted/speckled floors.
