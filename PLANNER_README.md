# course_planner — map-based route planning (Blue Raven)

Plans the DroneHacks route from a map file instead of hard-coded waypoints.
Pure Python + numpy (scipy optional). No ROS, no simulator needed to run it.

## Run it

```bash
python3 -m course_planner --route both --out plan.json --plot plan.png
python3 -m pytest tests/test_planner.py -q
```

`--route gate_b` or `--route window` picks the checkpoint 4 option. `--radius 0.27`
overrides the drone size. `--no-return` drops the return-bonus leg.

## How it works

1. `course_map.json` holds the course: walls, the window opening, the tunnel, boulders,
   the pad, the launch mark and the ordered checkpoint list (course frame: x East, y North, metres).
2. The map is rasterised at 2.5 cm and turned into a clearance field (distance to the nearest wall).
3. A* runs between successive checkpoint goals, only through cells at least
   `drone_radius + margin` (0.29 + 0.05 m) from any wall, with a cost that pulls the path to
   corridor centres.
4. Every line checkpoint gets an approach point and a point far enough past the line for the
   whole drone to be across. The tunnel is flown straight down its axis at its mid-height, with a
   1.5 s hover at the mouth. The window is approached square-on at its centre height.
5. Heights, speeds, hovers and headings are added (camera faces the direction of travel).
6. Output: course-frame waypoints plus PX4 local NED setpoints relative to the launch mark.

Change the map and the route follows. `test_plans_on_a_changed_course` checks exactly that.

## Before flying it: things to confirm

- **NED frame origin and heading.** `to_px4_ned` assumes PX4's local origin is the launch mark
  and +x points north (drone starts facing up the first corridor). Check `onboard/vio/frames.py`
  and `ev_bridge.py`: if the estimator uses the drone's initial heading instead, pass that heading.
- **Checkpoint 1 is tighter than the clearance allows.** The CP1 line is only 0.44 m from the north
  net. With 0.34 m clearance the drone centre gets 0.08 m past the line, but 0.26 m is needed for
  the whole drone to cross. Ask organizers how CP1 is judged; otherwise this spot needs a slow,
  deliberate creep with less margin.
- **Prop size** is confirmed at 7 in (radius 0.29 m, already the default in `Params`).
- **Checkpoint 4** follows the rules text (Gate B line or window), not the model's
  "Approach ring (stage 4 line)".

## Next step

Wire `px4_ned_from_launch` into the reference mission (`refnav/`) in place of `route.json`,
then fly it in the sim with the managed **Run saved file** so it gets a grade.

## Route files for the starter (refnav)

    python3 -m course_planner.make_routes      # -> routes/route_v1.json, route_fast.json, route_v2.json, route_v3.json

refnav flies the route's points as one smooth path (refnav/follow.py). Recording analysis: `python3 tools/analyze_bag.py <bag.mcap> --out <name>`.
