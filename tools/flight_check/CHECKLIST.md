# Real-drone flight test checklist (proposal)

Battery time and arming attempts are scarce, so every flight should answer **one** question and
leave a recording that proves the answer. This is a proposal for the team: the safety pilot and the
flight-code owner decide what actually flies. Thresholds marked *proposed* are starting points, not
rules.

## Before every flight

- [ ] Safety pilot ready on RC, kill switch tested on the ground, net closed, nobody inside.
- [ ] **People out of the camera's view.** The bench test (2026-11-06) saw 10–30 m VIO jumps when
      people moved in view or the lens was covered.
- [ ] Battery charged and recorded on the log sheet (below).
- [ ] Code version written down: `git rev-parse --short HEAD` on the drone's `~/team_ws/src`.
- [ ] This flight's question and pass criterion written on the log sheet *before* arming.
- [ ] VIO alive and still: on the ground, odometry ~30 Hz and not moving
      (`timeout 10s ros2 topic hz /visual_slam/tracking/odometry`).
- [ ] Recording started (next section). No recording, no flight.

## Record (inside the skybeta container)

```bash
ros2 bag record -s mcap -o ~/team_ws/bags/flight_NN \
  /visual_slam/tracking/odometry /skybeta/setpoint/pose /skybeta/setpoint/twist /skybeta/state \
  /fmu/in/vehicle_visual_odometry /tf /tf_static
```

- **Do not record the raw camera images by default.** One 1024×640 bgr8 camera at 30 Hz is about
  59 MB/s. Record a short image clip separately only when a flight needs one.
- `/fmu/in/vehicle_visual_odometry` is what PX4 actually receives; the analyzer reads its age
  (`timestamp - timestamp_sample`). Recording it only subscribes.
- Every extra node and recording costs Jetson CPU. On 2026-11-07 the vision age at PX4 went from
  75 ms to 200–400 ms under extra load (depth node + compressed stereo + an 18 MB/s recording) and
  the stack later stopped. Keep optional nodes (depth, floor line) off until the load test says
  they fit, and check **Data age** in every report.
- Add any team topics that exist by then (for example map-alignment output) and any
  `/fmu/out/*` topics you want. Recording only subscribes; it publishes nothing.
- Whether the container's rosbag2 has the `mcap` storage plugin is **not confirmed**. If `-s mcap`
  fails, record with the default storage and convert later on a laptop.
- Rule 6.4: recordings are for offline analysis and tuning only, never an input during a run.

## After every flight (on a laptop)

```bash
python3 tools/flight_check/analyze_flight.py bags/flight_NN --out flight_NN
```

Read four things first: **VIO events** (any jump or freeze), **data age** (anything over
200 ms means PX4 stopped trusting vision), **dropouts** on odometry and setpoints, and
**tracking error** (pose) or **speed error** (twist). Write the numbers on the log
sheet. Change one thing at a time between flights.

## Suggested order

Each step is worth doing only once the previous one passed.

| # | Flight | Question it answers | Proposed pass |
|---|---|---|---|
| 0 | **No flight:** drone on the ground, props off, 2 min recording while people walk around outside the net | Does VIO jump from movement behind the netting? | No jump events |
| 1 | Hover on VIO, 30 s at about 0.7 m | Does it hold still on camera-only tracking? | No jumps; horizontal drift p95 < 0.15 m |
| 2 | 2 m out and back along a tape measure | How much does VIO over- or under-read distance? | Back within 0.10 m of start; out-leg within 5 % of tape |
| 3 | Hover, then slow twist descent (about 0.2 m/s) to the floor | Does PX4 detect touchdown and disarm itself (`COM_DISARM_LAND` = 2 s)? CP5 depends on it | Disarms by itself within about 3 s of touchdown |
| 4 | One straight leg at 0.3 m/s | Does twist mode fly the commanded speed? | Speed error median < 0.05 m/s; no jumps |
| 5 | One corner at 0.3 m/s | How much does the drone overshoot or cut a corner? | Tracking error p95 well inside the narrowest clearance (about 0.28 m spare) |
| 6 | Full course at 0.3 m/s | Can it finish? | All checkpoints; no jumps |
| 7+ | Raise **one** speed parameter one step | Is the faster setting still safe? | Same as 6, tracking error not worse |

## Abort and stop rules (proposed)

- The pilot takes over at **any** unexpected motion. No discussion during flight.
- A VIO jump or freeze in the report: no further flights until the cause is understood.
- Two failed attempts at the same step: stop and change one thing, don't just retry.
- Battery below the team's agreed minimum: land.

## Log sheet

| # | Time | Battery | Code commit | Question | Result (numbers from the report) | Next change |
|---|---|---|---|---|---|---|
| | | | | | | |
