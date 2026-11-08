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

- **Never record, view or even `ros2 topic hz` the camera images on the drone**
  (`/oak/*/image_raw/compressed` and friends). Arnav's load test (2026-11-07): one subscriber to
  the compressed stereo makes the camera driver JPEG-encode 2 x 30 Hz inside the stack, and the
  vision age at PX4 climbs from about 60 ms to 200–340 ms within about 30 s, past PX4's 200 ms
  limit. The depth node alone is fine (about +7 ms). Logs: `~/team_ws/bench/load_test_144342`.
- `/fmu/in/vehicle_visual_odometry` is what PX4 actually receives; the analyzer reads its age
  (`timestamp - timestamp_sample`). Recording it only subscribes.
- Check **Data age** in every report. Optional nodes that read camera images (for example the
  floor-line observer, which subscribes to the raw left image) stay off in flight until a load
  test with them shows the vision age unchanged.
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
| 3 | Hover, then twist descent to the floor at **at least 0.3 m/s** (the hover mission uses 0.35, then a firmer press) | Does PX4 detect touchdown and disarm itself (`COM_DISARM_LAND` = 2 s)? CP5 depends on it | Disarms by itself within about 3 s of touchdown |
| 4 | One straight leg at 0.3 m/s | Does twist mode fly the commanded speed? | Speed error median < 0.05 m/s; no jumps |
| 5 | One corner at 0.3 m/s | How much does the drone overshoot or cut a corner? | Tracking error p95 well inside the narrowest clearance (about 0.28 m spare) |
| 6 | Full course at 0.3 m/s | Can it finish? | All checkpoints; no jumps |
| 7+ | Raise **one** speed parameter one step | Is the faster setting still safe? | Same as 6, tracking error not worse |

Landing note (Darion's end-to-end sim hover, 2026-11-07): PX4's land detector only declares
"landed" when the commanded descent is at least 1.1 x `LNDMC_Z_VEL_MAX` (0.275 m/s with the
v1.16 defaults). At 0.2 m/s the drone sat armed on the floor for 63 s. Check `LNDMC_Z_VEL_MAX`,
`MPC_LAND_CRWL` and `MPC_LAND_SPEED` on the drone before step 3.

**Before leaving the drone:** after a stack restart the competitor window runs the organizers'
example (arm + Position = a 5 m square at 1.5 m). Power down, or set `COMPETITOR_PKG` to the
team package first.

## Abort and stop rules (proposed)

- The pilot takes over at **any** unexpected motion. No discussion during flight.
- A VIO jump or freeze in the report: no further flights until the cause is understood.
- Two failed attempts at the same step: stop and change one thing, don't just retry.
- Battery below the team's agreed minimum: land.

## Log sheet

| # | Time | Battery | Code commit | Question | Result (numbers from the report) | Next change |
|---|---|---|---|---|---|---|
| | | | | | | |
