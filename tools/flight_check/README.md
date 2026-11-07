# tools/flight_check — real-drone flight tests (offline, read-only)

- [`CHECKLIST.md`](CHECKLIST.md): what to check, record and ask before and after each real flight,
  a suggested test order, and a log sheet.
- `analyze_flight.py`: one recording in, one report out.

```bash
python3 tools/flight_check/analyze_flight.py bags/flight_03 --out flight_03
```

It reads one `.mcap` file or a ros2 bag folder and writes `flight_03_report.txt`,
`flight_03_report.json` and `flight_03.png` (if matplotlib is installed). It reports:

| Section | What it means |
|---|---|
| Topics | count, rate and dropouts per topic. A dropout is a gap longer than `--gap-s` (0.25 s) *and* 3× the topic's own normal period |
| VIO events | `jump`: one step moved more than `--jump-m` (0.30 m) or faster than `--jump-speed-mps` (3 m/s, above the 1.5 m/s twist cap); `gap`: odometry stopped; `freeze`: the exact same position for at least `--freeze-s` (0.5 s), i.e. tracking stopped but output was held; `non_finite`: NaN/inf |
| Pose tracking | distance from each odometry sample to the pose setpoint in force then (horizontal and vertical; median, p95, max). **Assumes setpoint and odometry are in the same frame**; it warns when their frame ids differ |
| Twist tracking | commanded vs flown horizontal speed and vertical speed. Speed *magnitudes* do not depend on whether the twist is in a body or world frame, so no frame assumption is needed |
| State changes | `/skybeta/state` transitions, if it is a `std_msgs/String`; otherwise only its rate |

Topics default to the skybeta interface (`/visual_slam/tracking/odometry`, `/skybeta/setpoint/pose`,
`/skybeta/setpoint/twist`, `/skybeta/state`); each has a `--odom`/`--pose-sp`/`--twist-sp`/`--state`
option. For **sim** runs flown by refnav, use `tools/analyze_bag.py` instead.

## How it was checked

- `python3 -m pytest -q tests/test_flight_check.py`: 15 tests with hand-encoded CDR messages in a
  minimal MCAP file (decoders, dropouts, jumps, freezes, tracking maths, missing topics,
  read-only source check).
- Cross-checked once against an **independent** writer (the `mcap` + `mcap-ros2-support` packages,
  zstd-compressed chunks like rosbag2): frame ids, positions, a planted 12 m jump at t = 5 s and the
  state changes all came back exactly.

## Not yet known

- The real message types of `/skybeta/setpoint/*` and `/skybeta/state` are **assumed**
  (`PoseStamped`, `TwistStamped`, `String`). The report prints each topic's actual schema name;
  anything it cannot decode is counted but not analysed. Check the first real recording.
- Not yet run on a real recording.

Needs only numpy (matplotlib optional; zstandard/lz4 for compressed bags, laptop only). It uses
`tools/mcap_read.py`. No ROS, no node, no publisher.
