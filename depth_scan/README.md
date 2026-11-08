# depth_scan — on-camera depth → horizontal slice (LaserScan)

The product Chelsea specified for a map-free wall follower / centreline (bus, 18:3x): per bearing,
the **nearest** range over a height band of ±0.2 m around the camera, in `base_link`, stamped with
the depth frame's capture time, 0.2–3 m, ~1.5° bins, raw (holes stay holes).

**No map.** Input is only the depth image, its camera_info and the static TF base_link ← depth
frame. It does not read the course map, map_align or anything map-based (Arnav's directive), any
PX4 topic, or the camera images.

```
/oak/stereo/image_raw (16UC1 mm, 320x200, 30 Hz)  ─┐
/oak/stereo/camera_info                            ├─> depth_scan.node ─> /blueraven/depth/scan
TF base_link <- depth frame (static)               ─┘                     (sensor_msgs/LaserScan)
```

## Values per bin

| value | meaning |
|---|---|
| finite | nearest in-band return within [range_min, range_max], horizontal distance from base_link |
| `+inf` | returns in the band, all beyond range_max |
| `-inf` | returns in the band, all closer than range_min |
| `NaN` | **unknown**: no usable return (depth hole, net, blank texture, or fewer than `min_pixels`). Never treat as free |

Bearings: 0 = forward, left positive (ROS LaserScan). Min, not median, so a slanted wall reports
its nearest height. Hinson's depth is ~35 % valid pixels, so expect NaN bins; that is honest.

## Run

```bash
python3 -m pytest -q tests/test_depth_scan.py      # offline, no ROS
# on the drone, only with the team's agreement and after a props-off load check (EV age):
python3 -m depth_scan.node --ros-args -p depth_topic:=/oak/stereo/image_raw \
    -p info_topic:=/oak/stereo/camera_info
```

Parameters: `depth_topic`, `info_topic`, `out_topic` (`/blueraven/depth/scan`), `base_frame`
(`base_link`), `band_m` (0.20), `range_min` (0.20), `range_max` (3.0), `angle_increment_deg`
(1.5), `min_pixels` (2). The node refuses camera-image topics as input and `/fmu*` as output.

## Checked / not checked

- 16 tests on ray-cast depth images at 320×200 with the drone's camera offset: a wall ahead
  (ranges follow 1.5/cos b within the bin), a corridor (side walls at 0.6 m within 3 cm, far wall
  beyond range = +inf), a 30° slanted wall (reports its nearest height, 1.085 m vs 1.2 m at camera
  height), net holes stay NaN (never +inf), the floor stays out of the band when level, a 15°
  nose-down pitch puts the floor in the band unless `R_level` is given, 16UC1 = 32FC1, image
  padding and byte order, the standard optical-frame quaternion, about 1–2 ms per frame on a
  laptop CPU.
- **Not checked:** the real depth frame_id/TF on skybeta-09, Jetson timing, and CPU/latency impact
  in flight (run Arnav's EV-age check with it on). Attitude levelling is not wired into the node.
