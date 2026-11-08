# Flight analysis (real drone and T4 bags)

Offline tools used on 2026-11-07/08 to understand route 1's real flight (160449) and the T4 runs. They read a bag's poses
only (depth is skipped), so a 2 GB bag is fine.

1. **Extract** (inside a ROS 2 container that has px4_msgs, e.g. the T4's `dronehacks_gpu`):
   `python3 extract_flight.py <bag dir> > flight.csv`. It writes one line per message for PX4's fused pose
   (`vehicle_local_position[_v1]`), cuVSLAM odometry, refnav's twist and status, PX4's setpoints
   (`trajectory_setpoint`, `vehicle_local_position_setpoint`, `vehicle_attitude_setpoint`), its attitude, and
   (T4) the true pose.
2. **Report**: `python flight_report.py flight.csv [--real]` gives the climb drift (and how it grows with height),
   the hold (does it get back over the take-off point?), the offset per leg, the CP1 crossing (`--real`: on the tape
   map), PX4 setpoint vs achieved velocity in the body frame, the best-fit response delay and the push at it, and
   tilt vs PX4's tilt setpoint.
3. **Push or rotation?** `python push_or_rotation.py name=flight.csv:<lag>:<phase>` fits the achieved velocity to a
   constant body push, a rotated/scaled response and a general linear map.
4. **Plots**: `plot_real_flight.py` (flown vs planned path on the tape map + the fused pose over time). Its take-off
   pose constants are 160449's (odom (0.007, 0.007), yaw -0.4 deg): read them from the launch log for another flight.
5. `compare_tracking.py`: PX4 setpoint vs achieved, real vs T4.

Findings so far (details in presentation_notes.txt):
- 160449 (empty room): a steady body-frame push of ~0.1 m/s left + ~0.06 m/s back that PX4 never removed; refnav's
  P-only loop (0.5/s) held it at 0.22 m off the path. A constant push fits better than a rotated response; the
  best-fit response delay is ~1.0 s (refnav assumes 0.55 s).
- The T4's response delay is ~2.1 s, so position loops near it orbit; T4 hold results do not predict the real drone.
