"""MPPI GROUND CHECK on the drone (Darion's gate): does running MPPI on live depth starve cuVSLAM?

Run INSIDE the stack container, drone on the ground, NOT armed (this script has NO publishers at all: it cannot
command anything). Best with the depth recorder running too (e.g. 'fly_route.sh go' waiting at GO in another window,
or any 'ros2 bag record' of /oak/stereo/image_raw), since that is the flight's worst case.

    docker exec -it skybeta-stack bash -ic 'python3 /root/team_ws/bench/mppi_ground_check.py [baseline_s] [mppi_s]'

Phase 1 (default 20 s): measure only. Phase 2 (default 60 s): the real nav.local_nav MPPI at 10 Hz on live depth
(/oak/stereo/image_raw, stride 4) and live odometry, a dummy 2 m route ahead (nothing is sent anywhere).
Measured in both: /visual_slam/tracking/odometry rate, EV age at PX4 (/fmu/in/vehicle_visual_odometry
timestamp - timestamp_sample), MPPI ms. PASS = odometry >= 29 Hz and EV age p50 <= 60 ms during phase 2.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

for p in ("/root/team_ws/src/blueraven", "/root/team_ws/src/blueraven/skybeta"):
    sys.path.insert(0, p)
from nav.depth_points import R_FLU_OPTICAL, depth_to_odom          # noqa: E402
from nav.local_nav import LocalNav                                   # noqa: E402
from nav.pose_buffer import PoseBuffer                               # noqa: E402

BEST = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)


class Check(Node):
    def __init__(self, base_s: float, mppi_s: float) -> None:
        super().__init__("mppi_ground_check")                        # no publishers are created anywhere
        from nav_msgs.msg import Odometry
        from px4_msgs.msg import VehicleOdometry
        from sensor_msgs.msg import CameraInfo, Image
        self.base_s, self.mppi_s = base_s, mppi_s
        self.t0 = time.monotonic()
        self.odom_t: list[tuple[float, float]] = []                  # (monotonic, phase)
        self.ev_age: list[tuple[float, float]] = []
        self.mppi_ms: list[float] = []
        self.pose, self.vel, self.stamp = None, None, None
        self.K = None
        self.buf = PoseBuffer(horizon_s=2.0, max_gap_s=0.2)
        self.nav = None
        T = np.eye(4)
        T[:3, :3] = R_FLU_OPTICAL
        T[:3, 3] = (0.105, 0.038, 0.031)                             # base_link <- oak_left_camera_optical_frame
        self.T_base_cam = T
        self.create_subscription(Odometry, "/visual_slam/tracking/odometry", self.on_odom, BEST)
        self.create_subscription(VehicleOdometry, "/fmu/in/vehicle_visual_odometry", self.on_ev, BEST)
        self.create_subscription(CameraInfo, "/oak/stereo/camera_info", self.on_info, BEST)
        self.sub_img = None
        self.Image = Image
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f"phase 1: {base_s:.0f} s measuring only; then phase 2: {mppi_s:.0f} s MPPI on live depth")

    def phase(self) -> int:
        e = time.monotonic() - self.t0
        return 1 if e < self.base_s else (2 if e < self.base_s + self.mppi_s else 3)

    def on_odom(self, m) -> None:
        self.odom_t.append((time.monotonic(), self.phase()))
        p, q, v = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear
        st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.buf.add_position(st, [p.x, p.y, p.z])
        self.buf.add_attitude(st, [q.w, q.x, q.y, q.z])
        from nav.pose_buffer import quat_wxyz_to_mat
        vo = quat_wxyz_to_mat(q.w, q.x, q.y, q.z) @ np.array([v.x, v.y, v.z])   # base_link -> odom
        self.pose, self.vel, self.stamp, self.yaw = np.array([p.x, p.y, p.z]), vo, st, \
            math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def on_ev(self, m) -> None:
        if m.timestamp and m.timestamp_sample:
            self.ev_age.append(((m.timestamp - m.timestamp_sample) / 1e3, self.phase()))

    def on_info(self, m) -> None:
        self.K = np.array(m.k, dtype=float).reshape(3, 3)

    def on_depth(self, m) -> None:
        if self.nav is None or self.K is None:
            return
        D = np.frombuffer(m.data, dtype=np.uint16).reshape(m.height, m.width)
        st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        out = depth_to_odom(D, self.K, self.T_base_cam, self.buf, st, stride=4 if m.width >= 600 else 2)
        if out is not None:
            self.nav.add_depth(out[0], out[1], float(self.pose[2]), st)

    def tick(self) -> None:
        ph = self.phase()
        if ph == 2 and self.nav is None and self.pose is not None:
            fwd = np.array([math.cos(self.yaw), math.sin(self.yaw)])
            self.nav = LocalNav([self.pose[:2] + 2.0 * fwd], self.pose[:2], self.yaw)
            self.sub_img = self.create_subscription(self.Image, "/oak/stereo/image_raw", self.on_depth, BEST)
            self.get_logger().info("phase 2: MPPI looping on live depth (nothing is published)")
        if ph == 2 and self.nav is not None:
            t = time.monotonic()
            self.nav.step(self.pose[:2], self.vel[:2], self.get_clock().now().nanoseconds * 1e-9,
                          pose_stamp=self.stamp)
            self.mppi_ms.append((time.monotonic() - t) * 1e3)
        if ph == 3:
            self.report()
            raise SystemExit(0)

    def report(self) -> None:
        def hz(phase: int, dur: float) -> float:
            return sum(1 for _, p in self.odom_t if p == phase) / max(dur, 1e-6)

        def ev(phase: int) -> tuple[float, float]:
            a = np.array([x for x, p in self.ev_age if p == phase])
            return (float(np.percentile(a, 50)), float(np.percentile(a, 90))) if len(a) else (float("nan"),) * 2
        h1, h2 = hz(1, self.base_s), hz(2, self.mppi_s)
        e1, e2 = ev(1), ev(2)
        ms = np.array(self.mppi_ms) if self.mppi_ms else np.array([float("nan")])
        ok = h2 >= 29.0 and e2[0] <= 60.0
        print(f"BASELINE   odometry {h1:5.1f} Hz   EV age p50 {e1[0]:5.1f} ms p90 {e1[1]:5.1f} ms")
        print(f"WITH MPPI  odometry {h2:5.1f} Hz   EV age p50 {e2[0]:5.1f} ms p90 {e2[1]:5.1f} ms   "
              f"MPPI ms p50 {np.nanpercentile(ms, 50):.1f} p90 {np.nanpercentile(ms, 90):.1f} max {np.nanmax(ms):.1f} "
              f"(n={len(self.mppi_ms)}, depth frames {self.nav.depth_frames if self.nav else 0})")
        print("GATE (odometry >= 29 Hz, EV p50 <= 60 ms):", "PASS" if ok else "FAIL")


def main() -> None:
    base_s = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    mppi_s = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    rclpy.init()
    node = Check(base_s, mppi_s)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
