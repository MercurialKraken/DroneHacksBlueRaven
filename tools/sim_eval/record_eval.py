#!/usr/bin/env python3
"""SIM-ONLY evaluation recorder. Never import this from flight code.

Logs the estimator outputs, and optionally simulator ground truth, to CSV files
for analyze.py. Everything is stamped in sim time.

Output files:
  truth.csv     dronehacks_sim ground truth: world ENU position, body FLU -> ENU quaternion, velocity
  vio.csv       /visual_slam/tracking/odometry (cuVSLAM start frame, FLU) + pose variances
  px4.csv       /refnav/odom (PX4's own estimate, ENU; stamped at receipt)
  gravity.csv   /visual_slam/vis/gravity (appears once cuVSLAM's inertial init completes)
  features.csv  /visual_slam/vis/observations_count (tracked features per frame)
  health.csv    /estimator/healthy
  ev.csv        /fmu/in/vehicle_visual_odometry (latency = timestamp - timestamp_sample)

Usage, from the workspace in the companion terminal:
  python3 tools/sim_eval/record_eval.py --out runs/flight1              # live flight, with truth
  python3 tools/sim_eval/record_eval.py --out runs/x/replay --no-truth  # bag replay (see replay_vio.sh)
Stop with Ctrl+C.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from pathlib import Path
from typing import TYPE_CHECKING

import rclpy
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import Bool, Int32MultiArray

if TYPE_CHECKING:
    from typing import IO, Any

    from builtin_interfaces.msg import Time

# PX4 endpoints are BEST_EFFORT + TRANSIENT_LOCAL; a RELIABLE subscriber would receive nothing.
PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=50)
STEADY = ClockType.STEADY_TIME


def stamp_ns(stamp: Time) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class EvalRecorder(Node):
    def __init__(self, out: Path, truth: bool) -> None:
        super().__init__("sim_eval_recorder", parameter_overrides=[Parameter("use_sim_time", value=True)])
        out.mkdir(parents=True, exist_ok=True)
        self.out = out
        self.files: list[IO[str]] = []
        self.writers: dict[str, Any] = {}
        self.rows: dict[str, int] = {}
        pose = ["t_ns", "x", "y", "z", "qx", "qy", "qz", "qw"]
        self._table("vio", pose + ["var_x", "var_y", "var_z", "var_roll", "var_pitch", "var_yaw"])
        self._table("px4", pose)
        self._table("gravity", ["t_ns", "gx", "gy", "gz"])
        self._table("features", ["t_recv_ns", "n_obs"])
        self._table("health", ["t_recv_ns", "ok"])
        self._table("ev", ["t_recv_ns", "timestamp_us", "timestamp_sample_us", "pose_frame",
                           "x", "y", "z", "var_x", "var_y", "var_z"])

        self.create_subscription(Odometry, "/visual_slam/tracking/odometry", self._on_vio, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/refnav/odom", self._on_px4, qos_profile_sensor_data)
        self.create_subscription(Vector3Stamped, "/visual_slam/vis/gravity", self._on_gravity, 10)
        self.create_subscription(Int32MultiArray, "/visual_slam/vis/observations_count", self._on_features, 10)
        self.create_subscription(Bool, "/estimator/healthy", self._on_health, 10)
        from px4_msgs.msg import VehicleOdometry
        self.create_subscription(VehicleOdometry, "/fmu/in/vehicle_visual_odometry", self._on_ev, PX4_QOS)

        self._stack = contextlib.ExitStack()
        self.sim: Any = None
        if truth:
            from dronehacks_sim import Simulation, Unavailable, WorldChanged
            self._unavailable, self._world_changed = Unavailable, WorldChanged
            self.sim = self._stack.enter_context(Simulation())
            self._table("truth", ["t_ns", "x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "seq"])
            self.last_seq: int | None = None
            # Wall-clock polling, deduplicated by physics sequence (240 Hz sim > ~200 Hz wall).
            self.create_timer(0.004, self._poll_truth, clock=Clock(clock_type=STEADY))
        self.create_timer(1.0, self._flush, clock=Clock(clock_type=STEADY))
        self.create_timer(5.0, self._report, clock=Clock(clock_type=STEADY))
        self.get_logger().info(f"recording to {out}/ (truth={'on' if truth else 'off'}); Ctrl+C to stop")

    # ------------------------------------------------------------------ io
    def _table(self, name: str, header: list[str]) -> None:
        f = open(self.out / f"{name}.csv", "w", newline="")
        self.files.append(f)
        self.writers[name] = csv.writer(f)
        self.writers[name].writerow(header)
        self.rows[name] = 0

    def _write(self, name: str, row: list[Any]) -> None:
        self.writers[name].writerow(row)
        self.rows[name] += 1

    def _now(self) -> int:
        return self.get_clock().now().nanoseconds

    def _flush(self) -> None:
        for f in self.files:
            f.flush()

    def _report(self) -> None:
        self.get_logger().info("rows " + " ".join(f"{k}={v}" for k, v in self.rows.items()))

    def close(self) -> None:
        self._flush()
        for f in self.files:
            f.close()
        self._stack.close()

    # ------------------------------------------------------------- callbacks
    @staticmethod
    def _pose_row(m: Odometry) -> list[Any]:
        p, q = m.pose.pose.position, m.pose.pose.orientation
        return [stamp_ns(m.header.stamp), p.x, p.y, p.z, q.x, q.y, q.z, q.w]

    def _on_vio(self, m: Odometry) -> None:
        c = m.pose.covariance
        self._write("vio", self._pose_row(m) + [c[0], c[7], c[14], c[21], c[28], c[35]])

    def _on_px4(self, m: Odometry) -> None:
        self._write("px4", self._pose_row(m))

    def _on_gravity(self, m: Vector3Stamped) -> None:
        self._write("gravity", [stamp_ns(m.header.stamp), m.vector.x, m.vector.y, m.vector.z])

    def _on_features(self, m: Int32MultiArray) -> None:
        if len(m.data):
            self._write("features", [self._now(), int(m.data[0])])

    def _on_health(self, m: Bool) -> None:
        self._write("health", [self._now(), int(m.data)])

    def _on_ev(self, m: Any) -> None:
        self._write("ev", [self._now(), int(m.timestamp), int(m.timestamp_sample), int(m.pose_frame),
                           *(float(v) for v in m.position), *(float(v) for v in m.position_variance)])

    def _poll_truth(self) -> None:
        try:
            p = self.sim.pose()
        except self._unavailable:
            return
        except self._world_changed:
            self.get_logger().error("world was reset: stopping (one recording per world)")
            raise SystemExit(0)
        if p.sequence == self.last_seq:
            return
        self.last_seq = p.sequence
        self._write("truth", [p.sim_time_ns, *(float(v) for v in p.position_enu_m),
                              *(float(v) for v in p.orientation_xyzw),
                              *(float(v) for v in p.velocity_enu_m_s), p.sequence])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="output directory (created)")
    ap.add_argument("--no-truth", action="store_true", help="replay mode: do not read dronehacks_sim")
    args = ap.parse_args()
    rclpy.init()
    node = EvalRecorder(args.out, truth=not args.no_truth)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        pass
    finally:
        node._report()
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
