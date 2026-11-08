"""MPPI LINK CHECK on the drone: mppi_node (its own process) -> /blueraven/mppi/cmd -> refnav, on live depth.

Run INSIDE the stack container, drone on the ground, NOT armed. This script publishes ONLY /blueraven/mppi/start (the
message refnav sends at follow start); it has no /skybeta or /fmu publisher and cannot command anything. Start
mppi_node first (it publishes only /blueraven/mppi/*):

    nice -n 5 ros2 run blueraven_skybeta mppi_node --ros-args -p route_file:=<route.json> -p mppi_v_cruise_mps:=0.10 &
    python3 tools/mppi_link_check.py [pre_s] [follow_s]

Phase 1 (default 15 s): mppi_node's pre-flight loop (commands tagged start None). Phase 2 (default 45 s): a follow
start from the current pose, as refnav sends it; commands must carry its id. Measured: command rate, gaps (refnav
holds after mppi_follow.CMD_STALE_S, lands after CMD_LAND_S), age of each command at receipt, MPPI ms, states, and
odometry rate/age. PASS = phase-2 commands >= 9 Hz, every gap < CMD_STALE_S, age p90 <= 50 ms, all for this start,
odometry >= 29 Hz, and depth placed (a state other than WAIT/HOLD).
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

for p in ("/root/team_ws/src/blueraven", "/root/team_ws/src/blueraven/skybeta/blueraven_skybeta"):
    sys.path.insert(0, p)
from blueraven_skybeta.mppi_follow import CMD_STALE_S, CmdLink, yaw_enu  # noqa: E402

BEST = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
LATCHED = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


class Check(Node):
    def __init__(self, pre_s: float, follow_s: float) -> None:
        super().__init__("mppi_link_check")
        self.pre_s, self.follow_s, self.t0 = pre_s, follow_s, time.monotonic()
        self.pub_start = self.create_publisher(String, "/blueraven/mppi/start", LATCHED)   # the only publisher
        self.create_subscription(Odometry, "/visual_slam/tracking/odometry", self.on_odom, BEST)
        self.create_subscription(String, "/blueraven/mppi/cmd", self.on_cmd, 10)
        self.pose = None
        self.link = CmdLink()
        self.odom: list[tuple[int, float]] = []            # (phase, now - stamp ms)
        self.cmds: list[tuple[int, float, float, object, str, object]] = []   # (phase, recv, age ms, start, state, ms)
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f"phase 1: {pre_s:.0f} s pre-flight loop; phase 2: {follow_s:.0f} s after a follow start")

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def phase(self) -> int:
        e = time.monotonic() - self.t0
        return 1 if e < self.pre_s else (2 if e < self.pre_s + self.follow_s else 3)

    def on_odom(self, m: Odometry) -> None:
        p, q, st = m.pose.pose.position, m.pose.pose.orientation, m.header.stamp
        self.pose = (p.x, p.y, p.z, yaw_enu((q.x, q.y, q.z, q.w)))
        self.odom.append((self.phase(), (self.now() - st.sec - st.nanosec * 1e-9) * 1e3))

    def on_cmd(self, m: String) -> None:
        d = json.loads(m.data)
        now = self.now()
        self.link.on_cmd(d)
        self.cmds.append((self.phase(), now, (now - d["t"]) * 1e3, d["start"], d["state"], d.get("ms")))

    def tick(self) -> None:
        ph = self.phase()
        if ph == 2 and self.link.start_id is None and self.pose is not None:
            sid = self.link.begin(self.now())
            x, y, z, yaw = self.pose
            self.pub_start.publish(String(data=json.dumps({"id": sid, "pad": [x, y, z, yaw]})))
            self.get_logger().info(f"follow start {sid:.3f} from ({x:.2f}, {y:.2f}) yaw {math.degrees(yaw):.0f} deg")
        if ph == 3:
            self.report()
            raise SystemExit(0)

    def report(self) -> None:
        def pct(a, q):
            return float(np.percentile(a, q)) if len(a) else float("nan")
        ok = True
        for ph, dur in ((1, self.pre_s), (2, self.follow_s)):
            rows = [c for c in self.cmds if c[0] == ph]
            t = np.array([c[1] for c in rows])
            age = np.array([c[2] for c in rows])
            gaps = np.diff(t) if len(t) > 1 else np.array([math.inf])
            od = np.array([a for p, a in self.odom if p == ph])
            ms = np.array([c[5] for c in rows if c[5] is not None], float)
            states = sorted({c[4] for c in rows})
            ours = sum(1 for c in rows if c[3] == self.link.start_id) if ph == 2 else sum(1 for c in rows if c[3] is None)
            print(f"PHASE {ph} ({'pre-flight loop' if ph == 1 else 'follow'}): cmds {len(rows) / dur:5.1f} Hz, "
                  f"max gap {1e3 * gaps.max():.0f} ms, age at receipt p50 {pct(age, 50):.1f} p90 {pct(age, 90):.1f} "
                  f"max {age.max() if len(age) else float('nan'):.1f} ms, tagged right {ours}/{len(rows)}, "
                  f"MPPI ms p50 {pct(ms, 50):.1f} p90 {pct(ms, 90):.1f}, states {states}; "
                  f"odometry {len(od) / dur:5.1f} Hz, age p50 {pct(od, 50):.1f} p90 {pct(od, 90):.1f} ms")
            if ph == 2:
                ok = (len(rows) / dur >= 9.0 and gaps.max() < CMD_STALE_S and pct(age, 90) <= 50.0
                      and ours == len(rows) > 0 and len(od) / dur >= 29.0 and bool(set(states) - {"WAIT", "HOLD"}))
        print("GATE (cmds >= 9 Hz, gaps < %.2f s, age p90 <= 50 ms, all tagged, odometry >= 29 Hz, depth placed):"
              % CMD_STALE_S, "PASS" if ok else "FAIL")


def main() -> None:
    args = [float(a) for a in sys.argv[1:3]]
    rclpy.init()
    node = Check(*(args + [15.0, 45.0][len(args):]))
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
