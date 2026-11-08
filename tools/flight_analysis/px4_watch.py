"""READ-ONLY (no publishers): 20 s of PX4 heading/estimator state, to see whether yaw alignment is stuck while the drone
is still, and which estimator flags are set."""
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data as QOS


class W(Node):
    def __init__(self):
        super().__init__("arnav_px4_watch")
        import px4_msgs.msg as M
        self.es = self.lp = self.vs = None
        self.n_ev = 0
        self.create_subscription(M.EstimatorStatusFlags, "/fmu/out/estimator_status_flags", lambda m: setattr(self, "es", m), QOS)
        self.create_subscription(M.VehicleLocalPosition, "/fmu/out/vehicle_local_position", lambda m: setattr(self, "lp", m), QOS)
        self.create_subscription(M.VehicleStatus, "/fmu/out/vehicle_status_v1", lambda m: setattr(self, "vs", m), QOS)
        self.create_subscription(M.VehicleOdometry, "/fmu/in/vehicle_visual_odometry", self.on_ev, QOS)
        self.ev_q = None

    def on_ev(self, m):
        self.n_ev += 1
        self.ev_q = m.q


def yaw_from_q(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    rclpy.init()
    n = W()
    t0 = time.time(); nxt = 0.0
    while time.time() - t0 < 20.0:
        rclpy.spin_once(n, timeout_sec=0.05)
        if time.time() - t0 >= nxt and n.es is not None and n.lp is not None:
            nxt += 2.0
            lp, es = n.lp, n.es
            ev_yaw = yaw_from_q(n.ev_q) if n.ev_q is not None and all(math.isfinite(c) for c in n.ev_q) else float("nan")
            print(f"t+{time.time() - t0:4.1f}s yaw_align={es.cs_yaw_align} heading_good={lp.heading_good_for_control} "
                  f"ev_yaw_fused={es.cs_ev_yaw} PX4 heading={math.degrees(lp.heading):7.2f} deg EV yaw (NED)={math.degrees(ev_yaw):7.2f} deg "
                  f"v=({lp.vx:+.3f},{lp.vy:+.3f},{lp.vz:+.3f}) preflight_pass={getattr(n.vs, 'pre_flight_checks_pass', '?')}")
    es = n.es
    if es is not None:
        f = es.get_fields_and_field_types()
        on = {k: getattr(es, k) for k in f if k != "timestamp" and k != "timestamp_sample"
              and ((isinstance(getattr(es, k), bool) and getattr(es, k)) or (not isinstance(getattr(es, k), bool) and getattr(es, k)))}
        print("ESTIMATOR_STATUS_FLAGS set/nonzero:", on)
    print(f"EV messages to PX4 in 20 s: {n.n_ev}")
    n.destroy_node(); rclpy.try_shutdown()


if __name__ == "__main__":
    main()
