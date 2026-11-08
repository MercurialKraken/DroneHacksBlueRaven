import time, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data as Q
from rosidl_runtime_py.utilities import get_message
rclpy.init(); n = Node("arnav_vs")
t0 = time.time()
while time.time() - t0 < 3.0:
    rclpy.spin_once(n, timeout_sec=0.1)
ty = dict(n.get_topic_names_and_types()).get("/visual_slam/status", [None])[0]
got = []
n.create_subscription(get_message(ty), "/visual_slam/status", lambda m: got.append(m), Q)
t0 = time.time()
while time.time() - t0 < 3.0:
    rclpy.spin_once(n, timeout_sec=0.05)
if got:
    m = got[-1]
    print(f"vo_state {m.vo_state} (n={len(got)}), track ms {m.track_execution_time*1e3:.2f}, mean {m.track_execution_time_mean*1e3:.2f}, max {m.track_execution_time_max*1e3:.2f}")
