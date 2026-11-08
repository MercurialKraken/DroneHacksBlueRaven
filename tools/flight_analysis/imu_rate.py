"""READ-ONLY: is the OAK IMU published, at what rate, and what it reads."""
import time, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data as Q
from sensor_msgs.msg import Imu
rclpy.init(); n = Node("arnav_imu")
t0 = time.time()
while time.time() - t0 < 4.0:
    rclpy.spin_once(n, timeout_sec=0.1)
tops = [t for t, ty in n.get_topic_names_and_types() if ty[0] == "sensor_msgs/msg/Imu"]
print("imu topics:", tops)
st = {t: [] for t in tops}
for t in tops:
    n.create_subscription(Imu, t, lambda m, t=t: st[t].append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m.linear_acceleration.z, m.angular_velocity.z)), Q)
t0 = time.time()
while time.time() - t0 < 5.0:
    rclpy.spin_once(n, timeout_sec=0.01)
for t, v in st.items():
    if len(v) > 2:
        s = sorted(x[0] for x in v); g = sorted(b - a for a, b in zip(s, s[1:]))
        print(f"{t}: {len(v) / 5:.0f} Hz, stamp gap p50 {g[len(g)//2]*1e3:.1f} ms max {g[-1]*1e3:.1f} ms, acc z {v[-1][1]:.2f}, gyro z {v[-1][2]:.4f}")
    else:
        print(t, len(v), "msgs")
