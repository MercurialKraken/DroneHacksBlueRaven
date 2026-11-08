"""READ-ONLY: bytes/s and rate of every image / pointcloud / camera topic for 6 s (raw serialized size)."""
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data as Q
from rclpy.serialization import serialize_message
from rosidl_runtime_py.utilities import get_message

rclpy.init()
n = Node("arnav_bw")
t_disc = time.time()
while time.time() - t_disc < 6.0:
    rclpy.spin_once(n, timeout_sec=0.1)
tt = [(t, ty[0]) for t, ty in n.get_topic_names_and_types()
      if any(k in ty[0] for k in ("Image", "PointCloud2", "CameraInfo", "CompressedImage", "Imu", "Odometry"))]
st = {}
for t, ty in tt:
    st[t] = [0, 0]
    n.create_subscription(get_message(ty), t, lambda m, t=t: st[t].__setitem__(0, st[t][0] + 1) or st[t].__setitem__(1, st[t][1] + len(serialize_message(m))), Q)
t0 = time.time()
while time.time() - t0 < 6.0:
    rclpy.spin_once(n, timeout_sec=0.05)
tot = 0
for t, ty in sorted(tt):
    c, b = st[t]
    tot += b
    print(f"{t:45s} {ty.split('/')[-1]:16s} {c / 6:6.1f} Hz  {b / 6 / 1e6:8.2f} MB/s  ({(b / c / 1e3) if c else 0:8.1f} KB/msg)")
print(f"TOTAL received by this subscriber: {tot / 6 / 1e6:.1f} MB/s")
