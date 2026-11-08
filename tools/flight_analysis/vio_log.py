"""READ-ONLY: log cuVSLAM odometry (and status) at full rate to a CSV for N seconds; print a line every 2 s."""
import sys, time, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data as Q
from nav_msgs.msg import Odometry
import math
dur, out = float(sys.argv[1]), sys.argv[2]
rclpy.init(); n = Node("arnav_vio_log")
f = open(out, "w"); f.write("t_wall,stamp,x,y,z,yaw_deg\n")
last = [None]
def cb(m):
    p, q = m.pose.pose.position, m.pose.pose.orientation
    yaw = math.degrees(math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
    st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
    f.write(f"{time.time():.3f},{st:.3f},{p.x:.4f},{p.y:.4f},{p.z:.4f},{yaw:.2f}\n")
    last[0] = (p.x, p.y, p.z, yaw)
n.create_subscription(Odometry, "/visual_slam/tracking/odometry", cb, Q)
t0 = time.time(); nxt = 0
while time.time() - t0 < dur:
    rclpy.spin_once(n, timeout_sec=0.05)
    if time.time() - t0 >= nxt and last[0]:
        nxt += 2.0; x, y, z, yw = last[0]
        print(f"t+{time.time() - t0:5.1f}s  x {x:+.3f}  y {y:+.3f}  z {z:+.3f}  yaw {yw:+6.1f}", flush=True)
f.close(); n.destroy_node(); rclpy.try_shutdown()
