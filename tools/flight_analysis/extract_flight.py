"""Run INSIDE the T4 container: a real-drone bag's poses as CSV lines (kind,...), depth skipped.
lpos: PX4 EKF2 fused local position (NED); odom: cuVSLAM (ENU); twist: refnav's command; status: refnav JSON;
vstatus: PX4 nav/arming state."""
import json
import math
import sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=sys.argv[1], storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
want = ["/fmu/out/vehicle_local_position", "/fmu/out/vehicle_local_position_v1", "/fmu/in/trajectory_setpoint", "/sim/truth/odometry", "/fmu/out/vehicle_attitude", "/fmu/out/vehicle_attitude_setpoint", "/fmu/out/vehicle_attitude_setpoint_v1", "/fmu/out/vehicle_local_position_setpoint", "/visual_slam/tracking/odometry", "/skybeta/setpoint/twist",
        "/refnav/status", "/fmu/out/vehicle_status_v1"]
r.set_filter(rosbag2_py.StorageFilter(topics=[t for t in want if t in types]))
cls = {t: get_message(types[t]) for t in want if t in types}
while r.has_next():
    topic, data, t = r.read_next()
    m = deserialize_message(data, cls[topic])
    ts = t * 1e-9
    if topic == "/fmu/in/trajectory_setpoint":
        v = m.velocity
        print(f"tsp,{ts:.3f},{v[0]:.4f},{v[1]:.4f},{v[2]:.4f},{m.yawspeed:.4f}")
    elif topic == "/fmu/out/vehicle_attitude" or topic.startswith("/fmu/out/vehicle_attitude_setpoint"):
        q = list(m.q_d) if hasattr(m, "q_d") else list(m.q)
        w, x, y, z = (float(c) for c in q)
        roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
        print(f"{'attsp' if 'setpoint' in topic else 'att'},{ts:.3f},{roll:.3f},{pitch:.3f}")
    elif topic == "/fmu/out/vehicle_local_position_setpoint":
        print(f"lpsp,{ts:.3f},{m.vx:.4f},{m.vy:.4f},{m.vz:.4f}")
    elif topic == "/sim/truth/odometry":
        p = m.pose.pose.position
        print(f"truth,{ts:.3f},{p.x:.4f},{p.y:.4f},{p.z:.4f}")
    elif topic.startswith("/fmu/out/vehicle_local_position"):
        print(f"lpos,{ts:.3f},{m.x:.4f},{m.y:.4f},{m.z:.4f},{m.vx:.4f},{m.vy:.4f},{m.vz:.4f},{m.heading:.5f},"
              f"{int(m.xy_valid)},{m.eph:.4f}")
    elif topic == "/visual_slam/tracking/odometry":
        p, q = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        print(f"odom,{ts:.3f},{p.x:.4f},{p.y:.4f},{p.z:.4f},{yaw:.5f}")
    elif topic == "/skybeta/setpoint/twist":
        v, w = m.twist.linear, m.twist.angular
        print(f"twist,{ts:.3f},{v.x:.4f},{v.y:.4f},{v.z:.4f},{w.z:.4f}")
    elif topic == "/refnav/status":
        print(f"status,{ts:.3f},{json.dumps(json.loads(m.data), separators=(',', ':'))}")
    elif topic == "/fmu/out/vehicle_status_v1":
        print(f"vstatus,{ts:.3f},{m.nav_state},{m.arming_state}")
