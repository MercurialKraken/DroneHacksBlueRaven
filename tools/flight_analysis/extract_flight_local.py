"""Run LOCALLY (no ROS needed): extract_flight.py's CSV lines from a real-drone or T4 bag's .mcap file.

    py -m pip install --user mcap mcap-ros2-support
    py tools/flight_analysis/extract_flight_local.py <bag>.mcap > flight.csv

Same output, line for line, as extract_flight.py (which needs the T4 container's ROS 2 + px4_msgs). Messages are
decoded from the schemas rosbag2 embeds in the mcap file. Copy a no-depth bag (a few MB) to the laptop, read-only,
instead of copying bags around on the T4.
"""
import json
import math
import sys

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

WANT = ["/fmu/out/vehicle_local_position", "/fmu/out/vehicle_local_position_v1", "/fmu/in/trajectory_setpoint",
        "/sim/truth/odometry", "/fmu/out/vehicle_attitude", "/fmu/out/vehicle_attitude_setpoint",
        "/fmu/out/vehicle_attitude_setpoint_v1", "/fmu/out/vehicle_local_position_setpoint",
        "/visual_slam/tracking/odometry", "/skybeta/setpoint/twist", "/refnav/status", "/fmu/out/vehicle_status_v1"]


def line(topic: str, m, ts: float) -> str | None:
    if topic == "/fmu/in/trajectory_setpoint":
        v = m.velocity
        return f"tsp,{ts:.3f},{v[0]:.4f},{v[1]:.4f},{v[2]:.4f},{m.yawspeed:.4f}"
    if topic == "/fmu/out/vehicle_attitude" or topic.startswith("/fmu/out/vehicle_attitude_setpoint"):
        q = list(m.q_d) if hasattr(m, "q_d") else list(m.q)
        w, x, y, z = (float(c) for c in q)
        roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
        return f"{'attsp' if 'setpoint' in topic else 'att'},{ts:.3f},{roll:.3f},{pitch:.3f}"
    if topic == "/fmu/out/vehicle_local_position_setpoint":
        return f"lpsp,{ts:.3f},{m.vx:.4f},{m.vy:.4f},{m.vz:.4f}"
    if topic == "/sim/truth/odometry":
        p = m.pose.pose.position
        return f"truth,{ts:.3f},{p.x:.4f},{p.y:.4f},{p.z:.4f}"
    if topic.startswith("/fmu/out/vehicle_local_position"):
        return (f"lpos,{ts:.3f},{m.x:.4f},{m.y:.4f},{m.z:.4f},{m.vx:.4f},{m.vy:.4f},{m.vz:.4f},{m.heading:.5f},"
                f"{int(m.xy_valid)},{m.eph:.4f}")
    if topic == "/visual_slam/tracking/odometry":
        p, q = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return f"odom,{ts:.3f},{p.x:.4f},{p.y:.4f},{p.z:.4f},{yaw:.5f}"
    if topic == "/skybeta/setpoint/twist":
        v, w = m.twist.linear, m.twist.angular
        return f"twist,{ts:.3f},{v.x:.4f},{v.y:.4f},{v.z:.4f},{w.z:.4f}"
    if topic == "/refnav/status":
        return f"status,{ts:.3f},{json.dumps(json.loads(m.data), separators=(',', ':'))}"
    if topic == "/fmu/out/vehicle_status_v1":
        return f"vstatus,{ts:.3f},{m.nav_state},{m.arming_state}"
    return None


def main(path: str) -> None:
    sys.stdout.reconfigure(newline="\n")                     # same bytes as the T4's output, also on Windows
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, message, m in reader.iter_decoded_messages(topics=WANT, log_time_order=True):
            out = line(channel.topic, m, message.log_time * 1e-9)
            if out is not None:
                print(out)


if __name__ == "__main__":
    main(sys.argv[1])
