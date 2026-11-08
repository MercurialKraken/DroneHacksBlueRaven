"""Read a (possibly unclosed) MCAP bag as a raw stream and print a 1 Hz follow timeline.
usage: py -3 bag_read.py <bag_dir>"""
import glob
import json
import math
import sys

from mcap.stream_reader import StreamReader
from mcap.records import Schema, Channel, Message
from mcap_ros2.decoder import DecoderFactory

WANT = {"/visual_slam/tracking/odometry", "/blueraven/mppi/cmd", "/blueraven/mppi/start", "/fmu/out/vehicle_local_position",
        "/oak/stereo/camera_info", "/refnav/status", "/blueraven/mission/status", "/fmu/out/distance_sensor"}
f = sorted(glob.glob(sys.argv[1] + "/*.mcap"))[0]
dec = DecoderFactory()
schemas, chans, decs = {}, {}, {}
rows = {k: [] for k in WANT}
with open(f, "rb") as fh:
    try:
        for r in StreamReader(fh, skip_magic=False).records:
            if isinstance(r, Schema):
                schemas[r.id] = r
            elif isinstance(r, Channel):
                chans[r.id] = r
            elif isinstance(r, Message):
                ch = chans.get(r.channel_id)
                if ch is None or ch.topic not in WANT:
                    continue
                if ch.id not in decs:
                    decs[ch.id] = dec.decoder_for(ch.message_encoding, schemas[ch.schema_id])
                try:
                    m = decs[ch.id](r.data)
                except Exception:
                    continue
                rows[ch.topic].append((r.log_time * 1e-9, m))
    except Exception as e:                               # an unclosed bag ends mid-record
        print("stream ended:", type(e).__name__, str(e)[:80])
print({k: len(v) for k, v in rows.items()})


def yaw_q(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


od = [(t, m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m.pose.pose.position, yaw_q(m.pose.pose.orientation),
       m.twist.twist.linear) for t, m in rows["/visual_slam/tracking/odometry"]]
cmds = []
for t, m in rows["/blueraven/mppi/cmd"]:
    try:
        cmds.append((t, json.loads(m.data)))
    except Exception:
        pass
start = [json.loads(m.data) for t, m in rows["/blueraven/mppi/start"]]
t_follow = next((t for t, c in cmds if c.get("start") is not None), None)
print("start msgs:", start[:1], " first cmd with start:", t_follow)
# odometry jumps (bag time) and stamp lag vs receipt
prev = None
for t, st, p, y, v in od:
    if prev is not None:
        d = math.hypot(p.x - prev[2].x, p.y - prev[2].y)
        if d > 0.12:
            print(f"ODOM JUMP at bag {t:.2f}: {d:.2f} m in {st - prev[1]:.3f} s stamp, ({prev[2].x:.2f},{prev[2].y:.2f}) -> ({p.x:.2f},{p.y:.2f})")
    prev = (t, st, p)
t0 = od[0][0] if od else 0
print("odom: n", len(od), "rate", round(len(od) / max(od[-1][0] - od[0][0], 1e-9), 1) if od else 0)
lag = [t - st for t, st, *_ in od]
if lag:
    lag.sort()
    print(f"odom receipt - stamp: p50 {lag[len(lag)//2]:.3f} p90 {lag[int(.9*len(lag))]:.3f} max {lag[-1]:.3f} s")
print("\n t_bag   odom x    y    z   yaw   v_meas(x,y)   | cmd state  vx(=vn?) vy  e_lat  why")
ci = 0
nxt = None
for t, st, p, y, v in od:
    if nxt is None:
        nxt = (t_follow or t) - 3.0
    if t < nxt:
        continue
    nxt = t + 1.0
    while ci + 1 < len(cmds) and cmds[ci + 1][0] <= t:
        ci += 1
    c = cmds[ci][1] if cmds else {}
    print(f"{t - t0:6.1f}  {p.x:6.2f} {p.y:6.2f} {p.z:5.2f} {math.degrees(y):6.1f}  ({v.x:+.2f},{v.y:+.2f})  | "
          f"{str(c.get('state')):7s} {c.get('vn', c.get('vx', 0)) or 0:+.2f} {c.get('ve', c.get('vy', 0)) or 0:+.2f} "
          f"{c.get('e_lat') if c.get('e_lat') is None else round(c.get('e_lat'), 2)} {c.get('why')}")

# height: VIO odometry z vs PX4's local z (EKF2 height from the rangefinder) and the rangefinder distance
lp = [(t, m) for t, m in rows["/fmu/out/vehicle_local_position"]]
print("\n t_bag  odom_z  px4_z(up)  dist_bottom  ratio odom/px4")
j = 0
nxt = None
for t, st, p, y, v in od:
    if nxt is None:
        nxt = t
    if t < nxt:
        continue
    nxt = t + 5.0
    while j + 1 < len(lp) and lp[j + 1][0] <= t:
        j += 1
    if lp:
        m = lp[j][1]
        up = -m.z
        print(f"{t - t0:6.1f}  {p.z:6.2f}  {up:7.2f}   {getattr(m, 'dist_bottom', float('nan')):7.2f}   {p.z / up if abs(up) > 0.2 else float('nan'):.2f}")
