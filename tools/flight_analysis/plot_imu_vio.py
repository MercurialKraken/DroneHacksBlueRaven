"""Separate VIO and IMU (PX4) plots for a skybeta bag. The bag has no raw IMU topic: 'IMU' here is PX4's attitude
(gyro+accel AHRS; yaw is also fused with vision) and PX4's acceleration estimate. usage: plot_imu_vio.py <bag> <out.png>"""
import glob, math, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mcap.records import Channel, Message, Schema
from mcap.stream_reader import StreamReader
from mcap_ros2.decoder import DecoderFactory

want = {"/visual_slam/tracking/odometry", "/fmu/out/vehicle_attitude", "/fmu/out/vehicle_local_position", "/blueraven/mppi/cmd"}
dec, schemas, chans, decs, rows = DecoderFactory(), {}, {}, {}, {k: [] for k in want}
with open(sorted(glob.glob(sys.argv[1] + "/*.mcap"))[0], "rb") as fh:
    try:
        for r in StreamReader(fh, skip_magic=False).records:
            if isinstance(r, Schema): schemas[r.id] = r
            elif isinstance(r, Channel): chans[r.id] = r
            elif isinstance(r, Message):
                ch = chans.get(r.channel_id)
                if ch is None or ch.topic not in want: continue
                if ch.id not in decs: decs[ch.id] = dec.decoder_for(ch.message_encoding, schemas[ch.schema_id])
                try: rows[ch.topic].append((r.log_time * 1e-9, decs[ch.id](r.data)))
                except Exception: pass
    except Exception:
        pass
import json
od = rows["/visual_slam/tracking/odometry"]; t0 = od[0][0]
t_follow = next((t for t, m in rows["/blueraven/mppi/cmd"] if json.loads(m.data).get("start") is not None), t0) - t0
to = np.array([t for t, _ in od]) - t0
P = np.array([[m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z] for _, m in od])
q = np.array([[m.pose.pose.orientation.w, m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z] for _, m in od])
yaw_v = np.degrees(np.unwrap(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))))
V = np.array([[m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z] for _, m in od])
att = rows["/fmu/out/vehicle_attitude"]
ta = np.array([t for t, _ in att]) - t0
Q = np.array([list(m.q) for _, m in att])                                   # PX4 FRD->NED (w, x, y, z)
roll = np.degrees(np.arctan2(2 * (Q[:, 0] * Q[:, 1] + Q[:, 2] * Q[:, 3]), 1 - 2 * (Q[:, 1] ** 2 + Q[:, 2] ** 2)))
pitch = np.degrees(np.arcsin(np.clip(2 * (Q[:, 0] * Q[:, 2] - Q[:, 3] * Q[:, 1]), -1, 1)))
yaw_p = np.degrees(np.unwrap(np.arctan2(2 * (Q[:, 0] * Q[:, 3] + Q[:, 1] * Q[:, 2]), 1 - 2 * (Q[:, 2] ** 2 + Q[:, 3] ** 2))))
lp = rows["/fmu/out/vehicle_local_position"]
tl = np.array([t for t, _ in lp]) - t0
Acc = np.array([[m.ax, m.ay, m.az] for _, m in lp])

fig, ax = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
fig.suptitle(f"skybeta-10 scored MPPI run 2 ({sys.argv[1].split('/')[-1]}): VIO (left) and IMU / PX4 inertial (right)", fontsize=12)
for a in ax.flat:
    a.axvline(t_follow, color="k", ls=":", lw=1); a.grid(alpha=0.25)
ax[0, 0].plot(to, P[:, 0], label="x (forward at take-off = north)"); ax[0, 0].plot(to, P[:, 1], label="y (left = west)")
ax[0, 0].set_ylabel("m"); ax[0, 0].set_title("VIO position (cuVSLAM odometry)"); ax[0, 0].legend(fontsize=8)
ax[1, 0].plot(to, P[:, 2], color="#d02090"); ax[1, 0].set_ylabel("m"); ax[1, 0].set_title("VIO height")
ax[2, 0].plot(to, V[:, 0], label="v forward (body)"); ax[2, 0].plot(to, V[:, 1], label="v left (body)"); ax[2, 0].plot(to, V[:, 2], label="v up", alpha=0.6)
ax[2, 0].set_ylabel("m/s"); ax[2, 0].set_title("VIO velocity (body frame)"); ax[2, 0].legend(fontsize=8); ax[2, 0].set_xlabel("time in bag (s)")
ax[0, 1].plot(ta, roll, label="roll"); ax[0, 1].plot(ta, pitch, label="pitch")
ax[0, 1].set_ylabel("deg"); ax[0, 1].set_title("IMU attitude: roll / pitch (PX4 vehicle_attitude, gyro + accel)"); ax[0, 1].legend(fontsize=8)
ax[1, 1].plot(ta, yaw_p - yaw_p[0], label="PX4 yaw (IMU, fused with vision yaw)"); ax[1, 1].plot(to, -(yaw_v - yaw_v[0]), "--", label="VIO yaw (sign flipped: ENU vs NED)")
ax[1, 1].set_ylabel("deg from start"); ax[1, 1].set_title("Heading: IMU/PX4 vs VIO"); ax[1, 1].legend(fontsize=8)
ax[2, 1].plot(tl, Acc[:, 0], label="a north"); ax[2, 1].plot(tl, Acc[:, 1], label="a east"); ax[2, 1].plot(tl, Acc[:, 2], label="a down", alpha=0.5)
ax[2, 1].set_ylabel("m/s^2"); ax[2, 1].set_ylim(-1.5, 1.5); ax[2, 1].set_title("IMU acceleration (PX4 estimate, gravity removed)"); ax[2, 1].legend(fontsize=8); ax[2, 1].set_xlabel("time in bag (s)")
ax[0, 0].text(t_follow + 0.5, ax[0, 0].get_ylim()[0], "MPPI follow", fontsize=8)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=105)
print("saved", sys.argv[2], "| roll/pitch in follow p95 abs:", np.percentile(np.abs(roll[ta > t_follow]), 95).round(1), np.percentile(np.abs(pitch[ta > t_follow]), 95).round(1))
