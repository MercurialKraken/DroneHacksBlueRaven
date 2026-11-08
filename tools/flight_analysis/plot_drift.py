"""Plot skybeta-10 scored MPPI run 2 (route_mppi_full_160418) from its bag: where VIO believed the drone was, on the
course map with the nets, plus VIO height vs the rangefinder. usage: py -3 plot_drift.py <bag_dir> <out.png>"""
import glob
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mcap.records import Channel, Message, Schema  # noqa: E402
from mcap.stream_reader import StreamReader  # noqa: E402
from mcap_ros2.decoder import DecoderFactory  # noqa: E402

BAG, OUT = sys.argv[1], sys.argv[2]
START = np.array([-2.515, 1.466])                   # map take-off (Darion's mark), nose north
H = 3.048                                           # nets: the 20 ft arena square
ROUTE = json.load(open(sys.argv[3]))["route"] if len(sys.argv) > 3 else None

want = {"/visual_slam/tracking/odometry", "/fmu/out/vehicle_local_position", "/blueraven/mppi/cmd", "/blueraven/mppi/start"}
dec, schemas, chans, decs, rows = DecoderFactory(), {}, {}, {}, {k: [] for k in want}
with open(sorted(glob.glob(BAG + "/*.mcap"))[0], "rb") as fh:
    try:
        for r in StreamReader(fh, skip_magic=False).records:
            if isinstance(r, Schema):
                schemas[r.id] = r
            elif isinstance(r, Channel):
                chans[r.id] = r
            elif isinstance(r, Message):
                ch = chans.get(r.channel_id)
                if ch is None or ch.topic not in want:
                    continue
                if ch.id not in decs:
                    decs[ch.id] = dec.decoder_for(ch.message_encoding, schemas[ch.schema_id])
                try:
                    rows[ch.topic].append((r.log_time * 1e-9, decs[ch.id](r.data)))
                except Exception:
                    pass
    except Exception:
        pass                                        # the bag was never closed: it ends mid-record

od = np.array([(t, m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z)
               for t, m in rows["/visual_slam/tracking/odometry"]])
t0 = od[0, 0]
start_msgs = [json.loads(m.data) for _, m in rows["/blueraven/mppi/start"]]
t_follow = next((t for t, m in rows["/blueraven/mppi/cmd"] if json.loads(m.data).get("start") is not None), od[0, 0])
pad = start_msgs[0]["pad"] if start_msgs else [0.0, 0.0, 0.0, 0.0]
# odometry (x forward = north at take-off, y left = west) -> map: x = start_x - (y - pad_y), y = start_y + (x - pad_x)
mx = START[0] - (od[:, 2] - pad[1])
my = START[1] + (od[:, 1] - pad[0])
air = od[:, 3] > 0.3
fol = od[:, 0] >= t_follow

fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.15, 1]})
# course walls: the planning SDF sliced at 1.0 m
try:
    d = np.load(sys.argv[4]) if len(sys.argv) > 4 else None
except Exception:
    d = None
if d is not None:
    o, c, sdf = d["origin"], float(d["cell"]), d["planning"]
    k = int(round((1.0 - o[2]) / c))
    occ = sdf[:, :, k] < 0
    xs, ys = o[0] + c * np.arange(sdf.shape[0]), o[1] + c * np.arange(sdf.shape[1])
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    a1.scatter(X[occ], Y[occ], s=2, c="0.55", marker="s", linewidths=0, label="walls / obstacles (map, 1.0 m slice)")
a1.plot([-H, H, H, -H, -H], [-H, -H, H, H, -H], "r-", lw=2.5, label="nets (arena edge)")
fence = (START[0] - 0.533, START[0] + 5.413, START[1] - 4.514, START[1] + 1.432)   # fence15 in map (x0, x1, y0, y1)
a1.plot([fence[0], fence[1], fence[1], fence[0], fence[0]], [fence[2], fence[2], fence[3], fence[3], fence[2]],
        "r:", lw=1, label="MPPI virtual fence (now 15 cm tighter N/E)")
if ROUTE:
    rx = [START[0]] + [START[0] - p[1] for p in ROUTE]
    ry = [START[1]] + [START[1] + p[0] for p in ROUTE]
    a1.plot(rx, ry, "--o", color="#3060c0", ms=3, lw=1, label="route (from the take-off pose)")
a1.plot(mx[air & ~fol], my[air & ~fol], color="0.3", lw=1.5, label="VIO estimate: climb")
a1.plot(mx[fol], my[fol], color="#d02090", lw=2.2, label="VIO estimate: MPPI follow (where it THOUGHT it was)")
a1.plot(mx[-1], my[-1], "X", color="#d02090", ms=12)
a1.annotate(f"bag ends (unclosed; the logged 0.32 m VIO jump\ncame ~2 s later) VIO says {H - my[-1]:.2f} m from the north net", (mx[-1], my[-1]),
            xytext=(mx[-1] - 2.4, my[-1] - 1.6), arrowprops=dict(arrowstyle="->"), fontsize=9)
a1.annotate("pilot: the real drone was AT the north net\n(before the tunnel) -> VIO off by ~0.7-1 m",
            (mx[-1], H), xytext=(-2.9, 3.35), fontsize=9, color="red", arrowprops=dict(arrowstyle="->", color="red"))
jumps = [(i, math.hypot(mx[i] - mx[i - 1], my[i] - my[i - 1])) for i in range(1, len(mx))]
big = [(i, s) for i, s in jumps if s > 0.12]
for i, s in big:
    a1.plot([mx[i - 1], mx[i]], [my[i - 1], my[i]], color="orange", lw=3)
    a1.annotate(f"VIO jump {s:.2f} m", (mx[i], my[i]), fontsize=8, color="darkorange")
a1.set_aspect("equal")
a1.set_xlim(-3.4, 3.4)
a1.set_ylim(-3.4, 3.8)
a1.grid(alpha=0.25)
a1.set_xlabel("map x (m, east)")
a1.set_ylabel("map y (m, north)")
a1.set_title("Run 2 (route_mppi_full_160418): the VIO's own position estimate", fontsize=11)
a1.legend(fontsize=7.5, loc="lower left")

lp = rows["/fmu/out/vehicle_local_position"]
tl = np.array([t for t, _ in lp]) - t0
zl = np.array([-m.z for _, m in lp])
db = np.array([m.dist_bottom for _, m in lp])
a2.plot(od[:, 0] - t0, od[:, 3], color="#d02090", lw=2, label="VIO (cuVSLAM) height")
a2.plot(tl, db, color="0.2", lw=1.2, label="rangefinder distance to the floor (dist_bottom)")
a2.plot(tl, zl, color="#3060c0", lw=1, alpha=0.7, label="PX4 height estimate (rangefinder-based)")
a2.axvline(t_follow - t0, color="k", ls=":", lw=1)
a2.text(t_follow - t0 + 0.5, 0.15, "MPPI follow starts", fontsize=8)
a2.set_xlabel("time in bag (s)")
a2.set_ylabel("height (m)")
a2.set_ylim(-0.1, 1.5)
a2.grid(alpha=0.25)
a2.legend(fontsize=8, loc="lower right")
k = (tl > t_follow - t0 + 20)
if k.any():
    i = int(np.argmax(zl * k))
    j = int(np.argmin(np.abs((od[:, 0] - t0) - tl[i])))
    a2.annotate(f"VIO {od[j, 3]:.2f} m vs PX4 (rangefinder) {zl[i]:.2f} m: {zl[i] - od[j, 3]:.2f} m apart\n"
                "(raw rangefinder sits 0.10 m above the floor: compare the blue line)",
                (tl[i], zl[i]), xytext=(tl[i] - 40, 1.36), fontsize=9, arrowprops=dict(arrowstyle="->"))
a2.set_title("VIO height vs the rangefinder: the one direct drift measure in the bag", fontsize=11)
fig.suptitle("skybeta-10, scored MPPI run 2: MPPI tracked its route within ~0.1-0.15 m IN THE VIO FRAME; the VIO frame was wrong",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print("saved", OUT, "| follow points", int(fol.sum()), "| jumps >0.12 m:", [(round(s, 2)) for _, s in big],
      "| VIO end", (round(mx[-1], 2), round(my[-1], 2)), "north-net gap (VIO)", round(H - my[-1], 2))
