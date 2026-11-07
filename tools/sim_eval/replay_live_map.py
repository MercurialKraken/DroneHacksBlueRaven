#!/usr/bin/env python3
"""SIM-ONLY evaluation. Never import this from flight code.

Replays a recorded flight (rosbag2, mcap) through nav.live_map, read-only: no topics are published
and nothing is commanded. Two live layers run side by side on every depth frame:

  truth     pose from /sim/truth/odometry (course frame -> base_link) plus the camera's static TF.
            Unexplained returns here are real differences between the course map and the sim world
            (training props, layout changes), or depth errors. No pose gate.
  estimate  pose from TF map <- camera, built from the bag's /tf (px4_odom's odom -> base_link) and
            map -> odom from maps/course.yaml (--launch-key), through the same PoseGate as the node.
            Shows what the live map would have done in flight, including around VIO jumps.

    python3 tools/sim_eval/replay_live_map.py ~/bags/<run> --out runs/live_<run>

  corrected the estimate with nav.map_align_node's drift correction (/blueraven/odom_correction,
            the latest one at or before each frame) applied between odom and the camera, exactly as
            nav/live_map_node.py does. Only when the bag carries that topic.

Outputs in --out: frames.csv (one row per depth frame: both modes' counts, misfit, gate decision,
camera position error estimate - truth), obstacles_truth.npz and obstacles_estimate.npz (obstacle
voxel centres), summary.txt, and top-down PNGs when matplotlib is available.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nav.course_map import CourseMap  # noqa: E402
from nav.frames import correction_in_odom, load_launch, map_from_odom, yaw_from_quaternion  # noqa: E402
from nav.live_map import LiveObstacleLayer, LiveParams, PoseGate  # noqa: E402
from refnav.depth import decode_depth, quat_to_mat  # noqa: E402

DEPTH, INFO = "/camera/depth/image_raw", "/camera/depth/camera_info"
TRUTH, HEALTH = "/sim/truth/odometry", "/estimator/healthy"
CORRECTION = "/blueraven/odom_correction"


def stamp_s(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def tf_matrix(tr) -> np.ndarray:
    r, p = tr.rotation, tr.translation
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(r.w, r.x, r.y, r.z)
    T[:3, 3] = (p.x, p.y, p.z)
    return T


def open_bag(path: str):
    import rosbag2_py
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=""), rosbag2_py.ConverterOptions("", ""))
    except Exception:
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=path, storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    return reader, {t.name: t.type for t in reader.get_all_topics_and_types()}


def main() -> int:
    ap = argparse.ArgumentParser(description="SIM-ONLY: replay a bag through the live map")
    ap.add_argument("bag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(ROOT / "maps" / "course.yaml"))
    ap.add_argument("--course-map", default=str(ROOT / "maps" / "course_sdf.npz"))
    ap.add_argument("--launch-key", default="launch_sim")
    ap.add_argument("--depth-scale", type=float, default=1.0)
    ap.add_argument("--max-rate-hz", type=float, default=5.0)
    args = ap.parse_args()

    from rclpy.serialization import deserialize_message
    from rclpy.time import Time
    from rosidl_runtime_py.utilities import get_message
    from tf2_ros import Buffer
    from builtin_interfaces.msg import Duration as DurationMsg  # noqa: F401  (import check only)
    from rclpy.duration import Duration
    from geometry_msgs.msg import TransformStamped

    course = CourseMap.load(args.course_map)
    params = LiveParams(depth_scale=args.depth_scale)
    live = {"truth": LiveObstacleLayer(course, params=params), "estimate": LiveObstacleLayer(course, params=params)}
    gate = PoseGate()
    gate_cor = PoseGate()
    C = None                                                  # latest drift correction, in odom
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # map -> odom for the estimate chain, as the node would publish it
    T_mo = map_from_odom(load_launch(args.config, args.launch_key))
    buf = Buffer(cache_time=Duration(seconds=3600))
    mo = TransformStamped()
    mo.header.frame_id, mo.child_frame_id = "map", "odom"
    mo.transform.translation.x, mo.transform.translation.y, mo.transform.translation.z = (float(c) for c in T_mo[:3, 3])
    a = math.atan2(T_mo[1, 0], T_mo[0, 0])
    mo.transform.rotation.z, mo.transform.rotation.w = math.sin(a / 2), math.cos(a / 2)
    buf.set_transform_static(mo, "replay")

    reader, types = open_bag(args.bag)
    missing = [t for t in (DEPTH, INFO) if t not in types]
    if missing:
        raise SystemExit(f"bag lacks {missing}; it has {sorted(types)}")
    cls = {t: get_message(ty) for t, ty in types.items()}
    truth_t: list[float] = []
    truth_T: list[np.ndarray] = []
    K = None
    last_used = -math.inf
    rows = []
    skips: dict[str, int] = {}
    msgs = 0

    # Pass 1 would need the whole truth track for interpolation; the bag is in time order and truth
    # runs at 60 Hz, so the nearest earlier-or-equal truth sample is used (error <= 17 ms of motion).
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in cls:
            continue
        msgs += 1
        t_bag = t_ns * 1e-9
        if topic in ("/tf", "/tf_static"):
            m = deserialize_message(data, cls[topic])
            for tr in m.transforms:
                (buf.set_transform_static if topic == "/tf_static" else buf.set_transform)(tr, "bag")
        elif topic == TRUTH:
            m = deserialize_message(data, cls[topic])
            p, q = m.pose.pose.position, m.pose.pose.orientation
            T = np.eye(4)
            T[:3, :3] = quat_to_mat(q.w, q.x, q.y, q.z)
            T[:3, 3] = (p.x, p.y, p.z)
            truth_t.append(stamp_s(m.header))
            truth_T.append(T)
        elif topic == HEALTH:
            ok_h = bool(deserialize_message(data, cls[topic]).data)
            gate.on_health(t_bag, ok_h)
            gate_cor.on_health(t_bag, ok_h)
        elif topic == CORRECTION:
            m = deserialize_message(data, cls[topic])
            o = m.pose.orientation
            C = correction_in_odom(m.pose.position.x, m.pose.position.y, yaw_from_quaternion(o.x, o.y, o.z, o.w))
            if "corrected" not in live:
                live["corrected"] = LiveObstacleLayer(course, params=params)
        elif topic == INFO:
            m = deserialize_message(data, cls[topic])
            if m.k[0] > 0:
                K = np.array(m.k, dtype=float).reshape(3, 3)
        elif topic == DEPTH:
            if K is None or t_bag - last_used < 1.0 / args.max_rate_hz:
                continue
            m = deserialize_message(data, cls[topic])
            ts = stamp_s(m.header)
            row = {"t": round(ts, 3)}
            depth = decode_depth(m.encoding, m.height, m.width, m.step, bytes(m.data))
            try:
                T_bc = tf_matrix(buf.lookup_transform("base_link", m.header.frame_id, Time()).transform)
            except Exception as e:
                skips["no_camera_tf"] = skips.get("no_camera_tf", 0) + 1
                continue
            last_used = t_bag
            # truth
            T_true = None
            i = bisect.bisect_right(truth_t, ts) - 1
            if i >= 0 and ts - truth_t[i] <= 0.05:
                T_true = truth_T[i] @ T_bc
                st = live["truth"].integrate(depth, K, T_true)
                row.update(truth_explained=st.explained, truth_unexplained=st.unexplained, truth_offmap=st.off_map,
                           truth_misfit=st.misfit, truth_obstacles=st.obstacles)
            else:
                skips["no_truth"] = skips.get("no_truth", 0) + 1
            # estimate
            try:
                T_est = tf_matrix(buf.lookup_transform("map", m.header.frame_id,
                                                       Time(nanoseconds=int(ts * 1e9))).transform)
            except Exception:
                skips["no_estimate_tf"] = skips.get("no_estimate_tf", 0) + 1
                row["gate"] = "no TF"
                rows.append(row)
                continue
            ok, why = gate.check(t_bag, T_est[:3, 3])
            row["gate"] = why
            if T_true is not None:
                row["est_err_m"] = round(float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3])), 3)
            if ok:
                st = live["estimate"].integrate(depth, K, T_est)
                row.update(est_explained=st.explained, est_unexplained=st.unexplained, est_misfit=st.misfit,
                           est_rejected=st.rejected, est_obstacles=st.obstacles, est_trusted=live["estimate"].trusted)
            else:
                skips["gated"] = skips.get("gated", 0) + 1
            if C is not None:                                 # estimate + drift correction, as the node does
                T_cor = T_mo @ C @ np.linalg.inv(T_mo) @ T_est
                ok_c, why_c = gate_cor.check(t_bag, T_cor[:3, 3])
                row["cor_gate"] = why_c
                if T_true is not None:
                    row["cor_err_m"] = round(float(np.linalg.norm(T_cor[:3, 3] - T_true[:3, 3])), 3)
                if ok_c:
                    st = live["corrected"].integrate(depth, K, T_cor)
                    row.update(cor_explained=st.explained, cor_unexplained=st.unexplained, cor_misfit=st.misfit,
                               cor_rejected=st.rejected, cor_obstacles=st.obstacles)
            rows.append(row)

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "t", k))
    with open(out / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for mode, layer in live.items():
        np.savez_compressed(out / f"obstacles_{mode}.npz", points=layer.obstacle_points())

    def frac(r, mode):
        e, u = r.get(f"{mode}_explained"), r.get(f"{mode}_unexplained")
        return None if e is None or u is None or e + u == 0 else u / (e + u)

    lines = [f"bag {args.bag}: {msgs} messages, {len(rows)} depth frames used (max {args.max_rate_hz} Hz)",
             f"skips {skips}"]
    for mode, pre in (("truth", "truth"), ("estimate", "est"), ("corrected", "cor")):
        if mode not in live:
            lines.append("corrected: the bag has no /blueraven/odom_correction")
            continue
        fr = [x for x in (frac(r, pre) for r in rows) if x is not None]
        mf = [r[f"{pre}_misfit"] for r in rows if r.get(f"{pre}_misfit") is not None]
        lines.append(f"{mode:8s}: frames {len(fr)}, unexplained share median {np.median(fr) if fr else float('nan'):.3f} "
                     f"p95 {np.percentile(fr, 95) if fr else float('nan'):.3f}; misfit median "
                     f"{np.median(mf) if mf else float('nan'):.3f} max {max(mf) if mf else float('nan'):.3f}; "
                     f"final obstacle voxels {len(live[mode].obstacle_voxels())}, trusted {live[mode].trusted}")
    for key, label in (("est_err_m", "estimate"), ("cor_err_m", "corrected")):
        errs = [r[key] for r in rows if key in r]
        if errs:
            lines.append(f"camera position error, {label} vs truth: median {np.median(errs):.3f} m, "
                         f"p95 {np.percentile(errs, 95):.3f} m, max {max(errs):.3f} m")
    paused = [r["t"] for r in rows if r.get("gate", "ok") != "ok"]
    if paused:
        lines.append(f"gate paused {len(paused)} frames, first at t={paused[0]}, last at t={paused[-1]}")
    rejected = [r["t"] for r in rows if r.get("est_rejected")]
    if rejected:
        lines.append(f"misfit refused {len(rejected)} estimate frames, first at t={rejected[0]}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return 0
    lo, hi = course.bounds
    xs, ys = np.arange(lo[0], hi[0], 0.025), np.arange(lo[1], hi[1], 0.025)
    X, Y = np.meshgrid(xs, ys)
    Dm = course.distance(np.stack([X.ravel(), Y.ravel(), np.full(X.size, 1.0)], 1)).reshape(X.shape)
    fig, axes = plt.subplots(1, len(live), figsize=(6.5 * len(live), 6.5))
    for ax, (mode, layer) in zip(axes, live.items()):
        ax.contour(X, Y, Dm, levels=[0.02], colors="k", linewidths=0.7)
        P = layer.obstacle_points()
        if len(P):
            sc = ax.scatter(P[:, 0], P[:, 1], c=P[:, 2], s=4, cmap="viridis", vmin=0, vmax=2)
            fig.colorbar(sc, ax=ax, fraction=0.046, label="obstacle height (m)")
        ax.set_title(f"{mode}: {len(P)} live obstacle voxels over the course map (z 1.0 m outline)", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m, east)")
        ax.set_ylabel("y (m, north)")
    fig.tight_layout()
    fig.savefig(out / "obstacles.png", dpi=110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
