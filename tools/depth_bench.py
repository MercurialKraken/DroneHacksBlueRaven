"""Tape-measure check of a depth topic (Task C acceptance 1). ROS 2 (Humble/Jazzy).

Point the left camera square at a flat, textured target (a printed random pattern) at a measured distance,
then:
    python3 tools/depth_bench.py --distance 1.0 [--seconds 10] [--roi 0.2] [--topic /camera/depth/image_raw]

Prints, over the run: the median depth in the central ROI (roi = fraction of width/height), its error
against --distance, the ROI's valid-pixel fraction, the message rate, and the latency from the image stamp
to receipt here (same machine, same clock). Repeat at 1, 2 and 3 m.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refnav.depth import decode_depth  # noqa: E402


def roi_stats(depth_m: np.ndarray, roi: float) -> tuple[float, float]:
    """(median depth, valid fraction) of the central roi x roi window."""
    h, w = depth_m.shape
    hh, hw = max(1, int(h * roi / 2)), max(1, int(w * roi / 2))
    win = depth_m[h // 2 - hh:h // 2 + hh, w // 2 - hw:w // 2 + hw]
    ok = np.isfinite(win)
    return (float(np.median(win[ok])) if ok.any() else float("nan")), float(ok.mean())


def report(medians, valids, recv_t, lat_ms, distance=None) -> str:
    med = float(np.nanmedian(medians)) if medians else float("nan")
    lines = [f"frames {len(medians)}",
             f"median depth in ROI {med:.3f} m (frame-to-frame std {np.nanstd(medians):.3f} m)",
             f"valid fraction in ROI {np.mean(valids):.2f}" if valids else "valid fraction n/a"]
    if distance is not None:
        lines.append(f"error vs tape {med - distance:+.3f} m ({(med - distance) / distance * 100:+.1f}%)")
    if len(recv_t) > 1:
        lines.append(f"rate {(len(recv_t) - 1) / (recv_t[-1] - recv_t[0]):.2f} Hz")
    if lat_ms:
        lines.append(f"latency stamp->receipt median {np.median(lat_ms):.0f} ms, p95 {np.percentile(lat_ms, 95):.0f} ms")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--roi", type=float, default=0.2)
    ap.add_argument("--topic", default="/camera/depth/image_raw")
    a = ap.parse_args(argv)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    rclpy.init()
    node = Node("depth_bench")
    medians, valids, recv_t, lat_ms = [], [], [], []

    def cb(m: Image) -> None:
        now = node.get_clock().now().nanoseconds
        z = decode_depth(m.encoding, m.height, m.width, m.step, bytes(m.data))
        med, vf = roi_stats(z, a.roi)
        medians.append(med); valids.append(vf); recv_t.append(time.monotonic())
        lat_ms.append((now - (m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec)) * 1e-6)

    node.create_subscription(Image, a.topic, cb, qos_profile_sensor_data)
    end = time.monotonic() + a.seconds
    while time.monotonic() < end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
    print(report(medians, valids, recv_t, lat_ms, a.distance))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
