#!/usr/bin/env python3
"""Save one frame from each OAK camera as PNG and print brightness/contrast. Read-only: it
only subscribes, and changes nothing on the drone.

Run inside the skybeta container (host/run.sh shell), with this file copied to ~/team_ws:
    python3 /root/team_ws/snap_camera.py
It writes /root/team_ws/left.png and right.png, which is ~/team_ws/ on the drone's host.

How to read the numbers:
    brightness mean < 40     too dark for cuVSLAM
    contrast (std) < 15      almost blank: a wall, ceiling, table top or a covered lens
"""
from __future__ import annotations

import struct
import time
import zlib

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

OUT_DIR = "/root/team_ws"
TOPICS = {"left": "/oak/left/image_raw", "right": "/oak/right/image_raw"}


def to_rgb(msg: Image) -> np.ndarray:
    """sensor_msgs/Image (bgr8, rgb8 or mono8) -> (H, W, 3) uint8 RGB."""
    enc = msg.encoding.lower()
    channels = {"bgr8": 3, "rgb8": 3, "mono8": 1, "8uc1": 1}.get(enc)
    if channels is None:
        raise ValueError(f"unsupported encoding {msg.encoding}")
    rows = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    img = rows[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if channels == 1:
        return np.repeat(img, 3, axis=2)
    return img[:, :, ::-1] if enc == "bgr8" else img


def write_png(path: str, rgb: np.ndarray) -> None:
    """Minimal 8-bit RGB PNG writer (stdlib zlib only)."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + np.ascontiguousarray(rgb[y]).tobytes() for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def main() -> None:
    rclpy.init()
    node = Node("camera_snapshot")
    got: dict[str, Image] = {}
    for side, topic in TOPICS.items():
        node.create_subscription(Image, topic, lambda m, s=side: got.setdefault(s, m), qos_profile_sensor_data)
    deadline = time.monotonic() + 15.0
    while len(got) < len(TOPICS) and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    for side, topic in TOPICS.items():
        if side not in got:
            print(f"{side}: no image within 15 s on {topic}")
            continue
        rgb = to_rgb(got[side])
        gray = rgb.mean(axis=2)
        path = f"{OUT_DIR}/{side}.png"
        write_png(path, rgb)
        print(f"{side}: {rgb.shape[1]}x{rgb.shape[0]} {got[side].encoding}  brightness mean {gray.mean():.0f}/255  "
              f"contrast(std) {gray.std():.0f}  dark(<20) {100 * (gray < 20).mean():.0f}%  "
              f"saturated(>235) {100 * (gray > 235).mean():.0f}%  -> {path}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
