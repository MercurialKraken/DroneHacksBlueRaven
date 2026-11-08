#!/usr/bin/env python3
"""OFFLINE: the launch pose (x, y, yaw) from tape measurements taken FROM the drone on its start position.

    python tools/solve_launch.py --a F L --b F L [--write]

--a: forward and left (metres) from the drone's centre to Wall A's east end (the CP2 gap corner); --b: the same to
Wall B's north-east end. "Forward" is the direction the drone will face at take-off. The two ends' course
coordinates come from maps/course.yaml rebuilt_start, so two taped points give the start's position AND heading
(a 2D rigid fit); their distance checks the tape against the map. --write puts the result in course.yaml `launch`;
then `python tools/make_start_obj.py --all` regenerates the routes from it. Waypoints are relative to the actual
take-off pose, so this is what makes the route meet the real gap.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_start_obj import COURSE, start_geometry  # noqa: E402


def solve(taped: np.ndarray, course: np.ndarray) -> tuple[float, float, float]:
    """(x, y, yaw_deg) of the pose P with course = P * taped (2 x 2 arrays, rows are points)."""
    tc, cc = taped.mean(axis=0), course.mean(axis=0)
    H = (taped - tc).T @ (course - cc)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cc - R @ tc
    return float(t[0]), float(t[1]), math.degrees(math.atan2(R[1, 0], R[0, 0]))


def write_launch(text: str, x: float, y: float, yaw: float, note: str) -> str:
    i = text.index("\nlaunch:\n") + 1
    j = text.index("\nlaunch_sim:", i)
    block = text[i:j]
    block = re.sub(r"(\n  x: )[-0-9.]+", lambda m: f"{m.group(1)}{x:.3f}", block, count=1)
    block = re.sub(r"(\n  y: )[-0-9.]+", lambda m: f"{m.group(1)}{y:.3f}", block, count=1)
    block = re.sub(r"(\n  yaw_deg: )[-0-9.]+", lambda m: f"{m.group(1)}{yaw:.2f}", block, count=1)
    block = re.sub(r"(\n  confirmed: )\S+", lambda m: f"{m.group(1)}true", block, count=1)
    if "# taped:" in block:
        block = re.sub(r"\n  # taped:.*", f"\n  # taped: {note}", block, count=1)
    else:
        block = block.rstrip("\n") + f"\n  # taped: {note}\n"
    return text[:i] + block + text[j:]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", type=float, nargs=2, required=True, metavar=("F", "L"), help="to Wall A's east end (m)")
    ap.add_argument("--b", type=float, nargs=2, required=True, metavar=("F", "L"), help="to Wall B's NE end (m)")
    ap.add_argument("--write", action="store_true", help="put the result in maps/course.yaml launch")
    a = ap.parse_args(argv)
    text = COURSE.read_text(encoding="utf-8")
    g = start_geometry(yaml.safe_load(text))
    course = np.array([g["A"][1], g["B"][1]])
    taped = np.array([a.a, a.b])
    d_tape, d_map = float(np.linalg.norm(taped[1] - taped[0])), float(np.linalg.norm(course[1] - course[0]))
    x, y, yaw = solve(taped, course)
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    res = [float(np.linalg.norm(np.array([x, y]) + [c * f - s * l, s * f + c * l] - p)) for (f, l), p in zip(taped, course)]
    print(f"taped distance between the two ends {d_tape:.3f} m vs the map's {d_map:.3f} m "
          f"(difference {abs(d_tape - d_map) * 100:.1f} cm{'  <-- CHECK THE TAPE' if abs(d_tape - d_map) > 0.05 else ''})")
    print(f"launch: x {x:.3f}, y {y:.3f}, yaw {yaw:.2f} deg; fit residuals {res[0] * 100:.1f} / {res[1] * 100:.1f} cm")
    if a.write:
        note = (f"solved from tape (tools/solve_launch.py): Wall A end at forward {a.a[0]:.3f} left {a.a[1]:.3f}, "
                f"Wall B end at forward {a.b[0]:.3f} left {a.b[1]:.3f}; tape/map distance {d_tape:.3f}/{d_map:.3f} m")
        COURSE.write_bytes(write_launch(text, x, y, yaw, note).encode("utf-8"))
        print("written to maps/course.yaml launch; next: python tools/make_start_obj.py --all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
