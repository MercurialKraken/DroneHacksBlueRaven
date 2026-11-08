#!/usr/bin/env python3
"""OFFLINE: route v7, Darion's method to the first checkpoint (CP2), from maps/course.yaml.

    python tools/make_route_cp2_simple.py   # routes/route_v7_cp2_simple{,_sim}.json, copied into courses/

Take off and hover (the mission does that), fly STRAIGHT FORWARD from the take-off pose, turn RIGHT along a gradual
ARC (Chelsea, 2026-11-07: three targets, start / middle / end of a quarter circle of ARC_RADIUS_M, cut INSIDE the
corner so the drone keeps away from the end wall, Wall A, and turns 22.5 + 45 + 22.5 deg instead of 90 at once),
fly straight through the CP2 gap until the whole drone is across, land. The corner the arc rounds is
placed where the rightward leg is furthest from both gap walls (rebuilt_start's Wall A and Wall B, their exact
footprints: the leg runs along Wall A's south face and over Wall B's north-east end); the end is CROSS_MARGIN_M
beyond the drone radius past CP2's line, along its normal. One file per launch entry: `launch` (the real mark) and
`launch_sim` (the simulator's spawn). A new tape (tools/solve_launch.py --write, or new rebuilt_start numbers)
regenerates it: tools/make_start_obj.py --all runs this. Every leg is checked on maps/course_sdf.npz at flight
height against the planner's rule (drone radius + clearance margin + 0.01 m); nothing is written if one fails.
Waypoints only: the map is never read in flight.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from make_start_obj import COURSE, start_geometry, wall_footprint  # noqa: E402
from nav.course_map import CourseMap  # noqa: E402
from nav.frames import load_launch  # noqa: E402

NAME = "route_v7_cp2_simple"
COPY_TO = ROOT / "courses" / "start-2026-11-07-tape" / "routes"   # hinson/real-drone's per-course layout (same bytes)
HEIGHT_M, SPEED_MPS = 0.7, 0.2   # fixed 0.7 m; 0.2 m/s, a compromise from two T4 runs (refnav, 2026-11-07): at 0.3 the
                                  # legs drifted 0.15 m left (less since refnav's nose fix 2b9a76e: 0.08 m at 0.15);
                                  # at 0.15 the 90 deg corner became a yaw in place and cuVSLAM diverged (no parallax;
                                  # worse on skybeta-09, whose cuVSLAM has no IMU: Darion). On the arc this is ~16 deg/s
YAW_RATE_DPS = 30.0               # refnav's yaw-rate cap for this route (default 45): Darion's ask for the visual-only VIO.
                                  # NOTE flight.yaml's yaw_rate_dps (45) is a ROS parameter and WINS over this; FILLET_M
                                  # keeps the turn under ~20 deg/s whichever cap applies
CROSS_MARGIN_M = 0.20           # the end: the drone's centre this far beyond its radius past CP2's line
TURN_SHIFT_M = 0.0              # move the turn, so the whole rightward leg, this far forward (+: towards Wall A and
                                # the cone) or back (-: towards Wall B) from the best-clearance place
ARC_RADIUS_M = 0.7              # the turn's arc: it starts this far before the corner and ends this far after it
                                # (the wider, the more the drone translates per degree turned: VIO parallax)
FILLET_M = ARC_RADIUS_M         # refnav rounds each target's corner with an arc of at most this (its default 0.3 turns
                                # the 3 targets into three 0.3 m arcs, ~38 deg/s at 0.2 m/s); at the arc's radius it flies
                                # the arc itself (corner_cut_m 0.05 trims the middle one to ~0.6 m): ~19 deg/s
STEP_M = 0.005                  # leg sampling for the clearances


def samples(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    n = max(2, math.ceil(float(np.linalg.norm(q - p)) / STEP_M) + 1)
    return p + np.outer(np.linspace(0.0, 1.0, n), q - p)


def to_polygon(P: np.ndarray, poly) -> np.ndarray:
    """2D distance from points P (N, 2) to a convex polygon (corners in order); 0 inside."""
    poly = np.asarray(poly, dtype=float)
    d, cross = np.full(len(P), np.inf), []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ab = b - a
        t = np.clip((P - a) @ ab / (ab @ ab), 0.0, 1.0)
        d = np.minimum(d, np.linalg.norm(P - a - t[:, None] * ab, axis=1))
        cross.append(ab[0] * (P[:, 1] - a[1]) - ab[1] * (P[:, 0] - a[0]))
    c = np.array(cross)
    return np.where((c >= 0).all(axis=0) | (c <= 0).all(axis=0), 0.0, d)


def main() -> int:
    cfg = yaml.safe_load(COURSE.read_text(encoding="utf-8"))
    g = start_geometry(cfg)
    R = float(cfg["limits"]["drone_radius"])
    need = R + float(cfg["limits"]["clearance_margin"]) + 0.01             # the planner's rule, as route v6
    cp2 = next(c for c in cfg["checkpoints"] if c["id"] == "CP2")
    a, b = np.asarray(cp2["line"], dtype=float)
    n = np.asarray(cp2["direction"], dtype=float)
    n /= np.linalg.norm(n)
    if max(np.linalg.norm(a - g["A"][1]), np.linalg.norm(b - g["B"][1])) > 0.01:
        print("CP2's line is not rebuilt_start's two gap ends: run tools/make_start_obj.py first")
        return 1
    walls = {w: wall_footprint(*g[k], g["t"], g["gap_mid"]) for w, k in (("Wall A", "A"), ("Wall B", "B"))}
    h = g["cone_size"] / 2
    cone_base = [g["cone"] + d for d in ([-h, -h], [h, -h], [h, h], [-h, h])]
    cm = CourseMap.load(ROOT / "maps" / "course_sdf.npz")
    fly = {**json.loads((ROOT / "route.json").read_text(encoding="utf-8"))["params"],
           "height_m": HEIGHT_M, "speed_mps": SPEED_MPS, "yaw_rate_dps": YAW_RATE_DPS, "fillet_m": FILLET_M}
    past = R + CROSS_MARGIN_M
    for key, suffix in (("launch", ""), ("launch_sim", "_sim")):
        L = load_launch(COURSE, key)
        O, yaw = np.array([L["x"], L["y"]]), math.radians(L["yaw_deg"])
        f, r = np.array([math.cos(yaw), math.sin(yaw)]), np.array([math.sin(yaw), -math.cos(yaw)])   # forward, right
        if r @ n < 0.5:
            print(f"{key}: turning right from yaw {L['yaw_deg']} deg meets CP2 {math.degrees(math.acos(r @ n)):.0f} "
                  "deg off its normal: not this route")
            return 1

        def legs(F: float) -> tuple[np.ndarray, np.ndarray]:
            """The turn (F forward of take-off) and the end (on the rightward leg, `past` past CP2's line)."""
            C = O + F * f
            return C, C + (past - (C - a) @ n) / (r @ n) * r

        def clear(F: float) -> dict[str, float]:
            C, E = legs(F)
            P = np.vstack([samples(O, C), samples(C, E)])
            return {w: float(to_polygon(P, fp).min()) for w, fp in walls.items()}

        lo, hi = sorted(((a - O) @ f, (b - O) @ f))         # the rightward leg through Wall A's / Wall B's gap end
        F = float(max(np.arange(lo, hi, 0.0005), key=lambda F: min(clear(F).values()))) + TURN_SHIFT_M
        C, E = legs(F)
        right = float(np.linalg.norm(E - C))
        rad = ARC_RADIUS_M
        if not (0.0 < rad <= min(F - 0.3, right - 0.3)):
            print(f"{key}: ARC_RADIUS_M {rad} leaves no straight climb-out or exit leg: nothing written")
            return 1
        k = 1.0 - math.sqrt(0.5)                            # the arc's middle: rad * (1 - cos 45) in from both legs
        arc = [C - rad * f, C - k * rad * f + k * rad * r, C + rad * r]   # start, middle, end (map)
        poly = [O, *arc, E]
        ex = {w: float(min(to_polygon(samples(p, q), fp).min() for p, q in zip(poly, poly[1:])))
              for w, fp in walls.items()}
        z = L["z"] + HEIGHT_M
        on_map = [float(np.nanmin(cm.distance(np.c_[P, np.full(len(P), z)], "planning")))
                  for P in (samples(p, q) for p, q in zip(poly, poly[1:]))]
        print(f"{key}: forward {F:.3f} m, turn 90 deg right, {right:.3f} m: turn at map "
              f"({C[0]:+.3f}, {C[1]:+.3f}), land at ({E[0]:+.3f}, {E[1]:+.3f}), {past:.3f} m past CP2's line")
        print(f"  exact clearance {', '.join(f'{w} {d:.3f} m' for w, d in ex.items())}; map at z {z:.2f}: "
              f"legs {', '.join(f'{d:.3f}' for d in on_map)} m (rule {need:.2f} m); arc radius {rad} m, so the corner "
              f"({C[0]:+.3f}, {C[1]:+.3f}) is never reached")
        side = 1.0 if (g["cone"] - C) @ f > 0 else -1.0      # the cone is forward (+) or back (-) of the leg
        room = next((s for s in np.arange(0.0, 0.3, 0.001)
                     if min(clear(F - side * (s + 0.001)).values()) < need), 0.3)
        to_axis = float(np.linalg.norm(g["cone"] - (C + np.clip((g["cone"] - C) @ r, 0.0, right) * r)))
        to_base = float(to_polygon(samples(C, E), cone_base).min())
        over = "DIRECTLY OVER its base" if to_base == 0 else f"({to_base:.3f} m outside its {g['cone_size']} m base)"
        cone = f"the rightward leg passes {to_axis:.3f} m from the cone's centre {over}, {z - g['cone_h']:.2f} m above it"
        print(f"  {cone}; moving the leg up to {room:.3f} m further from it keeps {need:.2f} m from the exact walls "
              f"(TURN_SHIFT_M {-side * room:+.3f}; the map reads ~1 cm lower in the gap)")
        if min(on_map) < need or min(ex.values()) < need:
            print(f"{key}: below the {need:.2f} m clearance rule: nothing written")
            return 1
        route = [(F - rad, 0.0), (F - k * rad, -k * rad), (F, -rad), (F, -right)]
        src = (f"tools/make_route_cp2_simple.py on maps/course.yaml {cfg['course']['name']}, launch_key {key}, "
               f"{HEIGHT_M} m. Darion's method to CP2 with a gradual turn: take off, straight forward "
               f"{F - rad:.3f} m, a {rad} m-radius right arc (3 targets: start, middle, end) inside the corner at "
               f"forward {F:.3f} m, straight {right - rad:.3f} m through the CP2 gap, land {past:.3f} m past its line "
               f"(whole drone across + {CROSS_MARGIN_M:.2f} m). Turn at map ({C[0]:.3f}, {C[1]:.3f}), land at "
               f"({E[0]:.3f}, {E[1]:.3f}). Clearance (exact) {', '.join(f'{w} {d:.3f} m' for w, d in ex.items())}, "
               f"turn shift {TURN_SHIFT_M:+.3f} m from the best; map min {min(on_map):.3f} m. {cone[0].upper()}{cone[1:]}.")
        pts = ",\n".join(f"  [{fw:.3f}, {lf:.3f}]" for fw, lf in route)
        text = ('{\n "from": ' + json.dumps(src) + ',\n "frame": "metres from the take-off pose: [forward, left]; '
                'launch_key ' + key + '",\n "labels": ' + json.dumps(["arc start (turn right)", "arc middle (45 deg)", "arc end (heading through CP2)", "past CP2: land"]) +
                ',\n "route": [\n' + pts + '\n ],\n "params": ' + json.dumps(fly) + '\n}\n')
        json.loads(text)
        out = ROOT / "routes" / f"{NAME}{suffix}.json"
        out.write_text(text, encoding="utf-8")
        if COPY_TO.is_dir():
            shutil.copyfile(out, COPY_TO / out.name)
        print(f"wrote routes/{out.name}" + (f" and {COPY_TO.relative_to(ROOT).as_posix()}/{out.name}"
                                            if COPY_TO.is_dir() else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
