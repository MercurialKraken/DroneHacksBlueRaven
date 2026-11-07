#!/usr/bin/env python3
"""SIM-ONLY check. Never import this from flight code.

Compares the published course file's planning field (maps/course_sdf.npz) with the simulator's own
static collision voxels (dronehacks_sim), slice by slice at the given heights, and reports where
the drone actually spawned against the course YAML's launch mark. Answers Task 1's open questions:
is the transform between the course file and the sim world the identity, and does the sim's
course match the published layout?

Run in the companion (browser terminal, Ctrl+`), with the simulator running and the drone on the
pad. It only reads; it sends no commands.

    python3 tools/sim_eval/compare_course_voxels.py --out runs/course_compare

Expected differences: the sim's training props (Board_*, Crate_*) are not in the course file.
Apples to apples: the sim marks a voxel occupied if it INTERSECTS geometry, so a course voxel
counts as occupied when the course field at its centre is within half a voxel diagonal.

Outputs in --out: report.txt, and compare_z<h>.png when matplotlib is available
(black = both, red = sim only, blue = course file only).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nav.course_map import CourseMap  # noqa: E402


def slice_compare(cmap: CourseMap, grid, h: float, shift=(0.0, 0.0)):
    """(sim occupied 2D, course occupied 2D, x centres, y centres) on the sim's grid at height h."""
    res = float(grid.resolution_m)
    org = np.asarray(grid.origin_enu_m, dtype=float)
    occ = np.asarray(grid.occupied, dtype=bool)
    iz = int(math.floor((h - org[2]) / res))
    if not 0 <= iz < occ.shape[2]:
        raise ValueError(f"height {h} m is outside the sim map")
    xs = org[0] + (np.arange(occ.shape[0]) + 0.5) * res
    ys = org[1] + (np.arange(occ.shape[1]) + 0.5) * res
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    zc = org[2] + (iz + 0.5) * res
    pts = np.stack([X.ravel() - shift[0], Y.ravel() - shift[1], np.full(X.size, zc)], axis=1)
    d = cmap.distance(pts, "planning").reshape(X.shape)
    course = np.isfinite(d) & (d <= res * math.sqrt(3) / 2)
    known = np.isfinite(d)
    return occ[:, :, iz] & known, course, xs, ys


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / u if u else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description="SIM-ONLY: course file vs simulator collision voxels")
    ap.add_argument("--config", default=str(ROOT / "maps" / "course.yaml"))
    ap.add_argument("--heights", type=float, nargs="*", help="default: build.plot_heights from the config")
    ap.add_argument("--search", type=float, default=0.30, help="search +-this many metres of x/y shift")
    ap.add_argument("--out", default="runs/course_compare")
    args = ap.parse_args()

    from dronehacks_sim import Simulation  # sim-only import, kept out of module scope

    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cmap = CourseMap.load(cfg_path.parent / cfg["build"]["output"])
    heights = args.heights or cfg["build"].get("plot_heights", [1.0])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    with Simulation() as sim:
        pose = sim.pose()
        grid = sim.voxels
        res = float(grid.resolution_m)
        say(f"sim voxels: {np.shape(grid.occupied)} at {res} m, origin {np.round(grid.origin_enu_m, 3).tolist()}")
        say(f"course map: {cmap.meta.get('mesh')} sha256 {str(cmap.meta.get('mesh_sha256', ''))[:12]}..., "
            f"built {cmap.meta.get('built_utc')}")
        p = np.asarray(pose.position_enu_m, dtype=float)
        L = cfg.get("launch", {})
        say(f"drone spawn (sim ENU): {np.round(p, 3).tolist()} at sim time {pose.sim_time_ns} ns")
        if L:
            say(f"course launch mark:    [{L['x']}, {L['y']}]  ->  spawn - mark = "
                f"({p[0] - L['x']:+.3f}, {p[1] - L['y']:+.3f}) m; base_link z {p[2]:.3f} m")

        steps = np.arange(-args.search, args.search + 1e-9, res)
        for h in heights:
            sim_occ, course, xs, ys = slice_compare(cmap, grid, h)
            say(f"\nz = {h:.2f} m: IoU {iou(sim_occ, course):.3f}; sim-only {np.count_nonzero(sim_occ & ~course)}, "
                f"course-only {np.count_nonzero(course & ~sim_occ)}, both {np.count_nonzero(sim_occ & course)} voxels")
            best = max(((iou(*slice_compare(cmap, grid, h, (dx, dy))[:2]), dx, dy) for dx in steps for dy in steps))
            say(f"  best x/y shift (course -> sim) within +-{args.search} m: ({best[1]:+.2f}, {best[2]:+.2f}) m, "
                f"IoU {best[0]:.3f}")
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except ImportError:
                continue
            img = np.ones(sim_occ.shape + (3,))
            img[sim_occ & course] = (0, 0, 0)
            img[sim_occ & ~course] = (0.85, 0.1, 0.1)
            img[course & ~sim_occ] = (0.1, 0.3, 0.9)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(img.transpose(1, 0, 2), origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]))
            if L:
                ax.plot(L["x"], L["y"], "g^")
            ax.plot(p[0], p[1], "mx", ms=9)
            ax.set_title(f"z = {h:.2f} m: black both, red sim only, blue course file only; "
                         f"green mark, magenta spawn", fontsize=8)
            ax.set_xlabel("x (m, east)")
            ax.set_ylabel("y (m, north)")
            fig.tight_layout()
            fig.savefig(out / f"compare_z{h:.2f}.png", dpi=110)
            plt.close(fig)

    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out / 'report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
