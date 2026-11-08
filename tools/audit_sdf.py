#!/usr/bin/env python3
"""OFFLINE (laptop): how far the course map's planning field is from the EXACT distance to the mesh.

    python tools/audit_sdf.py courses/<name>/course.yaml   # -> courses/<name>/audit/report.txt + audit_z*.png

The map is built on a 5 cm grid (from 1.67 cm voxels) and read by trilinear interpolation, so it
carries a small error. This measures it, on a dense grid at flight heights everywhere within
`--band` of a surface, against trimesh's exact signed distance to the same planning pieces.

    error = map distance - exact distance
    error > 0  OPTIMISTIC: the map promises more room than there is (the dangerous direction)
    error < 0  conservative: the map gives away room (costs margin, not safety)

Per height it reports the error distribution and the worst optimistic point; per audit spot in
course.yaml (`audit_spots`) the worst optimistic error within its radius. A planner can then add
the measured worst optimistic error to its clearance margin. Needs trimesh with rtree (laptop only).
--spots-only audits just the spots' circles (about a minute); the whole course takes much longer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from nav.course_map import CourseMap, course_dir  # noqa: E402
from build_course_sdf import parse_obj_groups, select_groups  # noqa: E402


def planning_mesh(cfg: dict, here: Path) -> trimesh.Trimesh:
    m = cfg["course"]["mesh"]
    groups = {}
    for p in (m if isinstance(m, list) else [m]):                 # the published course + later changes
        groups.update(parse_obj_groups(here / p))
    spec = cfg["build"]["fields"]["planning"]
    names = select_groups(list(groups), spec.get("include", []), spec.get("exclude", []))
    tris = np.concatenate([groups[n] for n in names])
    return trimesh.Trimesh(vertices=tris.reshape(-1, 3), faces=np.arange(len(tris) * 3).reshape(-1, 3), process=True)


def exact_free_distance(mesh: trimesh.Trimesh, pts: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Unsigned distance to the nearest triangle. Only points the map calls free are audited, so no
    inside/outside test is needed - and trimesh's ray-parity test gives wrong signs where pieces
    touch or overlap (barrier segments, the tunnel against the barrier)."""
    out = np.empty(len(pts))
    for i in range(0, len(pts), chunk):
        out[i:i + chunk] = mesh.nearest.on_surface(pts[i:i + chunk])[1]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", nargs="?", default=str(course_dir() / "course.yaml"))
    ap.add_argument("--heights", type=float, nargs="*", default=[0.7, 1.0, 1.3])
    ap.add_argument("--spacing", type=float, default=0.025)
    ap.add_argument("--band", type=float, default=0.6, help="only points this close to a surface (m)")
    ap.add_argument("--out", help="default: audit/ next to the config")
    ap.add_argument("--spots-only", action="store_true",
                    help="audit only inside the audit_spots circles (fast; the whole course takes a long time)")
    args = ap.parse_args(argv)
    cfg_path = Path(args.config).resolve()
    here = cfg_path.parent
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cm = CourseMap.load(here / cfg["build"]["output"])
    mesh = planning_mesh(cfg, here)
    out = Path(args.out) if args.out else here / "audit"
    out.mkdir(parents=True, exist_ok=True)
    spots = cfg.get("audit_spots", [])
    lo, hi = cm.bounds
    xs = np.arange(lo[0] + 0.2, hi[0] - 0.2 + 1e-9, args.spacing)
    ys = np.arange(lo[1] + 0.2, hi[1] - 0.2 + 1e-9, args.spacing)
    X, Y = np.meshgrid(xs, ys)
    lines = [f"course map {cm.meta.get('mesh')} built {cm.meta.get('built_utc')}; exact = unsigned distance "
             f"to the {len(mesh.faces)} planning triangles; grid {args.spacing * 100:.1f} cm; band {args.band} m",
             "error = map - exact; > 0 is OPTIMISTIC (the map promises room that isn't there)"]
    per_height = {}
    for h in args.heights:
        P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, h)], axis=1)
        d_map = cm.distance(P, "planning")
        keep = np.isfinite(d_map) & (d_map > 0) & (d_map <= args.band)
        if args.spots_only:
            in_spot = np.zeros(len(P), dtype=bool)
            for s in spots:
                c, r = np.asarray(s["at"], dtype=float), float(s.get("radius", 0.5))
                in_spot |= np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) <= r
            keep &= in_spot
        P, d_map = P[keep], d_map[keep]
        d_ex = exact_free_distance(mesh, P)
        err = d_map - d_ex
        per_height[h] = (P, err, d_ex)
        i_opt = int(np.argmax(err))
        lines.append(f"\nz = {h:.2f} m: {len(P)} points | |error| median {np.median(np.abs(err)) * 100:.2f} cm, "
                     f"p95 {np.percentile(np.abs(err), 95) * 100:.2f} cm | optimistic: {np.mean(err > 0) * 100:.0f}% of "
                     f"points, p99 {np.percentile(err, 99) * 100:.2f} cm, max {err[i_opt] * 100:.2f} cm at "
                     f"({P[i_opt, 0]:+.3f}, {P[i_opt, 1]:+.3f}) where exact clearance is {d_ex[i_opt]:.3f} m | "
                     f"most conservative {err.min() * 100:.2f} cm")
        rel = (d_ex >= 0.2) & (d_ex <= 0.5)                         # where a planner's margin is decided
        if rel.any():
            lines.append(f"  at planning clearances 0.2-0.5 m: max optimistic {err[rel].max() * 100:+.2f} cm, "
                         f"p99 {np.percentile(err[rel], 99) * 100:+.2f} cm, median {np.median(err[rel]) * 100:+.2f} cm")
        for s in spots:
            c, r = np.asarray(s["at"], dtype=float), float(s.get("radius", 0.5))
            near = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) <= r
            if near.any():
                e = err[near]
                j = int(np.argmax(e))
                nr = near & rel
                at_rel = f"{err[nr].max() * 100:+.2f} cm" if nr.any() else "n/a"
                lines.append(f"  {s['name']:<24s} {near.sum():5d} pts | max optimistic {e[j] * 100:+.2f} cm "
                             f"(exact clearance there {d_ex[near][j]:.3f} m), at 0.2-0.5 m {at_rel} | median "
                             f"|error| {np.median(np.abs(e)) * 100:.2f} cm")
    worst = max(v[1].max() for v in per_height.values())
    worst_rel = max((v[1][(v[2] >= 0.2) & (v[2] <= 0.5)].max() for v in per_height.values()
                     if np.any((v[2] >= 0.2) & (v[2] <= 0.5))), default=float("nan"))
    lines.append(f"\nWORST optimistic error audited: {worst * 100:.2f} cm anywhere, {worst_rel * 100:.2f} cm at "
                 f"planning clearances (0.2-0.5 m). A margin planned on this map should include the latter.")
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return 0
    for h, (P, err, _) in per_height.items():
        fig, ax = plt.subplots(figsize=(8, 7.5))
        sc = ax.scatter(P[:, 0], P[:, 1], c=np.clip(err * 100, -2, 2), s=2, cmap="RdBu_r", vmin=-2, vmax=2)
        fig.colorbar(sc, ax=ax, fraction=0.046, label="map - exact (cm); red = optimistic")
        for s in spots:
            ax.add_patch(plt.Circle(s["at"], s.get("radius", 0.5), fill=False, color="k", lw=0.8, ls="--"))
            ax.text(s["at"][0], s["at"][1], s["name"], fontsize=6, ha="center")
        ax.set_aspect("equal")
        ax.set_title(f"planning field error vs exact geometry at z = {h:.2f} m (points within {args.band} m of a surface)",
                     fontsize=9)
        ax.set_xlabel("x (m, east)")
        ax.set_ylabel("y (m, north)")
        fig.tight_layout()
        fig.savefig(out / f"audit_z{h:.2f}.png", dpi=110)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
