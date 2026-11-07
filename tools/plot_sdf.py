#!/usr/bin/env python3
"""OFFLINE (laptop): top-down slices of the course's signed distance fields, to check by eye.

    python tools/plot_sdf.py maps/course.yaml            # -> maps/plots/sdf_z<height>.png

One PNG per height in build.plot_heights, with the planning field (left) and the localization
field (right):
  colour        signed distance (red inside objects, blue free; clipped at +-1 m)
  black line    the surface (drawn at 2 cm so thin shells like the tube and the net show)
  orange dashes where the drone's centre may go: distance = drone_radius + clearance_margin
  overlays      launch mark and heading, checkpoint lines with their direction of travel, the
                landing area and the pad's outer ring (all from the course YAML)
Needs matplotlib and pyyaml (laptop only).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nav.course_map import CourseMap  # noqa: E402

SURFACE_LINE_M = 0.02


def draw_course(ax: Any, cfg: dict[str, Any]) -> None:
    L = cfg.get("launch")
    if L:
        yaw = math.radians(L.get("yaw_deg", 0.0))
        ax.plot(L["x"], L["y"], "k^", ms=7)
        ax.annotate("", xy=(L["x"] + 0.4 * math.cos(yaw), L["y"] + 0.4 * math.sin(yaw)), xytext=(L["x"], L["y"]),
                    arrowprops={"arrowstyle": "->", "lw": 1.5})
        ax.text(L["x"] + 0.08, L["y"] - 0.15, "launch", fontsize=7)
    for cp in cfg.get("checkpoints", []):
        for opt in cp.get("options", [cp]):
            if "line" in opt:
                (x0, y0), (x1, y1) = opt["line"]
                ax.plot([x0, x1], [y0, y1], "-", color="lime", lw=2.5)
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                dx, dy = opt.get("direction", (0, 0))
                ax.annotate("", xy=(mx + 0.35 * dx, my + 0.35 * dy), xytext=(mx, my),
                            arrowprops={"arrowstyle": "->", "color": "green", "lw": 1.5})
                label = cp["id"] + (f" {opt['name']}" if opt is not cp else "")
                ax.text(mx + 0.05, my + 0.05, label, fontsize=7, color="darkgreen", weight="bold")
            if "centre" in opt and "outer_radius" in opt:
                ax.add_patch(plt.Circle(opt["centre"], opt["outer_radius"], fill=False, color="lime", lw=2))
                ax.text(opt["centre"][0], opt["centre"][1], cp["id"], fontsize=7, color="darkgreen",
                        weight="bold", ha="center")
    for name, z in cfg.get("zones", {}).items():
        if "centre" in z and "radius" in z:
            ax.add_patch(plt.Circle(z["centre"], z["radius"], fill=False, color="green", ls=":", lw=1.2))


def plot_height(cmap: CourseMap, cfg: dict[str, Any], h: float, out: Path) -> None:
    lo, hi = cmap.bounds
    step = cmap.cell / 2
    xs = np.arange(lo[0], hi[0] + step / 2, step)
    ys = np.arange(lo[1], hi[1] + step / 2, step)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, h)], axis=1)
    lim = cfg.get("limits", {})
    keep_out = lim.get("drone_radius", 0.0) + lim.get("clearance_margin", 0.0)
    fields = [f for f in ("planning", "localization") if f in cmap.fields] or list(cmap.fields)
    fig, axes = plt.subplots(1, len(fields), figsize=(7.5 * len(fields), 7.5), squeeze=False)
    for ax, field in zip(axes[0], fields):
        D = cmap.distance(pts, field).reshape(X.shape)
        im = ax.imshow(np.clip(D, -1, 1), extent=(xs[0], xs[-1], ys[0], ys[-1]), origin="lower",
                       cmap="RdBu", vmin=-1, vmax=1)
        # 2 cm rather than 0: a thin shell (the tube, the net) is ~0 on its voxels and never negative
        ax.contour(X, Y, D, levels=[SURFACE_LINE_M], colors="k", linewidths=0.8)
        if keep_out > 0:
            ax.contour(X, Y, D, levels=[keep_out], colors="orange", linewidths=0.9, linestyles="--")
        draw_course(ax, cfg)
        ax.set_title(f"{field} field at z = {h:.2f} m  (orange: {keep_out:.2f} m clearance)")
        ax.set_xlabel("x (m, east)")
        ax.set_ylabel("y (m, north)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="signed distance (m)")
    fig.suptitle(f"{cfg.get('course', {}).get('name', '')}  ·  {cmap.meta.get('mesh', '')}  ·  "
                 f"built {cmap.meta.get('built_utc', '?')}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", nargs="?", default=str(ROOT / "maps" / "course.yaml"))
    ap.add_argument("--out-dir", help="default: plots/ next to the config")
    args = ap.parse_args(argv)
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cmap = CourseMap.load(cfg_path.parent / cfg["build"]["output"])
    out_dir = Path(args.out_dir) if args.out_dir else cfg_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for h in cfg["build"].get("plot_heights", [1.0]):
        out = out_dir / f"sdf_z{h:.2f}.png"
        plot_height(cmap, cfg, float(h), out)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
