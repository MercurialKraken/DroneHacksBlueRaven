"""DEBUG ONLY: animate the raw MPPI controller on a fixed clearance grid (the tests/test_mppi.py worlds).

    python tools/mppi_vis_grid.py <scene> <out.gif> [obs_len]      scene: corner | pillar | corridor

Grey = inside obstacles (clearance < 0); dashed grey = where the drone's centre would touch (clearance = r_drone);
dashed blue = reference path; thin lines = sampled rollouts (bright = heavy, red = crashed); magenta = chosen plan;
circle = drone (r_drone). A frame titled BLOCKED (red) is one where every candidate crashed and MPPI commanded zero.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import matplotlib                                                                    # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                      # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter                        # noqa: E402
from matplotlib.collections import LineCollection                                   # noqa: E402

from nav.mppi import Mppi, MppiParams                                                # noqa: E402
from test_mppi import corridor_map, l_corridor_map, l_path, straight_path           # noqa: E402

SCENES = {
    "corner": lambda: (l_corridor_map(), *l_path(), ((-1.0, 5.5), (-1.0, 5.5))),
    "pillar": lambda: (corridor_map(width=2.2, pillar=(3.0, 0.0, 0.12)), *straight_path(), ((-1.0, 7.0), (-1.5, 1.5))),
    "corridor": lambda: (corridor_map(), *straight_path(), ((-1.0, 7.0), (-1.0, 1.0))),
}


def main() -> None:
    scene, out = sys.argv[1], sys.argv[2]
    ol = float(sys.argv[3]) if len(sys.argv) > 3 else MppiParams().obs_len
    grid, P, vref, ((x0, x1), (y0, y1)) = SCENES[scene]()
    m = Mppi(grid, P, vref, MppiParams(obs_len=ol))
    m.debug_samples = 160
    p = m.p
    a = math.exp(-p.dt / p.tau)
    b = p.tau * (1.0 - a)
    x, v = np.zeros(2), np.zeros(2)
    frames = []
    for _ in range(int(45.0 / p.dt)):
        r = m.step(x, v)
        frames.append((x.copy(), r))
        u = np.array(r.u)
        x = x + u * p.dt + (v - u) * b
        v = u + (v - u) * a
        if r.s >= m.length - 0.05 and np.linalg.norm(v) < 0.03:
            break
    track = np.array([f[0] for f in frames])
    blocked = [k for k, f in enumerate(frames) if f[1].blocked]
    D = grid.dist
    X, Y = np.meshgrid(grid.origin[0] + grid.cell * np.arange(D.shape[0]),
                       grid.origin[1] + grid.cell * np.arange(D.shape[1]), indexing="ij")
    fig, ax = plt.subplots(figsize=(7, 6.5), dpi=90)

    def draw(n: int) -> None:
        ax.clear()
        xk, r = frames[n]
        ax.contourf(X, Y, D, levels=[-10, 0], colors=["#9a9a9a"])
        ax.contour(X, Y, D, levels=[p.r_drone], colors=["#9a9a9a"], linestyles="--", linewidths=0.8)
        ax.plot(m.px, m.py, "--", color="#3060c0", lw=1)
        if len(r.samples):
            wv = r.sample_w / max(float(r.sample_w.max()), 1e-12)
            order = np.argsort(wv)
            cols = [((0.85, 0.2, 0.2, 0.35) if r.sample_crashed[k] else
                     plt.cm.viridis(0.15 + 0.85 * wv[k], alpha=0.15 + 0.75 * wv[k])) for k in order]
            ax.add_collection(LineCollection([r.samples[k] for k in order], colors=cols, linewidths=0.8))
            ax.plot(r.reference[:, 0], r.reference[:, 1], color="#20a040", lw=1.5)
        ax.plot(r.plan[:, 0], r.plan[:, 1], color="#d020c0", lw=2.5)
        ax.plot(track[: n + 1, 0], track[: n + 1, 1], color="#1f4fa0", lw=1.5)
        ax.add_patch(plt.Circle(xk, p.r_drone, fill=False, color="#1f4fa0", lw=1.5))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        clr = float(m.clearance(xk[None])[0])
        ax.set_title(f"{scene} obs_len {ol}  t {n * p.dt:4.1f} s  {'BLOCKED' if r.blocked else 'ok':7s}  clearance {clr:.3f} m"
                     f"  crash {100 * r.crash_frac:.0f}%", fontsize=9, family="monospace",
                     color="red" if r.blocked else "black")

    FuncAnimation(fig, draw, frames=range(0, len(frames), 2), interval=200).save(out, writer=PillowWriter(fps=8))
    if blocked:
        draw(blocked[0])
        fig.savefig(str(Path(out).with_suffix("")) + "_first_blocked.png")
        if blocked[0] > 5:
            draw(blocked[0] - 5)
            fig.savefig(str(Path(out).with_suffix("")) + "_before_blocked.png")
    print(f"{out}: {len(frames)} steps, blocked steps {len(blocked)} {blocked[:12]}, "
          f"min clearance {float(m.clearance(track).min()):.3f} m (r_drone {p.r_drone}), end s {frames[-1][1].s:.2f}/{m.length:.2f}")


if __name__ == "__main__":
    main()
