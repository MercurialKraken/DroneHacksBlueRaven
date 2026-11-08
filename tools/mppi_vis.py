"""DEBUG ONLY (never on the drone): animate MPPI flying a simulated course with a fake depth camera.

    python tools/mppi_vis.py [out.gif] [--noisy] [--scene new_start|cp1]

Per frame: true walls (grey) and nets (blue, invisible to depth), the low cone (orange, flown over), the depth grid's
obstacle cells (black), the waypoint lines (dashed), the re-anchored reference (green), sampled rollouts coloured by
their weight (bright = heavy; crashed ones red), the chosen plan (magenta), the drone (circle = 0.28 m radius, line =
heading = camera direction) and its track.
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

from nav.local_nav import LocalNav                                                   # noqa: E402
from sim_world import Drone                                                          # noqa: E402
from test_local_nav import clip_world, cp1_world, new_start_world                   # noqa: E402

SCENES = {
    "new_start": dict(world=new_start_world, start=(-2.515, 0.914), yaw=math.pi / 2,
                      wps=[(-2.45, 2.05), (-1.13, 2.30), (-0.49, 2.52)], view=((-3.3, 0.3), (0.3, 3.2))),
    "clip": dict(world=clip_world, start=(0.0, 0.0), yaw=0.0, wps=[(6.5, 0.0)], view=((-0.8, 7.6), (-1.4, 1.4))),
    "cp1": dict(world=cp1_world, start=(0.0, 0.0), yaw=0.0, wps=[(4.3, 0.25), (3.8, -3.6)],
                view=((-1.0, 5.2), (-5.2, 1.2))),
}


def run(scene: str, noisy: bool):
    sc = SCENES[scene]
    w = sc["world"]()
    nav = LocalNav(sc["wps"], sc["start"], sc["yaw"])
    nav.mppi.debug_samples = 160
    d = Drone(np.array(sc["start"], float), sc["yaw"])
    rng = np.random.default_rng(3)
    frames = []
    for k in range(700):
        t = 0.1 * k
        cam = d.cam()
        pts = w.depth_points(cam, d.yaw, noise_k=0.006 if noisy else 0.0, dropout=0.65 if noisy else 0.0,
                             outliers=0.01 if noisy else 0.0, rng=rng)
        nav.add_depth(pts, cam, d.z, t)
        c = nav.step(d.x, d.v, t)
        occ = nav.grid.occupied()
        ii, jj = np.nonzero(occ)
        frames.append(dict(t=t, x=d.x.copy(), yaw=d.yaw, v=float(np.hypot(*d.v)), cmd=c,
                           occ=np.c_[nav.grid.origin[0] + ii * nav.grid.p.cell, nav.grid.origin[1] + jj * nav.grid.p.cell]))
        if c.state == "ARRIVED":
            break
        d.step((c.vx, c.vy), c.yaw, 0.1)
    return w, nav, frames, sc


def render(scene: str, noisy: bool, out: str) -> None:
    w, nav, frames, sc = run(scene, noisy)
    (x0, x1), (y0, y1) = sc["view"]
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=90)
    X, Y = np.meshgrid(w.xs, w.ys, indexing="ij")
    track = np.array([f["x"] for f in frames])
    wp = np.r_[np.array(sc["start"])[None], np.array(sc["wps"])]

    def draw(n: int):
        ax.clear()
        f = frames[n]
        tall = w.h > 0.35
        ax.scatter(X[tall], Y[tall], s=1, c="#9a9a9a", marker="s", linewidths=0)
        ax.scatter(X[w.net], Y[w.net], s=1, c="#9cc3e6", marker="s", linewidths=0)
        low = (w.h > 0) & ~tall
        ax.scatter(X[low], Y[low], s=2, c="#f0a030", marker="s", linewidths=0)
        if len(f["occ"]):
            ax.scatter(f["occ"][:, 0], f["occ"][:, 1], s=3, c="k", marker="s", linewidths=0)
        ax.plot(wp[:, 0], wp[:, 1], ":", color="#3060c0", lw=1, marker="o", ms=4)
        ax.plot(nav.path[:, 0], nav.path[:, 1], "--", color="#3060c0", lw=1)
        r = f["cmd"].mppi
        if r is not None and len(r.samples):
            wv = r.sample_w / max(float(r.sample_w.max()), 1e-12)
            order = np.argsort(wv)
            segs = [r.samples[k] for k in order]
            cols = [((0.85, 0.2, 0.2, 0.35) if r.sample_crashed[k] else
                     plt.cm.viridis(0.15 + 0.85 * wv[k], alpha=0.15 + 0.75 * wv[k])) for k in order]
            ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.8))
            ax.plot(r.reference[:, 0], r.reference[:, 1], color="#20a040", lw=1.5)
            ax.plot(r.plan[:, 0], r.plan[:, 1], color="#d020c0", lw=2.5)
        ax.plot(track[: n + 1, 0], track[: n + 1, 1], color="#1f4fa0", lw=1.5)
        ax.add_patch(plt.Circle(f["x"], nav.mppi.p.r_drone, fill=False, color="#1f4fa0", lw=1.5))
        ax.plot([f["x"][0], f["x"][0] + 0.4 * math.cos(f["yaw"])], [f["x"][1], f["x"][1] + 0.4 * math.sin(f["yaw"])],
                color="#1f4fa0", lw=2)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        extra = ""
        if r is not None:
            extra = f"  ESS {r.ess:.0f}  side {r.side:+d}  crash {100 * r.crash_frac:.0f}%  {r.ms:.0f} ms"
        ax.set_title(f"t {f['t']:4.1f} s  {f['cmd'].state:7s}  |v| {f['v']:.2f} m/s{extra}", fontsize=9, family="monospace")

    step = 2
    anim = FuncAnimation(fig, draw, frames=range(0, len(frames), step), interval=200)
    anim.save(out, writer=PillowWriter(fps=8))
    # key frames as PNGs next to the GIF
    clear = w.clearance(track, min_height=0.35)
    k_min = int(np.argmin(clear))
    for name, n in (("approach", max(k_min - 25, 0)), ("tightest", k_min), ("end", len(frames) - 1)):
        draw(n)
        fig.savefig(str(Path(out).with_suffix("")) + f"_{name}.png")
    print(f"{out}: {len(frames)} steps, {frames[-1]['cmd'].state}, min clearance {clear.min():.3f} m at t={frames[k_min]['t']:.1f} s")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    scene = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--scene=")), "new_start")
    out = next((a for a in args if a.endswith(".gif")), f"mppi_{scene}.gif")
    render(scene, "--noisy" in sys.argv, out)
