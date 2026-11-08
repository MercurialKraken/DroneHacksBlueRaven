"""Time one MPPI navigation cycle on this machine (run it on the Jetson): depth -> grid -> distance -> MPPI step.

    python3 tools/time_mppi.py [samples] [horizon]

Synthetic but representative: a 1.2 m corridor with a right turn, 320x200 depth at stride 2 (~6k valid points per
frame after the ~35% valid fraction), K=1024, H=30 by default. Budget at 10 Hz: 100 ms for everything.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from nav.local_nav import LocalNav                                                  # noqa: E402
from nav.mppi import MppiParams                                                     # noqa: E402

try:
    import scipy  # noqa: F401
    EDT = "scipy"
except ImportError:
    EDT = "fallback (no scipy: slower)"


def main() -> None:
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    rng = np.random.default_rng(0)
    nav = LocalNav([(4.0, 0.0), (4.0, -3.5)], (0.0, 0.0), 0.0, mppi=MppiParams(samples=K, horizon=H))
    # walls of the corridor as point clouds at flight height (what a depth frame of it looks like)
    ys = np.r_[np.full(400, 0.6), np.full(400, -0.6)]
    xs = np.r_[np.linspace(0.3, 4.6, 400), np.linspace(0.3, 3.4, 400)]
    wall = np.c_[xs, ys, np.full(800, 0.7)]
    front = np.c_[np.full(300, 4.6), np.linspace(-0.6, 0.6, 300), np.full(300, 0.7)]
    base = np.r_[wall, front]
    pts = np.repeat(base, 8, axis=0) + rng.normal(0, 0.02, (len(base) * 8, 3))     # ~8.8k points
    cam = np.array([0.1, 0.0, 0.7])
    times = {"grid": [], "distance": [], "mppi": []}
    for k in range(40):
        t0 = time.perf_counter()
        nav.add_depth(pts, cam, 0.7, 0.1 * k)
        t1 = time.perf_counter()
        nav.mppi.set_grid(nav.grid.grid2d())
        nav._dirty = False
        t2 = time.perf_counter()
        nav.step((0.5 + 0.01 * k, 0.0), (0.3, 0.0), 0.1 * k)
        t3 = time.perf_counter()
        if k >= 5:                                                                  # skip warm-up
            times["grid"].append(t1 - t0)
            times["distance"].append(t2 - t1)
            times["mppi"].append(t3 - t2)
    tot = [a + b + c for a, b, c in zip(times["grid"], times["distance"], times["mppi"])]
    print(f"K={K} H={H} points/frame={len(pts)} distance transform: {EDT}")
    for name, v in times.items():
        print(f"  {name:9s} median {1e3 * np.median(v):6.1f} ms   p90 {1e3 * np.percentile(v, 90):6.1f} ms")
    print(f"  TOTAL     median {1e3 * np.median(tot):6.1f} ms   p90 {1e3 * np.percentile(tot, 90):6.1f} ms"
          f"   (budget 100 ms at 10 Hz)")


if __name__ == "__main__":
    main()
