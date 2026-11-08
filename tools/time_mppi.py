"""Time one MPPI navigation cycle on this machine (run it on the Jetson): depth -> grid -> distance -> MPPI step.
CPU (numpy) vs GPU (CuPy), over a sweep of sample counts.

    python3 tools/time_mppi.py                 # CPU and (if CuPy works) GPU, K = 1024 4096 16384, H = 30
    python3 tools/time_mppi.py --cpu           # CPU only
    python3 tools/time_mppi.py --gpu 1024 8192 # GPU only, these K

Synthetic but representative: a 1.2 m corridor with a right turn, ~8.8k depth points per frame (a 640x400 frame at
stride 4 after the ~35 % valid fraction), H=30 steps of 0.1 s. Budget at 10 Hz: 100 ms for everything. The depth
grid and its distance transform stay on the CPU either way (small); MPPI's rollouts move to the GPU with --gpu.
The FIRST GPU cycles include CuPy's kernel compilation: they're skipped as warm-up.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")
from nav.local_nav import FLIGHT_MPPI, LocalNav                                     # noqa: E402
from dataclasses import replace                                                     # noqa: E402

try:
    import scipy  # noqa: F401
    EDT = "scipy"
except ImportError:
    EDT = "fallback (no scipy: slower)"


def bench(K: int, H: int, xp, label: str, n: int = 40, warm: int = 8) -> None:
    rng = np.random.default_rng(0)
    nav = LocalNav([(4.0, 0.0), (4.0, -3.5)], (0.0, 0.0), 0.0, mppi=replace(FLIGHT_MPPI, samples=K, horizon=H), xp=xp)
    ys = np.r_[np.full(400, 0.6), np.full(400, -0.6)]
    xs = np.r_[np.linspace(0.3, 4.6, 400), np.linspace(0.3, 3.4, 400)]
    base = np.r_[np.c_[xs, ys, np.full(800, 0.7)], np.c_[np.full(300, 4.6), np.linspace(-0.6, 0.6, 300), np.full(300, 0.7)]]
    pts = np.repeat(base, 8, axis=0) + rng.normal(0, 0.02, (len(base) * 8, 3))
    cam = np.array([0.1, 0.0, 0.7])
    T = {"grid": [], "distance": [], "mppi": []}
    for k in range(n + warm):
        t0 = time.perf_counter()
        nav.add_depth(pts, cam, 0.7, 0.1 * k)
        t1 = time.perf_counter()
        nav.mppi.set_grid(nav.grid.grid2d())
        nav._dirty = False
        t2 = time.perf_counter()
        nav.step((0.5 + 0.01 * k, 0.0), (0.3, 0.0), 0.1 * k)
        if xp is not np:
            xp.cuda.Stream.null.synchronize()
        t3 = time.perf_counter()
        if k >= warm:
            T["grid"].append(t1 - t0)
            T["distance"].append(t2 - t1)
            T["mppi"].append(t3 - t2)
    tot = np.array(T["grid"]) + np.array(T["distance"]) + np.array(T["mppi"])
    print(f"{label:4s} K={K:6d} H={H}  grid {1e3 * np.median(T['grid']):5.1f}  distance {1e3 * np.median(T['distance']):4.1f}  "
          f"MPPI {1e3 * np.median(T['mppi']):6.1f} (p90 {1e3 * np.percentile(T['mppi'], 90):6.1f})  "
          f"TOTAL median {1e3 * np.median(tot):6.1f} ms  p90 {1e3 * np.percentile(tot, 90):6.1f} ms", flush=True)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    Ks = [int(a) for a in args] or [1024, 4096, 16384]
    H = 30
    do_cpu = "--gpu" not in sys.argv
    do_gpu = "--cpu" not in sys.argv
    print(f"distance transform: {EDT}; budget 100 ms at 10 Hz")
    if do_cpu:
        for K in Ks:
            bench(K, H, np, "CPU")
    if do_gpu:
        try:
            import cupy as cp
            cp.zeros(1).sum()
        except Exception as e:                                                      # noqa: BLE001
            print(f"GPU: CuPy not usable here ({type(e).__name__}: {e})")
            return
        for K in Ks:
            try:
                bench(K, H, cp, "GPU")
            except Exception as e:                                                  # noqa: BLE001
                print(f"GPU K={K}: FAILED {type(e).__name__}: {e}")
                raise


if __name__ == "__main__":
    main()
