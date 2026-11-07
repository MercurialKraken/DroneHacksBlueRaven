#!/usr/bin/env python3
"""SIM-ONLY: compare record_eval.py output against ground truth. Requires numpy; matplotlib is optional (for drift.png).

  python3 tools/sim_eval/analyze.py runs/flight1
  python3 tools/sim_eval/analyze.py runs/flight1 --vio runs/flight1/replay_1x/vio.csv --label 1x

Reports:
  - drift of cuVSLAM and PX4 against truth (yaw + translation aligned at the first common sample)
  - covariance cuVSLAM claims vs the error it actually has (a suggested position_floor)
  - when inertial init (gravity) happened relative to takeoff
  - low-feature stretches and unhealthy stretches, with where the drone was
  - vision latency into PX4
  - a clock check between truth and ROS time
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    Table = dict[str, np.ndarray]

LOW_FEATURES = 50          # tracked features per frame below which we call it "visually weak"
TAKEOFF_RISE_M = 0.15


def load(path: Path) -> Table | None:
    if not path.is_file():
        return None
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return None
    head = rows[0]
    body = [r for r in rows[1:] if len(r) == len(head)]          # drop a truncated last line
    if not body:
        return None
    data = np.array(body, dtype=float)
    return {k: data[:, i] for i, k in enumerate(head)}


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """(N, 4) xyzw -> (N, 3, 3)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def yaw_of(R: np.ndarray) -> np.ndarray:
    return np.arctan2(R[:, 1, 0], R[:, 0, 0])


def wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def truth_at(truth: Table, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tt = truth["t_ns"]
    p = np.stack([np.interp(t, tt, truth[k]) for k in "xyz"], axis=1)
    idx = np.clip(np.searchsorted(tt, t), 0, len(tt) - 1)
    R = quat_to_R(np.stack([truth[k][idx] for k in ("qx", "qy", "qz", "qw")], axis=1))
    return p, R


def rise_time(t: np.ndarray, z: np.ndarray) -> float | None:
    up = np.nonzero(z > z[0] + TAKEOFF_RISE_M)[0]
    return float(t[up[0]]) if len(up) else None


def intervals(t: np.ndarray, mask: np.ndarray, gap_s: float = 0.5) -> list[tuple[float, float]]:
    """Group the times where mask is true into [start, end] intervals (seconds)."""
    ts = t[mask] * 1e-9
    if not len(ts):
        return []
    out, start, prev = [], ts[0], ts[0]
    for x in ts[1:]:
        if x - prev > gap_s:
            out.append((start, prev))
            start = x
        prev = x
    out.append((start, prev))
    return out


def where(truth: Table, t_s: float) -> str:
    p, _ = truth_at(truth, np.array([t_s * 1e9]))
    return f"ENU ({p[0, 0]:+.2f}, {p[0, 1]:+.2f}, {p[0, 2]:+.2f})"


def evaluate(name: str, est: Table, truth: Table, out_dir: Path) -> dict[str, np.ndarray] | None:
    tt = truth["t_ns"]
    keep = (est["t_ns"] >= tt[0]) & (est["t_ns"] <= tt[-1])
    if keep.sum() < 10:
        print(f"[{name}] fewer than 10 samples overlap the truth log; skipped")
        return None
    t = est["t_ns"][keep]
    p_est = np.stack([est[k][keep] for k in "xyz"], axis=1)
    R_est = quat_to_R(np.stack([est[k][keep] for k in ("qx", "qy", "qz", "qw")], axis=1))
    p_tru, R_tru = truth_at(truth, t)

    # Both frames are gravity-aligned, so align yaw + translation on the first common sample.
    dyaw = float(yaw_of(R_tru[:1])[0] - yaw_of(R_est[:1])[0])
    c, s = math.cos(dyaw), math.sin(dyaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    p_w = p_est @ Rz.T
    p_w += p_tru[0] - p_w[0]
    e = p_w - p_tru
    e_h = np.hypot(e[:, 0], e[:, 1])
    e_yaw = np.degrees(wrap(yaw_of(R_est) + dyaw - yaw_of(R_tru)))
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p_tru, axis=0), axis=1))])
    dur = (t[-1] - t[0]) * 1e-9

    print(f"\n[{name}] {len(t)} samples, {dur:.1f} s, {dist[-1]:.2f} m flown")
    print(f"  horizontal error  final {e_h[-1]:.3f} m   max {e_h.max():.3f} m   RMS {np.sqrt(np.mean(e_h ** 2)):.3f} m"
          + (f"   final/distance {100 * e_h[-1] / dist[-1]:.2f} %" if dist[-1] > 0.5 else ""))
    print(f"  vertical error    final {e[-1, 2]:+.3f} m   max |{np.abs(e[:, 2]).max():.3f}| m")
    print(f"  yaw error         final {e_yaw[-1]:+.2f} deg   max |{np.abs(e_yaw).max():.2f}| deg"
          f"   RMS {np.sqrt(np.mean(e_yaw ** 2)):.2f} deg")
    marks = [m for m in (1, 2, 5, 10, 20) if m <= dist[-1]]
    if marks:
        print("  horizontal error after " + ", ".join(
            f"{m} m: {e_h[np.searchsorted(dist, m)]:.3f}" for m in marks))
    if "var_x" in est:
        sig = np.sqrt(np.median(est["var_x"][keep] + est["var_y"][keep]) / 2)
        rms = float(np.sqrt(np.mean(e_h ** 2) / 2))
        print(f"  claimed per-axis sigma {1000 * sig:.1f} mm vs actual per-axis RMS {1000 * rms:.1f} mm"
              + (f"  -> {rms / sig:.0f}x overconfident" if sig > 0 else ""))
        print(f"  suggested vision_bridge.yaml position_floor: [{rms ** 2:.2e}, {rms ** 2:.2e}, {rms ** 2:.2e}]"
              f"  (PX4 already floors at EKF2_EVP_NOISE 0.05 m -> 2.5e-03)")
    with open(out_dir / f"errors_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ns", "dist_m", "ex", "ey", "ez", "eyaw_deg"])
        w.writerows(np.column_stack([t, dist, e, e_yaw]).tolist())
    return {"t": t, "dist": dist, "e_h": e_h, "p_w": p_w, "p_tru": p_tru}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="directory with truth.csv from record_eval.py")
    ap.add_argument("--vio", type=Path, help="alternative vio.csv (e.g. from a replay)")
    ap.add_argument("--label", default=None, help="name for this vio variant in outputs")
    args = ap.parse_args()
    run = args.run
    truth = load(run / "truth.csv")
    if truth is None:
        raise SystemExit(f"no truth.csv in {run}")
    vio_path = args.vio or run / "vio.csv"
    label = args.label or ("vio" if args.vio is None else f"vio_{vio_path.parent.name}")
    out_dir = vio_path.parent
    vio, px4 = load(vio_path), load(run / "px4.csv")
    t0 = truth["t_ns"][0]
    print(f"truth: {len(truth['t_ns'])} samples, {(truth['t_ns'][-1] - t0) * 1e-9:.1f} s")

    # Clock check: truth and ROS stamps must share the sim clock.
    t_up = rise_time(truth["t_ns"], truth["z"])
    if t_up is not None and px4 is not None:
        t_up_px4 = rise_time(px4["t_ns"], px4["z"])
        if t_up_px4 is not None:
            print(f"clock check: PX4 sees takeoff {(t_up_px4 - t_up) * 1e-6:+.0f} ms after truth "
                  "(expect tens of ms: /refnav/odom is stamped at receipt)")
    print("takeoff: " + ("not detected" if t_up is None else f"{(t_up - t0) * 1e-9:.1f} s into the log"))

    results = {}
    for name, est in ((label, vio), ("px4", px4 if args.vio is None else None)):
        if est is not None:
            r = evaluate(name, est, truth, out_dir)
            if r is not None:
                results[name] = r

    grav = load(vio_path.parent / "gravity.csv")
    if grav is None:
        print("\ninertial init: NO gravity estimate in this log (cuVSLAM never initialised its IMU)")
    else:
        tg = grav["t_ns"][0]
        rel = "" if t_up is None else f", {(tg - t_up) * 1e-9:+.1f} s relative to takeoff"
        print(f"\ninertial init: first gravity at {(tg - t0) * 1e-9:.1f} s{rel}; "
              f"gravity = ({grav['gx'][-1]:+.2f}, {grav['gy'][-1]:+.2f}, {grav['gz'][-1]:+.2f})")

    feats = load(vio_path.parent / "features.csv")
    if feats is not None:
        n = feats["n_obs"]
        print(f"features/frame: p5 {np.percentile(n, 5):.0f}  p50 {np.median(n):.0f}  min {n.min():.0f}")
        for a, b in intervals(feats["t_recv_ns"], n < LOW_FEATURES):
            print(f"  below {LOW_FEATURES} from {a - t0 * 1e-9:.1f} to {b - t0 * 1e-9:.1f} s, "
                  f"at {where(truth, a)} -> {where(truth, b)}")

    health = load(run / "health.csv")
    if health is not None and args.vio is None:
        ok = health["ok"] > 0.5
        first_ok = np.argmax(ok) if ok.any() else len(ok)
        bad = np.zeros_like(ok)
        bad[first_ok:] = ~ok[first_ok:]
        spans = intervals(health["t_recv_ns"], bad)
        print(f"estimator health: {'never healthy' if not ok.any() else f'{len(spans)} unhealthy stretch(es) after first healthy'}")
        for a, b in spans:
            print(f"  unhealthy {a - t0 * 1e-9:.1f}-{b - t0 * 1e-9:.1f} s at {where(truth, a)}")

    ev = load(run / "ev.csv")
    if ev is not None and args.vio is None:
        lat = (ev["timestamp_us"] - ev["timestamp_sample_us"]) * 1e-3
        print(f"vision -> PX4 latency (capture to send): p50 {np.median(lat):.0f} ms  "
              f"p95 {np.percentile(lat, 95):.0f} ms  max {lat.max():.0f} ms")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\n(matplotlib not installed: skipped the plot; error tables are in {out_dir}/errors_*.csv)")
        return
    if results:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6))
        first = next(iter(results.values()))
        a1.plot(first["p_tru"][:, 0], first["p_tru"][:, 1], "k-", lw=2, label="truth")
        for name, r in results.items():
            a1.plot(r["p_w"][:, 0], r["p_w"][:, 1], "--", label=name)
            a2.plot(r["dist"], r["e_h"], label=name)
        a1.set_aspect("equal"); a1.set_xlabel("east (m)"); a1.set_ylabel("north (m)"); a1.legend(); a1.grid(alpha=.3)
        a2.set_xlabel("distance flown (m)"); a2.set_ylabel("horizontal error (m)"); a2.legend(); a2.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"drift_{label}.png", dpi=110)
        print(f"\nplot: {out_dir / f'drift_{label}.png'}")


if __name__ == "__main__":
    main()
