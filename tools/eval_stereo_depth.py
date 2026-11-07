"""Replay a T4 bag's stereo pairs through onboard.depth.stereo_depth and score the depth. Read-only, no ROS.

    python3 tools/eval_stereo_depth.py BAG.mcap [--every 10] [--scale 0.5]

Two scores per frame (only while flying, truth height > 0.3 m):
  * against the sim's own depth image (/camera/depth/image_raw, also SGBM): agreement where both are valid;
  * against the COURSE: every depth point placed with the sim's true pose (/sim/truth/odometry) and the
    camera extrinsic should lie on a surface of the course map ('planning' field: walls, nets, floor).
    The same score is given for the sim's depth, so the two are compared on equal terms.
Also the time per frame on THIS machine (not the Jetson's).
"""
from __future__ import annotations

import argparse, struct, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import eval_map_align as E  # noqa: E402  (its MCAP reader and CDR helpers)
from nav.course_map import CourseMap  # noqa: E402
from onboard.depth.stereo_depth import CameraModel, DepthParams, StereoDepthPipeline, to_gray  # noqa: E402
from refnav.depth import decode_depth, quat_to_mat  # noqa: E402

LEFT, RIGHT, LINFO, RINFO = "/stereo/left/image", "/stereo/right/image", "/stereo/left/camera_info", "/stereo/right/camera_info"
SIMD, TRUTH = "/camera/depth/image_raw", "/sim/truth/odometry"
R_BC = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)      # base_link (FLU) <- oak_left optical
T_BC = np.array([0.116, 0.037, 0.029])                             # from the bags' /tf_static


def stamp(b):
    s, ns = struct.unpack_from("<iI", b, 4)
    return s * 1_000_000_000 + ns


def image(b):
    o = E.skip_str(b, 12); o = E.a4(o); h, w = struct.unpack_from("<II", b, o); enc_end = E.skip_str(b, o + 8)
    o2 = E.a4(o + 8); el = struct.unpack_from("<I", b, o2)[0]; enc = bytes(b[o2 + 4:o2 + 4 + el - 1]).decode()
    o = E.a4(enc_end + 1); step = struct.unpack_from("<I", b, o)[0]; o += 4; m = struct.unpack_from("<I", b, o)[0]
    return enc, h, w, step, bytes(b[o + 4:o + 4 + m])


def info(b):
    o = E.skip_str(b, 12); o = E.a4(o); h, w = struct.unpack_from("<II", b, o); o = E.skip_str(b, o + 8)
    o = E.a4(o); nd = struct.unpack_from("<I", b, o)[0]; o = E.a8(o + 4); D = struct.unpack_from(f"<{nd}d", b, o); o += 8 * nd
    K = struct.unpack_from("<9d", b, o); R = struct.unpack_from("<9d", b, o + 72); P = struct.unpack_from("<12d", b, o + 144)
    return CameraModel.from_lists(w, h, K, D, R, P)


def course_residual(cm, depth, K, pose, stride=4, rmin=0.4, rmax=3.5):
    """|distance to the nearest course surface| of the depth points placed with the true pose."""
    fx, fy, cx, cy = K
    d = depth[::stride, ::stride]
    v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]] * stride
    ok = np.isfinite(d) & (d > rmin) & (d < rmax)
    z = d[ok]
    if not z.size:
        return np.array([])
    pc = np.stack([(u[ok] - cx) / fx * z, (v[ok] - cy) / fy * z, z], 1)
    t, q = pose
    Rmb = quat_to_mat(q[3], q[0], q[1], q[2])
    P = (pc @ R_BC.T + T_BC) @ Rmb.T + np.asarray(t)
    r = np.abs(cm.distance(P, "planning"))
    return r[np.isfinite(r)]                                         # outside the map grid: no distance


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("bag"); ap.add_argument("--every", type=int, default=10); ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--min-m", type=float, default=0.35); ap.add_argument("--max-m", type=float, default=6.0)
    a = ap.parse_args(argv)
    E.KEEP = {LEFT, RIGHT, LINFO, RINFO, SIMD, TRUTH}
    cm = CourseMap.load(ROOT / "maps" / "course_sdf.npz")
    pipe = None; Lm = Rm = None; pose = None
    lefts, rights, sims = {}, {}, {}
    n_pair = 0; rows = []
    for topic, t, b in E.stream(a.bag):
        if topic == LINFO and Lm is None:
            Lm = info(b)
        elif topic == RINFO and Rm is None:
            Rm = info(b)
        elif topic == TRUTH:
            o = E.skip_str(b, 12); o = E.skip_str(b, o); o = E.a8(o)
            p = struct.unpack_from("<7d", b, o); pose = (p[:3], p[3:7])
        elif topic in (LEFT, RIGHT, SIMD):
            (lefts if topic == LEFT else rights if topic == RIGHT else sims)[stamp(b)] = (b, pose)
        if pipe is None and Lm is not None and Rm is not None:
            pipe = StereoDepthPipeline(Lm, Rm, a.scale, DepthParams(min_m=a.min_m, max_m=a.max_m))
            print(f"calibration: {pipe.rect.source}, f {pipe.rect.f:.1f} px, baseline {pipe.rect.baseline_m:.4f} m, "
                  f"{pipe.size[0]}x{pipe.size[1]}, disparities {pipe.matcher.num_disp}")
        for s in [s for s in lefts if s in rights and s in sims]:
            (bl, pose_l), (br, _), (bs, _) = lefts.pop(s), rights.pop(s), sims.pop(s)
            n_pair += 1
            if n_pair % a.every or pipe is None or pose_l is None or pose_l[0][2] < 0.3:
                continue
            gl, gr = to_gray(*image(bl)), to_gray(*image(br))
            t0 = time.perf_counter(); z = pipe(gl, gr); dt = time.perf_counter() - t0
            sd = decode_depth(*image(bs))
            w, h = pipe.size
            import cv2
            sd_s = cv2.resize(sd, (w, h), interpolation=cv2.INTER_NEAREST)
            both = np.isfinite(z) & np.isfinite(sd_s) & (sd_s < 3.5)
            rel = np.abs(z[both] - sd_s[both]) / sd_s[both] if both.any() else np.array([np.nan])
            K = (pipe.K[0, 0], pipe.K[1, 1], pipe.K[0, 2], pipe.K[1, 2])
            Ks = (Lm.K[0, 0], Lm.K[1, 1], Lm.K[0, 2], Lm.K[1, 2])
            res_m = course_residual(cm, z, K, pose_l, stride=2)
            res_s = course_residual(cm, sd, Ks, pose_l, stride=4)
            band = (slice(int(0.1 * h), int(0.9 * h)), slice(int(0.15 * w), int(0.95 * w)))
            rows.append(dict(dt=dt, valid=np.isfinite(z[band]).mean(), valid_sim=np.isfinite(sd_s[band]).mean(),
                             rel_med=np.median(rel), rel_10=(rel < 0.10).mean(),
                             res_m=res_m, res_s=res_s))
        for d in (lefts, rights, sims):                       # drop anything unpaired older than 1 s
            if len(d) > 60:
                for k in sorted(d)[:-30]:
                    d.pop(k)
    if not rows:
        print("no flying stereo frames"); return
    cat = lambda k: np.concatenate([r[k] for r in rows if r[k].size])
    rm, rs = cat("res_m"), cat("res_s")
    dts = np.array([r["dt"] for r in rows])
    print(f"{Path(a.bag).name}: {len(rows)} frames (every {a.every}th pair, flying)")
    print(f"  valid pixels (central band)   mine {np.mean([r['valid'] for r in rows]):.2f}   sim depth {np.mean([r['valid_sim'] for r in rows]):.2f}")
    print(f"  vs sim depth (both valid, <3.5 m)   median rel diff {np.nanmedian([r['rel_med'] for r in rows]):.3f}   within 10% {np.mean([r['rel_10'] for r in rows]):.2f}")
    for name, r in (("mine", rm), ("sim ", rs)):
        print(f"  vs the course (0.4-3.5 m)  {name}: median {np.median(r)*100:.1f} cm, p90 {np.percentile(r, 90)*100:.1f} cm, "
              f"within 5 cm {np.mean(r < 0.05):.2f}, within 10 cm {np.mean(r < 0.10):.2f}  ({r.size} pts)")
    print(f"  time per frame on this machine: median {np.median(dts)*1000:.1f} ms, p95 {np.percentile(dts, 95)*1000:.1f} ms")


if __name__ == "__main__":
    main()
