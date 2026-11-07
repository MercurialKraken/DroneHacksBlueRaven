"""Replay a bag through nav.map_align.DriftCorrector and compare raw vs corrected pose with sim truth.
Read-only; no ROS (numpy, plus zstandard for zstd bags).

    python3 tools/eval_map_align.py BAG.mcap [--launch-key launch_sim] [--every 3]
"""
from __future__ import annotations

import argparse, json, math, mmap, struct, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nav.course_map import CourseMap  # noqa: E402
from nav.frames import load_launch  # noqa: E402
from nav.map_align import SE2, DriftCorrector, wrap  # noqa: E402
from nav.pose_buffer import ClockMap, PoseBuffer  # noqa: E402
from refnav import depth as D  # noqa: E402

KEEP = {"/camera/depth/image_raw", "/camera/depth/camera_info", "/fmu/out/vehicle_attitude",
        "/fmu/out/vehicle_local_position_v1", "/sim/truth/odometry", "/refnav/status"}


def stream(path):
    f = open(path, "rb"); mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ); chans = {}

    def s_(b, o):
        n = struct.unpack_from("<I", b, o)[0]; return bytes(b[o + 4:o + 4 + n]).decode(), o + 4 + n

    def walk(buf, start, end):
        i = start
        while i + 9 <= end:
            op = buf[i]; ln = struct.unpack_from("<Q", buf, i + 1)[0]; b0 = i + 9
            if op == 0x04:
                cid, sid = struct.unpack_from("<HH", buf, b0); t, _ = s_(buf, b0 + 4); chans[cid] = t
            elif op == 0x05:
                cid, seq, lt = struct.unpack_from("<HIQ", buf, b0)
                if chans.get(cid) in KEEP:
                    yield chans[cid], lt * 1e-9, bytes(buf[b0 + 22:b0 + ln])
            elif op == 0x06:
                usize = struct.unpack_from("<Q", buf, b0 + 16)[0]
                o = b0 + 28; comp, o = s_(buf, o); rlen = struct.unpack_from("<Q", buf, o)[0]
                if comp == "zstd":
                    import zstandard
                    raw = zstandard.ZstdDecompressor().decompress(bytes(buf[o + 8:o + 8 + rlen]), max_output_size=usize)
                    yield from walk(raw, 0, len(raw))
                elif comp:
                    raise SystemExit("unsupported compression " + comp)
                else:
                    yield from walk(buf, o + 8, o + 8 + rlen)
            i = b0 + ln
    yield from walk(mm, 8, len(mm))


def a4(o): return (o - 4 + 3) // 4 * 4 + 4
def a8(o): return (o - 4 + 7) // 8 * 8 + 4
def skip_str(b, o):
    o = a4(o); return o + 4 + struct.unpack_from("<I", b, o)[0]
def yaw_wxyz(w, x, y, z): return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("bag"); ap.add_argument("--launch-key", default="launch_sim"); ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--field", default=None, help="SDF field to align to (default: AlignParams.field)")
    ap.add_argument("--method", choices=("align", "mcl", "kf"), default="align", help="DriftCorrector (tau low-pass), particle filter, or Kalman")
    ap.add_argument("--kf", nargs="*", default=[], help="KfParams overrides, e.g. q_xy=1e-3 r_scale=100 (align.* via --align)")
    ap.add_argument("--interp", action="store_true", help="fit with the pose interpolated at the depth stamp (Task D) instead of the latest")
    ap.add_argument("--mcl", nargs="*", default=[], help="MclParams overrides, e.g. points_per_dof=75 walk_xy=0.01")
    ap.add_argument("--align", nargs="*", default=[], help="AlignParams/CorrectorParams overrides, e.g. yaw_min_gain=0")
    a = ap.parse_args(argv)
    cm = CourseMap.load(ROOT / "maps" / "course_sdf.npz")
    L = load_launch(ROOT / "maps" / "course.yaml", a.launch_key)
    start = SE2(L["x"], L["y"], math.radians(L["yaw_deg"] - 90.0))
    from nav.map_align import AlignParams, CorrectorParams
    cp = CorrectorParams(align=AlignParams(field=a.field) if a.field else AlignParams())
    for kv in a.align:
        k, v = kv.split("=", 1)
        tgt = cp.align if hasattr(cp.align, k) else cp
        setattr(tgt, k, type(getattr(tgt, k))(v))
    if a.method == "kf":
        from nav.map_kf import KalmanCorrector, KfParams
        kp = KfParams(align=cp.align)
        for kv in a.kf:
            k, v = kv.split("=", 1)
            setattr(kp, k, type(getattr(kp, k))(v))
        dc = KalmanCorrector(cm, start, kp)
        print(f"method: kalman, field {kp.align.field}, near_m {kp.align.near_m}, overrides {a.kf}")
    elif a.method == "mcl":
        from nav.map_mcl import MclParams, ParticleCorrector
        mp = MclParams(field=a.field) if a.field else MclParams()
        for kv in a.mcl:
            k, v = kv.split("=", 1)
            setattr(mp, k, type(getattr(mp, k))(v))
        dc = ParticleCorrector(cm, start, mp)
        print(f"method: particle filter, field {dc.p.field}, {dc.p.n} particles, overrides {a.mcl}")
    else:
        dc = DriftCorrector(cm, start, cp)
        print(f"method: align, field: {cp.align.field}")
    R_bc = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float); t_bc = np.array([0.116, 0.037, 0.029])
    K = att = pos = truth = phase = None; t0 = None; n = 0; rows = []; walls = []; totals = []; refused = 0
    buf, clk = PoseBuffer(1.0, 0.1), ClockMap()
    for topic, t, b in stream(a.bag):
        t0 = t if t0 is None else t0
        if topic == "/refnav/status":
            k = struct.unpack_from("<I", b, 4)[0]; phase = json.loads(b[8:8 + k - 1].decode()).get("phase")
        elif topic == "/fmu/out/vehicle_attitude":
            att = struct.unpack_from("<4f", b, 4 + 16)
            ts, tss = struct.unpack_from("<QQ", b, 4); clk.observe(t, ts); buf.add_attitude(clk.to_ros(tss), att)
        elif topic == "/fmu/out/vehicle_local_position_v1":
            pos = struct.unpack_from("<3f", b, 24)
            ts, tss = struct.unpack_from("<QQ", b, 4); clk.observe(t, ts); buf.add_position(clk.to_ros(tss), pos)
        elif topic == "/sim/truth/odometry":
            o = skip_str(b, 12); o = skip_str(b, o); o = a8(o)
            p = struct.unpack_from("<7d", b, o); truth = (p[0], p[1], p[2], yaw_wxyz(p[6], p[3], p[4], p[5]))
        elif topic == "/camera/depth/camera_info" and K is None:
            o = skip_str(b, 12); o = a4(o); o = skip_str(b, o + 8); o = a4(o)
            nd = struct.unpack_from("<I", b, o)[0]; o = a8(a8(o + 4) + 8 * nd)
            k = struct.unpack_from("<9d", b, o); K = (k[0], k[4], k[2], k[5])
        elif topic == "/camera/depth/image_raw" and K and att and pos and truth and phase not in (None, "wait", "arm", "done"):
            n += 1
            if n % a.every:
                continue
            o = skip_str(b, 12); o = a4(o); h, w = struct.unpack_from("<II", b, o); enc_end = skip_str(b, o + 8)
            o = a4(o + 8); el = struct.unpack_from("<I", b, o)[0]; enc = bytes(b[o + 4:o + 4 + el - 1]).decode()
            o = a4(enc_end + 1); step = struct.unpack_from("<I", b, o)[0]; o += 4; m = struct.unpack_from("<I", b, o)[0]
            pre0 = time.perf_counter()
            dep = D.decode_depth(enc, h, w, step, bytes(b[o + 4:o + 4 + m]))
            f_att, f_pos = att, pos
            if a.interp:
                sec, nsec = struct.unpack_from("<iI", b, 4)
                at = buf.at(sec + nsec * 1e-9)
                if at is None:
                    refused += 1
                    continue
                f_pos, f_att = tuple(at[0]), tuple(at[1])
            f = D.DepthFrame(dep, K, R_bc, t_bc, D.level_rotation(f_att), stride=4, border_px=24)
            pts = f.points(0.4, 3.5)
            f_odom = (f_pos[1], f_pos[0], -f_pos[2] + L["z"], wrap(math.pi / 2 - yaw_wxyz(*f_att)))   # PX4 NED -> odom ENU
            odom = (pos[1], pos[0], -pos[2] + L["z"], wrap(math.pi / 2 - yaw_wxyz(*att)))          # latest: for scoring
            raw = start.pose(odom)
            w0 = time.perf_counter()
            dc.update(t - t0, f_odom, pts)
            walls.append(time.perf_counter() - w0)
            totals.append(time.perf_counter() - pre0)
            T = dc.T
            cor = T.pose(odom)
            if a.method == "mcl":
                sx, sy, syaw = dc.spread
                info = (dc.last_explained >= dc.p.min_explained, f"{sx:.2f}/{sy:.2f}/{syaw:.1f}", 0, dc.last_explained)
            else:
                r = dc.last
                info = (r.ok, "".join("xyY"[i] if r.observable[i] else "-" for i in range(3)), r.inliers, r.rms_after)
            rows.append((t - t0, phase, math.hypot(raw[0] - truth[0], raw[1] - truth[1]), math.degrees(wrap(raw[3] - truth[3])),
                         math.hypot(cor[0] - truth[0], cor[1] - truth[1]), math.degrees(wrap(cor[3] - truth[3])), *info))
    A = np.array([(r[2], r[3], r[4], r[5]) for r in rows])
    air = np.array([r[1] not in ("climb", "land") for r in rows])
    mode = "pose at the depth stamp" if a.interp else "latest pose"
    print(f"{a.bag}\nframes {len(rows)} (every {a.every}th depth frame while flying, {mode}, {refused} refused); corrector {dc.n}")
    Wt = np.array(walls) * 1000
    print(f"[timing] update (align + filter) on this machine: p50 {np.percentile(Wt, 50):.1f} ms, "
          f"p95 {np.percentile(Wt, 95):.1f} ms, max {Wt.max():.1f} ms")
    Tt = np.array(totals) * 1000
    print(f"[timing] whole frame (decode + points + update): p50 {np.percentile(Tt, 50):.1f} ms, p95 {np.percentile(Tt, 95):.1f} ms, "
          f"max {Tt.max():.1f} ms -> {np.mean(Tt) * 5 / 10:.1f}% of one core at 5 Hz")
    for name, sel in (("all", np.ones(len(A), bool)), ("cruise (not climb/land)", air)):
        B = A[sel]
        print(f"[{name}] xy error  raw rms {np.sqrt(np.mean(B[:,0]**2)):.3f} max {B[:,0].max():.3f}  |  corrected rms {np.sqrt(np.mean(B[:,2]**2)):.3f} max {B[:,2].max():.3f} m")
        print(f"[{name}] yaw error raw rms {np.sqrt(np.mean(B[:,1]**2)):.2f} max {np.abs(B[:,1]).max():.2f}  |  corrected rms {np.sqrt(np.mean(B[:,3]**2)):.2f} max {np.abs(B[:,3]).max():.2f} deg")
    head = "fit spread(x/y/yaw) explained" if a.method == "mcl" else "fit obs  inl  rms"
    print(f"\n   t   phase   raw_xy raw_yaw | cor_xy cor_yaw | {head}")
    for r in rows[:: max(1, len(rows) // 32)]:
        print(f"{r[0]:6.1f} {str(r[1]):7s} {r[2]:6.3f} {r[3]:+6.2f} | {r[4]:6.3f} {r[5]:+6.2f} | {'ok ' if r[6] else 'rej'} {r[7]} {r[8]:4d} {r[9]:.3f}")


if __name__ == "__main__":
    main()
