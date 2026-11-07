"""Replay a bag through nav.map_align.DriftCorrector and compare raw vs corrected pose with sim truth.
Read-only; no ROS (numpy, plus zstandard for zstd bags).

    python3 tools/eval_map_align.py BAG.mcap [--launch-key launch_sim] [--every 3]
"""
from __future__ import annotations

import argparse, json, math, mmap, struct, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nav.course_map import CourseMap  # noqa: E402
from nav.frames import load_launch  # noqa: E402
from nav.map_align import SE2, DriftCorrector, wrap  # noqa: E402
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
    ap.add_argument("--method", choices=("align", "mcl"), default="align", help="DriftCorrector or the particle filter")
    a = ap.parse_args(argv)
    cm = CourseMap.load(ROOT / "maps" / "course_sdf.npz")
    L = load_launch(ROOT / "maps" / "course.yaml", a.launch_key)
    start = SE2(L["x"], L["y"], math.radians(L["yaw_deg"] - 90.0))
    from nav.map_align import AlignParams, CorrectorParams
    cp = CorrectorParams(align=AlignParams(field=a.field) if a.field else AlignParams())
    if a.method == "mcl":
        from nav.map_mcl import MclParams, ParticleCorrector
        dc = ParticleCorrector(cm, start, MclParams(field=a.field) if a.field else MclParams())
        print(f"method: particle filter, field {dc.p.field}, {dc.p.n} particles")
    else:
        dc = DriftCorrector(cm, start, cp)
        print(f"method: align, field: {cp.align.field}")
    R_bc = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float); t_bc = np.array([0.116, 0.037, 0.029])
    K = att = pos = truth = phase = None; t0 = None; n = 0; rows = []
    for topic, t, b in stream(a.bag):
        t0 = t if t0 is None else t0
        if topic == "/refnav/status":
            k = struct.unpack_from("<I", b, 4)[0]; phase = json.loads(b[8:8 + k - 1].decode()).get("phase")
        elif topic == "/fmu/out/vehicle_attitude":
            att = struct.unpack_from("<4f", b, 4 + 16)
        elif topic == "/fmu/out/vehicle_local_position_v1":
            pos = struct.unpack_from("<3f", b, 24)
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
            dep = D.decode_depth(enc, h, w, step, bytes(b[o + 4:o + 4 + m]))
            f = D.DepthFrame(dep, K, R_bc, t_bc, D.level_rotation(att), stride=4, border_px=24)
            pts = f.points(0.4, 3.5)
            odom = (pos[1], pos[0], -pos[2] + L["z"], wrap(math.pi / 2 - yaw_wxyz(*att)))   # PX4 NED -> odom ENU
            raw = start.pose(odom)
            T = dc.update(t - t0, odom, pts)
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
    print(f"{a.bag}\nframes {len(rows)} (every {a.every}th depth frame while flying); corrector {dc.n}")
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
