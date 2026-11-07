"""Replay a bag's EV stream (what the bridge sent PX4) through onboard.vio.touchdown_hold and compare the
published position with sim truth around landing. Read-only, no ROS.

    python3 tools/replay_touchdown_hold.py BAG.mcap [--engage 0.6 --ref 0.7 ...]
"""
from __future__ import annotations

import argparse, math, mmap, struct, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onboard.vio.touchdown_hold import TouchdownHold  # noqa: E402

KEEP = {"/fmu/in/vehicle_visual_odometry", "/fmu/out/distance_sensor", "/sim/truth/pose_px4", "/fmu/out/vehicle_land_detected",
        "/refnav/status"}


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


def a8(o): return (o - 4 + 7) // 8 * 8 + 4
def hdr_end(b):
    o = (12 - 4 + 3) // 4 * 4 + 4; return o + 4 + struct.unpack_from("<I", b, o)[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    for k in ("engage", "ref", "release"):
        ap.add_argument(f"--{k}", type=float, default=None)
    a = ap.parse_args(argv)
    ev, rng, truth, landed, land_phase = [], [], [], None, []
    t0 = None
    for tp, t, b in stream(a.bag):
        t0 = t if t0 is None else t0; r = t - t0
        if tp == "/fmu/in/vehicle_visual_odometry":
            ev.append((r, struct.unpack_from("<3f", b, 4 + 20)))            # timestamp, timestamp_sample, pose_frame(+pad), position
        elif tp == "/fmu/out/distance_sensor":
            rng.append((r, struct.unpack_from("<f", b, 4 + 20)[0]))
        elif tp == "/sim/truth/pose_px4":
            truth.append((r, struct.unpack_from("<3d", b, a8(hdr_end(b)))))
        elif tp == "/fmu/out/vehicle_land_detected" and landed is None and b[4 + 15]:
            landed = r
        elif tp == "/refnav/status":                                    # stand-in for PX4 nav_state AUTO_LAND
            n = struct.unpack_from("<I", b, 4)[0]
            land_phase.append((r, b'"phase": "land"' in b[8:8 + n] or b'"phase":"land"' in b[8:8 + n]))
    h = TouchdownHold(**{k + ("_m" if k != "release" else "_m"): v for k, v in (("engage", a.engage), ("ref", a.ref), ("release", a.release)) if v is not None})
    events = sorted([(r, 0, d) for r, d in rng] + [(r, 1, p) for r, p in ev] + [(r, 2, f) for r, f in land_phase],
                    key=lambda e: (e[0], e[1]))
    out = []
    for r, kind, v in events:
        if kind == 2:
            h.landing = h.landing or v                                  # once landing, stays landing
        elif kind == 0:
            h.on_range(r, v)
        else:
            p, held = h.filter(r, v)
            out.append((r, p, held, v))
    import bisect
    tt = [x[0] for x in truth]

    def err(r, p):
        i = min(bisect.bisect_left(tt, r), len(tt) - 1)
        return math.hypot(p[0] - truth[i][1][0], p[1] - truth[i][1][1])
    # the landing window: from 4 s before the first hold (or the end) to the end of the bag
    first_hold = next((r for r, _, held, _ in out if held), None)
    w0 = (first_hold if first_hold is not None else out[-1][0]) - 4.0
    W = [x for x in out if x[0] >= w0]
    raw_e = [err(r, v) for r, p, held, v in W]; pub_e = [err(r, p) for r, p, held, v in W]
    print(f"{Path(a.bag).name}: params engage {h.engage_m} ref {h.ref_m} release {h.release_m}; first hold at {first_hold}; PX4 landed flag at {landed}; {h.n}")
    print(f"  landing window ({len(W)} EV samples): raw EV error max {max(raw_e):.3f} final {raw_e[-1]:.3f} | published error max {max(pub_e):.3f} final {pub_e[-1]:.3f} m")


if __name__ == "__main__":
    main()
