"""Replay a bag's IMU and stereo stamps through the old and the new IMU/frame pairing, with an injected
executor stall, and count lost IMU. No ROS, no cuVSLAM; needs numpy (and zstandard for zstd bags).

    python3 tools/replay_imu_sync.py BAG.mcap --stall-at 80 --stall 1.2 [--images-first]

Arrival order = the bag's receive (log) time, which is the closest record of what a subscriber saw.
During the stall nothing is handled; afterwards the backlog is delivered images first (the worst
case, which is what the T4 run looked like) unless --imu-first.
"""
from __future__ import annotations

import argparse, mmap, struct, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onboard.slam.imu_sync import ImuFrameSync, legacy_release  # noqa: E402

TOPICS = {"/stereo/imu": "imu", "/stereo/left/image": "l", "/stereo/right/image": "r"}


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
                if chans.get(cid) in TOPICS:
                    sec, ns = struct.unpack_from("<iI", buf, b0 + 22 + 4)
                    yield TOPICS[chans[cid]], lt, sec * 1_000_000_000 + ns
            elif op == 0x06:
                usize = struct.unpack_from("<Q", buf, b0 + 16)[0]
                o = b0 + 28; comp, o = s_(buf, o); rlen = struct.unpack_from("<Q", buf, o)[0]
                if comp == "zstd":
                    import zstandard
                    raw = zstandard.ZstdDecompressor().decompress(bytes(buf[o + 8:o + 8 + rlen]), max_output_size=usize)
                    yield from walk(raw, 0, len(raw))
                elif comp:
                    raise SystemExit("unsupported chunk compression: " + comp)
                else:
                    yield from walk(buf, o + 8, o + 8 + rlen)
            i = b0 + ln
    yield from walk(mm, 8, len(mm))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("bag"); ap.add_argument("--stall-at", type=float, required=True); ap.add_argument("--stall", type=float, default=1.2)
    ap.add_argument("--imu-first", action="store_true"); ap.add_argument("--wait", type=float, default=0.08)
    a = ap.parse_args(argv)
    raw = list(stream(a.bag)); t0 = min(lt for _, lt, _ in raw)
    half, ev = {}, []
    for kind, lt, stamp in sorted(raw, key=lambda e: e[1]):
        now = (lt - t0) * 1e-9
        if kind == "imu":
            ev.append((now, "imu", stamp))
        else:
            h = half.setdefault(stamp, set()); h.add(kind)
            if h == {"l", "r"}:
                ev.append((now, "cam", stamp)); del half[stamp]
    end = a.stall_at + a.stall
    before = [e for e in ev if e[0] < a.stall_at]
    held = [e for e in ev if a.stall_at <= e[0] < end]
    after = [e for e in ev if e[0] >= end]
    held.sort(key=lambda e: (e[1] == "cam") if a.imu_first else (e[1] != "cam"))
    order = before + [(end, k, t) for _, k, t in held] + after

    q, lf, li, legacy_drop, legacy_frames = [], -1, -1, 0, 0
    for _, kind, t in order:
        if kind == "imu":
            q.append((t, None, None))
        elif t > lf:
            _, q, d, li = legacy_release(q, t, lf, li); legacy_drop += d; lf = t; legacy_frames += 1
    s = ImuFrameSync(wait_s=a.wait); frames = 0
    for now, kind, t in order:
        if kind == "imu":
            s.add_imu(t, None, None)
        else:
            s.add_frame(t, None, None, now)
        frames += len(s.ready(now))
    frames += len(s.ready(1e12))
    n_imu = sum(1 for e in ev if e[1] == "imu"); n_cam = sum(1 for e in ev if e[1] == "cam")
    print(f"bag: {n_imu} IMU, {n_cam} stereo pairs; stall {a.stall} s at {a.stall_at} s, backlog {'IMU' if a.imu_first else 'images'} first; "
          f"{sum(1 for e in held if e[1] == 'imu')} IMU and {sum(1 for e in held if e[1] == 'cam')} pairs held")
    print(f"old pairing: {legacy_drop} IMU dropped, {legacy_frames} frames tracked")
    print(f"new pairing: {s.n['imu_late']} IMU late, {frames} frames tracked, {s.n['frames_timeout']} timeouts, "
          f"{s.n['frames_waited']} frames waited, {s.n['frames_overflow']} frames overflowed")
    return legacy_drop, s.n["imu_late"]


if __name__ == "__main__":
    main()
