"""Replay a bag's cuVSLAM odometry through onboard.vio.vio_health.
Read-only, no ROS (zstandard for zstd bags).

    python3 tools/replay_vio_health.py BAG.mcap [BAG2 ...]

Reports samples withheld, the longest stretch with nothing passed to PX4 (PX4 drops vision after
EKF2_NOAID_TOUT = 1 s), and every stretch over 0.5 s.
"""
from __future__ import annotations

import mmap, struct, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onboard.vio.anchor import Anchor  # noqa: E402
from onboard.vio.vio_health import VioHealth  # noqa: E402


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
                if chans.get(cid) == "/visual_slam/tracking/odometry":
                    yield bytes(buf[b0 + 22:b0 + ln])                  # whole message: the covariance is ~0.4 KB in
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


def samples(bag):
    """(t, position, orientation xyzw, largest position variance)"""
    out, bad = [], 0
    for b in stream(bag):
        try:
            sec, ns = struct.unpack_from("<iI", b, 4)
            o = a4(12); n = struct.unpack_from("<I", b, o)[0]; o = a4(o + 4 + n); n = struct.unpack_from("<I", b, o)[0]; o = a8(o + 4 + n)
            pos = struct.unpack_from("<3d", b, o); q = struct.unpack_from("<4d", b, o + 24)
            cov = struct.unpack_from("<36d", b, o + 56) if len(b) >= o + 56 + 288 else None
        except struct.error:
            bad += 1; continue
        out.append((sec + ns * 1e-9, pos, q, max(cov[0], cov[7], cov[14]) if cov else None))
    if bad:
        print(f"  ({bad} unparseable odometry messages skipped)")
    return out


def run(S, max_var=0.05):
    """As the bridge does it: health check (with variance), re-anchoring, publish."""
    h, a = VioHealth(max_var_m2=max_var), Anchor()
    passed_t, withheld, last_out, steps, bad_var = [], 0, None, [], 0
    for t, p, q, var in S:
        ok, _ = h.update(t, p, var)
        if not ok:
            withheld += 1; continue
        if h.rebased and last_out is not None:
            a.rebase(last_out[0], last_out[1], p, q)
        out = a.apply(p, q)
        if last_out is not None:
            steps.append((t, __import__("math").dist(out[0], last_out[0])))
        if var is not None and var > max_var:
            bad_var += 1
        last_out = out; passed_t.append(t)
    gaps = [(x, y - x) for x, y in zip([S[0][0]] + passed_t, passed_t + [S[-1][0]])]
    long = [(round(x, 2), round(g, 2)) for x, g in gaps if g > 0.5]
    big = max(steps, key=lambda s: s[1]) if steps else (None, 0.0)
    return dict(withheld=withheld, gmax=max(g for _, g in gaps), long=long[:4], reanchored=a.n_rebased,
                max_step=(round(big[1], 3), round(big[0], 1) if big[0] else None), forwarded_over_gate=bad_var,
                raw_max_var=max((s[3] for s in S if s[3] is not None), default=None))


def main(argv=None):
    for bag in (argv or sys.argv[1:]):
        S = samples(bag)
        if not S:
            print(f"{Path(bag).name}: no odometry"); continue
        r = run(S)
        rv = "n/a" if r["raw_max_var"] is None else f"{r['raw_max_var']:.3g}"
        print(f"{Path(bag).name}: {len(S)} samples | withheld {r['withheld']} | longest gap {r['gmax']:.2f} s{' (PX4 drops EV)' if r['gmax'] > 1.0 else ''} "
              f"| re-anchored {r['reanchored']} | max published step {r['max_step'][0]} m at {r['max_step'][1]} | forwarded over gate {r['forwarded_over_gate']} "
              f"| raw max var {rv}")


if __name__ == "__main__":
    main()
