"""Replay a bag's cuVSLAM odometry through the old and the new onboard.vio.vio_health checks.
Read-only, no ROS (numpy-free; zstandard for zstd bags).

    python3 tools/replay_vio_health.py BAG.mcap [BAG2 ...]

For each check: samples withheld, the longest stretch with nothing passed to PX4 (PX4 drops vision
after EKF2_NOAID_TOUT = 1 s), and every stretch over 0.5 s.
"""
from __future__ import annotations

import math, mmap, struct, sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onboard.vio.vio_health import VioHealth  # noqa: E402


class LegacyVioHealth:
    """The check as it was before this change (compares against the last sample, 10 s window, 5 s quiet)."""

    def __init__(self, max_speed_mps=5.0, jump_m=1.0, window_s=10.0, max_jumps=1, quiet_s=5.0):
        self.max_speed, self.jump_m, self.window_s, self.max_jumps, self.quiet_s = max_speed_mps, jump_m, window_s, max_jumps, quiet_s
        self._last = None; self._events = deque(); self._last_bad_t = None

    def update(self, t, pos):
        bad = None
        if self._last is not None:
            t0, p0 = self._last; dt = t - t0; d = math.dist(p0, pos)
            if dt <= 0: bad = "time"
            elif d > self.jump_m: bad = "jump"
            elif d / dt > self.max_speed: bad = "speed"
        if self._last is None or t > self._last[0]:
            self._last = (t, tuple(pos))
        if bad:
            self._last_bad_t = t; self._events.append(t)
        while self._events and t - self._events[0] > self.window_s:
            self._events.popleft()
        if len(self._events) > self.max_jumps:
            return False, "window"
        if self._last_bad_t is not None and t - self._last_bad_t < self.quiet_s:
            return False, "settling"
        return True, "ok"


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
                    yield bytes(buf[b0 + 22:b0 + min(ln, 300)])
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
    out = []
    for b in stream(bag):
        sec, ns = struct.unpack_from("<iI", b, 4)
        o = a4(12); n = struct.unpack_from("<I", b, o)[0]; o = a4(o + 4 + n); n = struct.unpack_from("<I", b, o)[0]; o = a8(o + 4 + n)
        out.append((sec + ns * 1e-9, struct.unpack_from("<3d", b, o)))
    return out


def run(check, S):
    passed_t = []; withheld = 0
    for t, p in S:
        ok, _ = check.update(t, p)
        if ok: passed_t.append(t)
        else: withheld += 1
    gaps = [(a, b - a) for a, b in zip([S[0][0]] + passed_t, passed_t + [S[-1][0]])]
    long = [(round(a, 2), round(g, 2)) for a, g in gaps if g > 0.5]
    return withheld, max(g for _, g in gaps), long


def main(argv=None):
    for bag in (argv or sys.argv[1:]):
        S = samples(bag)
        print(f"{Path(bag).name}: {len(S)} odometry samples, {S[0][0]:.1f}-{S[-1][0]:.1f} s")
        for name, chk in (("old", LegacyVioHealth()), ("new", VioHealth())):
            w, gmax, long = run(chk, S)
            print(f"  {name}: withheld {w:4d}, longest stretch with nothing to PX4 {gmax:5.2f} s {'(PX4 would drop vision)' if gmax > 1.0 else ''}; stretches > 0.5 s: {long[:6]}")


if __name__ == "__main__":
    main()
