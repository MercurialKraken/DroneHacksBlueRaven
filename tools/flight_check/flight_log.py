"""Pure helpers for reading a real-drone flight recording offline. numpy only, no ROS.

Decodes the few ROS 2 message types the skybeta interface uses from raw CDR bytes (as stored in an
MCAP file), and computes the numbers a flight test needs: topic rates and dropouts, VIO jumps and
freezes, and how far the drone was from what it was told to do.

Read-only by construction: nothing here talks to ROS, PX4 or the drone.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- CDR decoding


class Cdr:
    """Little-endian ROS 2 CDR (XCDR1) reader. Alignment is relative to the byte after the 4-byte
    encapsulation header, as in rmw's serializer."""

    def __init__(self, buf: bytes):
        if len(buf) < 4:
            raise ValueError("CDR buffer shorter than its 4-byte header")
        if buf[1] != 0x01:
            raise ValueError(f"only little-endian CDR is supported (header byte {buf[1]:#x})")
        self.b = memoryview(buf)
        self.o = 4

    def _align(self, n: int) -> None:
        r = (self.o - 4) % n
        if r:
            self.o += n - r

    def _take(self, fmt: str, size: int):
        self._align(size)
        if self.o + struct.calcsize(fmt) > len(self.b):
            raise ValueError("CDR buffer truncated")
        v = struct.unpack_from(fmt, self.b, self.o)
        self.o += struct.calcsize(fmt)
        return v

    def i32(self) -> int:
        return self._take("<i", 4)[0]

    def u32(self) -> int:
        return self._take("<I", 4)[0]

    def u64(self) -> int:
        return self._take("<Q", 8)[0]

    def f64s(self, n: int) -> tuple[float, ...]:
        return self._take("<%dd" % n, 8)

    def string(self) -> str:
        n = self.u32()  # includes the terminating NUL
        if n == 0:
            return ""
        if self.o + n > len(self.b):
            raise ValueError("CDR string truncated")
        s = bytes(self.b[self.o : self.o + n - 1]).decode("utf-8", errors="replace")
        self.o += n
        return s


def _header(r: Cdr) -> tuple[float, str]:
    sec, nsec = r.i32(), r.u32()
    return sec + nsec * 1e-9, r.string()


def decode(schema: str, data: bytes) -> dict | None:
    """Decode one message by its schema name. Returns None for types this tool does not read."""
    s = schema.replace("/msg/", "/")
    r = Cdr(data)
    if s == "nav_msgs/Odometry":
        stamp, frame = _header(r)
        child = r.string()
        pos, quat = r.f64s(3), r.f64s(4)
        r.f64s(36)  # pose covariance
        lin, ang = r.f64s(3), r.f64s(3)
        return {
            "stamp": stamp,
            "frame": frame,
            "child": child,
            "pos": pos,
            "quat": quat,
            "lin": lin,
            "ang": ang,
        }
    if s == "geometry_msgs/PoseStamped":
        stamp, frame = _header(r)
        return {"stamp": stamp, "frame": frame, "pos": r.f64s(3), "quat": r.f64s(4)}
    if s == "geometry_msgs/Pose":
        return {"stamp": 0.0, "frame": "", "pos": r.f64s(3), "quat": r.f64s(4)}
    if s == "geometry_msgs/TwistStamped":
        stamp, frame = _header(r)
        return {"stamp": stamp, "frame": frame, "lin": r.f64s(3), "ang": r.f64s(3)}
    if s == "geometry_msgs/Twist":
        return {"stamp": 0.0, "frame": "", "lin": r.f64s(3), "ang": r.f64s(3)}
    if s == "std_msgs/String":
        return {"data": r.string()}
    if s == "px4_msgs/VehicleOdometry":
        # first two fields: uint64 timestamp, uint64 timestamp_sample (microseconds)
        ts, sample = r.u64(), r.u64()
        return {"px4_timestamp_us": ts, "px4_sample_us": sample}
    return None


# --------------------------------------------------------------------------- metrics


@dataclass(frozen=True)
class Thresholds:
    gap_s: float = 0.25  # no message for longer than this = a dropout
    jump_m: float = 0.30  # one-step position change larger than this = a jump
    jump_speed_mps: float = 3.0  # implied speed above this = a jump (twists are capped at 1.5)
    freeze_s: float = 0.5  # identical position for at least this long while messages arrive
    hold_s: float = 1.0  # a setpoint older than this is not compared against
    latency_s: float = 0.2  # older data than this is not fused by PX4 (organizers' EKF2_DELAY_MAX)


def topic_stats(times: np.ndarray, gap_s: float) -> dict:
    """Arrival times (s) -> count, mean rate, longest gap and the dropouts: gaps longer than
    gap_s AND longer than 3x the topic's own median period (so a 1 Hz status topic is not a
    dropout every second)."""
    t = np.sort(np.asarray(times, float))
    if t.size == 0:
        return {"count": 0, "rate_hz": 0.0, "duration_s": 0.0, "max_gap_s": None, "gaps": []}
    d = np.diff(t)
    dur = float(t[-1] - t[0])
    limit = max(gap_s, 3.0 * float(np.median(d))) if d.size else gap_s
    gaps = [
        {"t": round(float(t[i]), 3), "gap_s": round(float(d[i]), 3)}
        for i in np.flatnonzero(d > limit)
    ]
    return {
        "count": int(t.size),
        "rate_hz": round((t.size - 1) / dur, 2) if dur > 0 else 0.0,
        "duration_s": round(dur, 3),
        "max_gap_s": round(float(d.max()), 3) if d.size else None,
        "gaps": gaps,
    }


def vio_events(t: np.ndarray, p: np.ndarray, th: Thresholds = Thresholds()) -> list[dict]:
    """Odometry times (s) and positions (N,3) -> jump, gap, freeze and non-finite events."""
    t = np.asarray(t, float)
    p = np.asarray(p, float).reshape(-1, 3)
    if t.shape[0] != p.shape[0]:
        raise ValueError("times and positions differ in length")
    ev: list[dict] = []
    bad = ~np.isfinite(p).all(axis=1)
    for i in np.flatnonzero(bad):
        ev.append({"t": round(float(t[i]), 3), "kind": "non_finite"})
    ok = np.flatnonzero(~bad)
    for a, b in zip(ok[:-1], ok[1:]):
        dt = t[b] - t[a]
        step = float(np.linalg.norm(p[b] - p[a]))
        if dt > th.gap_s:
            ev.append(
                {
                    "t": round(float(t[a]), 3),
                    "kind": "gap",
                    "gap_s": round(float(dt), 3),
                    "step_m": round(step, 3),
                }
            )
        if dt < 0:  # stamps went backwards: report it, but don't call it a jump unless far
            ev.append(
                {"t": round(float(t[b]), 3), "kind": "stamp_backwards", "dt_s": round(float(dt), 3)}
            )
            if step > th.jump_m:
                ev.append(
                    {
                        "t": round(float(t[b]), 3),
                        "kind": "jump",
                        "step_m": round(step, 3),
                        "speed_mps": None,
                    }
                )
            continue
        speed = step / max(dt, 0.01)  # floor: stamp jitter
        if step > th.jump_m or speed > th.jump_speed_mps:
            ev.append(
                {
                    "t": round(float(t[b]), 3),
                    "kind": "jump",
                    "step_m": round(step, 3),
                    "speed_mps": round(speed, 2),
                }
            )
    # freezes: runs of bit-identical positions spanning freeze_s (tracking stopped, output held)
    i = 0
    while i < len(ok):
        j = i
        while j + 1 < len(ok) and np.array_equal(p[ok[j + 1]], p[ok[i]]):
            j += 1
        span = t[ok[j]] - t[ok[i]]
        if j > i and span >= th.freeze_s:
            ev.append(
                {
                    "t": round(float(t[ok[i]]), 3),
                    "kind": "freeze",
                    "duration_s": round(float(span), 3),
                }
            )
        i = j + 1
    return sorted(ev, key=lambda e: e["t"])


def _held(sp_t: np.ndarray, t: np.ndarray, hold_s: float) -> np.ndarray:
    """Index of the latest setpoint at or before each t, or -1 if none / older than hold_s."""
    idx = np.searchsorted(sp_t, t, side="right") - 1
    ok = idx >= 0
    age = np.where(ok, t - sp_t[np.clip(idx, 0, None)], np.inf)
    return np.where(ok & (age <= hold_s), idx, -1)


def _summary(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "median": round(float(np.median(x)), 3),
        "p95": round(float(np.percentile(x, 95)), 3),
        "max": round(float(x.max()), 3),
    }


def pose_tracking(odo_t, odo_p, sp_t, sp_p, hold_s: float = 1.0) -> dict:
    """Distance from each odometry sample to the pose setpoint in force at that moment.
    Assumes both are in the same frame (check the reported frame ids)."""
    odo_t, sp_t = np.asarray(odo_t, float), np.asarray(sp_t, float)
    odo_p, sp_p = np.asarray(odo_p, float).reshape(-1, 3), np.asarray(sp_p, float).reshape(-1, 3)
    if sp_t.size == 0 or odo_t.size == 0:
        return {"compared": 0}
    order = np.argsort(sp_t, kind="stable")
    sp_t, sp_p = sp_t[order], sp_p[order]
    k = _held(sp_t, odo_t, hold_s)
    m = k >= 0
    d = odo_p[m] - sp_p[k[m]]
    horiz = np.hypot(d[:, 0], d[:, 1])
    return {
        "compared": int(m.sum()),
        "horizontal_m": _summary(horiz),
        "vertical_m": _summary(np.abs(d[:, 2])),
        "series": {"t": odo_t[m], "horizontal_m": horiz, "vertical_m": d[:, 2]},
    }


def velocity(t, p, window_s: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference velocity over about window_s (robust to 30 Hz jitter)."""
    t = np.asarray(t, float)
    p = np.asarray(p, float).reshape(-1, 3)
    v = np.full_like(p, np.nan)
    lo = np.searchsorted(t, t - window_s / 2, side="left")
    hi = np.clip(np.searchsorted(t, t + window_s / 2, side="right") - 1, 0, len(t) - 1)
    dt = t[hi] - t[lo]
    ok = dt > 0
    v[ok] = (p[hi[ok]] - p[lo[ok]]) / dt[ok, None]
    return t, v


def twist_tracking(odo_t, odo_p, sp_t, sp_lin, hold_s: float = 0.5) -> dict:
    """Commanded vs flown speed. Compares horizontal speed magnitude and vertical speed, which do
    not depend on the twist's yaw frame (body or world), so no frame assumption is needed."""
    sp_t = np.asarray(sp_t, float)
    sp_lin = np.asarray(sp_lin, float).reshape(-1, 3)
    if sp_t.size == 0 or len(odo_t) < 3:
        return {"compared": 0}
    order = np.argsort(sp_t, kind="stable")
    sp_t, sp_lin = sp_t[order], sp_lin[order]
    t, v = velocity(odo_t, odo_p)
    k = _held(sp_t, t, hold_s)
    m = (k >= 0) & np.isfinite(v).all(axis=1)
    cmd, fly = sp_lin[k[m]], v[m]
    cmd_h, fly_h = np.hypot(cmd[:, 0], cmd[:, 1]), np.hypot(fly[:, 0], fly[:, 1])
    return {
        "compared": int(m.sum()),
        "cmd_horizontal_mps": _summary(cmd_h),
        "flown_horizontal_mps": _summary(fly_h),
        "horizontal_speed_error_mps": _summary(np.abs(fly_h - cmd_h)),
        "vertical_speed_error_mps": _summary(np.abs(fly[:, 2] - cmd[:, 2])),
        "series": {"t": t[m], "cmd_h": cmd_h, "fly_h": fly_h},
    }


def state_changes(times, texts) -> list[dict]:
    """String state messages -> only the moments the text changed."""
    out, last = [], None
    for t, s in zip(times, texts):
        if s != last:
            out.append({"t": round(float(t), 3), "state": s})
            last = s
    return out


def latency(t: np.ndarray, age_s: np.ndarray, limit_s: float = 0.2) -> dict:
    """Per-message data age (s) at times t (s) -> summary, fraction over limit_s, and the onset:
    the first 1 s window whose median age exceeds limit_s (the overload signature: age jumps and
    stays high). Ages can come from arrival minus header stamp, or PX4's own timestamp minus
    timestamp_sample."""
    t = np.asarray(t, float)
    a = np.asarray(age_s, float)
    ok = np.isfinite(a) & np.isfinite(t)
    t, a = t[ok], a[ok]
    if a.size == 0:
        return {"n": 0}
    onset = None
    bins = np.floor(t - t[0]).astype(int)
    for b in np.unique(bins):
        if np.median(a[bins == b]) > limit_s:
            onset = round(float(t[bins == b][0]), 3)
            break
    pct = lambda q: round(float(np.percentile(a, q)) * 1e3, 1)  # noqa: E731
    return {
        "n": int(a.size),
        "median_ms": pct(50),
        "p90_ms": pct(90),
        "max_ms": round(float(a.max()) * 1e3, 1),
        "over_limit_fraction": round(float(np.mean(a > limit_s)), 3),
        "limit_ms": round(limit_s * 1e3, 1),
        "first_slow_second_t": onset,
    }
