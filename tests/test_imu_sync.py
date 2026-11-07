"""onboard.slam.imu_sync: frames wait for IMU coverage, so a callback-order shuffle after a stall loses
no IMU, while a missing IMU stream can only delay a frame by wait_s."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onboard.slam.imu_sync import ImuFrameSync, legacy_release  # noqa: E402

IMU_HZ, CAM_HZ = 460.0, 15.0
NS = 1_000_000_000


def streams(seconds: float):
    imu = [int(i * NS / IMU_HZ) for i in range(1, int(seconds * IMU_HZ))]
    cam = [int(i * NS / CAM_HZ) for i in range(1, int(seconds * CAM_HZ))]
    return imu, cam


def arrivals(seconds: float, stall_at: float | None = None, stall_s: float = 1.2, images_first: bool = True):
    """Callback order as one executor delivers it: (wall_s, kind, stamp_ns). Normally each message is
    handled when it is captured. During a stall nothing is handled; afterwards the backlog is handled
    with every queued image before any queued IMU (the worst order)."""
    imu, cam = streams(seconds)
    ev = [(t / NS, "imu", t) for t in imu] + [(t / NS, "cam", t) for t in cam]
    ev.sort(key=lambda e: (e[0], e[1] == "cam"))
    if stall_at is None:
        return ev
    end = stall_at + stall_s
    before = [e for e in ev if e[0] < stall_at]
    held = [e for e in ev if stall_at <= e[0] < end]
    after = [e for e in ev if e[0] >= end]
    held.sort(key=lambda e: (e[1] != "cam") if images_first else (e[1] == "cam"))
    return before + [(end, k, t) for _, k, t in held] + after


def run_sync(events, wait_s=0.08):
    s = ImuFrameSync(wait_s=wait_s)
    out = []
    for now, kind, t in events:
        if kind == "imu":
            s.add_imu(t, (0, 0, 9.8), (0, 0, 0))
        else:
            s.add_frame(t, "L", "R", now)
        out += s.ready(now)
    out += s.ready(1e9)
    return s, out


def run_legacy(events):
    q, last_f, last_i, dropped, frames = [], -1, -1, 0, 0
    for _, kind, t in events:
        if kind == "imu":
            q.append((t, None, None))
        elif t > last_f:
            reg, q, d, last_i = legacy_release(q, t, last_f, last_i)
            dropped += d
            last_f = t
            frames += 1
    return dropped, frames


def test_steady_streams_lose_nothing_and_never_wait():
    ev = arrivals(10.0)
    s, out = run_sync(ev)
    imu, cam = streams(10.0)
    assert [r.t_ns for r in out] == cam
    assert s.n["imu_late"] == 0 and s.n["frames_timeout"] == 0
    assert sum(len(r.imu) for r in out) == sum(1 for t in imu if t <= cam[-1])
    prev = -1
    for r in out:                                     # each frame gets exactly (previous frame, this frame]
        assert all(prev < x[0] <= r.t_ns for x in r.imu)
        prev = r.t_ns


def test_a_stall_that_delivers_images_first_drops_imu_in_the_old_code_but_not_here():
    ev = arrivals(10.0, stall_at=5.0, stall_s=1.2, images_first=True)
    legacy_dropped, _ = run_legacy(ev)
    assert legacy_dropped > 400                       # about 1.2 s x 460 Hz: the T4 failure mode
    s, out = run_sync(ev)
    assert s.n["imu_late"] == 0
    assert s.n["frames_timeout"] == 0                 # the backlog's IMU arrives right behind its images
    _, cam = streams(10.0)
    assert [r.t_ns for r in out] == cam               # nothing lost, still in order


def test_imu_first_after_a_stall_is_fine_in_both():
    ev = arrivals(10.0, stall_at=5.0, stall_s=1.2, images_first=False)
    assert run_legacy(ev)[0] == 0
    s, _ = run_sync(ev)
    assert s.n["imu_late"] == 0


def test_a_dead_imu_never_freezes_tracking():
    _, cam = streams(3.0)
    s = ImuFrameSync(wait_s=0.08)
    released = []
    for t in cam:                                     # images only, handled as they are captured
        now = t / NS
        s.add_frame(t, "L", "R", now)
        released += s.ready(now)
        released += s.ready(now + 0.1)                # the node's timer, 100 ms later
    assert [r.t_ns for r in released] == cam
    assert all(r.timed_out and r.imu == [] for r in released)
    assert s.n["frames_timeout"] == len(cam)


def test_a_frame_waits_at_most_wait_s():
    s = ImuFrameSync(wait_s=0.08)
    s.add_imu(100, 0, 0)
    s.add_frame(200, "L", "R", now_s=1.0)
    assert s.ready(1.05) == []                        # no IMU at or after 200 yet, still inside wait_s
    r = s.ready(1.081)
    assert len(r) == 1 and r[0].timed_out and [x[0] for x in r[0].imu] == [100]


def test_imu_older_than_a_released_frame_is_late_and_counted():
    s = ImuFrameSync(wait_s=0.0)
    s.add_frame(200, "L", "R", 0.0)
    assert len(s.ready(0.0)) == 1                     # released on timeout with no IMU
    s.add_imu(150, 0, 0)
    assert s.n["imu_late"] == 1 and s.imu == []


def test_frames_release_in_stamp_order_even_if_they_arrive_out_of_order():
    s = ImuFrameSync()
    s.add_frame(300, "L3", "R3", 0.0)
    s.add_frame(200, "L2", "R2", 0.0)
    for t in (150, 250, 350):
        s.add_imu(t, 0, 0)
    out = s.ready(0.0)
    assert [r.t_ns for r in out] == [200, 300]
    assert [[x[0] for x in r.imu] for r in out] == [[150], [250]]
