"""refnav.core + refnav.depth, without ROS: a fake PX4 (kinematic, obeys Offboard/arm/land) and a
ray-cast depth camera in a world of walls. Each safety behaviour in core.py's docstring is pinned
here, and the same flight on two different clocks (simulation seconds, Unix seconds) must make
the same decisions."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from devsim.world import DEPTH_BORDER_PX  # noqa: E402
from refnav import core as C  # noqa: E402
from refnav import depth as D  # noqa: E402
from refnav import follow as F  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    Wall = tuple[int, float, float, float]      # (axis, value, lo, hi): see render_depth
    LogRow = tuple[float, str, bool, tuple[str, ...], np.ndarray]   # (t, phase, stream, commands, pos)

W, H, FX = 160, 100, 106.0                  # a small 74 deg HFOV camera
FLOOR_D = 0.18                              # the floor below the body origin of a landed aircraft (as in the simulator)
K = (FX, FX, W / 2, H / 2)
R_BODY_CAM = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # FLU <- optical, forward camera
T_BODY_CAM = np.array([0.16, 0.0, 0.0])


def q_yaw(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def render_depth(pos_ned: Sequence[float] | np.ndarray, yaw: float, walls: Iterable[Wall]) -> np.ndarray:
    """Depth (m, NaN = no return) seen by the forward camera of a level aircraft at pos_ned/yaw.
    walls: [(axis, value, lo, hi)] vertical planes, axis 0 = n = value, axis 1 = e = value,
    extending lo..hi along the other horizontal axis and from the floor (d = 0) up to d = -2.5."""
    vs, us = np.mgrid[0:H, 0:W].astype(np.float64)
    d_opt = np.stack([(us - K[2]) / K[0], (vs - K[3]) / K[1], np.ones_like(us)], -1)
    d_flu = d_opt @ R_BODY_CAM.T
    d_frd = d_flu * np.array([1.0, -1.0, -1.0])
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    d_ned = d_frd @ Rz.T
    o = np.asarray(pos_ned) + Rz @ (T_BODY_CAM * np.array([1.0, -1.0, -1.0]))
    best = np.full((H, W), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        tf = (FLOOR_D - o[2]) / d_ned[..., 2]               # the floor, 0.18 m below the body when landed
        best = np.where((tf > 0), np.minimum(best, tf), best)
        for axis, val, lo, hi in walls:
            t = (val - o[axis]) / d_ned[..., axis]
            p = o + d_ned * t[..., None]
            other = p[..., 1 - axis]
            hit = (t > 0) & (other >= lo) & (other <= hi) & (p[..., 2] <= 0) & (p[..., 2] >= -2.5)
            best = np.where(hit, np.minimum(best, t), best)
    best[~np.isfinite(best)] = np.nan
    return best.astype(np.float32)


class FakePX4:
    """Obeys the mission the way PX4 does at this level of detail: Offboard only after setpoints
    have been streaming, arm, track the setpoint at <= 1 m/s and 60 deg/s, AUTO_LAND descends
    and disarms on the floor."""

    def __init__(self, pos: Sequence[float] = (0.0, 0.0, 0.0), yaw: float = 0.0) -> None:
        self.pos: np.ndarray = np.array(pos, float)
        self.yaw: float = yaw
        self.armed, self.nav, self.landed = False, 4, True
        self.stream_since: float | None = None

    def apply(self, t: float, out: C.Output, dt: float) -> None:
        if out.stream:
            self.stream_since = t if self.stream_since is None else self.stream_since
        else:
            self.stream_since = None
        for c in out.commands:
            if c == "offboard" and self.stream_since is not None and t - self.stream_since >= 0.5:
                self.nav = C.NAV_OFFBOARD
            elif c == "arm":
                self.armed = True
            elif c == "land":
                self.nav = C.NAV_AUTO_LAND
            elif c == "disarm_force" and self.landed:
                self.armed = False
        if self.armed and self.nav == C.NAV_OFFBOARD and out.setpoint is not None:
            n, e, d, yaw = out.setpoint
            step = np.array([n, e, d]) - self.pos
            dist = np.linalg.norm(step)
            self.pos += step * min(1.0, 1.0 * dt / dist) if dist > 1e-9 else 0
            err = float(F.wrap(yaw - self.yaw))
            self.yaw += max(-math.radians(60) * dt, min(math.radians(60) * dt, err))
            self.landed = self.pos[2] > -0.05
        elif self.nav == C.NAV_AUTO_LAND and self.armed:
            self.pos[2] = min(0.0, self.pos[2] + 0.7 * dt)
            if self.pos[2] >= 0.0:
                self.landed, self.armed = True, False

    def vehicle(self) -> C.Vehicle:
        return C.Vehicle(pos=tuple(self.pos), heading=self.yaw, pos_ok=True, q=q_yaw(self.yaw),
                         armed=self.armed, nav_state=self.nav, range_m=max(0.0, -float(self.pos[2])))


ROOM = [(0, 15.0, -15.0, 15.0), (0, -15.0, -15.0, 15.0), (1, 15.0, -15.0, 15.0), (1, -15.0, -15.0, 15.0)]


def fly(params: C.Params, walls: Iterable[Wall] = (), t0: float = 0.0, seconds: float = 120.0, dt: float = 0.05,
        health: Callable[[float], bool | None] = lambda t: True,
        depth_on: Callable[[float], bool] = lambda t: True,
        nav_override: Callable[[float], int | None] | None = None, yaw0: float = 0.0,
        phantom: Callable[[float], bool] = lambda t: False,
        depth_fn: Callable[[float, np.ndarray], np.ndarray] | None = None,
        on_tick: Callable[[float, C.Mission], object] | None = None
        ) -> tuple[C.Mission, FakePX4, list[LogRow]]:
    """depth_fn(t, img) -> img replaces what the camera delivers (blank, sparse, ...);
    on_tick(t, mission) runs before each step (outside requests)."""
    walls = list(walls) + ROOM                         # indoors: a 30 x 30 m room around everything
    if params.depth_border_px == C.Params().depth_border_px:
        params.depth_border_px = DEPTH_BORDER_PX
    m = C.Mission(params)
    px4 = FakePX4(yaw=yaw0)
    log: list[LogRow] = []
    t = 0.0
    next_depth = 0.0
    while t < seconds and m.phase != "done":
        now = t0 + t
        if health(t) is not None:
            m.on_health(now, health(t))
        if t >= next_depth and depth_on(t):
            img = render_depth(px4.pos, px4.yaw, walls)
            if phantom(t):
                img[6:40, 56:104] = 0.6                  # a stereo false match: 0.6 m, near the top, ONE frame
            if depth_fn is not None:
                img = depth_fn(t, img)
            m.on_depth(now, img, K, R_BODY_CAM, T_BODY_CAM, q_yaw(px4.yaw), now=now)
            next_depth = t + 1 / 7.5
        v = px4.vehicle()
        if nav_override is not None and nav_override(t) is not None:
            px4.nav = nav_override(t)
            v.nav_state = px4.nav
        if on_tick is not None:
            on_tick(t, m)
        out = m.step(now, v)
        log.append((t, out.phase, out.stream, tuple(out.commands), px4.pos.copy()))
        px4.apply(now, out, dt)
        t += dt
    return m, px4, log


ROUTE = [(3.0, 0.0), (3.0, 3.0)]            # forward 3 m, then 3 m to the left


def test_nominal_route_completes_and_stops_streaming_once_px4_lands() -> None:
    m, px4, log = fly(C.Params(route=ROUTE))
    assert m.result == "complete", (m.result, m.why)
    assert m.follower.points_passed == 2
    phases = [p for _, p, *_ in log]
    assert phases.index("arm") < phases.index("climb") < phases.index("follow") < phases.index("land")
    # the takeoff heading is north (yaw 0): forward 3 m = n 3; left 3 m = e -3
    assert np.allclose(m.targets[1], (3.0, -3.0), atol=1e-9)
    land_auto = [i for i, (_, p, *_r) in enumerate(log) if p == "land" and px4.nav == C.NAV_AUTO_LAND]
    first_auto = next(i for i, (t, p, s, c, pos) in enumerate(log) if p == "land" and not s)
    assert all(not s for _, p, s, *_ in log[first_auto:])            # never streams again
    assert not px4.armed and px4.landed


def test_route_is_in_the_takeoff_frame() -> None:
    m, px4, log = fly(C.Params(route=[(2.0, 0.0)]), yaw0=math.radians(90))
    assert m.result == "complete"
    assert np.allclose(m.targets[0], (0.0, 2.0), atol=1e-9)          # facing east: forward = +e


def test_no_health_never_arms() -> None:
    m, px4, log = fly(C.Params(route=ROUTE, wait_s=20.0), health=lambda t: None)
    assert m.result.startswith("refused") and not px4.armed
    assert all("arm" not in c for *_x, c, _p in log)


def test_unhealthy_estimator_in_the_air_lands() -> None:
    m, px4, log = fly(C.Params(route=ROUTE), health=lambda t: t < 12.0)
    assert m.result.startswith("landed: estimator"), m.result
    assert m.follower.points_passed < 2 and not px4.armed


def test_health_gap_restarts_the_quiet_window() -> None:
    h = C.Health(quiet_s=3.0, stale_s=1.0)
    for k in range(40):
        h.on_msg(k * 0.1, True)
    assert h.arm_ok(3.95)[0]
    h.on_msg(6.0, True)                                  # 2 s silent, then healthy again
    assert not h.arm_ok(6.05)[0]


def test_no_translation_without_live_depth_and_landing_when_it_stays_silent() -> None:
    # depth stops at t = 14 s (mid-route): the reference freezes, then the aircraft lands
    m, px4, log = fly(C.Params(route=[(12.0, 0.0)], speed_mps=0.5), depth_on=lambda t: t < 14.0)
    assert m.result.startswith("landed: depth silent"), m.result
    pos_at_cut = [pos for t, p, s, c, pos in log if abs(t - 14.6) < 0.026][0]
    before_land = [pos for t, p, s, c, pos in log if p == "follow" and t > 15.0]
    assert before_land and max(abs(q[0] - pos_at_cut[0]) for q in before_land) < 0.3
    assert m.counts["depth_holds"] > 0


def test_obstacle_ahead_stops_and_lands_where_it_is() -> None:
    wall = [(0, 4.5, -5.0, 5.0)]                          # a wall across the route at n = 4.5
    m, px4, log = fly(C.Params(route=[(8.0, 0.0)]), walls=wall)
    assert m.result.startswith("landed: route blocked"), m.result
    assert px4.pos[0] < 4.5 - 0.6                         # stopped short of the wall
    assert m.counts["obstacle_stops"] == 1


def test_a_wall_beside_the_route_is_not_an_obstacle() -> None:
    walls = [(1, 1.0, -5.0, 10.0), (1, -1.0, -5.0, 10.0)]   # a 2 m wide corridor along the route
    m, px4, log = fly(C.Params(route=[(5.0, 0.0)]), walls=walls)
    assert m.result == "complete", m.result


def test_pilot_takeover_is_never_fought() -> None:
    # 13 s in, someone switches PX4 to Position mode (nav_state 2)
    m, px4, log = fly(C.Params(route=ROUTE), nav_override=lambda t: 2 if t >= 13.0 else None)
    assert m.result.startswith("yielded"), m.result
    after = [(s, c) for t, p, s, c, pos in log if t > 13.0]
    assert all(not s and not c for s, c in after)


def test_same_decisions_on_any_clock() -> None:
    # a tick that is exact in binary (1/32 s) and thresholds off its grid: at Unix-time magnitudes
    # a 50 ms tick is not representable, and a decision that lands EXACTLY on a threshold may
    # round either way - float rounding of the inputs, not a dependence on the clock
    prm = dict(route=ROUTE, health_quiet_s=3.013, stream_lead_s=1.013, cmd_resend_s=1.013)
    a = fly(C.Params(**prm), t0=0.0, dt=1 / 32)
    b = fly(C.Params(**prm), t0=1.76e9, dt=1 / 32)       # Unix time, as on hardware
    pa = [(round(t, 3), p, c) for t, p, s, c, pos in a[2]]
    pb = [(round(t, 3), p, c) for t, p, s, c, pos in b[2]]
    assert pa == pb and a[0].result == b[0].result == "complete"


# ------------------------------------------------------------------ depth.py
def _straight(length: float, bearing: float = 0.0, step: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Path samples (forward, right) along a bearing from the aircraft, and their distances."""
    along = np.arange(0.0, length + 1e-9, step)
    return np.stack([along * math.cos(bearing), along * math.sin(bearing)], 1), along


def _free(img: np.ndarray, path: tuple[np.ndarray, np.ndarray], q=(1.0, 0.0, 0.0, 0.0), K_=K, stride: int = 2,
          hits: int = 3) -> tuple[float, bool]:
    f = D.DepthFrame(img, K_, R_BODY_CAM, T_BODY_CAM, D.level_rotation(q), stride=stride)
    return D.path_clearance(f, f.points(0.3, 6.0), *path, 0.6, 0.4, hits, 0.6, blind_m=0.9)


def test_path_clearance_levels_a_tilted_aircraft_and_ignores_the_floor() -> None:
    walls = [(0, 3.0, -5.0, 5.0)]
    pos = (0.0, 0.0, -1.2)
    img = render_depth(pos, 0.0, walls)
    free, obstacle = _free(img, _straight(4.0))
    assert obstacle and free == pytest.approx(3.0 - 0.6, abs=0.06)    # the wall, a corridor half-width early
    q = (math.cos(math.radians(3.0)), 0.0, math.sin(math.radians(3.0)), 0.0)    # the same image, 6 deg nose-up
    assert _free(img, _straight(4.0), q)[1]
    free, obstacle = _free(render_depth(pos, 0.0, []), _straight(4.0))
    assert not obstacle                                               # the floor is never an obstacle


def test_an_obstacle_counts_only_beside_the_path_it_is_on() -> None:
    walls = [(0, 2.0, 0.8, 1.2)]                           # a post 2 m ahead, 1 m to the right
    img = render_depth((0.0, 0.0, -1.2), 0.0, walls)
    assert not _free(img, _straight(3.0))[1]
    assert _free(img, _straight(3.0, math.atan2(1.0, 2.0)))[1]


def test_decode_depth_encodings() -> None:
    mm = np.array([[0, 1000], [2500, 65535]], np.uint16)
    a = D.decode_depth("16UC1", 2, 2, 4, mm.tobytes())
    assert np.isnan(a[0, 0]) and a[0, 1] == pytest.approx(1.0) and a[1, 0] == pytest.approx(2.5)
    m = np.array([[np.nan, 1.5], [0.0, np.inf]], np.float32)
    b = D.decode_depth("32FC1", 2, 2, 8, m.tobytes())
    assert b[0, 1] == pytest.approx(1.5) and np.isnan(b[0, 0]) and np.isnan(b[1, 0]) and np.isnan(b[1, 1])
    with pytest.raises(ValueError):
        D.decode_depth("rgb8", 1, 1, 3, b"\0\0\0")


def test_a_one_frame_false_match_does_not_stop_the_route() -> None:
    hits: list[float] = []

    def phantom(t: float) -> bool:                                   # one frame every ~2 s while flying
        on = 12.0 < t and int(t * 7.5) % 15 == 0
        if on:
            hits.append(t)
        return on
    prm = dict(route=[(4.0, 0.0), (4.0, 3.0)], depth_border_px=2, min_obstacle_px=1200)   # 24 sampled points: big enough
    m, px4, log = fly(C.Params(**prm), phantom=phantom)
    assert hits and m.result == "complete", m.result
    assert m.counts["unconfirmed_obstacles"] > 0       # it saw them, and did not believe a single frame
    one = fly(C.Params(**dict(prm, confirm_frames=1)), phantom=phantom)[0]
    assert one.result.startswith("landed: route blocked")   # without confirmation the same flight stops


def test_border_crop_removes_edge_rows() -> None:
    img = np.full((H, W), np.nan, np.float32)
    img[0:3, 60:100] = 0.6
    q = (1.0, 0.0, 0.0, 0.0)
    keep = D.DepthFrame(img, K, R_BODY_CAM, T_BODY_CAM, D.level_rotation(q), stride=1, border_px=0).points()
    crop = D.DepthFrame(img, K, R_BODY_CAM, T_BODY_CAM, D.level_rotation(q), stride=1, border_px=4).points()
    assert len(keep) == 120 and len(crop) == 0


def test_sparse_points_are_not_an_obstacle_but_a_pole_is() -> None:
    """min_obstacle_px: a few scattered depth returns do not stop the aircraft; a 5 cm pole does."""
    Kr = (842.0, 842.0, 640.0, 400.0)
    hits = max(3, round(C.Params().min_obstacle_px / 8 ** 2))
    img = np.full((800, 1280), 5.0, np.float32)           # the real camera's size, a wall 5 m ahead
    img[300:308, 600:616] = 0.63                           # 8 x 16 px of junk
    free, obstacle = _free(img, _straight(5.5), K_=Kr, stride=8, hits=hits)
    assert obstacle and free == pytest.approx(5.16 - 0.6, abs=0.06)      # the wall, not the junk at 0.63 m
    pole = np.full((800, 1280), 5.0, np.float32)
    w = int(0.05 / 3.0 * 842) + 1                          # a 5 cm pole 3 m ahead
    pole[100:700, 640 - w // 2:640 + w // 2 + 8] = 3.0
    free, obstacle = _free(pole, _straight(5.5), K_=Kr, stride=8, hits=hits)
    assert obstacle and free == pytest.approx(3.16 - 0.6, abs=0.06)


def test_a_px4_failsafe_landing_mid_route_is_never_streamed_over() -> None:
    """PX4 switching itself to AUTO_LAND during the route (a failsafe) is a takeover like a pilot's:
    the mission stops streaming at once (PX4's land detector reads the setpoint topic)."""
    m, px4, log = fly(C.Params(route=ROUTE), nav_override=lambda t: C.NAV_AUTO_LAND if t >= 13.0 else None)
    assert m.result.startswith("yielded"), m.result
    assert all(not s and not c for t, p, s, c, pos in log if t > 13.0)



# ------------------------------------------------- unusable depth is never permission to move
def _advance(log: Sequence[LogRow], t_from: float) -> float:
    """How far the aircraft moved forward (north) after t_from."""
    p = [pos for t, ph, st, c, pos in log if t >= t_from]
    return max(q[0] for q in p) - p[0][0] if p else 0.0


@pytest.mark.parametrize("kind", ["blank", "sparse", "future", "stale"])
def test_unusable_depth_never_moves_the_aircraft_and_ends_in_a_landing(kind: str) -> None:
    """From t = 11 s (the route has just started) every depth frame is unusable: blank (no
    return anywhere), sparse (5 % of pixels return), stamped 1 s in the future, or stamped 1 s
    in the past. The aircraft must not move along the route, and must land when no usable depth
    has arrived for depth_lost_land_s."""
    rng = np.random.default_rng(0)

    def bad(t: float, img: np.ndarray) -> np.ndarray:
        if t < 11.0:
            return img
        if kind == "blank":
            return np.full_like(img, np.nan)
        if kind == "sparse":
            out = np.full_like(img, np.nan)
            m = rng.random(img.shape) < 0.05
            out[m] = img[m]
            return out
        return img
    m = C.Mission(C.Params(route=[(6.0, 0.0)], speed_mps=0.5, depth_border_px=DEPTH_BORDER_PX))
    px4 = FakePX4()
    log, t, next_depth = [], 0.0, 0.0
    while t < 40.0 and m.phase != "done":
        m.on_health(t, True)
        if t >= next_depth:
            img = bad(t, render_depth(px4.pos, px4.yaw, ROOM))
            stamp = t
            if t >= 11.0 and kind == "future":
                stamp = t + 1.0
            if t >= 11.0 and kind == "stale":
                stamp = t - 1.0
            m.on_depth(stamp, img, K, R_BODY_CAM, T_BODY_CAM, q_yaw(px4.yaw), now=t)
            next_depth = t + 1 / 7.5
        out = m.step(t, px4.vehicle())
        log.append((t, out.phase, out.stream, tuple(out.commands), px4.pos.copy()))
        px4.apply(t, out, 0.05)
        t += 0.05
    t_go = next(t for t, ph, *_ in log if ph == "follow")
    assert t_go < 11.0, t_go                              # it was flying the route when depth went bad
    pos_at_11 = next(pos for t, *_r, pos in log if t >= 11.0)
    ahead = max(pos[0] for t, *_r, pos in log if t >= 11.0) - pos_at_11[0]
    # what was already commanded (half the lookahead at most) may complete; nothing beyond it
    assert ahead < m.p.look_max_m / 2 + 0.05, ahead
    assert m.result.startswith("landed: depth silent"), m.result
    key = {"blank": "depth_unusable", "sparse": "depth_unusable", "future": "depth_future", "stale": "depth_stale"}[kind]
    assert m.counts[key] > 0


def test_a_blank_frame_does_not_refresh_depth_and_says_why() -> None:
    m = C.Mission(C.Params(route=[(3.0, 0.0)], depth_border_px=DEPTH_BORDER_PX))
    ok = m.on_depth(1.0, render_depth(np.array([0, 0, -1.2]), 0.0, ROOM), K, R_BODY_CAM, T_BODY_CAM, q_yaw(0.0), now=1.0)
    assert ok and m.depth_t == 1.0
    assert not m.on_depth(1.5, np.full((H, W), np.nan, np.float32), K, R_BODY_CAM, T_BODY_CAM, q_yaw(0.0), now=1.5)
    assert m.depth_t == 1.0 and m.counts["depth_unusable"] == 1


def test_a_path_out_of_view_is_unknown_not_free() -> None:
    """A good frame looking AWAY from the path keeps the camera alive but is no evidence for it."""
    img = render_depth(np.array([0, 0, -1.2]), 0.0, ROOM)
    free, obstacle = _free(img, _straight(3.0, math.radians(90)))
    assert not obstacle and free < 1.0                     # stops at the first sample past the blind range


def test_coverage_counts_far_returns_and_ignores_missing_ones() -> None:
    far = np.full((H, W), 9.0, np.float32)                 # everything 9 m away: beyond 6 m, still a return
    free, obstacle = _free(far, _straight(2.0), stride=4)
    assert not obstacle and free == pytest.approx(2.0)
    holed = far.copy()
    holed[30:70, 60:100] = np.nan                          # the middle of the image returned nothing
    free, obstacle = _free(holed, _straight(2.0), stride=4)
    assert not obstacle and free < 1.9


# ---- the interactive runtime's takeoff/land example and landing requests (2026-10-17) ----------------
def test_takeoff_hover_land_with_an_empty_route() -> None:
    m, px4, log = fly(C.Params(route=[], height_m=1.2, hover_s=4.0))
    assert m.result == "complete", (m.result, m.why)
    hover = [t for t, p, *_ in log if p == "hover"]
    assert hover and hover[-1] - hover[0] >= 4.0 - 0.1               # held the whole hover_s
    assert max(-pos[2] for *_, pos in log) == pytest.approx(1.2, abs=0.2)
    assert not px4.armed and px4.landed


def test_hover_zero_goes_straight_from_the_climb_to_the_path() -> None:
    m, px4, log = fly(C.Params(route=ROUTE))
    assert "hover" not in {p for _, p, *_ in log} and m.result == "complete"


def test_a_landing_request_in_the_air_lands_through_the_mission() -> None:
    asked: list[float] = []

    def ask(t: float, m: C.Mission) -> None:
        if m.phase == "hover" and not asked:
            asked.append(t)
            m.request_land("operator")
    m, px4, log = fly(C.Params(route=[], hover_s=30.0), on_tick=ask)
    assert asked and m.result == "landed: landing requested: operator", m.result
    land = [t for t, p, *_ in log if p == "land"]
    assert land[0] - asked[0] < 0.2                                   # at once, not after the hover
    assert not px4.armed and px4.landed


def test_a_landing_request_before_arming_never_arms() -> None:
    m, px4, log = fly(C.Params(route=[(3.0, 0.0)]), on_tick=lambda t, m: m.request_land("stop") if t > 0.5 else None)
    assert m.result.startswith("refused: landing requested: stop"), m.result
    assert not any("arm" in c for *_, c, _ in log) and not px4.armed


def _sky(t: float, img: np.ndarray) -> np.ndarray:
    """The field case (acc2 run_001): the camera works but sees mostly sky - depth on ~15 % of the pixels."""
    img = img.copy()
    img[: int(H * 0.85)] = np.nan
    return img


def test_too_little_depth_never_arms_and_says_how_much_there_was() -> None:
    """The wait names the measured fraction and the floor - from the start, and in the final refusal."""
    seen: list[str] = []
    m, px4, log = fly(C.Params(route=[], wait_s=20.0), depth_fn=_sky, seconds=30.0,
                      on_tick=lambda t, m: seen.append(m.why) if 5.0 < t < 5.1 else None)
    assert not px4.armed
    assert "depth on " in seen[0] and "this mission needs 30% (min_valid_frac)" in seen[0], seen
    assert m.result.startswith("refused: waiting for live depth: the latest frame was refused: depth on"), m.result
    st = m.status(30.0)
    assert st["depth_valid_frac"] < 0.3 and "min_valid_frac" in st["depth_reject"]
    m2, px4b, _ = fly(C.Params(route=[], min_valid_frac=0.05), depth_fn=_sky, seconds=60.0)
    assert m2.result == "complete" and m2.depth_reject is None, (m2.result, m2.why)


def test_no_depth_at_all_is_said_as_such() -> None:
    m, px4, log = fly(C.Params(route=[], wait_s=5.0), depth_on=lambda t: False, seconds=10.0)
    assert m.result == "refused: waiting for live depth: no depth frame has arrived after 5 s", m.result


class StuckPX4(FakePX4):
    """PX4 whose land detector never agrees (its estimate ran away on the pad): it only disarms when forced."""

    def apply(self, t: float, out: C.Output, dt: float) -> None:
        armed = self.armed
        super().apply(t, out, dt)
        if armed and not self.armed and "disarm_force" not in out.commands:
            self.armed = True


def test_touchdown_on_the_rangefinder_disarms_when_px4_will_not() -> None:
    import test_refnav_core as T
    real = T.FakePX4
    T.FakePX4 = StuckPX4
    try:
        m, px4, log = fly(C.Params(route=ROUTE))
    finally:
        T.FakePX4 = real
    assert m.result == "complete", (m.result, m.why)
    forced = [t for t, p, s, c, pos in log if "disarm_force" in c]
    assert forced and all(-pos[2] < 0.05 for t, p, s, c, pos in log if "disarm_force" in c)


def test_never_forces_a_disarm_in_the_air() -> None:
    m = C.Mission(C.Params(route=ROUTE))
    m.phase, m.t_phase, m.land_auto_seen = "land", 0.0, True
    v = C.Vehicle(pos=(0.0, 0.0, -0.8), heading=0.0, pos_ok=True, q=q_yaw(0.0), armed=True,
                  nav_state=C.NAV_AUTO_LAND, range_m=0.8)
    for k in range(200):
        out = m.step(k * 0.05, v)
        assert "disarm_force" not in out.commands
    v.range_m = None                                    # no reading is not a touchdown either
    for k in range(200, 400):
        assert "disarm_force" not in m.step(k * 0.05, v).commands


# ---- the path, its params and its route ----------------------------------------------------------
def test_the_path_is_built_on_the_ground_before_arming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building takes longer than a tick (and PX4 drops Offboard after 1 s of silence): never in the air."""
    builds: list[tuple[str, float | None]] = []
    profile = F.speed_profile
    mission: list[C.Mission] = []

    def watch(t: float, m: C.Mission) -> None:
        if not mission:
            mission.append(m)
            build = m.path_builder
            m.path_builder = lambda pts: builds.append(("path", m.phase, m.t_cmd)) or build(pts)
        if m.phase == "arm":
            assert m.path is not None and m.speed is not None
    monkeypatch.setattr(F, "speed_profile", lambda *a, **k: builds.append(("speed", mission[0].phase,
                                                                         mission[0].t_cmd)) or profile(*a, **k))
    m, px4, log = fly(C.Params(route=ROUTE), on_tick=watch)
    assert m.result == "complete", (m.result, m.why)
    assert builds == [("path", "wait", None), ("speed", "wait", None)]     # once, before any control request
    assert m.follower.path is m.path


def test_the_plan_waits_for_a_settled_heading() -> None:
    """The route is laid out along the take-off heading: a plan made before the estimator settles is rotated."""
    headings: list[float] = []

    def watch(t: float, m: C.Mission) -> None:
        if m.home is not None and not headings:
            headings.append(m.home[3])
            assert m.health.arm_ok(t)[0], "planned before the estimator was healthy"
    m, px4, log = fly(C.Params(route=ROUTE), on_tick=watch)
    assert m.result == "complete" and headings


def test_requalifying_on_the_ground_keeps_the_path() -> None:
    """A requalify (here: depth stale once at arm) must not rebuild: a rebuild blocks the tick, staling the inputs it
    then requalifies on, which loops forever."""
    builds: list[str] = []
    stale_once: list[bool] = []
    wrapped: list[C.Mission] = []

    def watch(t: float, m: C.Mission) -> None:
        if not wrapped:
            wrapped.append(m)
            build = m.path_builder
            m.path_builder = lambda pts: builds.append(m.phase) or build(pts)
        if m.phase == "arm" and not stale_once:
            stale_once.append(True)
            m.depth_ready_since = None                  # depth drops out once between readiness and arming
    m, px4, log = fly(C.Params(route=ROUTE), on_tick=watch)
    assert m.result == "complete", (m.result, m.why)
    assert stale_once and builds == ["wait"]


def test_a_reversal_route_flies_to_its_point_and_back() -> None:
    cmds: list[F.Command] = []
    m, px4, log = fly(C.Params(route=[(2.0, 0.0), (0.0, 0.0)]),
                      on_tick=lambda t, m: cmds.append(m.cmd) if m.cmd is not None else None)
    assert m.result == "complete" and m.follower.points_passed == 2, (m.result, m.why)
    assert min(math.hypot(pos[0] - 2.0, pos[1]) for *_, pos in log) <= m.p.arrive_m
    assert np.isfinite([[*c.pos, *c.vel, *c.acc, c.yaw] for c in cmds]).all()


def _never_armed(log: Sequence[LogRow]) -> bool:
    return not any("arm" in c for *_, c, _ in log)


@pytest.mark.parametrize("bad", ["short", "nan", "raises"])
def test_a_degenerate_path_refuses_before_arming(bad: str) -> None:
    def builder(pts: list[tuple[float, float]]) -> F.Path:
        if bad == "raises":
            return F.fillet_path([pts[0], pts[0]], 0.3)
        if bad == "short":
            return F.fillet_path([pts[0], (pts[0][0] + 0.005, pts[0][1])], 0.3)
        p = F.fillet_path(pts, 0.3)
        p.k[3] = math.nan
        return p

    def swap(t: float, m: C.Mission) -> None:
        m.path_builder = builder
    m, px4, log = fly(C.Params(route=ROUTE), on_tick=swap)
    assert m.result.startswith("refused: "), m.result
    assert ("no path" if bad == "raises" else "degenerate path") in m.result
    assert _never_armed(log) and not px4.armed


@pytest.mark.parametrize("route, why", [
    ([(0.005, 0.0), (2.0, 0.0)], "point 1 (0.005, 0.0) is within 1 cm of the take-off point"),
    ([(2.0, 0.0), (2.0, 0.009)], "point 2 (2.0, 0.009) is within 1 cm of point 1")])
def test_a_route_with_a_repeated_point_is_refused_at_load(route: list[tuple[float, float]], why: str) -> None:
    with pytest.raises(ValueError, match=re.escape(why)):
        C.params_from_doc({"route": route})


@pytest.mark.parametrize("lo, hi", [("tight_m", "roomy_m"), ("xtrack_slow_m", "xtrack_stop_m"),
                                    ("face_tol_deg", "yaw_slow_deg"), ("eph_ok_m", "eph_stop_m")])
def test_params_that_contradict_each_other_are_refused(lo: str, hi: str) -> None:
    with pytest.raises(ValueError, match=f"{lo} .* must be less than {hi}"):
        C.params_from_doc({"route": ROUTE}, {lo: getattr(C.Params(), hi)})


def test_aircraft_params_win_over_the_route_file() -> None:
    doc = {"route": [[1, 0]], "params": {"speed_mps": 0.4, "yaw_rate_dps": 90, "hold_s": 0.2}}
    prm, unknown = C.params_from_doc(doc, {"yaw_rate_dps": 45.0, "depth_topic": "/x", "min_obstacle_px": 900.0})
    assert prm.route == [(1.0, 0.0)] and prm.speed_mps == 0.4 and prm.yaw_rate_dps == 45.0
    assert prm.min_obstacle_px == 900 and isinstance(prm.min_obstacle_px, int)
    assert unknown == ["hold_s"]                     # the aircraft's other parameters are not the route's business


def test_devsim_flies_the_aircraft_params() -> None:
    from devsim.run import load_params
    from main import HEIGHT_M, HOVER_S
    from refnav.launch import flight_params
    root = Path(__file__).resolve().parents[1]
    prm = load_params(root / "route.json")
    assert (prm.height_m, prm.hover_s, prm.depth_border_px) == (HEIGHT_M, HOVER_S, DEPTH_BORDER_PX)
    assert prm.yaw_rate_dps == flight_params(root)["yaw_rate_dps"]


def test_blind_arms_and_completes_with_no_depth_messages_at_all() -> None:
    m, px4, log = fly(C.Params(route=ROUTE, depth_mode="blind"), depth_on=lambda t: False)
    assert m.result == "complete", (m.result, m.why)
    assert m.follower.points_passed == 2 and m.counts["depth_frames"] == 0 and not px4.armed


def test_blind_never_stops_on_an_obstacle() -> None:
    wall = [(0, 4.5, -5.0, 5.0)]
    m, px4, log = fly(C.Params(route=[(8.0, 0.0)], depth_mode="blind"), walls=wall)
    assert m.counts["obstacle_stops"] == 0
    assert not (m.result or "").startswith("landed: route blocked")
    assert px4.pos[0] > 4.5 - 0.6                         # flew where assisted mode stopped


def test_blind_uses_no_map_align_correction() -> None:
    m = C.Mission(C.Params(route=ROUTE, depth_mode="blind"))
    m.on_correction(1.0, 0.5, 0.5, 0.2)
    assert (m.corr.x, m.corr.y, m.corr.yaw) == (0.0, 0.0, 0.0) and m.corr_t is None
    a = C.Mission(C.Params(route=ROUTE))
    a.on_correction(1.0, 0.5, 0.5, 0.2)
    assert a.corr.x == 0.5 and a.corr_t == 1.0


def test_assisted_is_the_default_and_depth_mode_is_validated() -> None:
    assert C.Params().depth_mode == "assisted"
    assert C.params_from_doc({"route": ROUTE}, {"depth_mode": "blind"})[0].depth_mode == "blind"
    with pytest.raises(ValueError, match="depth_mode must be one of"):
        C.params_from_doc({"route": ROUTE}, {"depth_mode": "sonar"})


def test_flight_yaml_documents_both_modes_and_the_depth_topic() -> None:
    from refnav.launch import flight_params
    fp = flight_params(Path(__file__).resolve().parents[1])
    assert fp["depth_mode"] == "assisted" and fp["depth_topic"] == "/camera/depth/image_raw"
    assert C.params_from_doc({"route": ROUTE}, fp)[0].depth_mode == "assisted"
