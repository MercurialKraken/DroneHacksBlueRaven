"""Wall-seek: closed-loop tests against a fake drone (velocity lag + dead time, optional constant body push like
skybeta-09's 0.1 m/s back-left) and a fake corridor depth sensor, on the tape course's Wall A geometry (pad frame:
launch at (0, 0) facing north; Wall A's face 1.677 m north, spanning east -0.228 .. 1.176)."""
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blueraven_wallseek import wallseek_core as W  # noqa: E402
from blueraven_wallseek import wallseek_depth as D  # noqa: E402

WALL_N, WALL_E0, WALL_E1 = 1.677, -0.228, 1.176
HALF = 0.34
WALL_B = ((-0.152, 0.768), (0.801, 1.658))      # pad (n, e): map (-1.747, 0.762) -> NE tip (-0.857, 1.715)


def seg_dist(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    t = max(0.0, min(1.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)))
    return math.hypot(px - ax - t * (bx - ax), py - ay - t * (by - ay))


class Sim:
    def __init__(self, p=None, tau=0.55, dead=0.25, push=(0.0, 0.0), wall=(WALL_N, WALL_E0, WALL_E1), far=3.5,
                 depth_ok=lambda t: True, nav_override=None, arm_t=2.0, seed=1):
        self.m = W.WallSeek(p or W.Params())
        self.tau, self.dead, self.push, self.wall, self.far = tau, dead, push, wall, far
        self.depth_ok, self.nav_override, self.arm_t = depth_ok, nav_override, arm_t
        self.pos, self.vel = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        self.queue, self.t, self.dt = [], 0.0, 0.05
        self.armed, self.offboard, self.t_press = False, False, None
        self.rng = random.Random(seed)
        self.min_wall = math.inf
        self.trace, self.depth_seq, self.t_depth, self.ahead = [], 0, -1e9, None
        self.twists = 0

    def corridor(self):
        n, e, _ = self.pos
        if self.wall is not None:
            wn, e0, e1 = self.wall
            if e + HALF > e0 and e - HALF < e1 and wn > n:
                return wn - n + self.rng.gauss(0, 0.01)
        return self.far

    def run(self, t_max=120.0):
        while self.t < t_max and self.m.phase not in ("done", "yielded"):
            self.step()
        return self

    def step(self):
        t = self.t
        if not self.armed and t >= self.arm_t and self.m.phase == "ready":
            self.armed = self.offboard = True
        nav = 14 if self.offboard else 2
        if self.nav_override and t >= self.nav_override[0]:
            nav, self.offboard = self.nav_override[1], False
        if round(t / 0.0333) != round((t - self.dt) / 0.0333) and self.depth_ok(t):    # ~30 Hz depth
            self.ahead, self.t_depth, self.depth_seq = self.corridor(), t, self.depth_seq + 1
        state = {"armed": self.armed, "offboard_active": self.offboard, "nav_state": nav, "px4_link": True}
        i = W.Inputs(now=t, pos=tuple(self.pos), vel=(self.vel[0], self.vel[1]), yaw_err=0.0, t_odom=t, state=state,
                     t_state=t, ahead_m=self.ahead, t_depth=self.t_depth, depth_seq=self.depth_seq)
        out = self.m.step(i)
        cmd = (0.0, 0.0, 0.0)
        if out.twist is not None and self.offboard:
            self.twists += 1
            cmd = out.twist[:3]
        self.queue.append((t, cmd))
        while len(self.queue) > 1 and self.queue[1][0] <= t - self.dead:
            self.queue.pop(0)
        c = self.queue[0][1] if self.queue[0][0] <= t - self.dead else (0.0, 0.0, 0.0)
        airborne = self.pos[2] > 0.005 or c[2] > 0
        for k in range(3):
            target = c[k] + (self.push[k] if k < 2 and airborne else 0.0)
            self.vel[k] += (target - self.vel[k]) * min(1.0, self.dt / self.tau)
            self.pos[k] += self.vel[k] * self.dt
        if self.pos[2] < 0:
            self.pos[2], self.vel[2] = 0.0, 0.0
        if self.pos[2] <= 0.0 and self.armed and c[2] <= -0.3:            # PX4's land detector: pressing on the ground
            self.t_press = self.t_press if self.t_press is not None else t
            if t - self.t_press >= 1.0:
                self.armed = self.offboard = False
        else:
            self.t_press = None
        if self.wall is not None:
            wn, e0, e1 = self.wall
            ex = min(max(self.pos[1], e0), e1)
            if self.pos[0] <= wn:
                self.min_wall = min(self.min_wall, math.hypot(wn - self.pos[0], self.pos[1] - ex) if ex != self.pos[1]
                                    else wn - self.pos[0])
        self.min_b = min(getattr(self, "min_b", math.inf), seg_dist(self.pos[:2], *WALL_B))
        self.trace.append((t, self.m.phase, tuple(self.pos)))
        self.t += self.dt


def phase_span(sim, phase):
    ts = [r for r in sim.trace if r[1] == phase]
    return ts[0], ts[-1]


@pytest.mark.parametrize("plant", ["drone", "t4"])
def test_full_mission_stops_translates_lands_past_the_end(plant):
    p = W.Params()
    kw = {}
    if plant == "t4":
        p.lag_s, kw = 1.5, {"tau": 0.9, "dead": 0.6}
    s = Sim(p, **kw).run()
    assert s.m.phase == "done", s.m.log
    seq = [x.split(" -> ")[1].split(":")[0] for x in s.m.log]
    assert seq[:7] == ["ready", "climb", "approach", "settle", "translate", "past", "land"], seq
    # standoff held along the wall: the centre stays 0.50 +- 0.06 m from Wall A's face while beside it
    beside = [r for r in s.trace if r[1] == "translate" and r[2][1] + HALF < WALL_E1]
    devs = [abs((WALL_N - r[2][0]) - p.standoff_m) for r in beside]
    assert beside and max(devs) < 0.06, max(devs)
    assert s.min_wall > 0.29 + 0.10, s.min_wall                      # never within 10 cm of contact
    # landed past the end: the drone's left side is clear of Wall A's end by about past_end
    _, last = phase_span(s, "land")
    assert last[2][1] - 0.29 > WALL_E1 + 0.2, last
    assert abs(last[2][2]) < 0.01                                      # on the ground


def test_body_push_like_skybeta09_is_absorbed_by_the_depth_standoff():
    s = Sim(push=(-0.07, -0.07)).run()                                 # back + left (west) ~0.1 m/s
    assert s.m.phase == "done", s.m.log
    beside = [r for r in s.trace if r[1] == "translate" and r[2][1] + HALF < WALL_E1 and r[0] > 0]
    assert max(abs((WALL_N - r[2][0]) - 0.5) for r in beside) < 0.10
    assert s.min_wall > 0.29 + 0.05
    assert any("past the wall's end" in x for x in s.m.log), s.m.log   # finished by the wall's end, not a timeout
    assert s.min_b > 0.29 + 0.05, s.min_b                              # Wall B's tip (unseen: behind-right) stays clear
    assert s.trace[-1][2][1] - 0.29 > WALL_E1 + 0.2, s.trace[-1]      # landed with its left side clear of the end


def test_no_wall_lands_before_the_forward_limit():
    s = Sim(wall=None).run()
    assert s.m.phase == "done"
    assert any("no wall within" in x for x in s.m.log), s.m.log
    assert max(r[2][0] for r in s.trace) < W.Params().max_forward_m + 0.15


def test_endless_wall_lands_at_the_right_limit():
    s = Sim(wall=(WALL_N, -1.0, 10.0)).run()
    assert any("did not end within" in x for x in s.m.log), s.m.log
    assert max(r[2][1] for r in s.trace) < W.Params().max_right_m + 0.15


def test_pilot_takeover_yields_and_stops_sending():
    s = Sim(nav_override=(20.0, 2)).run(40)
    assert s.m.phase == "yielded"
    out = s.m.step(W.Inputs(now=41, pos=(0, 0, 0.4), vel=(0, 0), yaw_err=0, t_odom=41, state={"armed": True,
                    "nav_state": 2}, t_state=41, ahead_m=1.0, t_depth=41))
    assert out.twist is None


def test_depth_dropout_holds_then_lands():
    s = Sim(depth_ok=lambda t: t < 12.0).run()
    assert any("depth silent" in x for x in s.m.log), s.m.log


def test_armed_before_ready_sends_nothing():
    m = W.WallSeek()
    st = {"armed": True, "nav_state": 2, "px4_link": True}
    out = m.step(W.Inputs(now=0, pos=(0, 0, 0), vel=(0, 0), yaw_err=0, t_odom=0, state=st, t_state=0,
                          ahead_m=None, t_depth=-1e9))
    assert m.phase == "done" and out.twist is None


def test_corridor_depth_sees_the_wall_and_ignores_the_floor():
    K, H, Wd = (401.5, 401.4, 313.8, 193.4), 400, 640
    v, u = np.mgrid[0:H, 0:Wd].astype(np.float32)
    fx, fy, cx, cy = K
    depth = np.full((H, Wd), 0.40, np.float32)                          # a wall plane 0.40 m in front of the camera
    floor_y = 0.43                                                      # floor 0.43 m below the camera
    ray_y = (v - cy) / fy
    with np.errstate(divide="ignore"):
        floor_z = np.where(ray_y > 0, floor_y / ray_y, np.inf)
    depth = np.minimum(depth, floor_z).astype(np.float32)
    a = D.ahead(depth, K)
    assert a.range_m == pytest.approx(0.40 + 0.1053, abs=0.01)
    open_ = np.minimum(np.full((H, Wd), np.nan, np.float32), floor_z)  # only floor: nothing ahead
    assert math.isinf(D.ahead(open_.astype(np.float32), K).range_m)
    half = depth.copy()
    half[:, : int(cx - fx * 0.5 / 0.40)] = np.nan                       # wall only on the far left: outside the corridor
    half[:, int(cx - fx * 0.5 / 0.40):] = np.minimum(np.full_like(half[:, int(cx - fx * 0.5 / 0.40):], 3.0),
                                                       floor_z[:, int(cx - fx * 0.5 / 0.40):])
    assert D.ahead(half, K).range_m > 2.0


def test_decode_16uc1_millimetres():
    a = np.array([[0, 400], [1000, 65535]], dtype="<u2")
    d = D.decode("16UC1", 2, 2, 4, a.tobytes())
    assert math.isnan(d[0, 0]) and d[0, 1] == pytest.approx(0.4) and d[1, 0] == pytest.approx(1.0)
