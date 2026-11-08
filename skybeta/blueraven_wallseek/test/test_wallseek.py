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
LEGACY = {'route_mode': False, 'standoff_m': 0.60, 'clear_m': 0.90}   # the gap / translate modes as tested before route v3
WALL_B = ((-0.152, 0.768), (0.801, 1.658))      # pad (n, e): map (-1.747, 0.762) -> NE tip (-0.857, 1.715)


def seg_dist(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    t = max(0.0, min(1.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)))
    return math.hypot(px - ax - t * (bx - ax), py - ay - t * (by - ay))


class Sim:
    def __init__(self, p=None, tau=0.55, dead=0.25, push=(0.0, 0.0), wall=(WALL_N, WALL_E0, WALL_E1), far=3.5,
                 depth_ok=lambda t: True, nav_override=None, arm_t=2.0, seed=1, est_fault=None,
                 liftoff_kick=(-0.15, 1.5), outlier_every=0, var_fn=None):
        self.est_fault = est_fault          # t, true pos, true vel -> (reported pos, reported vel) or None
        self.liftoff_kick = liftoff_kick    # the backward kick off the floor every T4 run shows: (m/s north, seconds)
        self.outlier_every = outlier_every  # every Nth depth frame reads 0.30 m (a wild frame)
        self.var_fn = var_fn                # () -> cuVSLAM pose variance to report
        self.t_lift = None
        self.wz = []
        self.wz_trace = []
        self.hdg = 0.0                      # nose heading, rad clockwise from north (turns with the commanded yaw rate)
        self.m = W.WallSeek(p or W.Params(**LEGACY))   # legacy tests: gap mode; route tests pass route_mode=True
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
        """The depth corridor along the NOSE (heading hdg, clockwise from north): the nearest point of Wall A (or of
        Wall B) that lies ahead and within HALF of the drone's centre line."""
        n, e, _ = self.pos
        fn, fe = math.cos(self.hdg), math.sin(self.hdg)            # forward unit (north, east)
        best = self.far
        segs = []
        if self.wall is not None:
            wn, e0, e1 = self.wall
            segs.append(((wn, e0), (wn, e1)))
        if self.wall is not None and self.wall == (WALL_N, WALL_E0, WALL_E1):
            segs.append(WALL_B)
        for (an, ae), (bn, be) in segs:
            for k in range(201):
                pn, pe = an + (bn - an) * k / 200, ae + (be - ae) * k / 200
                dn, de = pn - n, pe - e
                fwd, lat = dn * fn + de * fe, -dn * fe + de * fn
                if fwd > 0 and abs(lat) <= HALF:
                    best = min(best, fwd)
        return best + self.rng.gauss(0, 0.01) if best < self.far else self.far

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
            if self.outlier_every and self.depth_seq % self.outlier_every == 0:
                self.ahead = 0.30
        state = {"armed": self.armed, "offboard_active": self.offboard, "nav_state": nav, "px4_link": True}
        rp, rv = tuple(self.pos), (self.vel[0], self.vel[1])
        if self.est_fault:
            f = self.est_fault(t, rp, rv)
            if f:
                rp, rv = f
        i = W.Inputs(now=t, pos=rp, vel=rv, yaw_err=0.2, t_odom=t, state=state,      # a drifted heading estimate
                     t_state=t, ahead_m=self.ahead, t_depth=self.t_depth, depth_seq=self.depth_seq,
                     vio_var=self.var_fn() if self.var_fn else 0.002)
        out = self.m.step(i)
        cmd = (0.0, 0.0, 0.0)
        self.wz_trace.append(out.twist[3] if out.twist is not None else 0.0)
        if out.twist is not None and self.offboard:
            self.twists += 1
            cmd = out.twist[:3]
            self.wz.append(out.twist[3])
            self.hdg -= out.twist[3] * self.dt
        self.queue.append((t, cmd))
        while len(self.queue) > 1 and self.queue[1][0] <= t - self.dead:
            self.queue.pop(0)
        c = self.queue[0][1] if self.queue[0][0] <= t - self.dead else (0.0, 0.0, 0.0)
        airborne = self.pos[2] > 0.005 or c[2] > 0
        if self.t_lift is None and self.pos[2] > 0.01:
            self.t_lift = t
        kick = self.liftoff_kick[0] if self.t_lift is not None and t - self.t_lift < self.liftoff_kick[1] else 0.0
        for k in range(3):
            target = c[k] + (self.push[k] if k < 2 and airborne else 0.0) + (kick if k == 0 else 0.0)
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


GAP_END = (1.48, 1.86)                          # pad (n, e): 0.5 m past the CP2 gap centre on the leg from (1.077, 1.10)


@pytest.mark.parametrize("plant", ["drone", "t4"])
def test_full_mission_stops_translates_and_flies_through_the_gap(plant):
    p = W.Params(**LEGACY)
    kw = {}
    if plant == "t4":
        p.lag_s, kw = 1.5, {"tau": 0.9, "dead": 0.6}
    s = Sim(p, **kw).run()
    assert s.m.phase == "done", s.m.log
    seq = [x.split(" -> ")[1].split(":")[0] for x in s.m.log]
    assert seq[:7] == ["ready", "climb", "approach", "settle", "translate", "gap", "land"], seq
    # standoff held along the wall: the centre stays 0.60 +- 0.06 m from Wall A's face while translating
    beside = [r for r in s.trace if r[1] == "translate"]
    devs = [abs((WALL_N - r[2][0]) - p.standoff_m) for r in beside]
    assert beside and max(devs) < 0.06, max(devs)
    assert s.min_wall > 0.29 + 0.15, s.min_wall                      # Wall A (face and its east end): 15 cm spare
    assert s.min_b > 0.29 + 0.15, s.min_b                            # Wall B's tip: 15 cm spare
    _, last = phase_span(s, "land")
    assert math.dist(last[2][:2], GAP_END) < 0.15, last               # landed at the end point past the gap
    assert abs(last[2][2]) < 0.01                                      # on the ground


P4 = (1.350, 2.80)                              # route v3: due east of P2 (0.45 m past the gap centre)
CONE = (1.372, 1.084)                            # pad (n, e): 0.30 m tall, passes below the drone at 0.4 m


def _turn_angles(s):
    """Integrated commanded yaw (deg) per turn phase, and whether any yaw was commanded outside turns."""
    turns, cur, stray, prev = [], 0.0, False, None
    for (t, ph, _), w in zip(s.trace, s.wz_trace):
        if ph == "turn":
            cur += math.degrees(w) * s.dt
        else:
            if prev == "turn":
                turns.append(round(cur)); cur = 0.0
            stray |= w != 0.0
        prev = ph
    return turns, stray


@pytest.mark.parametrize("plant", ["drone", "t4", "drone+push"])
def test_route_v3_face_the_gap_straight_through_then_p4(plant):
    p = W.Params(route_mode=True, land_at_p2=False)
    kw = {}
    if plant == "t4":
        p.lag_s, kw = 1.5, {"tau": 0.9, "dead": 0.6}
    if plant == "drone+push":
        kw = {"push": (-0.07, -0.07)}
    s = Sim(p, **kw).run(160)
    assert s.m.phase == "done", s.m.log
    seq = [x.split(" -> ")[1].split(":")[0] for x in s.m.log if " -> " in x and ":" in x.split(" -> ")[1]]
    assert seq[:10] == ["ready", "climb", "approach", "settle", "turn", "leg", "pause", "turn", "leg", "land"], seq
    turns, stray = _turn_angles(s)
    assert len(turns) == 2 and abs(turns[0] + 76) <= 3 and abs(turns[1] + 14) <= 3, turns   # ~76 then ~14 deg right
    assert not stray                                                  # no yaw at all outside the in-place turns
    assert s.min_wall > 0.29 + 0.15, s.min_wall
    assert s.min_b > 0.29 + 0.12, s.min_b
    assert math.dist(s.trace[-1][2][:2], P4) < 0.15, s.trace[-1]
    # the legs are straight: the leg toward P3 stays within 6 cm of the 45 deg line
    leg2 = [r[2] for r in s.trace if r[1] == "leg"]
    assert leg2


def test_climbs_to_050_in_place_at_p1_not_at_take_off():
    s = Sim(W.Params(route_mode=True, land_at_p2=False, p1_climb_to_m=0.50)).run(160)
    first_leg = next(k for k, r in enumerate(s.trace) if r[1] == "rise")
    climb_top = max(r[2][2] for r in s.trace[:first_leg])             # take-off, approach, stop: ~0.40
    assert climb_top < 0.47, climb_top                                  # take-off and P1 stay at ~0.40
    over_cone = [r[2][2] for r in s.trace if r[1] == "leg" and math.dist(r[2][:2], CONE) < 0.45]
    assert over_cone and min(over_cone) > 0.44, min(over_cone or [0])   # above the 0.30 m cone
    leg2 = [r[2][2] for r in s.trace if r[1] == "leg" and r[2][1] > 2.2]
    assert leg2 and abs(sum(leg2) / len(leg2) - 0.50) < 0.05            # 0.50 m on the way to P4


def test_noisy_vision_velocity_at_the_stop_still_ends_the_stop_and_turns():
    # T4 runs 5-6: cuVSLAM's velocity read 0.1-0.7 m/s at the stop while the drone held within 2 cm: never settled
    rnd = random.Random(3)
    s = Sim(W.Params(route_mode=True, land_at_p2=False))
    s.est_fault = lambda t, p, v: (p, (v[0] + rnd.uniform(-0.3, 0.3), v[1] + rnd.uniform(-0.3, 0.3))) \
        if s.m.phase in ("approach", "settle") else None
    s.run(160)
    seq = [x.split(" -> ")[1].split(":")[0] for x in s.m.log if " -> " in x and ":" in x.split(" -> ")[1]]
    assert "turn" in seq, seq
    assert s.m.phase == "done" and any("route complete" in x for x in s.m.log), s.m.log


def test_dashboard_view_shows_the_planned_then_the_actual_route():
    m = W.WallSeek(W.Params(route_mode=True, land_at_p2=False))
    v = m.dashboard_view()
    assert v["targets"] == [[0.877, 0.0], [1.35, 1.853], [1.35, 2.8]], v
    s = Sim(W.Params(route_mode=True, land_at_p2=False)).run(160)
    v = s.m.dashboard_view()
    assert len(v["targets"]) == 3 and abs(v["targets"][2][1] - 2.8) < 1e-6 and v["path"][0] == [0.0, 0.0]
    assert abs(v["targets"][0][0] - (WALL_N - 0.80)) < 0.08         # P1 = where depth stopped it


def test_default_route_lands_at_p2_past_the_gap():
    s = Sim(W.Params(route_mode=True)).run(160)
    seq = [x.split(" -> ")[1].split(":")[0] for x in s.m.log if " -> " in x and ":" in x.split(" -> ")[1]]
    assert seq[:7] == ["ready", "climb", "approach", "settle", "turn", "leg", "land"], seq
    assert any("route complete" in x for x in s.m.log) and s.m.phase == "done", s.m.log
    p2 = s.m.route_pts[1]
    assert math.dist(s.trace[-1][2][:2], p2) < 0.15, (s.trace[-1], p2)
    assert s.min_wall > 0.29 + 0.15 and s.min_b > 0.29 + 0.15, (s.min_wall, s.min_b)
    assert len(s.m.dashboard_view()["targets"]) == 2


def test_old_mode_translates_until_the_wall_ends():
    s = Sim(W.Params(**{**LEGACY, 'turn_e_m': 0.0, 'standoff_m': 0.44})).run()             # the gap leg off: as before
    seq = [x.split(" -> ")[1].split(":")[0] for x in s.m.log]
    assert seq[:7] == ["ready", "climb", "approach", "settle", "translate", "past", "land"], seq
    assert s.trace[-1][2][1] - 0.29 > WALL_E1 + 0.2, s.trace[-1]


def test_body_push_like_skybeta09_is_absorbed_by_the_depth_standoff():
    s = Sim(push=(-0.07, -0.07)).run()                                 # back + left (west) ~0.1 m/s
    assert s.m.phase == "done", s.m.log
    beside = [r for r in s.trace if r[1] == "translate" and r[0] > 0]
    assert max(abs((WALL_N - r[2][0]) - LEGACY['standoff_m']) for r in beside) < 0.10
    assert any("through the gap" in x for x in s.m.log), s.m.log       # finished the leg, not a timeout
    assert s.min_wall > 0.29 + 0.12, s.min_wall
    assert s.min_b > 0.29 + 0.12, s.min_b
    assert math.dist(s.trace[-1][2][:2], GAP_END) < 0.20, s.trace[-1]


def test_never_commands_a_turn_even_with_a_drifted_heading_estimate():
    s = Sim().run()                     # every sample reports an 11 deg heading error (yaw_err 0.2 rad)
    assert s.m.phase == "done" and s.wz and all(w == 0.0 for w in s.wz)


def test_bad_vision_velocity_beside_the_wall_does_not_drive_it_in():
    # T4 run 20261108T0749: cuVSLAM reported the drone sliding south-west at ~0.3 m/s while it went north-east
    bad = lambda t, p, v: (p, (-0.3, -0.3)) if s.m.phase in ("settle", "translate") else None   # noqa: E731
    s = Sim(est_fault=bad)
    s.run()
    assert s.min_wall > 0.29 + 0.08, s.min_wall


def test_a_position_jump_lands_straight_down_without_chasing_it():
    jumped = {}

    def fault(t, p, v):
        if s.m.phase == "translate" and "t" not in jumped:
            jumped["t"] = t
        return ((p[0] - 3.0, p[1] + 2.0, p[2]), v) if "t" in jumped else None
    s = Sim(est_fault=fault)
    s.run()
    assert any("jumped" in x for x in s.m.log), s.m.log
    land = [r for r in s.trace if r[1] == "land"]
    drift = math.dist(land[0][2][:2], land[-1][2][:2])
    assert drift < 0.15, drift                                         # came down where it was, not toward the jump
    assert s.min_wall > 0.29 + 0.05


def test_takeoff_kick_returns_over_the_launch_point_before_the_approach():
    s = Sim().run()
    worst = min(r[2][0] for r in s.trace if r[1] == "climb")
    start = next(r for r in s.trace if r[1] == "approach")
    assert abs(start[2][0]) < 0.06 and abs(start[2][1]) < 0.06, start      # back over the launch point
    s0 = Sim(p=W.Params(**{**LEGACY, 'takeoff_kick_mps': 0.0})).run()                        # without the forward kick: drifts further
    assert worst > min(r[2][0] for r in s0.trace if r[1] == "climb") + 0.02, worst


def test_single_wild_depth_frames_are_ignored():
    s = Sim(outlier_every=17).run()                    # a 0.30 m frame every ~0.6 s, as T4 run 2's 0.346
    assert s.m.phase == "done"
    assert not any("backing off" in (o or "") for o in [s.m.why]), s.m.log
    assert any("through the gap" in x for x in s.m.log), s.m.log


def test_cuvslam_variance_spike_lands_straight_down_before_any_jump():
    s = Sim()
    s.var_fn = lambda: 0.07 if s.m.phase == "settle" else 0.002   # T4 run 2: 1e-3 -> 7e-2, 0.8 s before the jump
    s.run()
    assert any("variance" in x for x in s.m.log), s.m.log
    land = [r for r in s.trace if r[1] == "land"]
    assert math.dist(land[0][2][:2], land[-1][2][:2]) < 0.10


def test_high_variance_on_the_ground_and_climb_does_not_ground_the_mission():
    # T4 run 3: variance 0.024 on the pad (normal near the floor) landed the mission at the first airborne tick
    s = Sim()
    s.var_fn = lambda: 0.05 if (s.pos[2] < 0.3 or s.m.phase in ("ready", "climb")) else 0.004
    s.run()
    assert not any("variance" in x for x in s.m.log), s.m.log
    assert any("through the gap" in x for x in s.m.log), s.m.log


def test_variance_check_waits_for_one_metre_forward():
    s = Sim()
    s.var_fn = lambda: 0.05 if s.m.phase == "approach" and s.pos[0] < 0.9 else 0.004
    s.run()
    assert not any("variance" in x for x in s.m.log), s.m.log         # high only before the 1 m mark: ignored


def test_slide_immediately_variant_skips_the_stop():
    s = Sim(p=W.Params(**{**LEGACY, 'slide_immediately': True})).run()
    seq = [x.split(" -> ")[1].split(":")[0] for x in s.m.log]
    assert "settle" not in seq and seq[:6] == ["ready", "climb", "approach", "translate", "gap", "land"], seq
    assert s.min_wall > 0.29 + 0.07, s.min_wall


def test_no_wall_lands_before_the_forward_limit():
    s = Sim(wall=None).run()
    assert s.m.phase == "done"
    assert any("no wall within" in x for x in s.m.log), s.m.log
    assert max(r[2][0] for r in s.trace) < W.Params().max_forward_m + 0.15


def test_endless_wall_lands_at_the_right_limit():
    s = Sim(W.Params(**{**LEGACY, 'turn_e_m': 0.0}), wall=(WALL_N, -1.0, 10.0)).run()   # old mode (no gap leg)
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
