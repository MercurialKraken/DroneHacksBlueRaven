"""Wall-seek mission (Darion's plan): take off, fly forward until a wall is straight ahead, stop at a set distance
from it, translate right holding that distance (measured by depth, not by the position estimate) until the wall
ends, carry on a little past its end, land. The nose never turns. PURE: no ROS, no numpy.

Frame ('pad'): origin at the take-off pose, north = the nose's heading on the pad, east = its right, up = up.
The node converts pad velocities to the interface's ENU twist (and applies the sim's frame offset).

Phases: wait -> ready -> climb -> approach -> settle -> translate -> past -> land -> done
        (or 'yielded' the moment PX4 leaves Offboard while airborne: a pilot or a failsafe has it; nothing is sent).

Landing copies refnav_skybeta.Descent (Hinson's, flown on skybeta-09): brake, descend at 0.35 m/s (PX4's land
detector needs >= 0.275), press at 0.6 m/s after 3 s on the ground, until PX4 disarms or Offboard ends."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

NAV_OFFBOARD = 14
AIRBORNE = ("climb", "approach", "settle", "translate", "past")


@dataclass
class Params:
    height_m: float = 0.40          # flight height above the pad (Darion)
    standoff_m: float = 0.44        # drone CENTRE to the wall's face while stopped/translating (Darion + Hinson:
                                    # ~15 cm past the radius to both Wall A and Wall B's tip)
    approach_mps: float = 0.15      # forward speed toward the wall
    translate_mps: float = 0.10     # rightward speed along the wall
    past_end_m: float = 0.35        # carry on this far right after the wall has ended
    clear_m: float = 0.90           # 'wall ahead' = corridor range under this (centre); over it for clear_frames = ended
    clear_frames: int = 5           # consecutive depth frames
    settle_s: float = 2.0           # the complete hover stop before translating
    settle_tol_m: float = 0.05      # within this of the standoff ...
    settle_mps: float = 0.05        # ... and slower than this
    max_forward_m: float = 1.30     # no wall by this far forward: hover and land (real launch: Wall A ~1.68 m ahead)
    max_right_m: float = 2.0        # the wall never ends within this: hover and land
    contact_m: float = 0.36         # a surface this close (centre; drone radius 0.29): back off at once
    k_wall: float = 0.5             # 1/s, standoff loop on the depth range
    k_xy: float = 0.5               # 1/s, position holds on the estimate (refnav's KP_XY)
    # integral terms: skybeta-09 has a steady ~0.1 m/s body push PX4 never removes (route 1, 160449); a P-only loop
    # holds it as a standing error (0.1 / 0.5 = 0.2 m), so the standoff would never settle. The integral learns it.
    ki: float = 0.15                # 1/s^2 ... per metre of error per second
    i_max_mps: float = 0.12         # the integral's share is capped
    i_zone_m: float = 0.10          # ... and it only learns within this of the target ...
    i_stuck_mps: float = 0.03       # ... or when stuck off it (slower than this ...
    i_stuck_s: float = 1.5          # ... for this long)
    approach_timeout_s: float = 40.0
    translate_timeout_s: float = 45.0
    k_z: float = 1.0                # 1/s (refnav's KP_Z)
    climb_mps: float = 0.30
    climb_tol_m: float = 0.05
    climb_hold_s: float = 1.0
    xy_hold_cap_mps: float = 0.10
    lag_s: float = 0.55             # twist -> velocity lag: 0.55 on the drone, 1.5 on the T4 (refnav's VEL_LAG_S)
    yaw_rate_dps: float = 30.0
    kp_yaw: float = 1.5
    vio_heading_hold: bool = False  # False: yaw rate 0, PX4 holds heading on the IMU gyro (see _wz)
    range_rate_window_s: float = 0.4
    jump_m: float = 0.25            # one odometry step this large = the estimate jumped: land straight down
    jump_mps: float = 1.5           # or an implied speed this high (we fly <= 0.2 m/s)
    depth_hold_s: float = 0.5       # depth older than this: hold still ...
    depth_land_s: float = 2.0       # ... older than this: land
    odom_hold_s: float = 0.45       # refnav's HOLD_S (skybeta-09 odometry arrives bunched, gaps to 0.35 s)
    odom_land_s: float = 1.0
    state_land_s: float = 1.0
    descend_mps: float = 0.35
    press_mps: float = 0.60
    ground_wait_s: float = 3.0
    ground_band_m: float = 0.06
    stall_s: float = 1.0
    stall_m: float = 0.03
    brake_mps: float = 0.10
    brake_max_s: float = 2.0
    land_hold_mps: float = 0.15
    # take-off: every flight drifts back ~0.2 m in the first ~1.5 s off the floor (T4 truth, both wall-seek runs;
    # skybeta-09 160449: 0.34 m back). A SMALL forward feed-forward for the first second cancels part of it; any
    # overshoot forward is harmless (the depth standoff decides where the drone stops)
    takeoff_kick_mps: float = 0.06
    takeoff_kick_s: float = 1.2
    liftoff_m: float = 0.03         # 'lifted off' = this far above the pad
    home_tol_m: float = 0.05        # start the approach only back over the launch point (both axes within this) ...
    home_wait_s: float = 6.0        # ... or after this long at height anyway
    depth_median_n: int = 5         # depth: median of the last N frames (one wild frame can't move it)
    vio_var_m2: float = 0.02        # cuVSLAM's own pose variance (x or y) above this (~14 cm sigma; normal ~2e-3) ...
    vio_var_count: int = 2          # ... on this many samples in a row: land straight down (T4 run 2 gave 0.8 s warning)
    slide_immediately: bool = False # variant: no hover stop at the wall, start translating as soon as it is at the standoff


@dataclass
class Inputs:
    now: float
    pos: tuple[float, float, float] | None      # pad (north, east, up), metres
    vel: tuple[float, float] | None             # pad (north, east), m/s
    yaw_err: float | None                       # pad heading minus current heading, rad (ENU sense: +CCW)
    t_odom: float
    state: dict[str, Any] | None                # /skybeta/state
    t_state: float
    ahead_m: float | None                       # depth corridor range from the drone's centre (inf: nothing)
    t_depth: float                              # receive time of that depth frame
    depth_seq: int = 0                          # increments per depth frame (counts frames, not ticks)
    vio_var: float | None = None                # max(pose covariance xx, yy) of that odometry sample, m^2
    vo_failed: bool = False                     # cuVSLAM reports vo_state FAILED (real drone's /visual_slam/status)


@dataclass
class Output:
    phase: str
    why: str = ""
    twist: tuple[float, float, float, float] | None = None   # pad (vn, ve), ENU vz, yaw rate (rad/s, +CCW); None: send nothing


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _cap2(x: float, y: float, cap: float) -> tuple[float, float]:
    n = math.hypot(x, y)
    return (x * cap / n, y * cap / n) if n > cap else (x, y)


@dataclass
class WallSeek:
    p: Params = field(default_factory=Params)
    phase: str = "wait"
    why: str = "waiting for odometry, /skybeta/state and an explicit disarmed report"
    t_phase: float | None = None
    offboard_seen: bool = False
    # phase memory
    t_in_tol: float | None = None
    t_at_height: float | None = None
    clear_count: int = 0
    last_seq: int = -1
    e_clear: float | None = None
    n_hold: float | None = None
    t_last: float | None = None
    e_ref: float | None = None      # translate's advancing east reference
    rr: list[tuple[float, float]] = field(default_factory=list)   # (t_depth, range) for the range rate
    rr_seq: int = -1
    vn_depth: float = 0.0           # closing speed toward the wall from depth, m/s
    prev_pos: tuple[float, float, float] | None = None
    t_prev_odom: float = -1e9
    blind: bool = False             # landing without the position estimate (it jumped)
    dbuf: list[float] = field(default_factory=list)   # last depth ranges (inf as 10.0) for the median
    dseq: int = -1
    ahead_f: float | None = None    # the filtered range the mission uses
    t_liftoff: float | None = None
    t_var: float = -1e9
    var_count: int = 0
    t_slow: dict = field(default_factory=dict)   # per integral: since when the drone has been slow on that axis
    dt: float = 0.0
    i_n: float = 0.0                # integral share, north (standoff loop and north holds)
    i_e: float = 0.0                # integral share, east (east holds)
    # landing
    land_target: tuple[float, float] | None = None
    t_land: float | None = None
    braked: bool = False
    t_ground: float | None = None
    zs: list[tuple[float, float]] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ helpers
    def _set(self, phase: str, now: float, why: str) -> None:
        if phase != self.phase:
            self.log.append(f"{now:.2f} {self.phase} -> {phase}: {why}")
        self.phase, self.why, self.t_phase = phase, why, now

    def _land(self, i: Inputs, why: str) -> None:
        self.land_target = (i.pos[0], i.pos[1]) if i.pos else None
        self.t_land, self.braked, self.t_ground, self.zs = i.now, False, None, []
        self._set("land", i.now, why)

    def _wz(self, i: Inputs) -> float:
        """Yaw rate: ZERO by default, so PX4's rate loop holds the heading on the IMU gyro. The vision heading drifted
        ~11 deg in 2 s facing Wall A at 0.44 m in the T4 (sim run 20261108T0749) and a heading hold on it turned the
        real drone 24 deg into the wall. vio_heading_hold=True brings that hold back (not recommended near walls)."""
        if not self.p.vio_heading_hold:
            return 0.0
        cap = math.radians(self.p.yaw_rate_dps)
        return _clamp(self.p.kp_yaw * (i.yaw_err or 0.0), -cap, cap)

    def _filter_depth(self, i: Inputs) -> None:
        """Median of the last depth_median_n frames: a single wild frame (T4 run 2: 0.346 m against a true 0.407)
        can't move it; a real change shows in every frame and passes within ~3 frames (0.1 s)."""
        if i.depth_seq == self.dseq or i.ahead_m is None:
            return
        self.dseq = i.depth_seq
        v = i.ahead_m if math.isfinite(i.ahead_m) else 10.0
        self.dbuf = (self.dbuf + [v])[-max(1, self.p.depth_median_n):]
        m = sorted(self.dbuf)[len(self.dbuf) // 2]
        self.ahead_f = math.inf if m >= 10.0 else m

    def _track_range(self, i: Inputs) -> None:
        """Closing speed toward the wall from the (filtered) depth range itself (not the vision velocity, which went
        wrong beside the wall): the range's slope over ~range_rate_window_s of frames, + = moving toward it."""
        if i.depth_seq == self.rr_seq:
            return
        self.rr_seq = i.depth_seq
        r_f = self.ahead_f
        if r_f is None or not math.isfinite(r_f) or r_f >= self.p.clear_m:
            self.rr, self.vn_depth = [], 0.0
            return
        self.rr = [(t, r) for t, r in self.rr if i.t_depth - t <= self.p.range_rate_window_s] + [(i.t_depth, r_f)]
        span = self.rr[-1][0] - self.rr[0][0]
        if span < 0.6 * self.p.range_rate_window_s or len(self.rr) < 4:
            self.vn_depth = 0.0
            return
        n = len(self.rr)                                          # least-squares slope of range over time
        mt = sum(t for t, _ in self.rr) / n
        mr = sum(r for _, r in self.rr) / n
        den = sum((t - mt) ** 2 for t, _ in self.rr)
        self.vn_depth = -sum((t - mt) * (r - mr) for t, r in self.rr) / den if den > 0 else 0.0

    def _jumped(self, i: Inputs) -> str | None:
        """The vision position stepped (cuVSLAM diverging beside the wall jumped metres per sample): why, or None."""
        if i.pos is None or i.t_odom == self.t_prev_odom:
            return None
        prev, self.prev_pos, t_prev, self.t_prev_odom = self.prev_pos, i.pos, self.t_prev_odom, i.t_odom
        if prev is None:
            return None
        step = math.dist(prev, i.pos)
        dt = i.t_odom - t_prev
        if step > self.p.jump_m or (dt > 0 and step / dt > self.p.jump_mps and step > 0.05):
            return f"position estimate jumped {step:.2f} m in {dt:.2f} s"
        return None

    def _vz(self, i: Inputs) -> float:
        return _clamp(self.p.k_z * (self.p.height_m - i.pos[2]), -self.p.climb_mps, self.p.climb_mps)

    def _integrate(self, name: str, err: float, speed: float = 0.0) -> float:
        """Learns a steady push near the target (|err| < i_zone_m), or anywhere the drone is STUCK off it (moving
        slower than i_stuck_mps: a push holding it there). Not while it is still closing a big error: integrating
        the approach itself would wind up and overshoot into the wall."""
        cap = self.p.i_max_mps
        if self.t_liftoff is None or self.t_last - self.t_liftoff < self.p.takeoff_kick_s + 1.0:
            return getattr(self, name)       # the lift-off kick is a transient, not a steady push: don't learn it
        slow = abs(speed) < self.p.i_stuck_mps
        self.t_slow[name] = (self.t_slow.get(name) or self.t_last) if slow else None
        stuck = slow and self.t_last - self.t_slow[name] >= self.p.i_stuck_s   # braking near the target is brief
        if abs(err) >= self.p.i_zone_m and not stuck:
            return getattr(self, name)
        v = _clamp(getattr(self, name) + self.p.ki * err * self.dt, -cap, cap)
        setattr(self, name, v)
        return v

    def _hold(self, target: float, now_pos: float, vel: float, axis: str = "i_e") -> float:
        err = target - now_pos
        cap = self.p.xy_hold_cap_mps
        return _clamp(self.p.k_xy * (err - self.p.lag_s * vel) + self._integrate(axis, err, vel), -cap, cap)

    def _east_ve(self, e: float, ve_meas: float) -> float:
        """Translate rightward at translate_mps against any push: a reference east position advancing at
        translate_mps (never more than 0.2 m ahead of the drone), tracked with the east integral."""
        p = self.p
        self.e_ref = min((self.e_ref if self.e_ref is not None else e) + p.translate_mps * self.dt, e + 0.20)
        err = self.e_ref - e
        return _clamp(p.translate_mps + p.k_xy * (err - p.lag_s * (ve_meas - p.translate_mps))
                      + self._integrate("i_e", err), 0.0, 2 * p.translate_mps)

    def _wall_vn(self, rng: float, vn_meas: float, lo: float, hi: float) -> float:
        """Standoff loop on the depth range, with the velocity lag predicted out (vn_meas = the closing speed from
        DEPTH, self.vn_depth), plus the learned push."""
        err = rng - self.p.standoff_m
        lag = self.p.lag_s + (self.p.depth_median_n // 2) / 30.0      # + the median's own delay (~2 frames at 30 Hz)
        return _clamp(self.p.k_wall * (err - lag * vn_meas) + self._integrate("i_n", err, vn_meas), lo, hi)

    # ------------------------------------------------------------------ step
    def step(self, i: Inputs) -> Output:
        p = self.p
        if self.t_phase is None:
            self.t_phase = i.now
        self.dt = 0.0 if self.t_last is None else _clamp(i.now - self.t_last, 0.0, 0.25)
        self.t_last = i.now
        jump = self._jumped(i)
        self._filter_depth(i)
        self._track_range(i)
        if i.t_odom != self.t_var:                    # cuVSLAM's own uncertainty, counted per odometry sample
            self.t_var = i.t_odom
            bad = i.vio_var is not None and i.vio_var > p.vio_var_m2
            self.var_count = self.var_count + 1 if bad else 0
        st = i.state or {}
        armed, offboard, nav = st.get("armed"), st.get("offboard_active"), st.get("nav_state")
        if self.phase in ("done", "yielded"):
            return Output(self.phase, self.why)

        # ---- our own landing ends when PX4 disarms (it leaves Offboard at the same moment: that is not a takeover)
        if self.phase == "land" and armed is False:
            self._set("done", i.now, "landed: PX4 disarmed")
            return Output(self.phase, self.why)
        # ---- PX4 left Offboard while we fly: never fight a pilot or a failsafe
        if self.phase in AIRBORNE + ("land",):
            if nav == NAV_OFFBOARD:
                self.offboard_seen = True
            if self.offboard_seen and nav is not None and nav != NAV_OFFBOARD:
                self._set("yielded", i.now, f"PX4 left Offboard (nav_state {nav}): a pilot or a failsafe has it")
                return Output(self.phase, self.why)

        fresh_state = i.state is not None and i.now - i.t_state <= p.state_land_s and st.get("px4_link") is not False
        odom_age = i.now - i.t_odom

        # ---- ground: wait for inputs, then stream a still hold for the pilot to arm into
        if self.phase == "wait":
            if armed is True:
                self._set("done", i.now, "refused: armed before this mission was ready (nothing sent)")
                return Output(self.phase, self.why)
            if i.pos is None or odom_age > p.odom_land_s or not fresh_state or armed is not False:
                return Output(self.phase, self.why)
            self._set("ready", i.now, "streaming a still hold: the safety pilot may arm (Offboard on arm)")
        if self.phase == "ready":
            if armed and nav == NAV_OFFBOARD and offboard:
                self.offboard_seen = True
                self._set("climb", i.now, f"Offboard: climbing to {p.height_m:.2f} m")
            elif i.pos is None or odom_age > p.odom_land_s or not fresh_state:
                self._set("wait", i.now, "inputs went stale before arming")
                return Output(self.phase, self.why)
            else:
                return Output(self.phase, self.why, (0.0, 0.0, 0.0, 0.0))

        # ---- airborne supervision
        if self.phase in AIRBORNE:
            if i.vo_failed or self.var_count >= p.vio_var_count:
                self.blind = True
                self._land(i, ("cuVSLAM reports tracking FAILED" if i.vo_failed else
                               f"cuVSLAM's own variance {i.vio_var:.3g} m^2 > {p.vio_var_m2:g} for {self.var_count} samples")
                           + ": landing straight down before the estimate jumps")
            elif jump:
                self.blind = True
                self._land(i, jump + ": landing straight down, not chasing the estimate")
            elif not fresh_state:
                self._land(i, f"/skybeta/state silent for {i.now - i.t_state:.2f} s")
            elif i.pos is None or odom_age > p.odom_land_s:
                self._land(i, f"odometry silent for {odom_age:.2f} s")
            elif odom_age > p.odom_hold_s:
                return Output(self.phase, f"holding: odometry {odom_age:.2f} s old", (0.0, 0.0, 0.0, 0.0))
        if self.phase in ("approach", "settle", "translate"):
            d_age = i.now - i.t_depth
            if d_age > p.depth_land_s:
                self._land(i, f"depth silent for {d_age:.2f} s")
            elif d_age > p.depth_hold_s:
                return Output(self.phase, f"holding: depth {d_age:.2f} s old", (0.0, 0.0, 0.0, self._wz(i)))
            elif self.ahead_f is not None and self.ahead_f < p.contact_m:
                vn = -p.xy_hold_cap_mps
                return Output(self.phase, f"backing off: surface {self.ahead_f:.2f} m ahead",
                              (vn, 0.0, self._vz(i), self._wz(i)))

        n, e, u = i.pos if i.pos else (0.0, 0.0, 0.0)
        vn_m, ve_m = i.vel if i.vel else (0.0, 0.0)
        rng = self.ahead_f if self.ahead_f is not None else math.inf
        wall = rng < p.clear_m

        if self.phase == "climb":
            if self.t_liftoff is None and u > p.liftoff_m:
                self.t_liftoff = i.now
            kick = p.takeoff_kick_mps if self.t_liftoff is not None and i.now - self.t_liftoff < p.takeoff_kick_s else 0.0
            if abs(u - p.height_m) < p.climb_tol_m:
                self.t_at_height = self.t_at_height if self.t_at_height is not None else i.now
                held = i.now - self.t_at_height
                home = abs(n) < p.home_tol_m and abs(e) < p.home_tol_m
                if held >= p.climb_hold_s and (home or held >= p.home_wait_s):
                    self._set("approach", i.now, f"at {u:.2f} m, {math.hypot(n, e):.2f} m from the launch point: "
                                                 f"forward until a wall is {p.standoff_m:.2f} m ahead")
            else:
                self.t_at_height = None
            if self.phase == "climb":
                return Output(self.phase, self.why, (self._hold(0.0, n, vn_m, "i_n") + kick, self._hold(0.0, e, ve_m),
                                                     self._vz(i), self._wz(i)))

        if self.phase == "approach":
            if not wall and n >= p.max_forward_m - 0.02:
                self._land(i, f"no wall within {p.max_forward_m:.2f} m forward")
            elif i.now - self.t_phase > p.approach_timeout_s:
                self._land(i, f"approach did not settle within {p.approach_timeout_s:.0f} s")
            else:
                if wall:
                    vn = self._wall_vn(rng, self.vn_depth, -p.xy_hold_cap_mps, p.approach_mps)
                    if p.slide_immediately and abs(rng - p.standoff_m) < p.settle_tol_m:
                        self.clear_count, self.last_seq = 0, i.depth_seq
                        self._set("translate", i.now, f"wall {rng:.2f} m ahead: sliding right at once (no stop)")
                    elif abs(rng - p.standoff_m) < p.settle_tol_m and math.hypot(vn_m, ve_m) < p.settle_mps:
                        self.t_in_tol = None
                        self._set("settle", i.now, f"wall {rng:.2f} m ahead: hover stop {p.settle_s:.0f} s")
                else:   # nothing ahead yet: go, braking on the estimate before the forward limit
                    vn = _clamp(p.k_xy * (p.max_forward_m - n - p.lag_s * vn_m), -p.xy_hold_cap_mps, p.approach_mps)
                if self.phase == "approach":
                    return Output(self.phase, self.why, (vn, self._hold(0.0, e, ve_m), self._vz(i), self._wz(i)))

        if self.phase == "settle":
            if not wall:
                self._set("approach", i.now, "lost the wall while settling")
                return Output(self.phase, self.why, (0.0, self._hold(0.0, e, ve_m), self._vz(i), self._wz(i)))
            ok = abs(rng - p.standoff_m) < p.settle_tol_m and math.hypot(vn_m, ve_m) < p.settle_mps
            self.t_in_tol = (self.t_in_tol if self.t_in_tol is not None else i.now) if ok else None
            if self.t_in_tol is not None and i.now - self.t_in_tol >= p.settle_s:
                self.clear_count, self.last_seq = 0, i.depth_seq
                self._set("translate", i.now, f"stopped {rng:.2f} m from the wall: translating right")
            else:
                vn = self._wall_vn(rng, self.vn_depth, -p.xy_hold_cap_mps, p.xy_hold_cap_mps)
                return Output(self.phase, self.why, (vn, self._hold(0.0, e, ve_m), self._vz(i), self._wz(i)))

        if self.phase == "translate":
            if i.depth_seq != self.last_seq:                     # count FRAMES, not ticks
                self.last_seq = i.depth_seq
                self.clear_count = 0 if wall else self.clear_count + 1
            if e >= p.max_right_m:
                self._land(i, f"the wall did not end within {p.max_right_m:.2f} m right")
            elif i.now - self.t_phase > p.translate_timeout_s:
                self._land(i, f"translate did not finish within {p.translate_timeout_s:.0f} s")
            elif self.clear_count >= p.clear_frames:
                self.e_clear, self.n_hold = e, n
                self._set("past", i.now, f"wall ended at east {e:.2f} m: {p.past_end_m:.2f} m more, then land")
            else:
                vn = (self._wall_vn(rng, self.vn_depth, -p.xy_hold_cap_mps, p.xy_hold_cap_mps) if wall
                      else self._hold(n, n, vn_m, "i_n"))
                return Output(self.phase, self.why, (vn, self._east_ve(e, ve_m), self._vz(i), self._wz(i)))

        if self.phase == "past":
            if e >= self.e_clear + p.past_end_m or e >= p.max_right_m:
                self._land(i, f"{e - self.e_clear:.2f} m past the wall's end")
            elif self.ahead_f is not None and self.ahead_f < p.standoff_m and i.now - i.t_depth <= p.depth_hold_s:
                self._land(i, f"something {self.ahead_f:.2f} m ahead after the wall ended")
            else:
                return Output(self.phase, self.why,
                              (self._hold(self.n_hold, n, vn_m, "i_n"), self._east_ve(e, ve_m), self._vz(i), self._wz(i)))

        if self.phase == "land":
            return self._descend(i, armed, offboard)
        return Output(self.phase, self.why)

    # ------------------------------------------------------------------ landing (refnav_skybeta.Descent)
    def _descend(self, i: Inputs, armed: Any, offboard: Any) -> Output:
        p = self.p
        if not (armed and offboard):
            self._set("done", i.now, "landed: PX4 disarmed or Offboard ended" if armed is False else "Offboard ended")
            return Output(self.phase, self.why)
        if self.blind:          # the estimate jumped: straight down on time alone, pressing once surely down
            press = i.now - self.t_land > p.height_m / p.descend_mps + p.ground_wait_s
            return Output(self.phase, self.why, (0.0, 0.0, -(p.press_mps if press else p.descend_mps), 0.0))
        if i.pos is None or i.now - i.t_odom > p.odom_hold_s:
            return Output(self.phase, self.why, (0.0, 0.0, -p.descend_mps, 0.0))   # blind: straight down
        n, e, u = i.pos
        vn_m, ve_m = i.vel or (0.0, 0.0)
        vn = ve = 0.0
        if self.land_target is not None and self.t_ground is None:     # + the push learned in flight (i_n, i_e)
            vn, ve = _cap2(p.k_xy * (self.land_target[0] - n) + self.i_n, p.k_xy * (self.land_target[1] - e) + self.i_e,
                           p.land_hold_mps)
        if not self.braked and math.hypot(vn_m, ve_m) > p.brake_mps and i.now - self.t_land < p.brake_max_s:
            return Output(self.phase, self.why, (vn, ve, 0.0, 0.0))
        self.braked = True
        self.zs = [(t, z) for t, z in self.zs if i.now - t <= p.stall_s] + [(i.now, u)]
        stalled = i.now - self.zs[0][0] >= p.stall_s * 0.9 and self.zs[0][1] - self.zs[-1][1] < p.stall_m
        on_ground = u <= p.ground_band_m or stalled
        self.t_ground = (self.t_ground if self.t_ground is not None else i.now) if on_ground else None
        press = self.t_ground is not None and i.now - self.t_ground >= p.ground_wait_s
        if self.t_ground is not None:
            vn = ve = 0.0
        return Output(self.phase, self.why, (vn, ve, -(p.press_mps if press else p.descend_mps), 0.0))


# ---------------------------------------------------------------------- frames (as refnav_skybeta / frame_offset)
class Pad:
    """North along the nose on the pad, origin there: the odometry pose captured until Offboard begins."""

    def __init__(self, x: float, y: float, z: float, yaw: float) -> None:
        self.p0, self.yaw0 = (x, y, z), yaw
        self.c, self.s = math.cos(yaw), math.sin(yaw)

    def vector(self, vx: float, vy: float) -> tuple[float, float]:
        """odom ENU (x, y) -> pad (north, east)."""
        return self.c * vx + self.s * vy, self.s * vx - self.c * vy

    def to_odom(self, vn: float, ve: float) -> tuple[float, float]:
        """pad (north, east) -> odom ENU (x, y): the same matrix, its own inverse."""
        return self.vector(vn, ve)

    def pose(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        n, e = self.vector(x - self.p0[0], y - self.p0[1])
        return n, e, z - self.p0[2]


def yaw_enu_from_xyzw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def twist_for_interface(vx: float, vy: float, offset_rad: float) -> tuple[float, float]:
    """Chelsea's frame_offset.twist_for_interface: the interface rotates poses by odom_to_ned_yaw_offset_rad but
    not twists, so pre-rotate (identity at 0.0, the real drone's value; -1.5708 on the T4)."""
    if offset_rad == 0.0:
        return vx, vy
    c, s = math.cos(-offset_rad), math.sin(-offset_rad)
    return c * vx - s * vy, s * vx + c * vy
