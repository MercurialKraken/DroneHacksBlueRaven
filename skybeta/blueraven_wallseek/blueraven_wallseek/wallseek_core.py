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
AIRBORNE = ("climb", "approach", "settle", "translate", "past", "gap", "turn", "pause", "leg", "rise")


@dataclass
class Params:
    height_m: float = 0.40          # flight height above the pad (Darion)
    standoff_m: float = 0.80        # drone CENTRE to Wall A's face at P1 (Darion, route v3; was 0.60, before 0.44 kept
                                    # cuVSLAM too close to a plain wall in every T4 run). Straight on at 0.60 would pass
                                    # Wall B's tip at 0.28 m (< radius 0.29), hence the gap leg below.
    # the gap leg (Darion): translate east until turn_e_m (vision estimate), then fly straight through the CP2 gap's
    # centre, on gap_past_m beyond it, and land. Launch-relative numbers on the tape course (pad: north = forward from
    # the launch mark, east = right): best turn at 1.10-1.12 m east -> 28.6 deg north of east (the gap normal is 28.8),
    # 0.50 m from both Wall A's end and Wall B's tip. On the T4 the spawn is 0.2 m further north: gap_n_m 1.039.
    # ROUTE MODE (Darion, default): after the stop at P1 (0.60 m from Wall A on depth), straight legs with turns of
    # exactly 90/45 deg made IN PLACE at turn_rate_dps (the gyro's rate loop): 90 right -> east to P2 -> 45 left ->
    # through the CP2 gap to P3 -> 45 right -> east to P4 (about halfway to the tunnel) -> land. Checked on the course
    # SDF at 0.4 m: >= 0.39 m from every wall (tightest: the barrier behind the launch); the 0.30 m CP2 cone passes
    # ~0.10 m BELOW the drone (height is from the pad, so the drone's bottom is at 0.40 m). T4: p3_n_m 1.357.
    route_mode: bool = True         # v3 (Darion): P1 at standoff 0.80 -> face P2 -> straight through the gap -> P4
    p2_past_m: float = 0.45         # P2: this far past the CP2 gap centre, on the line from P1 (whole drone across)
    p4_e_m: float = 2.80            # P4: due east of P2, about halfway to the tunnel
    # Darion: climb DURING the P1 -> P2 leg (not at take-off), in proportion to the leg's progress, to the tunnel's
    # height, then hold it. The tube's free space is centred ~0.95 m above the floor (course SDF: 0.71 m clearance
    # at z 0.9-1.0, only 0.36 m at 0.4), and height is from the pad (centre ~0.07 m higher). Over the 0.30 m CP2 cone
    # (~60% along the leg) the drone is at ~0.67 m (T4 v3 run at 0.4 m: its centre dipped to 0.33 m there).
    leg_climb_to_m: float = 0.0     # gradual climb during the gap leg: OFF (Darion: climb at P1 instead)
    p1_climb_to_m: float = 0.0      # climb in place at P1 before turning: OFF (Darion 2026-11-08: the organizers removed
                                    # the CP2 cone, fly the whole route at 0.40); 0.50 cleared the cone when it was there
    land_at_p2: bool = True         # Darion: the route ends at P2 (past the CP2 gap) and lands there; False: on to P4
    turn_rate_dps: float = 15.0     # slow in-place turns (a 45 deg/s in-place yaw diverged cuVSLAM on the T4)
    turn_dwell_s: float = 1.0       # still, after each turn
    pause_s: float = 1.5            # stop at each waypoint before turning
    leg_mps: float = 0.12
    turn_e_m: float = 1.10          # (gap mode, route_mode False) 0 disables the gap leg: translate until the wall ends
    gap_n_m: float = 1.239
    gap_e_m: float = 1.417
    gap_past_m: float = 0.50
    approach_mps: float = 0.15      # forward speed toward the wall
    translate_mps: float = 0.10     # rightward speed along the wall
    past_end_m: float = 0.35        # carry on this far right after the wall has ended
    clear_m: float = 1.30           # 'wall ahead' = corridor range under this (centre); over it for clear_frames = ended
                                    # (1.30: the standoff loop takes over 0.5 m before a 0.80 m stop)
    clear_frames: int = 5           # consecutive depth frames
    settle_s: float = 2.0           # the complete hover stop before translating
    settle_tol_m: float = 0.05      # within this of the standoff ...
    settle_mps: float = 0.05        # ... and closing on the wall slower than this (from depth)
    settle_max_s: float = 6.0       # ... or after this long within 2 x settle_tol_m anyway
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
    vio_var_m2: float = 0.03        # cuVSLAM's own pose variance (x or y) above this (normal in flight ~3e-3-9e-3) ...
    vio_var_count: int = 2          # ... on this many samples in a row: land straight down (T4 run 2: 0.8 s warning)
    var_check_after_m: float = 1.0  # ... checked only after lift-off AND once this far forward (Darion): near the floor
                                    # the variance is naturally 0.01-0.05 (it grounded T4 run 3 on the pad). The T4's
                                    # spawn is 0.2 m further north, so the same spot there is 0.8 m.
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
    var_armed: bool = False         # the cuVSLAM-variance check is live (after lift-off and var_check_after_m forward)
    leg: tuple | None = None        # current straight leg: (start (n, e), end (n, e), unit (n, e), length)
    s_ref: float = 0.0
    plan: list = field(default_factory=list)   # route mode: remaining steps
    route_pts: list | None = None   # route mode: the actual P1, P2, P4 once P1 is fixed
    heading: float = 0.0            # route mode: commanded pad heading, deg clockwise from north
    h_target: float | None = None   # current height target (None: height_m)
    leg_index: int = 0              # route mode: 1 = P1 -> P2 (through the gap), 2 = P2 -> P4
    ph: list = field(default_factory=list)       # (t, north, east) for the position-slope velocity
    t_vel: float = -1e9
    vel_est: tuple = (0.0, 0.0)
    turn: tuple | None = None       # (sign, degrees, hold point)
    turned: float = 0.0             # degrees commanded so far in this turn
    last_wz: float = 0.0            # deg/s commanded last tick
    t_turned: float | None = None
    hold_pt: tuple | None = None
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

    def _track_vel(self, i: Inputs) -> None:
        """Velocity (pad north, east) as the least-squares slope of the last ~0.3 s of odometry positions."""
        if i.pos is None or i.t_odom == self.t_vel:
            return
        self.t_vel = i.t_odom
        self.ph = [q for q in self.ph if i.t_odom - q[0] <= 0.3] + [(i.t_odom, i.pos[0], i.pos[1])]
        m = len(self.ph)
        if m < 3 or self.ph[-1][0] - self.ph[0][0] < 0.12:
            return
        mt = sum(q[0] for q in self.ph) / m
        den = sum((q[0] - mt) ** 2 for q in self.ph)
        if den <= 0:
            return
        self.vel_est = tuple(sum((q[0] - mt) * q[k] for q in self.ph) / den for k in (1, 2))

    def _jumped(self, i: Inputs) -> str | None:
        """The vision position stepped (cuVSLAM diverging beside the wall jumped metres per sample): why, or None."""
        if i.pos is None or i.t_odom == self.t_prev_odom:
            return None
        prev, self.prev_pos, t_prev, self.t_prev_odom = self.prev_pos, i.pos, self.t_prev_odom, i.t_odom
        if prev is None:
            return None
        step = math.dist(prev, i.pos)
        dt = i.t_odom - t_prev
        # a fast step counts only above 10 cm: T4 run 6 landed on a 7 cm one (1.75 m/s over 0.04 s) beside the wall
        if step > self.p.jump_m or (dt > 0 and step / dt > self.p.jump_mps and step > 0.10):
            return f"position estimate jumped {step:.2f} m in {dt:.2f} s"
        return None

    def _vz(self, i: Inputs) -> float:
        target = self.h_target if self.h_target is not None else self.p.height_m
        return _clamp(self.p.k_z * (target - i.pos[2]), -self.p.climb_mps, self.p.climb_mps)

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
        self._track_vel(i)
        self._filter_depth(i)
        self._track_range(i)
        if (not self.var_armed and self.t_liftoff is not None and i.pos is not None
                and i.pos[0] >= p.var_check_after_m):
            self.var_armed = True                     # latched: stays on once past the mark
        if i.t_odom != self.t_var:                    # cuVSLAM's own uncertainty, counted per odometry sample
            self.t_var = i.t_odom
            bad = self.var_armed and i.vio_var is not None and i.vio_var > p.vio_var_m2
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
        # velocity from the POSITION's own slope (0.3 s), not cuVSLAM's twist field: beside the wall the twist read
        # 0.1-0.7 m/s while the position held within 2 cm (T4 runs 5-6), and the legs steer by position anyway
        vn_m, ve_m = self.vel_est
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
                    elif abs(rng - p.standoff_m) < p.settle_tol_m and abs(self.vn_depth) < p.settle_mps:
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
            # 'still' from DEPTH (the wall distance steady), not cuVSLAM's velocity: beside the wall that read 0.1-0.7 m/s
            # while the drone held within 2 cm (T4 runs 5-6), so the stop never ended. Cap: settle_max_s near the target.
            ok = abs(rng - p.standoff_m) < p.settle_tol_m and abs(self.vn_depth) < p.settle_mps
            self.t_in_tol = (self.t_in_tol if self.t_in_tol is not None else i.now) if ok else None
            capped = i.now - self.t_phase >= p.settle_max_s and abs(rng - p.standoff_m) < 2 * p.settle_tol_m
            if (self.t_in_tol is not None and i.now - self.t_in_tol >= p.settle_s) or capped:
                if p.route_mode:
                    self._route_start(i, n, e)
                else:
                    self.clear_count, self.last_seq = 0, i.depth_seq
                    self._set("translate", i.now, f"stopped {rng:.2f} m from the wall: translating right")
            else:
                vn = self._wall_vn(rng, self.vn_depth, -p.xy_hold_cap_mps, p.xy_hold_cap_mps)
                return Output(self.phase, self.why, (vn, self._hold(0.0, e, ve_m), self._vz(i), self._wz(i)))

        if self.phase == "translate":
            if i.depth_seq != self.last_seq:                     # count FRAMES, not ticks
                self.last_seq = i.depth_seq
                self.clear_count = 0 if wall else self.clear_count + 1
            if p.turn_e_m > 0 and e >= p.turn_e_m:              # Darion: before Wall B's tip, angle through the gap
                self._start_gap(i, n, e)
            elif e >= p.max_right_m:
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

        if self.phase == "turn":
            sign, angle, (hn, he) = self.turn
            self.turned += abs(self.last_wz) * self.dt            # the angle commanded so far (deg), tick by tick
            rem = angle - self.turned
            if rem <= 1e-3 and self.t_turned is None:
                self.t_turned = i.now
            if self.t_turned is not None and i.now - self.t_turned >= p.turn_dwell_s:
                self._next_step(i, n, e)
            else:
                # the gyro's rate loop turns it; the last tick is trimmed so the turn is exactly `angle`
                rate = 0.0 if rem <= 1e-3 else min(p.turn_rate_dps, rem / max(self.dt, 0.02))
                self.last_wz = rate
                wz = sign * math.radians(rate)
                return Output(self.phase, self.why,
                              (self._hold(hn, n, vn_m, "i_n"), self._hold(he, e, ve_m), self._vz(i), wz))

        if self.phase == "rise":
            hn, he = self.hold_pt
            if abs(u - self.h_target) < p.climb_tol_m:
                self.t_at_height = self.t_at_height if self.t_at_height is not None else i.now
                if i.now - self.t_at_height >= p.climb_hold_s:
                    self._next_step(i, n, e)
            else:
                self.t_at_height = None
            if self.phase == "rise":
                return Output(self.phase, self.why,
                              (self._hold(hn, n, vn_m, "i_n"), self._hold(he, e, ve_m), self._vz(i), 0.0))

        if self.phase == "pause":
            hn, he = self.hold_pt
            if i.now - self.t_phase >= p.pause_s:
                self._next_step(i, n, e)
            else:
                return Output(self.phase, self.why,
                              (self._hold(hn, n, vn_m, "i_n"), self._hold(he, e, ve_m), self._vz(i), 0.0))

        if self.phase == "leg":
            (tn, te), (gn, ge), (un, ue), total = self.leg
            s_now = (n - tn) * un + (e - te) * ue
            if s_now >= total - 0.03:
                self._next_step(i, n, e)
            elif self.ahead_f is not None and self.ahead_f < p.contact_m and i.now - i.t_depth <= p.depth_hold_s:
                self._land(i, f"surface {self.ahead_f:.2f} m ahead on a leg")
            elif i.now - self.t_phase > p.translate_timeout_s:
                self._land(i, f"leg did not finish within {p.translate_timeout_s:.0f} s")
            else:
                if self.leg_index == 1 and p.leg_climb_to_m > 0:    # the leg through the gap: climb with progress
                    frac = _clamp(s_now / max(total, 1e-6), 0.0, 1.0)
                    self.h_target = p.height_m + (p.leg_climb_to_m - p.height_m) * frac
                return Output(self.phase, self.why, (*self._leg_cmd(n, e, vn_m, ve_m, p.leg_mps), self._vz(i), 0.0))

        if self.phase == "gap":
            (tn, te), (gn, ge), (un, ue), total = self.leg
            s_now = (n - tn) * un + (e - te) * ue                 # progress along the leg
            if s_now >= total - 0.03:
                self._land(i, f"through the gap: {math.hypot(n - gn, e - ge):.2f} m from the end point")
            elif e >= p.max_right_m + 0.3:
                self._land(i, f"east {e:.2f} m: past the limit on the gap leg")
            elif self.ahead_f is not None and self.ahead_f < p.contact_m and i.now - i.t_depth <= p.depth_hold_s:
                self._land(i, f"surface {self.ahead_f:.2f} m ahead on the gap leg")
            elif i.now - self.t_phase > p.translate_timeout_s:
                self._land(i, f"gap leg did not finish within {p.translate_timeout_s:.0f} s")
            else:
                v = p.translate_mps
                self.s_ref = min(self.s_ref + v * self.dt, s_now + 0.20, total)
                rn, re = tn + self.s_ref * un, te + self.s_ref * ue      # moving reference along the straight leg
                ff = v if self.s_ref < total else 0.0
                cap = 2 * p.translate_mps
                vn = _clamp(ff * un + p.k_xy * ((rn - n) - p.lag_s * (vn_m - ff * un)) + self._integrate("i_n", rn - n, vn_m - ff * un),
                            -cap, cap)
                ve = _clamp(ff * ue + p.k_xy * ((re - e) - p.lag_s * (ve_m - ff * ue)) + self._integrate("i_e", re - e, ve_m - ff * ue),
                            -cap, cap)
                return Output(self.phase, self.why, (vn, ve, self._vz(i), self._wz(i)))

        if self.phase == "land":
            return self._descend(i, armed, offboard)
        return Output(self.phase, self.why)

    def dashboard_view(self) -> dict:
        """What the T4 dashboard draws (it reads refnav's /refnav/status: 'targets' and 'path' as pad [north, east],
        'look' = the point being steered to). Before P1 is fixed by depth: the PLANNED route (P1 at its nominal 0.48 m
        south-west of P2's 45 deg leg); after: the actual one."""
        p = self.p
        if self.route_pts is not None:
            pts = self.route_pts
        else:                                                # nominal P1: Wall A's face is 0.438 m north of the gap centre
            n1 = p.gap_n_m + 0.438 - p.standoff_m
            p2, p4 = self._route_points(n1, 0.0)
            pts = [(n1, 0.0), p2] + ([] if p.land_at_p2 else [p4])
        look = None
        if self.phase in ("leg", "gap") and self.leg is not None:
            (tn, te), _, (un, ue), _ = self.leg
            look = [round(tn + self.s_ref * un, 3), round(te + self.s_ref * ue, 3)]
        elif self.phase == "turn" and self.turn is not None:
            look = [round(c, 3) for c in self.turn[2]]
        elif self.phase == "pause" and self.hold_pt is not None:
            look = [round(c, 3) for c in self.hold_pt]
        return {"phase": self.phase, "why": self.why, "program": "wallseek",
                "targets": [[round(n, 3), round(e, 3)] for n, e in pts],
                "path": [[0.0, 0.0]] + [[round(n, 3), round(e, 3)] for n, e in pts], "look": look}

    # ------------------------------------------------------------------ route mode (Darion): straight legs, 90/45 turns
    def _route_points(self, n1: float, e1: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """P2 = past the CP2 gap's centre on the straight line from P1 through that centre; P4 = due east of P2."""
        p = self.p
        gn, ge = p.gap_n_m, p.gap_e_m
        d = math.hypot(gn - n1, ge - e1)
        un, ue = (gn - n1) / d, (ge - e1) / d
        p2 = (gn + p.p2_past_m * un, ge + p.p2_past_m * ue)
        return p2, (p2[0], p.p4_e_m)

    def _route_start(self, i: Inputs, n: float, e: float) -> None:
        """Stopped at P1 (the depth standoff, Darion: 0.80 m from Wall A). One slow in-place turn from north to face P2
        (through the CP2 gap's centre: ~76 deg right), straight to P2, turn to face P4 (~14 deg right), straight to
        P4, land. Course SDF at 0.4 m: the P1-P2 leg clears Wall A's end and Wall B's tip by 0.47 m; the 0.30 m cone
        passes under the drone. (Aiming at the old P3 instead clipped Wall A's end with 1.3 cm spare.)"""
        p2, p4 = self._route_points(n, e)
        self.route_pts = [(n, e), p2, p4]
        self.heading = 0.0                                   # pad heading, deg clockwise from north (nose on the pad)
        self.plan = ([("rise", self.p.p1_climb_to_m)] if self.p.p1_climb_to_m > 0 else []) + \
            [("face", p2), ("leg", p2)] + ([] if self.p.land_at_p2 else [("pause", None), ("face", p4), ("leg", p4)])
        if self.p.land_at_p2:
            self.route_pts = [(n, e), p2]
        self.log.append(f"{i.now:.2f} route: P1 ({n:.2f}, {e:.2f}) -> P2 ({p2[0]:.2f}, {p2[1]:.2f}) -> "
                        f"P4 ({p4[0]:.2f}, {p4[1]:.2f}) [pad north, east]")
        self._next_step(i, n, e)

    def _next_step(self, i: Inputs, n: float, e: float) -> None:
        if not self.plan:
            self._land(i, "route complete")
            return
        kind, arg = self.plan.pop(0)
        if kind == "rise":                                   # Darion: at P1 climb straight up (in place) first
            self.h_target, self.hold_pt, self.t_at_height = arg, (n, e), None
            self._set("rise", i.now, f"climbing in place to {arg:.2f} m at P1")
            return
        if kind == "face":                                   # turn in place to point the nose at `arg`, from here
            bearing = math.degrees(math.atan2(arg[1] - e, arg[0] - n))      # clockwise from north
            delta = (bearing - self.heading + 180.0) % 360.0 - 180.0       # + = clockwise = right
            self.heading = bearing
            if abs(delta) < 0.5:
                self._next_step(i, n, e)
                return
            kind, arg = "turn", -delta                       # turn's convention: + = left (counter-clockwise)
        if kind == "turn":
            sign = 1.0 if arg > 0 else -1.0                  # + = left (counter-clockwise), - = right
            self.turn = (sign, abs(arg), (n, e))
            self.turned, self.last_wz, self.t_turned = 0.0, 0.0, None
            self._set("turn", i.now, f"turning {abs(arg):.1f} deg {'left' if arg > 0 else 'right'} in place at "
                                     f"{self.p.turn_rate_dps:.0f} deg/s")
        elif kind == "pause":
            self.hold_pt = (n, e)
            self._set("pause", i.now, f"stopped at ({n:.2f}, {e:.2f}) for {self.p.pause_s:.1f} s")
        else:
            dn, de = arg[0] - n, arg[1] - e
            d = math.hypot(dn, de)
            u = (dn / d, de / d) if d > 1e-6 else (1.0, 0.0)
            self.leg, self.s_ref = ((n, e), arg, u, d), 0.0
            self.leg_index += 1
            if self.leg_index > 1 and self.p.leg_climb_to_m > 0:
                self.h_target = self.p.leg_climb_to_m        # after the gap leg: hold the tunnel height
            self._set("leg", i.now, f"straight {d:.2f} m to ({arg[0]:.2f}, {arg[1]:.2f}), heading "
                                    f"{math.degrees(math.atan2(u[0], u[1])):.0f} deg north of east")

    def _leg_cmd(self, n: float, e: float, vn_m: float, ve_m: float, v: float) -> tuple[float, float]:
        """Carrot on a stick: a reference point advancing along the straight leg at v (never > 0.2 m ahead of the
        drone's progress), tracked with P + the learned push; feed-forward v along the leg."""
        p = self.p
        (tn, te), _, (un, ue), total = self.leg
        s_now = (n - tn) * un + (e - te) * ue
        self.s_ref = min(self.s_ref + v * self.dt, s_now + 0.20, total)
        rn, re = tn + self.s_ref * un, te + self.s_ref * ue
        ff = v if self.s_ref < total else 0.0
        cap = 2 * v
        vn = _clamp(ff * un + p.k_xy * ((rn - n) - p.lag_s * (vn_m - ff * un)) + self._integrate("i_n", rn - n, vn_m - ff * un),
                    -cap, cap)
        ve = _clamp(ff * ue + p.k_xy * ((re - e) - p.lag_s * (ve_m - ff * ue)) + self._integrate("i_e", re - e, ve_m - ff * ue),
                    -cap, cap)
        return vn, ve

    def _start_gap(self, i: Inputs, n: float, e: float) -> None:
        """A straight leg from here through the CP2 gap's centre, on past it by gap_past_m (nose still north)."""
        p = self.p
        dn, de = p.gap_n_m - n, p.gap_e_m - e
        d = math.hypot(dn, de)
        un, ue = dn / d, de / d
        end = (p.gap_n_m + p.gap_past_m * un, p.gap_e_m + p.gap_past_m * ue)
        self.leg = ((n, e), end, (un, ue), d + p.gap_past_m)
        self.s_ref = 0.0
        self._set("gap", i.now, f"at east {e:.2f} m: straight through the gap, {math.degrees(math.atan2(un, ue)):.0f} deg "
                                f"north of east, to N {end[0]:.2f} E {end[1]:.2f}")

    # ------------------------------------------------------------------ landing (refnav_skybeta.Descent)
    def _descend(self, i: Inputs, armed: Any, offboard: Any) -> Output:
        p = self.p
        if not (armed and offboard):
            self._set("done", i.now, "landed: PX4 disarmed or Offboard ended" if armed is False else "Offboard ended")
            return Output(self.phase, self.why)
        if self.blind:          # the estimate jumped: straight down on time alone, pressing once surely down
            h = self.h_target if self.h_target is not None else p.height_m
            press = i.now - self.t_land > h / p.descend_mps + p.ground_wait_s
            return Output(self.phase, self.why, (0.0, 0.0, -(p.press_mps if press else p.descend_mps), 0.0))
        if i.pos is None or i.now - i.t_odom > p.odom_hold_s:
            return Output(self.phase, self.why, (0.0, 0.0, -p.descend_mps, 0.0))   # blind: straight down
        n, e, u = i.pos
        vn_m, ve_m = self.vel_est
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
