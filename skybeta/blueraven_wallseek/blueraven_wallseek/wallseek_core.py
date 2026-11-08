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
    standoff_m: float = 0.50        # drone CENTRE to the wall's face while stopped/translating (Darion)
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
    i_zone_m: float = 0.10          # ... and it only learns within this of the target
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
        cap = math.radians(self.p.yaw_rate_dps)
        return _clamp(self.p.kp_yaw * (i.yaw_err or 0.0), -cap, cap)

    def _vz(self, i: Inputs) -> float:
        return _clamp(self.p.k_z * (self.p.height_m - i.pos[2]), -self.p.climb_mps, self.p.climb_mps)

    def _integrate(self, name: str, err: float) -> float:
        """Learns a steady push only near the target (|err| < i_zone_m): far errors are the approach itself, and
        integrating them would wind up and overshoot into the wall."""
        cap = self.p.i_max_mps
        if abs(err) >= self.p.i_zone_m:
            return getattr(self, name)
        v = _clamp(getattr(self, name) + self.p.ki * err * self.dt, -cap, cap)
        setattr(self, name, v)
        return v

    def _hold(self, target: float, now_pos: float, vel: float, axis: str = "i_e") -> float:
        err = target - now_pos
        cap = self.p.xy_hold_cap_mps
        return _clamp(self.p.k_xy * (err - self.p.lag_s * vel) + self._integrate(axis, err), -cap, cap)

    def _east_ve(self, e: float, ve_meas: float) -> float:
        """Translate rightward at translate_mps against any push: a reference east position advancing at
        translate_mps (never more than 0.2 m ahead of the drone), tracked with the east integral."""
        p = self.p
        self.e_ref = min((self.e_ref if self.e_ref is not None else e) + p.translate_mps * self.dt, e + 0.20)
        err = self.e_ref - e
        return _clamp(p.translate_mps + p.k_xy * (err - p.lag_s * (ve_meas - p.translate_mps))
                      + self._integrate("i_e", err), 0.0, 2 * p.translate_mps)

    def _wall_vn(self, rng: float, vn_meas: float, lo: float, hi: float) -> float:
        """Standoff loop on the depth range, with the velocity lag predicted out, plus the learned push."""
        err = rng - self.p.standoff_m
        return _clamp(self.p.k_wall * (err - self.p.lag_s * vn_meas) + self._integrate("i_n", err), lo, hi)

    # ------------------------------------------------------------------ step
    def step(self, i: Inputs) -> Output:
        p = self.p
        if self.t_phase is None:
            self.t_phase = i.now
        self.dt = 0.0 if self.t_last is None else _clamp(i.now - self.t_last, 0.0, 0.25)
        self.t_last = i.now
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
            if not fresh_state:
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
            elif i.ahead_m is not None and i.ahead_m < p.contact_m:
                vn = -p.xy_hold_cap_mps
                return Output(self.phase, f"backing off: surface {i.ahead_m:.2f} m ahead", (vn, 0.0, self._vz(i), self._wz(i)))

        n, e, u = i.pos if i.pos else (0.0, 0.0, 0.0)
        vn_m, ve_m = i.vel if i.vel else (0.0, 0.0)
        rng = i.ahead_m if i.ahead_m is not None else math.inf
        wall = rng < p.clear_m

        if self.phase == "climb":
            if abs(u - p.height_m) < p.climb_tol_m:
                self.t_at_height = self.t_at_height if self.t_at_height is not None else i.now
                if i.now - self.t_at_height >= p.climb_hold_s:
                    self._set("approach", i.now, f"at {u:.2f} m: forward until a wall is {p.standoff_m:.2f} m ahead")
            else:
                self.t_at_height = None
            if self.phase == "climb":
                return Output(self.phase, self.why,
                              (self._hold(0.0, n, vn_m, "i_n"), self._hold(0.0, e, ve_m), self._vz(i), self._wz(i)))

        if self.phase == "approach":
            if not wall and n >= p.max_forward_m - 0.02:
                self._land(i, f"no wall within {p.max_forward_m:.2f} m forward")
            elif i.now - self.t_phase > p.approach_timeout_s:
                self._land(i, f"approach did not settle within {p.approach_timeout_s:.0f} s")
            else:
                if wall:
                    vn = self._wall_vn(rng, vn_m, -p.xy_hold_cap_mps, p.approach_mps)
                    if abs(rng - p.standoff_m) < p.settle_tol_m and math.hypot(vn_m, ve_m) < p.settle_mps:
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
                vn = self._wall_vn(rng, vn_m, -p.xy_hold_cap_mps, p.xy_hold_cap_mps)
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
                vn = (self._wall_vn(rng, vn_m, -p.xy_hold_cap_mps, p.xy_hold_cap_mps) if wall
                      else self._hold(n, n, vn_m, "i_n"))
                return Output(self.phase, self.why, (vn, self._east_ve(e, ve_m), self._vz(i), self._wz(i)))

        if self.phase == "past":
            if e >= self.e_clear + p.past_end_m or e >= p.max_right_m:
                self._land(i, f"{e - self.e_clear:.2f} m past the wall's end")
            elif i.ahead_m is not None and i.ahead_m < p.standoff_m and i.now - i.t_depth <= p.depth_hold_s:
                self._land(i, f"something {i.ahead_m:.2f} m ahead after the wall ended")
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
