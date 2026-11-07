"""refnav.core - the reference mission's flight logic. PURE: no ROS, no clock reads, no files.

Everything arrives through calls that carry their own time, on ONE clock chosen by whoever runs
it: simulation time in the simulator and in replays of recorded data (ROS use_sim_time + /clock),
system time on hardware. The same code therefore runs against all three.

The mission: wait until PX4 has a valid position and heading, the estimator reports healthy and
the depth camera is live -> stream a hold setpoint, request Offboard and arm -> climb -> fly a
pre-planned route of legs (each leg: turn to face it, check the depth ahead, advance a carrot at a
bounded speed while the live depth shows the way clear) -> land.

Safety behaviour (each one is a test in tests/test_refnav_core.py):
* fail closed on estimator health: absent, stale or unhealthy refuses to arm; unhealthy or silent
  for health_bad_land_s in the air lands;
* no translation without OBSERVED free space: the carrot advances only while a depth frame no older
  than depth_stale_s returned depth on at least min_corridor_coverage of the rays crossing the
  corridor ahead (no return is unknown space, never free space); a blank, sparse (valid fraction
  < min_valid_frac), future-stamped, stale-on-arrival or out-of-order frame refreshes nothing; no
  usable depth for depth_lost_land_s lands;
* an obstacle closer than stop_m ahead stops the aircraft and ends the route (it lands where it
  is: route blocked; planning around obstacles is the student's job);
* PX4 leaving Offboard while airborne (a pilot or a failsafe) is never fought: the mission stops
  streaming and commanding and ends ("yielded");
* PX4 owns the landing: once AUTO_LAND is observed the mission stops streaming setpoints (PX4's
  land detector reads the trajectory_setpoint topic; a position-only stream blanks its
  commanded-descent test and the aircraft never detects touchdown);
* a landing REQUEST from outside (the interactive runtime's `land`, request_land) goes through this
  same landing: airborne it lands where it is; before arming it ends without arming; while arming it
  ends with a normal checked disarm. It never adds a controller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import depth as D

if TYPE_CHECKING:
    from typing import Any

    import numpy as np

    Intrinsics = tuple[float, float, float, float]      # pinhole (fx, fy, cx, cy) in pixels
    QuatWXYZ = tuple[float, float, float, float]        # PX4 attitude, FRD body -> NED, w first

NAV_OFFBOARD = 14
NAV_AUTO_LAND = 18
AIRBORNE = ("climb", "hover", "turn", "check", "go", "hold")


@dataclass
class Params:
    route: list[tuple[float, float]] = field(default_factory=list)   # [(forward_m, left_m)] from the takeoff point and heading
    # Per leg: camera looks this many deg right (+) / left (-) of travel. Depth checks stay on the travel
    # direction. A right offset moves the corridor into the left image columns, where stereo returns no
    # depth (columns < 80 of 640 in run_0613), so it lowers corridor coverage; see test_a_right_camera_offset...
    leg_yaw_off_deg: list[float] = field(default_factory=lambda: [0.0])
    height_m: float = 1.2
    speed_mps: float = 0.6
    lead_max_m: float = 1.0                     # the carrot never leads the aircraft by more
    yaw_rate_dps: float = 45.0
    face_tol_deg: float = 8.0
    arrive_m: float = 0.3
    hold_s: float = 1.0
    climb_tol_m: float = 0.15
    climb_rate_mps: float = 0.5
    hover_s: float = 0.0                        # hold at height_m this long after the climb (0: straight on)
    health_quiet_s: float = 3.0                 # healthy this long before arming
    health_stale_s: float = 1.0
    health_bad_land_s: float = 1.0
    depth_stale_s: float = 0.5
    depth_lost_land_s: float = 3.0
    stop_m: float = 0.9
    corridor_half_m: float = 0.6
    band_m: float = 0.4
    min_obstacle_px: int = 1200                 # an obstacle covers at least this many image pixels (~35 x 35):
                                                # a 5 cm pole does out to ~4 m, sparse stereo junk does not
    depth_stride: int = 8                       # every n-th pixel of the depth image is used
    min_valid_frac: float = 0.3                 # below this a frame is unusable (flight data: >= 0.76)
    min_corridor_coverage: float = 0.6          # of the corridor's rays must return depth (flight data: >= 0.86)
    depth_future_tol_s: float = 0.05            # a capture stamp later than now + this is refused
    depth_border_px: int = 24                   # image edge ignored (stereo matching is weakest there)
    confirm_frames: int = 2                     # an obstacle must be in this many consecutive frames to stop
    stream_lead_s: float = 1.0                  # setpoints stream this long before Offboard is requested
    wait_s: float = 300.0                       # nothing ready this long: give up without arming
    arm_wait_s: float = 30.0
    cmd_resend_s: float = 1.0
    land_timeout_s: float = 60.0


@dataclass
class Vehicle:
    """The latest PX4 reports, updated by the adapter as messages arrive."""
    pos: tuple[float, float, float] | None = None          # NED metres (vehicle_local_position)
    heading: float | None = None      # rad
    pos_ok: bool = False              # xy_valid and z_valid and a heading (good for control, or PX4 yaw-aligned)
    q: QuatWXYZ | None = None            # attitude, FRD -> NED, (w, x, y, z)
    armed: bool | None = None
    nav_state: int | None = None
    landed: bool | None = None


@dataclass
class Output:
    phase: str
    setpoint: tuple[float, float, float, float] | None = None     # (n, e, d, yaw) for the trajectory setpoint
    stream: bool = False              # publish offboard_control_mode + the setpoint this tick
    commands: list[str] = field(default_factory=list)   # "offboard" | "arm" | "land" | "disarm" (checked)
    events: list[str] = field(default_factory=list)


class Health:
    """Estimator health from a boolean stream, failing closed: nothing heard, stale or unhealthy
    is never permission, and a gap restarts the quiet window."""

    def __init__(self, quiet_s: float, stale_s: float) -> None:
        self.quiet_s, self.stale_s = float(quiet_s), float(stale_s)
        self.last_t = self.last_ok = self.ok_since = self.bad_since = None  # times: float | None; last_ok: bool | None

    def on_msg(self, t: float, ok: bool) -> None:
        ok = bool(ok)
        if ok:
            if not self.last_ok or self.ok_since is None or (self.last_t is not None and t - self.last_t > self.stale_s):
                self.ok_since = t
            self.bad_since = None
        else:
            self.ok_since = None
            if self.bad_since is None:
                self.bad_since = t
        self.last_t, self.last_ok = t, ok

    def fresh(self, now: float) -> bool:
        return self.last_t is not None and now - self.last_t <= self.stale_s

    def arm_ok(self, now: float) -> tuple[bool, str]:
        if self.last_t is None:
            return False, "no estimator health received"
        if not self.fresh(now):
            return False, f"estimator health stale ({now - self.last_t:.1f} s)"
        if not self.last_ok:
            return False, "estimator unhealthy"
        if now - self.ok_since < self.quiet_s:
            return False, f"estimator healthy for only {now - self.ok_since:.1f}/{self.quiet_s:.0f} s"
        return True, "ok"

    def bad_for(self, now: float) -> float:
        if self.last_t is None:
            return math.inf
        if not self.fresh(now):
            return now - (self.last_t + self.stale_s)
        if not self.last_ok:
            return now - self.bad_since
        return 0.0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Mission:
    def __init__(self, params: Params) -> None:
        self.p: Params = params
        self.health: Health = Health(params.health_quiet_s, params.health_stale_s)
        self.phase: str = "wait"
        self.why: str = ""
        self.sp: list[float] | None = None    # [n, e, d, yaw]
        self.home: tuple[float, float, float, float] | None = None              # (n, e, d, heading) at the arm request
        self.targets: list[tuple[float, float]] = []             # NED (n, e) of each leg's end
        self.leg: int = 0
        self.legs_done: int = 0
        self.t_phase: float | None = None
        self.t_started: float | None = None         # the first time this mission saw (its clock): where a replay starts
        self.t_last: float | None = None
        self.t_cmd: float | None = None
        self.offboard_seen: bool = False
        self.land_reason: str | None = None
        self.land_auto_seen: bool = False
        self.result: str | None = None            # final: "complete" | "blocked" | "landed: <why>" | "refused: <why>" | "yielded: <why>"
        self.depth_ready_since: float | None = None # start of continuously usable depth before control requests
        self.depth_t: float | None = None           # capture stamp of the latest USABLE depth frame (depth alive)
        self.frame: D.DepthFrame | None = None             # that frame (refnav.depth.DepthFrame) and its obstacle points
        self.depth_pts: np.ndarray | None = None
        self.corr_t: float | None = None            # stamp of the latest frame that OBSERVED the current corridor
        self._judged_t: float | None = None         # the latest frame judged against the corridor (each frame once)
        self.last_coverage: float | None = None
        self.leg_start: tuple[float, float] | None = None
        self.leg_dir: tuple[float, float] | None = None
        self.leg_len: float = 0.0
        self.leg_heading: float = 0.0
        self.leg_yaw: float = 0.0                   # heading held on this leg: leg_heading + its camera offset
        self.s: float = 0.0
        self.counts: dict[str, int] = {"depth_frames": 0, "obstacle_stops": 0, "depth_holds": 0, "unconfirmed_obstacles": 0,
                       "depth_unusable": 0, "depth_future": 0, "depth_stale": 0, "depth_out_of_order": 0,
                       "corridor_unobserved": 0}
        self.clr_hist: list[tuple[float, float]] = []            # [(depth stamp, clearance)] along the current leg, newest last
        self.last_clearance: float | None = None
        self.land_request: str | None = None      # why an outside landing request was made (request_land), until acted on
        self.depth_reject: tuple[str, str] | None = None      # (key, detail): why the latest refused depth frame was refused
        self.last_valid_frac: float | None = None   # fraction of pixels returning depth in the latest decoded frame

    def request_land(self, why: str = "requested") -> None:
        """An outside request to end the flight through the mission's own landing (no second controller)."""
        if self.land_request is None:
            self.land_request = str(why) or "requested"

    def depth_refused(self, key: str, detail: str) -> None:
        """A depth frame was refused (here, or before it reached the mission: no calibration, no
        attitude, an undecodable image). The reason is what the wait reports, with its numbers."""
        self.depth_reject = (str(key), str(detail))
        # Obsolete transport packets do not contradict the newer usable stream.
        # Its existing freshness deadline still expires if current data stops.
        if key not in ("out_of_order", "stale"):
            self.depth_ready_since = None

    # ------------------------------------------------------------------ inputs
    def on_health(self, t: float, ok: bool) -> None:
        self.health.on_msg(t, ok)

    def on_depth(self, stamp: float, depth_m: np.ndarray, K: Intrinsics, R_body_cam: np.ndarray,
                 t_body_cam: np.ndarray, q_wxyz: QuatWXYZ | None, now: float | None = None) -> bool:
        """One depth frame with its own calibration; stamp = capture time on the mission clock, now =
        that clock at arrival (None: not judged for being from the future). -> True if it was usable."""
        if q_wxyz is None:
            self.depth_refused("attitude_missing", "no attitude for this depth frame")
            return False
        stamp = float(stamp)
        if now is not None and stamp > now + self.p.depth_future_tol_s:
            self.counts["depth_future"] += 1
            self.depth_refused("future", f"stamped {stamp - now:.2f} s in the future "
                                         f"(depth_future_tol_s {self.p.depth_future_tol_s:g} s)")
            return False
        if now is not None and now - stamp > self.p.depth_stale_s:
            self.counts["depth_stale"] += 1                  # too old to act on when it arrives
            self.depth_refused("stale", f"{now - stamp:.2f} s old on arrival (depth_stale_s {self.p.depth_stale_s:g} s)")
            return False
        if self.depth_t is not None and stamp <= self.depth_t:
            self.counts["depth_out_of_order"] += 1
            self.depth_refused("out_of_order", "stamped no later than the last frame used")
            return False
        f = D.DepthFrame(depth_m, K, R_body_cam, t_body_cam, D.level_rotation(q_wxyz), stride=self.p.depth_stride,
                         border_px=self.p.depth_border_px)
        self.last_valid_frac = f.valid_frac
        if f.valid_frac < self.p.min_valid_frac:
            self.counts["depth_unusable"] += 1
            self.depth_refused("too_little_depth", f"depth on {f.valid_frac:.0%} of the pixels, this mission needs "
                                                   f"{self.p.min_valid_frac:.0%} (min_valid_frac)")
            return False
        if (self.depth_ready_since is None or self.depth_t is None
                or stamp - self.depth_t > self.p.depth_stale_s):
            self.depth_ready_since = stamp
        self.frame, self.depth_t, self.depth_pts = f, stamp, f.points(0.3, 6.0)
        self.depth_reject = None                             # a refusal is only reported until a frame is used
        self.counts["depth_frames"] += 1
        return True

    def depth_fresh(self, now: float) -> bool:
        """A usable depth frame is recent (the camera is alive), whatever it looked at."""
        return self.depth_t is not None and now - self.depth_t <= self.p.depth_stale_s

    def depth_ready_for_arm(self, now: float) -> bool:
        """Same quiet interval as estimator health, measured in usable capture history.

        A rare good snapshot among unusable frames cannot authorize motors. This
        only gates the first control request; airborne/authority policy is unchanged.
        """
        return (self.depth_fresh(now) and self.depth_ready_since is not None
                and math.isfinite(self.p.health_quiet_s) and self.p.health_quiet_s >= 0
                and self.depth_t - self.depth_ready_since >= self.p.health_quiet_s)

    def corridor_fresh(self, now: float) -> bool:
        """A recent frame OBSERVED the corridor of the current leg."""
        return self.corr_t is not None and now - self.corr_t <= self.p.depth_stale_s

    # ------------------------------------------------------------------ step
    def _set(self, phase: str, now: float, out: Output, why: str = "") -> None:
        out.events.append(f"{self.phase} -> {phase}" + (f": {why}" if why else ""))
        self.phase, self.t_phase = phase, now
        if why:
            self.why = why

    def _end(self, now: float, out: Output, result: str) -> None:
        self.result = result
        self._set("done", now, out, result)

    def _land(self, now: float, out: Output, why: str) -> None:
        self.land_reason = why
        self._set("land", now, out, why)

    def _resend(self, now: float) -> bool:
        if self.t_cmd is None or now - self.t_cmd >= self.p.cmd_resend_s:
            self.t_cmd = now
            return True
        return False

    def _requalify(self, now: float, out: Output, why: str) -> None:
        # Cancel the old priming interval and its held position. The original
        # preflight deadline remains t_started; this cannot renew its budget.
        self.sp, self.home, self.targets = None, None, []
        self._set("wait", now, out, why)

    def step(self, now: float, v: Vehicle) -> Output:
        p = self.p
        out = Output(self.phase)
        dt = 0.0 if self.t_last is None else min(max(now - self.t_last, 0.0), 0.25)
        self.t_last = now
        if self.t_phase is None:
            self.t_phase = now
        if self.t_started is None:
            self.t_started = now
        if self.phase == "done":
            return out

        if self.phase in ("wait", "arm") and self.t_cmd is None:
            if v.armed is True:
                self._end(now, out, "yielded: vehicle armed before this mission requested control")
            elif now - self.t_started > p.wait_s:
                self._end(now, out, f"refused: {self.why or 'not ready'} after {p.wait_s:.0f} s")
            elif self.phase == "arm" and v.armed is None:
                self._requalify(now, out, "waiting for an explicit disarmed PX4 report")

        # ---- airborne supervision, before anything else may command -------------------
        if self.phase in AIRBORNE or (self.phase == "land" and not self.land_auto_seen):
            # only the mission's OWN landing may put PX4 in AUTO_LAND; anywhere else that is PX4 (a
            # failsafe) or a pilot taking over, and streaming on would blank PX4's land detector
            allowed = (NAV_OFFBOARD, NAV_AUTO_LAND) if self.phase == "land" else (NAV_OFFBOARD,)
            if self.offboard_seen and v.nav_state is not None and v.nav_state not in allowed:
                self._end(now, out, f"yielded: PX4 left Offboard (nav_state {v.nav_state}): a pilot or a failsafe has it")
                out.phase = self.phase
                return out
        if self.land_request is not None and self.phase in ("wait", "arm") + AIRBORNE:
            why = f"landing requested: {self.land_request}"
            if self.phase == "wait":
                self._end(now, out, f"refused: {why} before arming")
            elif self.phase == "arm":
                if v.armed:
                    out.commands.append("disarm")          # a normal, checked disarm: on the ground, never forced
                self._end(now, out, f"refused: {why} while arming")
            else:
                self._land(now, out, why)
            self.land_request = None
            out.phase = self.phase
        if self.phase in AIRBORNE:
            bad = self.health.bad_for(now)
            if bad > p.health_bad_land_s:
                self._land(now, out, f"estimator unhealthy or silent for {bad:.1f} s")
            elif not v.pos_ok or v.pos is None:
                self._land(now, out, "PX4 position or heading invalid")
            elif self.depth_t is None or now - self.depth_t > p.depth_lost_land_s:
                self._land(now, out, f"depth silent for more than {p.depth_lost_land_s:.0f} s")

        ph = self.phase
        if ph == "wait":
            ok, why = self.health.arm_ok(now)
            if now - self.t_started > p.wait_s:
                self._end(now, out, f"refused: {self.why or 'not ready'} after {p.wait_s:.0f} s")
            elif v.armed is not False:
                self.why = "waiting for an explicit disarmed PX4 report"
            elif not v.pos_ok or v.pos is None or v.heading is None:
                self.why = "waiting for PX4 position and heading"
            elif not ok:
                self.why = why
            elif not self.depth_fresh(now):
                self.why = "waiting for live depth: " + (
                    f"the latest frame was refused: {self.depth_reject[1]}" if self.depth_reject else
                    "no depth frame has arrived" if self.depth_t is None else
                    f"the latest usable frame is {now - self.depth_t:.1f} s old")
            elif not self.depth_ready_for_arm(now):
                self.why = f"waiting for {p.health_quiet_s:g} s of continuously usable depth"
            else:
                n, e, d = v.pos
                self.home = (n, e, d, v.heading)
                self.sp = [n, e, d, v.heading]
                c, s = math.cos(v.heading), math.sin(v.heading)
                self.targets = [(n + f * c + l * s, e + f * s - l * c) for f, l in p.route]
                self._set("arm", now, out)
        elif ph == "arm":
            ok, why = self.health.arm_ok(now)
            if not ok:
                if self.t_cmd is None and v.armed is False:
                    self._requalify(now, out, f"{why}; requalifying before any control request")
                else:
                    self._end(now, out, f"refused: {why} before arming")
            elif self.t_cmd is None and not self.depth_ready_for_arm(now):
                if v.armed is False:
                    self._requalify(now, out, "depth not continuously usable; requalifying before any control request")
                else:
                    self._end(now, out, "yielded: depth lost before this mission requested control")
            elif now - self.t_phase > p.arm_wait_s:
                if v.armed:
                    out.commands.append("disarm")          # a normal, checked disarm: on the ground, never forced
                self._end(now, out, f"refused: not armed in Offboard after {p.arm_wait_s:.0f} s")
            elif v.armed and v.nav_state == NAV_OFFBOARD:
                self.offboard_seen = True
                self._set("climb", now, out)
            elif now - self.t_phase >= p.stream_lead_s and self._resend(now):
                out.commands += ["offboard", "arm"]
        elif ph == "climb":
            target = self.home[2] - p.height_m
            self.sp[2] = max(target, self.sp[2] - p.climb_rate_mps * dt)
            if self.sp[2] <= target + 1e-9 and abs(v.pos[2] - target) < p.climb_tol_m:
                if p.hover_s > 0:
                    self._set("hover", now, out, f"holding {p.hover_s:.0f} s at {p.height_m:.1f} m")
                else:
                    self._next_leg(now, out, v)
        elif ph == "hover":
            if now - self.t_phase >= p.hover_s:
                self._next_leg(now, out, v)
        elif ph == "turn":
            err = _wrap(self.leg_yaw - self.sp[3])
            step = math.radians(p.yaw_rate_dps) * dt
            self.sp[3] = _wrap(self.sp[3] + max(-step, min(step, err)))
            if abs(err) < 1e-3 and abs(_wrap(self.leg_yaw - v.heading)) < math.radians(p.face_tol_deg):
                self._set("check", now, out)
        elif ph == "check":
            # a depth frame captured AFTER the turn ended looks along this leg
            if self.depth_t is not None and self.depth_t >= self.t_phase and self.depth_fresh(now):
                if self.clr_hist and self.clr_hist[0][0] < self.t_phase:
                    self.clr_hist, self.corr_t = [], None    # frames from before the turn ended look elsewhere
                clr = self._clearance(v)                     # None: not enough observed frames yet, wait
                if clr is not None and self.corridor_fresh(now):
                    self.last_clearance = clr
                    if clr < p.stop_m:
                        self.counts["obstacle_stops"] += 1
                        self._land(now, out, f"route blocked: obstacle {clr:.2f} m ahead at leg {self.leg + 1}")
                    else:
                        self.s = 0.0
                        self._set("go", now, out)
        elif ph == "go":
            self._go(now, dt, out, v)
        elif ph == "hold":
            if now - self.t_phase >= p.hold_s:
                self.legs_done += 1
                self.leg += 1
                self._next_leg(now, out, v)

        if self.phase == "land":
            if v.nav_state == NAV_AUTO_LAND:
                self.land_auto_seen = True
            if not self.land_auto_seen and self._resend(now):
                out.commands.append("land")
            if self.land_auto_seen and v.armed is False:
                why = self.land_reason or ""
                self._end(now, out, "complete" if why == "route complete" else f"landed: {why}")
            elif now - self.t_phase > p.land_timeout_s:
                self._end(now, out, f"landed: {self.land_reason}; landing not confirmed within {p.land_timeout_s:.0f} s")

        out.phase = self.phase
        out.setpoint = tuple(self.sp) if self.sp is not None else None
        # PX4 owns the landing once AUTO_LAND is seen: nothing streams from then on
        out.stream = self.phase in ("arm",) + AIRBORNE or (self.phase == "land" and not self.land_auto_seen)
        return out

    def _next_leg(self, now: float, out: Output, v: Vehicle) -> None:
        if self.leg >= len(self.targets):
            self._land(now, out, "route complete")
            return
        start = (self.sp[0], self.sp[1])
        tn, te = self.targets[self.leg]
        dn, de = tn - start[0], te - start[1]
        self.leg_len = math.hypot(dn, de)
        self.leg_start = start
        self.leg_dir = (dn / self.leg_len, de / self.leg_len) if self.leg_len > 1e-6 else (1.0, 0.0)
        self.leg_heading = math.atan2(de, dn) if self.leg_len > 1e-6 else self.sp[3]
        offs = self.p.leg_yaw_off_deg
        off = math.radians(offs[self.leg]) if self.leg < len(offs) else 0.0
        self.leg_yaw = _wrap(self.leg_heading + off)
        self.clr_hist, self.corr_t = [], None
        self._set("turn", now, out, f"leg {self.leg + 1}/{len(self.targets)}")

    def _clearance(self, v: Vehicle) -> float | None:
        """Clearance along the leg CONFIRMED by the last confirm_frames frames that OBSERVED the
        corridor: the largest of their clearances, so an obstacle counts only when every one of
        those frames shows it (a stereo false match lasts a frame; a wall does not go away). A frame
        whose rays through the corridor (at stop_m + lead_max_m ahead) mostly returned nothing has
        observed nothing there and is not evidence of free space. None = unknown (do not move)."""
        p = self.p
        bearing = _wrap(self.leg_heading - v.heading)        # positive = to the right (NED yaw)
        if self.frame is not None and self._judged_t != self.depth_t:
            self._judged_t = self.depth_t
            cov, _ = self.frame.corridor_coverage(bearing, p.corridor_half_m, p.band_m, p.stop_m + p.lead_max_m)
            self.last_coverage = cov
            if cov >= p.min_corridor_coverage:
                hits = max(3, round(p.min_obstacle_px / p.depth_stride ** 2))
                c = D.clearance(self.depth_pts, bearing, p.corridor_half_m, p.band_m, hits)
                self.clr_hist = (self.clr_hist + [(self.depth_t, c)])[-max(1, p.confirm_frames):]
                self.corr_t = self.depth_t
            else:
                self.counts["corridor_unobserved"] += 1
        if len(self.clr_hist) < self.p.confirm_frames:
            return None
        confirmed = max(c for _, c in self.clr_hist)
        if min(c for _, c in self.clr_hist) < self.p.stop_m <= confirmed:
            self.counts["unconfirmed_obstacles"] += 1
        return confirmed

    def _go(self, now: float, dt: float, out: Output, v: Vehicle) -> None:
        p = self.p
        n0, e0 = self.leg_start
        un, ue = self.leg_dir
        proj = (v.pos[0] - n0) * un + (v.pos[1] - e0) * ue
        clr = self._clearance(v) if self.depth_fresh(now) else None
        if clr is not None and not self.corridor_fresh(now):
            clr = None                                       # the latest evidence of this corridor is too old
        if clr is not None:
            self.last_clearance = clr
            if clr < p.stop_m:
                # stop where the aircraft is (not where the carrot was)
                self.sp[0], self.sp[1] = v.pos[0], v.pos[1]
                self.counts["obstacle_stops"] += 1
                self._land(now, out, f"route blocked: obstacle {clr:.2f} m ahead on leg {self.leg + 1}")
                return
            self.s = min(self.s + p.speed_mps * dt, self.leg_len, max(proj, 0.0) + p.lead_max_m)
        else:
            self.counts["depth_holds"] += 1                  # no live evidence: the carrot waits
        self.sp[0], self.sp[1] = n0 + un * self.s, e0 + ue * self.s
        tn, te = self.targets[self.leg]
        if self.s >= self.leg_len - 1e-6 and math.hypot(v.pos[0] - tn, v.pos[1] - te) < p.arrive_m:
            self._set("hold", now, out, f"arrived leg {self.leg + 1}")

    def status(self, now: float) -> dict[str, Any]:
        return {"phase": self.phase, "why": self.why, "result": self.result, "started_t": self.t_started, "leg": self.leg,
                "legs": len(self.targets), "legs_done": self.legs_done, "home": self.home,
                "targets": [list(t) for t in self.targets], "setpoint": list(self.sp) if self.sp else None,
                "clearance_m": (None if self.last_clearance is None or math.isinf(self.last_clearance)
                                else round(self.last_clearance, 2)),
                "depth_age_s": None if self.depth_t is None else round(now - self.depth_t, 3),
                "corridor_age_s": None if self.corr_t is None else round(now - self.corr_t, 3),
                "corridor_coverage": None if self.last_coverage is None else round(self.last_coverage, 2),
                "depth_valid_frac": None if self.last_valid_frac is None else round(self.last_valid_frac, 3),
                "depth_reject": None if self.depth_reject is None else self.depth_reject[1],
                "health_bad_s": (None if math.isinf(self.health.bad_for(now)) else round(self.health.bad_for(now), 2)),
                **self.counts}
