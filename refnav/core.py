"""refnav.core - the reference mission's flight logic. PURE: no ROS, no clock reads, no files.

Everything arrives through calls that carry their own time, on ONE clock chosen by whoever runs
it: simulation time in the simulator and in replays of recorded data (ROS use_sim_time + /clock),
system time on hardware. The same code therefore runs against all three.

The mission: wait until PX4 has a valid position and heading, the estimator reports healthy and
the depth camera is live -> stream a hold setpoint, request Offboard and arm -> climb -> follow one
smooth path through the route's points (refnav.follow), moving only into space the live depth has
observed free -> land.

Safety behaviour (each one is a test in tests/test_refnav_core.py):
* fail closed on estimator health: absent, stale or unhealthy refuses to arm; unhealthy or silent
  for health_bad_land_s in the air lands;
* no translation without OBSERVED free space: the reference only moves along path the latest depth
  frame (no older than depth_stale_s) observed, where at least min_corridor_coverage of the rays
  towards it returned depth (no return is unknown space, never free space), and brakes to stop there;
  a blank, sparse (valid fraction < min_valid_frac), future-stamped, stale-on-arrival or out-of-order
  frame refreshes nothing; no usable depth for depth_lost_land_s lands;
* an obstacle on the path closer than stop_m in confirm_frames consecutive frames ends the route (it
  lands where it is: route blocked);
* no progress along the path for stall_s lands, so a frozen reference cannot hover forever;
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
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from . import depth as D
from . import follow as F

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    Intrinsics = tuple[float, float, float, float]      # pinhole (fx, fy, cx, cy) in pixels
    QuatWXYZ = tuple[float, float, float, float]        # PX4 attitude, FRD body -> NED, w first

NAV_OFFBOARD = 14
NAV_AUTO_LAND = 18
AIRBORNE = ("climb", "hover", "follow")


@dataclass
class Params:
    route: list[tuple[float, float]] = field(default_factory=list)   # [(forward_m, left_m)] from the takeoff point and heading
    height_m: float = 1.2
    speed_mps: float = 0.6
    accel_mps2: float = 0.5                     # speeding up and braking along the path
    yaw_rate_dps: float = 45.0
    face_tol_deg: float = 8.0
    arrive_m: float = 0.3                       # the route is complete within this of its last point
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
    # PX4 disarms only if its land detector agrees, which needs a sane estimate; VIO can die on the pad (camera
    # inches off a blank floor) and the estimate then runs away. Touchdown is judged from the rangefinder instead.
    touchdown_m: float = 0.12                   # rangefinder at or under this, level, for touchdown_s: force the disarm
    touchdown_s: float = 8.0                    # PX4 confirms and disarms itself well inside this when it can
    touchdown_after_s: float = 2.0              # never sooner than this after the landing began
    touchdown_tilt_deg: float = 15.0
    stall_s: float = 45.0                       # no progress along the path this long: land
    fillet_m: float = 0.3                       # largest corner arc; less where the legs are short
    a_lat_mps2: float = 0.25                    # sideways acceleration in turns: PX4 lags tighter ones and cuts the corner
    tight_m: float = 0.40                       # where the mapped path passes this close to a surface it crawls...
    roomy_m: float = 0.60                       # ...rising to full speed by this clearance
    crawl_mps: float = 0.15
    look_min_m: float = 0.3                     # lookahead at rest; it grows by look_s per m/s made while on track
    look_s: float = 1.5
    look_max_m: float = 1.2
    xtrack_slow_m: float = 0.12                 # off the path by more than this: slow down...
    xtrack_stop_m: float = 0.35                 # ...and stop at this
    yaw_slow_deg: float = 60.0                  # heading error at which forward speed reaches zero
    eph_ok_m: float = 0.1                       # PX4 horizontal position std: full speed below this...
    eph_stop_m: float = 0.5                     # ...stopped at this


@dataclass
class Vehicle:
    """The latest PX4 reports, updated by the adapter as messages arrive."""
    pos: tuple[float, float, float] | None = None          # NED metres (vehicle_local_position)
    heading: float | None = None      # rad
    pos_ok: bool = False              # xy_valid and z_valid and a heading (good for control, or PX4 yaw-aligned)
    vel: tuple[float, float] = (0.0, 0.0)  # NED m/s (vehicle_local_position vx, vy)
    range_m: float | None = None      # downward rangefinder (distance_sensor), independent of the estimate
    eph: float = 0.0                  # PX4's horizontal position standard deviation, m
    q: QuatWXYZ | None = None            # attitude, FRD -> NED, (w, x, y, z)
    armed: bool | None = None
    nav_state: int | None = None
    landed: bool | None = None


@dataclass
class Output:
    phase: str
    setpoint: tuple[float, float, float, float] | None = None     # (n, e, d, yaw) for the trajectory setpoint
    velocity: tuple[float, float] | None = None   # (vn, ve) feed-forward with the setpoint, follow mode only
    accel: tuple[float, float] | None = None      # (an, ae) feed-forward, follow mode only
    yawspeed: float | None = None                 # rad/s feed-forward: the slew this tick, follow mode only
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


class Mission:
    def __init__(self, params: Params, path_builder: Callable[[list[tuple[float, float]]], F.Path] | None = None) -> None:
        self.p: Params = params
        self.path_builder = path_builder or (lambda corners: F.fillet_path(corners, params.fillet_m))   # NED corners -> path
        self.health: Health = Health(params.health_quiet_s, params.health_stale_s)
        self.phase: str = "wait"
        self.why: str = ""
        self.sp: list[float] | None = None    # [n, e, d, yaw]
        self.home: tuple[float, float, float, float] | None = None              # (n, e, d, heading) at the arm request
        self.targets: list[tuple[float, float]] = []             # NED (n, e) of each route point
        self.t_phase: float | None = None
        self.t_started: float | None = None         # the first time this mission saw (its clock): where a replay starts
        self.t_last: float | None = None
        self.t_cmd: float | None = None
        self.offboard_seen: bool = False
        self.land_reason: str | None = None
        self.land_auto_seen: bool = False
        self.t_touch: float | None = None           # since when the rangefinder has shown the aircraft on the ground
        self.result: str | None = None            # final: "complete" | "blocked" | "landed: <why>" | "refused: <why>" | "yielded: <why>"
        self.depth_ready_since: float | None = None # start of continuously usable depth before control requests
        self.depth_t: float | None = None           # capture stamp of the latest USABLE depth frame (depth alive)
        self.frame: D.DepthFrame | None = None             # that frame (refnav.depth.DepthFrame) and its obstacle points
        self.depth_pts: np.ndarray | None = None
        self.judged_t: float | None = None          # the latest depth frame judged against the path (each frame once)
        self.blocked_frames: int = 0                # consecutive judged frames with an obstacle inside stop_m
        self.s_limit: float = 0.0                   # path distance the latest judged frame shows free to stop in
        self.counts: dict[str, int] = {"depth_frames": 0, "obstacle_stops": 0, "depth_holds": 0, "unconfirmed_obstacles": 0,
                       "depth_unusable": 0, "depth_future": 0, "depth_stale": 0, "depth_out_of_order": 0, "stalls": 0}
        self.last_clearance: float | None = None
        self.land_request: str | None = None      # why an outside landing request was made (request_land), until acted on
        self.depth_reject: tuple[str, str] | None = None      # (key, detail): why the latest refused depth frame was refused
        self.last_valid_frac: float | None = None   # fraction of pixels returning depth in the latest decoded frame
        self.best_s: float = 0.0                    # furthest the aircraft has come along the path
        self.t_best: float = 0.0                    # when that was
        self.follower: F.Follower | None = None
        self.corr: F.SE2 = F.SE2()                  # map-based drift correction of PX4's estimate (identity until one arrives)
        self.corr_t: float | None = None
        self.cmd: F.Command | None = None

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
    def on_correction(self, t: float, x: float, y: float, yaw: float) -> None:
        """A drift correction of PX4's estimate in its local NED frame. The latest one holds until replaced."""
        self.corr, self.corr_t = F.SE2(x, y, yaw), t

    def _corrected(self, v: Vehicle) -> Vehicle:
        n, e = self.corr.point(v.pos[0], v.pos[1])
        return replace(v, pos=(n, e, v.pos[2]), heading=float(F.wrap(v.heading + self.corr.yaw)), vel=self.corr.vector(*v.vel))

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
            elif not v.pos_ok or v.pos is None or v.heading is None:
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
                    self._start(now, out, v)
        elif ph == "hover":
            if now - self.t_phase >= p.hover_s:
                self._start(now, out, v)
        elif ph == "follow":
            self._follow(now, dt, out, v)

        if self.phase == "land":
            if v.nav_state == NAV_AUTO_LAND:
                self.land_auto_seen = True
            if not self.land_auto_seen and self._resend(now):
                out.commands.append("land")
            if self.land_auto_seen and v.armed and self._on_ground(now, v) and self._resend(now):
                out.commands.append("disarm_force")         # on the ground for touchdown_s: PX4 may never agree
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

    def _start(self, now: float, out: Output, v: Vehicle) -> None:
        p = self.p
        if not self.targets:
            self._land(now, out, "route complete")
            return
        path = self.path_builder([(self.sp[0], self.sp[1])] + self.targets)
        prof = F.speed_profile(path, p.speed_mps, math.radians(p.yaw_rate_dps), p.a_lat_mps2, p.accel_mps2,
                               (p.tight_m, p.roomy_m, p.crawl_mps))
        lim = F.Limits(accel=p.accel_mps2, look_min=p.look_min_m, look_s=p.look_s, look_max=p.look_max_m,
                       face_tol=math.radians(p.face_tol_deg), yaw_slow=math.radians(p.yaw_slow_deg),
                       xtrack_slow=p.xtrack_slow_m, xtrack_stop=p.xtrack_stop_m, end_tol=p.arrive_m)
        self.follower = F.Follower(path, prof, lim, self.targets)
        self.best_s, self.t_best, self.s_limit = 0.0, now, 0.0
        self._set("follow", now, out, f"path {path.length:.1f} m through {len(self.targets)} points")

    def _judge_depth(self, v: Vehicle) -> bool:
        """Judge the newest depth frame against the path ahead, once per frame: set s_limit, where the reference must
        be able to stop (in observed free space, stop_m short of an obstacle). -> True when the route is blocked."""
        p, fol = self.p, self.follower
        if self.frame is None or self.judged_t == self.depth_t:
            return False
        self.judged_t = self.depth_t
        ne, along = fol.ahead(p.stop_m + p.look_max_m)
        c, sn = math.cos(v.heading), math.sin(v.heading)
        dn, de = ne[:, 0] - v.pos[0], ne[:, 1] - v.pos[1]
        fr = np.stack([dn * c + de * sn, -dn * sn + de * c], 1)          # level heading frame: forward, right
        hits = max(3, round(p.min_obstacle_px / p.depth_stride ** 2))
        # closer than stop_m the view is mostly floor, which stereo often misses: those samples were judged from further back
        free, obstacle = D.path_clearance(self.frame, self.depth_pts, fr, along, p.corridor_half_m, p.band_m, hits,
                                          p.min_corridor_coverage, blind_m=p.stop_m)
        self.last_clearance = free
        close = obstacle and free < p.stop_m
        self.blocked_frames = self.blocked_frames + 1 if close else 0
        self.counts["unconfirmed_obstacles"] += 0 < self.blocked_frames < p.confirm_frames
        if close:
            self.s_limit = fol.s_ref
        elif not obstacle and fol.s_hat + free >= fol.path.length - 1e-6:
            self.s_limit = math.inf                          # clear to the end: the speed profile brings it to rest
        else:
            self.s_limit = fol.s_hat + free - (p.stop_m if obstacle else 0.0)
        return self.blocked_frames >= p.confirm_frames

    def _follow(self, now: float, dt: float, out: Output, raw: Vehicle) -> None:
        """The path lives in the corrected frame; what goes to PX4 is mapped back into its own estimate's frame."""
        p, fol, v, back = self.p, self.follower, self._corrected(raw), self.corr.inverse()
        if not self.depth_fresh(now):
            self.s_limit = fol.s_ref                         # no live evidence: the reference waits
        elif self._judge_depth(v):
            self.counts["obstacle_stops"] += 1
            self.sp[0], self.sp[1] = raw.pos[0], raw.pos[1]
            self._land(now, out, f"route blocked: obstacle {self.last_clearance:.2f} m ahead at {fol.s_hat:.1f} m along the path")
            return
        if self.s_limit <= fol.s_ref:
            self.counts["depth_holds"] += 1
        doubt = (v.eph - p.eph_ok_m) / (p.eph_stop_m - p.eph_ok_m)
        self.cmd = fol.step((v.pos[0], v.pos[1]), v.vel, v.heading, dt, self.s_limit, doubt)
        before = self.sp[3]
        self._slew_yaw(self.cmd.yaw - self.corr.yaw, dt)
        self.sp[0], self.sp[1] = back.point(*self.cmd.pos)
        out.velocity, out.accel = back.vector(*self.cmd.vel), back.vector(*self.cmd.acc)
        out.yawspeed = float(F.wrap(self.sp[3] - before)) / dt if dt > 0 else 0.0
        if fol.s_hat > self.best_s + 0.1:
            self.best_s, self.t_best = fol.s_hat, now
        if now - self.t_best > p.stall_s:
            self.counts["stalls"] += 1
            self._land(now, out, f"no progress along the path for {p.stall_s:.0f} s at {fol.s_hat:.1f} m")
        elif self.cmd.done:
            self._land(now, out, "route complete")

    def _on_ground(self, now: float, v: Vehicle) -> bool:
        p = self.p
        level = v.q is not None and 1.0 - 2.0 * (v.q[1] ** 2 + v.q[2] ** 2) >= math.cos(math.radians(p.touchdown_tilt_deg))
        low = v.range_m is not None and v.range_m <= p.touchdown_m and now - self.t_phase >= p.touchdown_after_s
        self.t_touch = (self.t_touch if self.t_touch is not None else now) if (low and level) else None
        return self.t_touch is not None and now - self.t_touch >= p.touchdown_s

    def _slew_yaw(self, target: float, dt: float) -> None:
        step = math.radians(self.p.yaw_rate_dps) * dt
        self.sp[3] = F.wrap(self.sp[3] + max(-step, min(step, F.wrap(target - self.sp[3]))))

    def status(self, now: float) -> dict[str, Any]:
        return {"phase": self.phase, "why": self.why, "result": self.result, "started_t": self.t_started,
                "points": len(self.targets), "points_passed": self.follower.points_passed if self.follower else 0,
                "home": self.home, "correction_age_s": None if self.corr_t is None else round(now - self.corr_t, 2),
                "targets": [list(t) for t in self.targets], "setpoint": list(self.sp) if self.sp else None,
                "clearance_m": (None if self.last_clearance is None or math.isinf(self.last_clearance)
                                else round(self.last_clearance, 2)),
                "depth_age_s": None if self.depth_t is None else round(now - self.depth_t, 3),
                "depth_valid_frac": None if self.last_valid_frac is None else round(self.last_valid_frac, 3),
                "depth_reject": None if self.depth_reject is None else self.depth_reject[1],
                "health_bad_s": (None if math.isinf(self.health.bad_for(now)) else round(self.health.bad_for(now), 2)),
                **self._follow_status(), **self.counts}

    def _follow_status(self) -> dict[str, Any]:
        if self.follower is None:
            return {}
        fol, c = self.follower, self.cmd
        st = {"path": fol.path.downsample(0.1), "path_len": round(fol.path.length, 2), "s_ref": round(fol.s_ref, 2),
              "s_hat": round(fol.s_hat, 2), "correction": [round(self.corr.x, 3), round(self.corr.y, 3), round(self.corr.yaw, 4)]}
        if c is not None:
            st.update(xtrack=round(c.xtrack, 3), scale=round(c.scale, 2), lead=round(c.lead, 2), yaw_cmd=round(c.yaw, 3),
                      look=[round(c.look[0], 3), round(c.look[1], 3)])
        return st
