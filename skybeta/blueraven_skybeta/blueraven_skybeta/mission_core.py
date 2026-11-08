"""The real-drone hover for the organizers' skybeta stack. PURE: no ROS, no clock reads.

The drone runs the organizers' stack (OAK -> cuVSLAM -> PX4) and their skybeta_flight_interface,
which owns Offboard. We only read /visual_slam/tracking/odometry and /skybeta/state and publish
/skybeta/setpoint/pose or /skybeta/setpoint/twist (docs/competitor_guide.md in their repo). The
node wraps this class; everything that decides what to publish is here, so it is unit-tested.

Sequence (the safety pilot arms, then switches to Position; the interface enters Offboard ONCE):
  INIT           nothing is published. Wait for fresh state and odometry, PX4 reporting landed, and
                 the drone sitting still on the launch mark for settle_s, then fix map <- odom from
                 the current pose (the odom origin is wherever the stack started, not the mark).
  WAIT_OFFBOARD  publish a pose at the current position every tick (the interface only enters
                 Offboard if our node sent a setpoint in the last second). No movement. If odometry
                 is lost or jumps here, the stack may have restarted with a new origin (it restarts
                 on clock steps): back to INIT, which fixes map <- odom again and re-sends it.
  TAKEOFF        offboard_active: a pose at the take-off point with z ramped up to the flight
                 height. Poses persist, so a stalled node leaves the drone holding.
  HOVER          the same pose for hover_s, then DESCEND over the take-off point. In TAKEOFF and
                 HOVER, drifting more than hold_max_err_m from the take-off point means PX4 holds somewhere else than we command (the guide's
                 EKF-vs-odom frame offset): DESCEND straight down at once, without steering.
  DESCEND        a downward twist (descend_mps), refreshed every tick (twists expire after 1 s),
                 until PX4 reports landed and disarmed.
  DONE / ABORTED nothing is ever published again. Abort: offboard_active drops (the pilot or a
                 failsafe has the vehicle), /skybeta/state or odometry goes stale, or odometry
                 jumps (a reset: every position it gives is now in another frame). On abort the node
                 also calls /skybeta/hold. Silence means a re-arm cannot hand control back to this
                 node: the interface requires a setpoint from it in the last second.

Frames: odom is the stack's ENU startup frame; `map` is the course frame (maps/course.yaml).
map <- odom is yaw + translation (nav.frames.map_from_startup_odom).
Drift: INIT's map <- odom is handed to map_align once (Command.map_from_odom_init, published latched
on nav.frames.MAP_FROM_ODOM_INIT_TOPIC); map_align starts from it and publishes the corrected
map <- odom (MAP_FROM_ODOM_TOPIC), which on_map_from_odom takes in place of INIT's. When those stop,
the last one stays in use: drift since then is uncorrected, but nothing jumps back.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    """The team repository (nav/, maps/): found above this file (a --symlink-install build, the
    skybeta default, resolves back to the source), or named by BLUERAVEN_REPO."""
    import os
    env = os.environ.get("BLUERAVEN_REPO")
    for p in ([Path(env)] if env else []) + list(Path(__file__).resolve().parents):
        if (p / "nav" / "course_map.py").exists():
            return p
    raise ImportError("blueraven_skybeta needs the team repository (nav/, maps/): clone it into "
                      "~/team_ws/src or set BLUERAVEN_REPO")


if str(_repo_root()) not in sys.path:
    sys.path.insert(0, str(_repo_root()))
from nav.frames import MAP_FROM_ODOM_STALE_S, map_from_startup_odom, pose_xyz_yaw  # noqa: E402


@dataclass
class MissionParams:
    hover_s: float = 5.0
    hold_max_err_m: float = 0.40        # TAKEOFF/HOVER: further than this from the take-off point means
                                        # PX4 isn't holding where we think (frame offset?): land straight down
    height_m: float = 0.70              # flight height above the launch point
    climb_mps: float = 0.30
    descend_mps: float = 0.35           # PX4's land detector counts it as descending only at commanded vz >=
                                        # 1.1 x LNDMC_Z_VEL_MAX (0.25 with the defaults: 0.275 m/s). At 0.2 it
                                        # never reports landed and never disarms (Darion's T4 sim, 2026-11-07)
    ground_press_mps: float = 0.60      # on the ground this long without 'landed': press down firmer
    ground_wait_s: float = 3.0
    ground_band_m: float = 0.06         # "on the ground": odom z within this of the take-off height
    descend_xy_hold: bool = False       # DESCEND: also pull xy back over the landing point with a twist P-term.
                                        # Off by default: a twist is rotated by the UNMEASURED frame offset (the
                                        # interface doesn't correct twists), and a P-loop rotated ~90 deg orbits,
                                        # >90 deg diverges. Turn on only with odom_to_ned_yaw_offset_rad measured
    kp_xy: float = 1.0                  # 1/s: velocity per metre of error to the reference
    takeoff_tol_m: float = 0.08
    takeoff_settle_s: float = 5.0       # after the climb ramp: wait this long for the height, then go on anyway
                                        # (PX4 may hold a steady offset: rangefinder vs VIO z). Never hang there
    min_climb_frac: float = 0.5         # ...but only if it climbed at least this share of the way; else land
    state_stale_s: float = 0.6          # /skybeta/state is 5 Hz
    odom_stale_s: float = 1.0           # 30 Hz, but skybeta-09 bags showed arrival gaps up to 0.39 s under load
                                        # (2026-11-07); 0.5 s would abort a hover on one stall
    frame_offset_max_m: float = 0.10    # INIT/WAIT_OFFBOARD: /skybeta/state odom_enu vs local_enu must agree in
                                        # x/y, or the streamed pose is a far target in PX4's frame (up to 12 m/s)
    odom_jump_m: float = 0.15           # a sample this far from the last one AND faster than odom_jump_mps
    odom_jump_mps: float = 3.0          # is a reset (the stack restarts on clock steps), not motion
    odom_jump_deg: float = 30.0         # same for heading, with odom_jump_dps
    odom_jump_dps: float = 360.0
    settle_s: float = 2.0               # the drone must sit still this long before map <- odom is fixed
    settle_tol_m: float = 0.02
    settle_tol_deg: float = 1.0
    corr_stale_s: float = MAP_FROM_ODOM_STALE_S   # map_align's map <- odom older than this is reported stale


@dataclass
class Command:
    """What to publish this tick. At most one of pose / twist; neither once DONE or ABORTED."""
    phase: str
    pose: tuple[float, float, float] | None = None          # odom ENU position
    pose_yaw: float | None = None                            # odom ENU yaw; None = hold heading
    twist: tuple[float, float, float, float] | None = None   # odom ENU (vx, vy, vz, yaw_rate)
    call_hold: bool = False                                  # call /skybeta/hold (once, on abort)
    map_from_odom_init: np.ndarray | None = None             # publish once, latched (the tick INIT ends)
    why: str = ""
    events: list[str] = field(default_factory=list)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class SkybetaMission:
    def __init__(self, params: MissionParams, launch: dict[str, float]) -> None:
        self.p = params
        self.launch = launch
        self.z_map = launch["z"] + params.height_m
        self.t_hover: float | None = None
        self.t_ramp_done: float | None = None
        self.z_takeoff_start: float | None = None
        self.why_height: str | None = None
        self.t_on_ground: float | None = None
        self.descend_reason: str | None = None
        self.land_xy_odom: tuple[float, float] | None = None   # hover: descend over this odom point
        self.hold_xy_on_descent = True      # false after a hold-error: straight down, no xy correction
        self.phase = "INIT"
        self.why = "waiting for state and odometry"
        self.state: dict[str, Any] | None = None
        self.t_state = -math.inf
        self.odom: tuple[float, float, float, float] | None = None   # (x, y, z, yaw) in odom
        self.t_odom = -math.inf
        self.s_odom: float | None = None    # its capture stamp
        self.settle: list[tuple[float, tuple[float, float, float, float]]] = []
        self.T_map_odom: np.ndarray | None = None
        self.T_odom_map: np.ndarray | None = None
        self.T_init: np.ndarray | None = None
        self.t_fixed = -math.inf            # when INIT last fixed map <- odom
        self.n_init = 0
        self.odom_jump: str | None = None   # set by on_odom, consumed by step
        self.t_corr = -math.inf             # last map_align correction taken
        self.n_corr = 0
        self.corr_step_max_m = 0.0          # largest move of the drone's map position by one correction
        self.takeoff: tuple[float, float, float, float] | None = None   # (x, y, z_target, z_ref) odom
        self.t_last: float | None = None
        self.hold_called = False

    # ------------------------------------------------------------------ inputs
    def on_state(self, t: float, state: dict[str, Any]) -> None:
        self.state, self.t_state = dict(state), float(t)

    def on_odom(self, t: float, x: float, y: float, z: float, yaw: float, stamp: float | None = None) -> None:
        """t = receive time (staleness); stamp = the message's capture stamp, which times the jump
        check (samples delivered in a burst arrive together but were captured apart)."""
        if not all(math.isfinite(v) for v in (x, y, z, yaw)):
            return
        s_prev, self.s_odom = self.s_odom, stamp
        if self.odom is not None:
            dt = (stamp - s_prev) if (stamp is not None and s_prev is not None and stamp > s_prev) else t - self.t_odom
            dt = max(dt, 1.0 / 30.0)
            d = math.dist(self.odom[:3], (x, y, z))
            dyaw = abs(_wrap(yaw - self.odom[3]))
            if (d > self.p.odom_jump_m and d / dt > self.p.odom_jump_mps) or (
                    dyaw > math.radians(self.p.odom_jump_deg) and dyaw / dt > math.radians(self.p.odom_jump_dps)):
                self.odom_jump = (f"odometry jumped {d:.2f} m / {math.degrees(dyaw):.0f} deg in {dt:.2f} s "
                                  f"(a stack restart resets it)")
        self.odom, self.t_odom = (float(x), float(y), float(z), float(yaw)), float(t)
        if self.phase == "INIT":
            self.settle.append((float(t), self.odom))
            while self.settle and t - self.settle[0][0] > 2.0 * self.p.settle_s:
                self.settle.pop(0)

    def on_map_from_odom(self, t: float, T: Any, stamp: float | None = None) -> str | None:
        """map_align's corrected map <- odom (MAP_FROM_ODOM_TOPIC), used from the next tick on.
        Takes x, y and yaw; z stays INIT's (map_align corrects only those three, and this way a bad z
        can't move the flight height). Ignored until INIT has fixed its own, which map_align starts
        from, and if stamped before the latest INIT (computed against an older odom origin).
        map_align gates its own updates; the largest step is reported. Returns why a message
        was ignored, or None if it was taken."""
        if self.T_map_odom is None:
            return "before INIT fixed map <- odom"
        if stamp is not None and stamp < self.t_fixed:
            return "stamped before the latest INIT"
        T = np.asarray(T, dtype=float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            return "not a finite 4x4 transform"
        T_new = pose_xyz_yaw(T[0, 3], T[1, 3], self.T_map_odom[2, 3], math.atan2(T[1, 0], T[0, 0]))
        if self.odom is not None:
            p = np.array([*self.odom[:3], 1.0])
            step = float(np.linalg.norm((T_new @ p - self.T_map_odom @ p)[:2]))
            self.corr_step_max_m = max(self.corr_step_max_m, step)
        self._set_T(T_new)
        self.t_corr, self.n_corr = float(t), self.n_corr + 1
        return None

    def status(self, t: float) -> dict[str, Any]:
        """For /blueraven/mission/status: phase, progress, and which map <- odom is in use."""
        src = None
        if self.T_map_odom is not None:
            src = "init" if self.n_corr == 0 else ("live" if t - self.t_corr <= self.p.corr_stale_s else "stale")
        moved = None
        if self.T_init is not None and self.odom is not None:
            p = np.array([*self.odom[:3], 1.0])
            moved = round(float(np.linalg.norm((self.T_map_odom @ p - self.T_init @ p)[:2])), 3)
        return {"phase": self.phase, "why": self.why, "inits": self.n_init, "map_from_odom": src,
                "corr_age_s": None if self.n_corr == 0 else round(t - self.t_corr, 2),
                "corr_n": self.n_corr, "corr_step_max_m": round(self.corr_step_max_m, 3),
                "corr_from_init_m": moved}

    # --------------------------------------------------------------- helpers
    def _set_T(self, T: np.ndarray) -> None:
        self.T_map_odom = T
        self.T_odom_map = np.linalg.inv(T)

    def _fresh(self, t: float) -> tuple[bool, str]:
        if self.state is None or t - self.t_state > self.p.state_stale_s:
            return False, "/skybeta/state stale"
        if self.odom is None or t - self.t_odom > self.p.odom_stale_s:
            return False, "odometry stale"
        return True, "ok"

    def _frame_offset(self, st: dict[str, Any]) -> float | None:
        """x/y distance between the stack's odometry and PX4's local position (both ENU, in
        /skybeta/state). None when either is missing (older stacks, sims without them): unchecked.
        Height is not compared: PX4 takes z from the rangefinder, so it differs by design."""
        a, b = st.get("odom_enu"), st.get("local_enu")
        try:
            return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
        except (TypeError, ValueError, IndexError):
            return None

    def _settled(self, t: float) -> bool:
        win = [o for (ts, o) in self.settle if t - ts <= self.p.settle_s]
        if not self.settle or t - self.settle[0][0] < self.p.settle_s or len(win) < 3:
            return False
        P = np.array([o[:3] for o in win])
        yaws = np.array([o[3] for o in win])
        return (float(np.max(np.linalg.norm(P - P.mean(axis=0), axis=1))) <= self.p.settle_tol_m
                and float(np.max(np.abs([_wrap(a - yaws[0]) for a in yaws]))) <= math.radians(self.p.settle_tol_deg))

    def _end(self, phase: str, why: str, out: Command, hold: bool) -> Command:
        out.events.append(f"{self.phase} -> {phase}: {why}")
        self.phase, self.why = phase, why
        out.phase, out.why = phase, why
        out.pose = out.twist = None
        if hold and not self.hold_called:
            out.call_hold = self.hold_called = True
        return out

    def _set(self, phase: str, why: str, out: Command) -> None:
        out.events.append(f"{self.phase} -> {phase}: {why}")
        self.phase, self.why = phase, why

    # ------------------------------------------------------------------ step
    def step(self, t: float) -> Command:
        p = self.p
        dt = 0.0 if self.t_last is None else min(max(t - self.t_last, 0.0), 0.2)
        self.t_last = t
        out = Command(self.phase, why=self.why)
        if self.phase in ("DONE", "ABORTED"):
            return out
        fresh, why = self._fresh(t)
        st = self.state or {}
        jump, self.odom_jump = self.odom_jump, None

        if self.phase == "WAIT_OFFBOARD":
            # Before take-off a lost or reset odometry means the stack may have restarted with a new
            # origin, so INIT's map <- odom is no longer true: fix it again (still on the mark).
            odom_lost = self.odom is None or t - self.t_odom > p.odom_stale_s
            if jump or odom_lost:
                self._set("INIT", f"{jump or 'odometry lost'} before take-off: re-fixing map <- odom", out)
                self.T_map_odom = self.T_odom_map = self.T_init = None
                self.n_corr, self.t_corr, self.corr_step_max_m = 0, -math.inf, 0.0
                self.settle = [(self.t_odom, self.odom)] if (self.odom and jump) else []

        if self.phase == "INIT":
            if jump:
                self.settle = [(self.t_odom, self.odom)]
            if not fresh:
                self.why = f"waiting: {why}"
            elif st.get("landed") is not True:
                self.why = "waiting: /skybeta/state doesn't report landed (map <- odom is fixed on the mark)"
            elif (off := self._frame_offset(st)) is not None and off > p.frame_offset_max_m:
                self.why = (f"waiting: odometry and PX4's local position disagree by {off:.2f} m in x/y "
                            f"(> {p.frame_offset_max_m:.2f}): restart the stack with the drone level and still")
            elif not self._settled(t):
                self.why = f"waiting for the drone to sit still for {p.settle_s:.0f} s on the launch mark"
            else:
                self._set_T(map_from_startup_odom(self.launch, self.odom))
                self.T_init = self.T_map_odom.copy()
                self.t_fixed, self.n_init = t, self.n_init + 1
                out.map_from_odom_init = self.T_init.copy()
                self._set("WAIT_OFFBOARD", "map <- odom fixed; streaming a hold at the current position", out)
            out.phase, out.why = self.phase, self.why
            if self.phase == "INIT":
                return out

        if self.phase == "WAIT_OFFBOARD":
            if not fresh:                      # never hand a handover to a node that can't see
                self.why = f"not streaming: {why}"
                out.why = self.why
                return out
            off = self._frame_offset(st)
            if off is not None and off > p.frame_offset_max_m and st.get("offboard_active") is not True:
                # the streamed hold would be a far target in PX4's frame: no setpoints, so no Offboard
                self.why = f"not streaming: odometry and PX4's local position disagree by {off:.2f} m in x/y"
                out.why = self.why
                return out
            if st.get("offboard_active") is True:
                x, y, z, _ = self.odom
                z_target = float(self.T_odom_map[2, 3] + self.z_map) if abs(self.T_odom_map[2, 2] - 1) < 1e-9 else z
                self.takeoff = (x, y, z_target, z)
                self.z_takeoff_start, self.t_ramp_done = z, None
                self._set("TAKEOFF", f"Offboard: climbing to {p.height_m:.2f} m", out)
            else:
                x, y, z, _ = self.odom
                out.pose = (x, y, z)
                self.why = "streaming a hold; waiting for the pilot to arm and switch to Position"
                out.phase, out.why = self.phase, self.why
                return out

        if self.phase == "DESCEND":
            return self._descend(t, out, jump)

        # ---- active phases: abort first ------------------------------------------------
        if not fresh:
            return self._end("ABORTED", why, out, hold=True)
        if jump:
            return self._end("ABORTED", jump, out, hold=True)
        if st.get("offboard_active") is not True:
            return self._end("ABORTED", "Offboard ended: the pilot or a failsafe has the vehicle", out, hold=True)

        x, y, z, _ = self.odom
        if self.phase in ("TAKEOFF", "HOVER"):
            x0, y0 = self.takeoff[:2]
            err = math.hypot(x - x0, y - y0)
            if err > p.hold_max_err_m:          # PX4 isn't holding where we command: don't steer, land
                self.hold_xy_on_descent = False
                self._set("DESCEND", f"{err:.2f} m from the take-off point while holding it "
                                     f"(PX4 frame offset?): landing straight down", out)
        if self.phase == "TAKEOFF":
            x0, y0, z_target, z_ref = self.takeoff
            z_ref = min(z_target, z_ref + p.climb_mps * dt)
            self.takeoff = (x0, y0, z_target, z_ref)
            out.pose = (x0, y0, min(z_target, max(z_ref, z)))   # never chase a high vision z above the target
            at_height = z_ref >= z_target and abs(z - z_target) < p.takeoff_tol_m
            if z_ref >= z_target and self.t_ramp_done is None:
                self.t_ramp_done = t
            settled_off = (not at_height and self.t_ramp_done is not None
                           and t - self.t_ramp_done >= p.takeoff_settle_s)
            if settled_off:
                need = z_target - self.z_takeoff_start
                climbed = z - self.z_takeoff_start
                if need > 0 and climbed < p.min_climb_frac * need:
                    self.land_xy_odom = (x0, y0)
                    self._set("DESCEND", f"climbed only {climbed:.2f} of {need:.2f} m in "
                                         f"{p.takeoff_settle_s:.0f} s after the ramp: landing", out)
                    out.pose = None
                    at_height = False
                else:
                    self.why_height = f"holding {z - z_target:+.2f} m off the target height (PX4 vs odom z?)"
                    at_height = True
            if at_height and self.phase == "TAKEOFF":
                note = f"; {self.why_height}" if self.why_height else ""
                self.t_hover = t
                self._set("HOVER", f"at {p.height_m:.2f} m; hovering {p.hover_s:.0f} s{note}", out)
                out.phase, out.why = self.phase, self.why
                return out
            if self.phase == "TAKEOFF":                        # still climbing (a DESCEND falls through)
                out.phase, out.why = self.phase, self.why
                return out

        if self.phase == "HOVER":
            x0, y0, z_target, _ = self.takeoff
            if t - self.t_hover >= p.hover_s:
                self.land_xy_odom = (x0, y0)
                self._set("DESCEND", f"hovered {p.hover_s:.0f} s; descending", out)
            else:
                out.pose = (x0, y0, z_target)
                out.phase, out.why = self.phase, self.why
                return out

        if self.phase == "DESCEND":
            return self._descend(t, out, None)
        return out

    def _descend(self, t: float, out: Command, jump: str | None) -> Command:
        """DESCEND: straight down (an xy pull only with descend_xy_hold) until PX4 reports landed and disarmed.
        It never aborts and never calls /skybeta/hold: the interface would then hold its last safe position,
        i.e. climb back to the hover height. It needs no odometry, so odometry going stale or jumping near
        the floor (the cameras lose texture there) doesn't stop it. On the ground without 'landed' for
        ground_wait_s it presses down harder (PX4's land detector wants a firm downward command)."""
        p = self.p
        st = self.state or {}
        if self.descend_reason is None:
            self.descend_reason = self.why
        state_fresh = self.state is not None and t - self.t_state <= p.state_stale_s
        if state_fresh:
            if st.get("landed") is True and st.get("armed") is False:
                return self._end("DONE", "landed and disarmed", out, hold=False)
            if st.get("offboard_active") is not True:
                if st.get("landed") is True:
                    return self._end("DONE", "landed; Offboard ended", out, hold=False)
                return self._end("ABORTED", "Offboard ended during the descent: the pilot or a failsafe has it",
                                 out, hold=False)
        odom_ok = self.odom is not None and t - self.t_odom <= p.odom_stale_s and not jump
        vz, now = -p.descend_mps, "descending"
        if odom_ok and self.z_takeoff_start is not None and self.odom[2] - self.z_takeoff_start <= p.ground_band_m:
            self.t_on_ground = t if self.t_on_ground is None else self.t_on_ground
            if t - self.t_on_ground >= p.ground_wait_s:
                vz = -p.ground_press_mps
                now = (f"on the ground {t - self.t_on_ground:.0f} s and PX4 hasn't reported landed: pressing down "
                       f"(pilot: disarm if it stays armed)")
        elif odom_ok:
            self.t_on_ground = None
        if not odom_ok:
            now = "descending straight down without odometry (stale or jumped)"
        if not state_fresh:
            now += "; /skybeta/state stale, can't see landed/armed"
        ox = oy = 0.0
        if odom_ok and self.hold_xy_on_descent and p.descend_xy_hold and self.land_xy_odom is not None:
            x, y, _, _ = self.odom
            ox, oy = p.kp_xy * (self.land_xy_odom[0] - x), p.kp_xy * (self.land_xy_odom[1] - y)
        out.twist = (ox, oy, vz, 0.0)
        self.why = f"{self.descend_reason} | {now}"
        out.phase, out.why = self.phase, self.why
        return out
