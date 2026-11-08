"""The real-drone mission for the organizers' skybeta stack. PURE: no ROS, no clock reads.

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
  HOVER          (mission "hover", the first flights) the same pose for hover_s, then DESCEND over
                 the take-off point. In TAKEOFF and HOVER, drifting more than hold_max_err_m from
                 the take-off point means PX4 holds somewhere else than we command (the guide's
                 EKF-vs-odom frame offset): DESCEND straight down at once, without steering.
  FOLLOW         (mission "course") a TWIST every tick: the reference's feed-forward velocity plus a P term towards
                 the reference point, and a yaw rate towards the path heading ahead. Poses are
                 position-only through the interface, so a pose reference would crawl.
  DESCEND        a downward twist (descend_mps), xy held over the end of the route, refreshed every
                 tick (twists expire after 1 s), until PX4 reports landed and disarmed.
  DONE / ABORTED nothing is ever published again. Abort: offboard_active drops (the pilot or a
                 failsafe has the vehicle), /skybeta/state or odometry goes stale, or odometry
                 jumps (a reset: every position it gives is now in another frame). On abort the node
                 also calls /skybeta/hold. Silence means a re-arm cannot hand control back to this
                 node: the interface requires a setpoint from it in the last second.

Frames: odom is the stack's ENU startup frame. The route and the course map are in `map` (the
course frame, maps/course.yaml). map <- odom is yaw + translation (nav.frames.map_from_startup_odom).
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
from nav.path_geometry import build_path  # noqa: E402
from nav.wall_centring import CentringParams, WallCentring  # noqa: E402


@dataclass
class MissionParams:
    mission: str = "hover"              # "hover": take off, hold hover_s over the take-off point, land
                                        # (first flights: no route, no map needed). "course": fly the route.
    hover_s: float = 5.0
    hold_max_err_m: float = 0.40        # TAKEOFF/HOVER: further than this from the take-off point means
                                        # PX4 isn't holding where we think (frame offset?): land straight down
    height_m: float = 0.70              # flight height above the launch point
    cruise_mps: float = 0.30
    climb_mps: float = 0.30
    descend_mps: float = 0.20
    descend_xy_hold: bool = False       # DESCEND: also pull xy back over the landing point with a twist P-term.
                                        # Off by default: a twist is rotated by the UNMEASURED frame offset (the
                                        # interface doesn't correct twists), and a P-loop rotated ~90 deg orbits,
                                        # >90 deg diverges. Turn on only with odom_to_ned_yaw_offset_rad measured
    max_speed_mps: float = 1.5          # the interface caps twists here too
    kp_xy: float = 1.0                  # 1/s: velocity per metre of error to the reference
    kp_z: float = 1.0
    kp_yaw: float = 1.5
    max_yaw_rate_dps: float = 45.0      # VIO lost track on faster turns in the sims
    face_tol_deg: float = 25.0          # slow the reference down when the nose is further off than this
    lead_max_m: float = 0.30            # the reference never leads the aircraft by more
    look_m: float = 0.30                # the nose faces the path this far ahead of the reference
    yaw_rate_k: float = 1.0             # curvature speed limit: v <= max_yaw_rate / |k|
    arrive_m: float = 0.10
    takeoff_tol_m: float = 0.08
    takeoff_settle_s: float = 5.0       # after the climb ramp: wait this long for the height, then go on anyway
                                        # (PX4 may hold a steady offset: rangefinder vs VIO z). Never hang there
    min_climb_frac: float = 0.5         # ...but only if it climbed at least this share of the way; else land
    state_stale_s: float = 0.6          # /skybeta/state is 5 Hz
    odom_stale_s: float = 0.5           # 30 Hz, but skybeta-09's bench saw gaps up to ~0.3 s (2026-11-07)
    odom_jump_m: float = 0.15           # a sample this far from the last one AND faster than odom_jump_mps
    odom_jump_mps: float = 3.0          # is a reset (the stack restarts on clock steps), not motion
    odom_jump_deg: float = 30.0         # same for heading, with odom_jump_dps
    odom_jump_dps: float = 360.0
    settle_s: float = 2.0               # the drone must sit still this long before map <- odom is fixed
    settle_tol_m: float = 0.02
    settle_tol_deg: float = 1.0
    corr_stale_s: float = MAP_FROM_ODOM_STALE_S   # map_align's map <- odom older than this is reported stale
    drone_radius: float = 0.28
    margin: float = 0.10
    max_radius_m: float = 1.0           # largest corner arc; 0 makes every corner a stop-and-turn
    wall_centring: bool = False         # course: shift the reference toward the REAL corridor centre seen in
                                        # depth (nav.wall_centring); needs on_depth_points. Off until benched
    centring_cap_m: float = 0.15
    pose_buffer_s: float = 2.0          # odometry kept to place a depth frame at its capture time
    pose_gap_s: float = 0.15            # refuse to interpolate across a bigger odometry gap


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
    def __init__(self, params: MissionParams, launch: dict[str, float], corners_map: Any = None,
                 course_map: Any = None) -> None:
        self.p = params
        self.launch = launch
        self.z_map = launch["z"] + params.height_m
        if params.mission not in ("hover", "course"):
            raise ValueError(f"mission must be 'hover' or 'course', not {params.mission!r}")
        self.path = None
        if params.mission == "course":
            if corners_map is None or len(corners_map) < 2:
                raise ValueError("mission 'course' needs a route of at least 2 corners")
            C = np.asarray(corners_map, dtype=float)
            self.path = build_path(C[:, :2], course_map, params.drone_radius, params.margin, z=self.z_map,
                                   max_radius=params.max_radius_m)
        self.centring = (WallCentring(course_map, CentringParams(cap_m=params.centring_cap_m))
                         if (params.wall_centring and params.mission == "course" and course_map is not None) else None)
        self.poses: list[tuple[float, float, float, float, float]] = []   # (stamp, x, y, z, yaw) in odom
        self.t_hover: float | None = None
        self.t_ramp_done: float | None = None
        self.z_takeoff_start: float | None = None
        self.why_height: str | None = None
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
        self.yaw_off = 0.0                  # map yaw = odom yaw + yaw_off
        self.t_fixed = -math.inf            # when INIT last fixed map <- odom
        self.n_init = 0
        self.odom_jump: str | None = None   # set by on_odom, consumed by step
        self.t_corr = -math.inf             # last map_align correction taken
        self.n_corr = 0
        self.corr_step_max_m = 0.0          # largest move of the drone's map position by one correction
        self.takeoff: tuple[float, float, float, float] | None = None   # (x, y, z_target, z_ref) odom
        self.s_ref = 0.0
        self.s_proj = 0.0
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
        st = float(stamp) if stamp is not None else float(t)
        if not self.poses or st > self.poses[-1][0]:
            self.poses.append((st, *self.odom))
            while self.poses and st - self.poses[0][0] > self.p.pose_buffer_s:
                self.poses.pop(0)
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

    def pose_at(self, stamp: float) -> tuple[tuple[float, float, float, float], float] | None:
        """Odometry (x, y, z, yaw) interpolated at a capture stamp, and the yaw rate there; None if the stamp
        is outside the buffer or falls in a gap over pose_gap_s (never extrapolates)."""
        P = self.poses
        if len(P) < 2 or not P[0][0] <= stamp <= P[-1][0]:
            return None
        i = next(k for k in range(1, len(P)) if P[k][0] >= stamp)
        (t0, *a), (t1, *b) = P[i - 1], P[i]
        if t1 - t0 > self.p.pose_gap_s:
            return None
        f = 0.0 if t1 == t0 else (stamp - t0) / (t1 - t0)
        dyaw = _wrap(b[3] - a[3])
        pose = (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), _wrap(a[3] + f * dyaw))
        return pose, (dyaw / (t1 - t0) if t1 > t0 else 0.0)

    def on_depth_points(self, t: float, stamp: float, pts_base: Any) -> str | None:
        """Wall points from one depth frame, in base_link (x forward, y left, z up), captured at `stamp`.
        Feeds wall centring; returns why the frame wasn't used, or None."""
        if self.centring is None:
            return "wall centring is off"
        if self.phase != "FOLLOW" or self.T_map_odom is None:
            return f"not following ({self.phase})"
        at = self.pose_at(stamp)
        if at is None:
            return "no odometry at the frame's capture time"
        (x, y, z, yaw), rate = at
        B = np.asarray(pts_base, dtype=float).reshape(-1, 3)
        c, s = math.cos(yaw), math.sin(yaw)
        odom = np.c_[c * B[:, 0] - s * B[:, 1] + x, s * B[:, 0] + c * B[:, 1] + y, B[:, 2] + z]
        T = self.T_map_odom
        pts_map = odom @ T[:3, :3].T + T[:3, 3]
        drone = (T @ [x, y, z, 1.0])[:3]
        s_at = self.path.project(drone[:2], self.s_proj - 0.5, self.s_proj + 0.5)
        m = self.centring.measure(pts_map, drone, self.path, s_at, yaw_rate=rate)
        self.centring.update(t, m)
        return None if m.ok else m.why

    def status(self, t: float) -> dict[str, Any]:
        """For /blueraven/mission/status: phase, progress, and which map <- odom is in use."""
        src = None
        if self.T_map_odom is not None:
            src = "init" if self.n_corr == 0 else ("live" if t - self.t_corr <= self.p.corr_stale_s else "stale")
        moved = None
        if self.T_init is not None and self.odom is not None:
            p = np.array([*self.odom[:3], 1.0])
            moved = round(float(np.linalg.norm((self.T_map_odom @ p - self.T_init @ p)[:2])), 3)
        return {"mission": self.p.mission, "phase": self.phase, "why": self.why, "s_ref": round(self.s_ref, 2),
                "length": None if self.path is None else round(float(self.path.length), 2),
                "inits": self.n_init, "map_from_odom": src,
                "corr_age_s": None if self.n_corr == 0 else round(t - self.t_corr, 2),
                "corr_n": self.n_corr, "corr_step_max_m": round(self.corr_step_max_m, 3),
                "corr_from_init_m": moved,
                "centring_m": None if self.centring is None else round(self.centring.offset, 3),
                "centring_why": None if (self.centring is None or self.centring.last is None) else self.centring.last.why}

    # --------------------------------------------------------------- helpers
    def _set_T(self, T: np.ndarray) -> None:
        self.T_map_odom = T
        self.T_odom_map = np.linalg.inv(T)
        self.yaw_off = math.atan2(T[1, 0], T[0, 0])

    def _fresh(self, t: float) -> tuple[bool, str]:
        if self.state is None or t - self.t_state > self.p.state_stale_s:
            return False, "/skybeta/state stale"
        if self.odom is None or t - self.t_odom > self.p.odom_stale_s:
            return False, "odometry stale"
        return True, "ok"

    def _settled(self, t: float) -> bool:
        win = [o for (ts, o) in self.settle if t - ts <= self.p.settle_s]
        if not self.settle or t - self.settle[0][0] < self.p.settle_s or len(win) < 3:
            return False
        P = np.array([o[:3] for o in win])
        yaws = np.array([o[3] for o in win])
        return (float(np.max(np.linalg.norm(P - P.mean(axis=0), axis=1))) <= self.p.settle_tol_m
                and float(np.max(np.abs([_wrap(a - yaws[0]) for a in yaws]))) <= math.radians(self.p.settle_tol_deg))

    def _to_map(self, x: float, y: float, z: float) -> np.ndarray:
        return (self.T_map_odom @ [x, y, z, 1.0])[:3]

    def _vec_to_odom(self, vx: float, vy: float) -> tuple[float, float]:
        R = self.T_odom_map[:2, :2]
        v = R @ [vx, vy]
        return float(v[0]), float(v[1])

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

        # ---- active phases: abort first ------------------------------------------------
        if not fresh:
            return self._end("ABORTED", why, out, hold=True)
        if jump:
            return self._end("ABORTED", jump, out, hold=True)
        if self.phase == "DESCEND" and st.get("landed") is True and st.get("armed") is False:
            return self._end("DONE", "landed and disarmed", out, hold=False)
        if st.get("offboard_active") is not True:
            if self.phase == "DESCEND" and st.get("landed") is True:
                return self._end("DONE", "landed; Offboard ended", out, hold=False)
            return self._end("ABORTED", "Offboard ended: the pilot or a failsafe has the vehicle", out, hold=True)

        x, y, z, yaw = self.odom
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
            out.pose = (x0, y0, max(z_ref, z))
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
                if p.mission == "hover":
                    self.t_hover = t
                    self._set("HOVER", f"at {p.height_m:.2f} m; hovering {p.hover_s:.0f} s{note}", out)
                    out.phase, out.why = self.phase, self.why
                    return out
                self.s_ref = self.s_proj = self.path.project(self._to_map(x, y, z)[:2], 0.0, 0.5)
                self._set("FOLLOW", "at flight height; following the route", out)
                out.pose = None
            elif self.phase == "TAKEOFF":                      # still climbing (a DESCEND falls through)
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

        pm = self._to_map(x, y, z)
        yaw_map = _wrap(yaw + self.yaw_off)
        if self.phase == "FOLLOW":
            path = self.path
            self.s_proj = path.project(pm[:2], self.s_proj - 0.25, self.s_ref + 0.25)
            k = float(path.curvature_at(self.s_ref))
            v = p.cruise_mps
            if k != 0.0:
                v = min(v, math.radians(p.max_yaw_rate_dps) * p.yaw_rate_k / abs(k)) if math.isfinite(k) else 0.0
            psi_look = float(path.heading_at(self.s_ref + p.look_m))
            yaw_err = _wrap(psi_look - yaw_map)
            v *= max(0.0, 1.0 - abs(yaw_err) / math.radians(p.face_tol_deg)) if abs(yaw_err) > math.radians(5.0) else 1.0
            self.s_ref = min(path.length, self.s_ref + v * dt, self.s_proj + p.lead_max_m)
            if not math.isfinite(k) and abs(yaw_err) <= math.radians(5.0):
                self.s_ref = min(path.length, self.s_ref + 0.02)        # past a sharp corner once facing out
            ref = path.sample(self.s_ref)
            psi = float(ref["psi"])
            vff = (v * math.cos(psi), v * math.sin(psi))
            shift = 0.0
            if self.centring is not None:                      # toward the REAL corridor centre seen in depth
                shift = self.centring.update(t, None)           # advances hold/decay; frames update it too
            rx, ry = float(ref["x"]) - shift * math.sin(psi), float(ref["y"]) + shift * math.cos(psi)
            ex, ey, ez = rx - pm[0], ry - pm[1], self.z_map - pm[2]
            vx, vy = vff[0] + p.kp_xy * ex, vff[1] + p.kp_xy * ey
            vz = p.kp_z * ez
            n = math.sqrt(vx * vx + vy * vy + vz * vz)
            if n > p.max_speed_mps:
                vx, vy, vz = (c * p.max_speed_mps / n for c in (vx, vy, vz))
            wz = max(-math.radians(p.max_yaw_rate_dps), min(math.radians(p.max_yaw_rate_dps), p.kp_yaw * yaw_err))
            ox, oy = self._vec_to_odom(vx, vy)
            out.twist = (ox, oy, vz, wz)
            end = path.point_at(path.length)
            if self.s_ref >= path.length - 1e-6 and math.hypot(end[0] - pm[0], end[1] - pm[1]) < p.arrive_m:
                self._set("DESCEND", "route complete; descending", out)
            out.phase, out.why = self.phase, self.why
            return out

        if self.phase == "DESCEND":
            ox = oy = 0.0
            if not (self.hold_xy_on_descent and p.descend_xy_hold):
                pass                                                   # straight down, no steering
            elif self.land_xy_odom is not None:                        # hover: over the take-off point
                ox, oy = p.kp_xy * (self.land_xy_odom[0] - x), p.kp_xy * (self.land_xy_odom[1] - y)
            elif self.path is not None:                                # course: over the route's end
                end = self.path.point_at(self.path.length)
                ox, oy = self._vec_to_odom(p.kp_xy * (end[0] - pm[0]), p.kp_xy * (end[1] - pm[1]))
            out.twist = (ox, oy, -p.descend_mps, 0.0)
            out.phase, out.why = self.phase, self.why
            return out
        return out
