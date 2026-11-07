"""The route through the course's checkpoints, planned at startup from the map. Numpy only; deterministic.

This replaces a fixed waypoint file. The route comes from the published course (the planning SDF built from it,
maps/course_sdf.npz) and maps/course.yaml (the launch pose, the checkpoint lines in order, the limits). Change
either and the route changes with it; there are no waypoints anywhere in the code.

1. Slice the planning field at the flight height into a 2-D clearance grid, in whole millimetres, on the map's
   own 5 cm cells.
2. Each checkpoint line becomes two goals at its most open point: an approach point before it, and an exit
   point past it, as far as the whole-drone rule asks (centre `drone_radius` past the line) and the clearance
   allows. CP4 takes the configured option, and the other options' lines are closed so the route can't use
   them. The route ends at the pad's centre.
3. A* between successive goals, 8-connected, through cells at least `clearance` from every surface, with a cost
   that prefers the middle of corridors.
4. Each leg is cut down to its corners: straight runs collapse wherever the straight line stays clear, and open
   areas are shortcut where the line keeps `clearance + roomy`. Goals are always kept.

Same map, same yaml, same height and option -> the same route, bit for bit, on any machine: clearances are
integers (mm), A* costs are integers, and ties break on the cell index. So the sims (x86) and the Jetson (ARM)
fly the identical route; `RoutePlan.digest` is a short hash to compare.

The output is refnav's route format, [(forward_m, left_m)] from the launch pose in its heading.

    python3 -m nav.route_planner --launch-key launch_sim --height 0.7 --cp4 window [--plot plan.png]
"""
from __future__ import annotations

import hashlib
import heapq
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nav.course_map import CourseMap

    XY = tuple[float, float]

ROOT = Path(__file__).resolve().parents[1]
MM = 1000                                       # millimetres per metre: the planner's integer unit
# 8-connected moves (di, dj, length in thousandths of a cell)
_MOVES = ((1, 0, 1000), (-1, 0, 1000), (0, 1, 1000), (0, -1, 1000),
          (1, 1, 1414), (1, -1, 1414), (-1, 1, 1414), (-1, -1, 1414))


@dataclass(frozen=True)
class PlannerParams:
    drone_radius_m: float = 0.28        # rotor arm plus prop (course.yaml limits.drone_radius)
    clearance_m: float = 0.39           # hard minimum for the drone's centre: radius + margin + 1 cm map error
    roomy_m: float = 0.0                # shortcut where the line keeps clearance + roomy (refnav's band re-centres
                                        # the path afterwards; 0.10 gave 32 points and +0.5 s, see tools/compare_routes)
    centre_pref_m: float = 0.55         # clearance beyond which a cell counts as the middle of a corridor
    centre_weight: float = 8.0          # how strongly A* prefers the middle of corridors
    approach_m: float = 0.30            # approach point this far before each line
    cross_extra_m: float = 0.05         # exit this far beyond "whole drone across"
    snap_m: float = 0.25                # a goal may move this far to the nearest free cell
    simplify_tol_m: float = 0.03        # straight-run tolerance when collapsing a leg
    min_spacing_m: float = 0.05         # consecutive route points at least this far apart
    keep_centre_m: float | None = None  # None: collapse/shortcut wherever the line keeps `clearance` (tight around
                                        # wall ends). A number: a straight line may replace part of the centred
                                        # A* path only if it loses at most this much of that part's clearance
                                        # (for a physical course that isn't exactly the published one)


@dataclass
class RoutePlan:
    route: list[XY]                     # refnav format: (forward_m, left_m) from the launch pose, launch excluded
    map_xy: list[XY]                    # the same points in the map frame (x east, y north)
    labels: list[str]                   # one per point: '' or the goal it is
    launch_xy: XY                       # where the route starts, in the map frame
    z: float                            # map height the route was planned at
    cp4: str                            # the CP4 option flown
    crossings: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def digest(self) -> str:
        """Short hash of the route to the millimetre: equal digests mean the identical route."""
        text = ";".join(f"{a:.3f},{b:.3f}" for a, b in self.route) + f"|{self.z:.3f}|{self.cp4}"
        return hashlib.sha256(text.encode()).hexdigest()[:12]

    @property
    def length_m(self) -> float:
        pts = [self.launch_xy] + self.map_xy
        return float(sum(math.dist(a, b) for a, b in zip(pts, pts[1:])))

    def summary(self) -> str:
        cross = ", ".join(
            f"{c['checkpoint']} {c.get('past_m', c.get('end_from_centre_m'))}" for c in self.crossings)
        return (f"{len(self.route)} points, {self.length_m:.1f} m at z {self.z:.2f}, CP4 {self.cp4}, "
                f"digest {self.digest}; crossings: {cross}")


# ---------------------------------------------------------------------- the grid
class ClearanceGrid:
    """The planning field at one height: whole millimetres on the map's own x-y cells ([i, j] = x, y)."""

    def __init__(self, cm: CourseMap, z: float, field_name: str = "planning") -> None:
        F = cm.fields[field_name]
        ox, oy, oz = (float(v) for v in cm.origin)
        self.c = float(cm.cell)
        nz = F.shape[2]
        k = (z - oz) / self.c
        if not 0.0 <= k <= nz - 1:
            raise ValueError(f"height {z:.2f} m is outside the map's z range")
        k0 = min(int(math.floor(k)), nz - 2)
        f = k - k0
        D = F[:, :, k0].astype(np.float64) * (1.0 - f) + F[:, :, k0 + 1].astype(np.float64) * f
        self.mm = np.floor(D * MM + 0.5).astype(np.int64)
        self.ox, self.oy = ox, oy
        self.nx, self.ny = self.mm.shape

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor((x - self.ox) / self.c + 0.5)), int(math.floor((y - self.oy) / self.c + 0.5))

    def xy(self, ij: tuple[int, int]) -> XY:
        return self.ox + ij[0] * self.c, self.oy + ij[1] * self.c

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny

    def at(self, x: float, y: float) -> int:
        """Clearance of the nearest cell, mm; outside the map is blocked (-1)."""
        i, j = self.cell(x, y)
        return int(self.mm[i, j]) if self.inside(i, j) else -1

    def segment_min(self, a: Sequence[float], b: Sequence[float]) -> int:
        """Lowest clearance (mm) along the segment, sampled every half cell."""
        n = max(2, int(math.dist(a, b) / (self.c / 2)) + 1)
        return min(self.at(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])) for t in np.linspace(0.0, 1.0, n))

    def close_segment(self, a: Sequence[float], b: Sequence[float], half_width: float) -> None:
        """Block every cell within half_width of the segment (a line the route must not use)."""
        xs = self.ox + np.arange(self.nx)[:, None] * self.c
        ys = self.oy + np.arange(self.ny)[None, :] * self.c
        ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
        dx, dy = bx - ax, by - ay
        t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / max(dx * dx + dy * dy, 1e-12), 0.0, 1.0)
        near = np.hypot(xs - (ax + t * dx), ys - (ay + t * dy)) <= half_width
        self.mm[near] = np.minimum(self.mm[near], -1)


# ------------------------------------------------------------------------ search
def _snap(g: ClearanceGrid, free: list[list[bool]], x: float, y: float, max_m: float) -> tuple[int, int]:
    """The nearest free cell to (x, y) within max_m; ties go to the lowest (i, j)."""
    i0, j0 = g.cell(x, y)
    n = int(max_m / g.c)
    best = None
    for di in range(-n, n + 1):
        for dj in range(-n, n + 1):
            i, j = i0 + di, j0 + dj
            if g.inside(i, j) and free[i][j]:
                key = (di * di + dj * dj, i, j)
                if best is None or key < best:
                    best = key
    if best is None:
        raise ValueError(f"no free cell within {max_m:.2f} m of ({x:.2f}, {y:.2f})")
    return best[1], best[2]


def _astar(g: ClearanceGrid, free: list[list[bool]], pen: list[list[int]],
           start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Integer-cost A*: deterministic on any machine."""
    gi, gj = goal

    def h(i: int, j: int) -> int:                       # octile distance: admissible, since pen >= 1000
        dx, dy = abs(i - gi), abs(j - gj)
        return 1000 * max(dx, dy) + 414 * min(dx, dy)

    best = {start: 0}
    came: dict[tuple[int, int], tuple[int, int]] = {}
    heap = [(h(*start), 0, start[0] * g.ny + start[1], start)]
    while heap:
        _, cost, _, cur = heapq.heappop(heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if cost > best[cur]:
            continue
        for di, dj, step in _MOVES:
            i, j = cur[0] + di, cur[1] + dj
            if not (0 <= i < g.nx and 0 <= j < g.ny) or not free[i][j]:
                continue
            nc = cost + step * pen[i][j] // 1000
            if nc < best.get((i, j), 1 << 62):
                best[(i, j)] = nc
                came[(i, j)] = cur
                heapq.heappush(heap, (nc + h(i, j), nc, i * g.ny + j, (i, j)))
    raise ValueError("no path between two goals: a gap is narrower than the drone plus its margin")


def _kept(need_mm: int, span_min_mm: int, keep_mm: int | None) -> int:
    """The clearance a straight line must keep to replace a stretch of path whose own minimum is span_min_mm."""
    return need_mm if keep_mm is None else max(need_mm, span_min_mm - keep_mm)


def _rdp(g: ClearanceGrid, pts: list[XY], tol: float, need_mm: int,
         cl: list[int] | None = None, keep_mm: int | None = None) -> list[XY]:
    """Ramer-Douglas-Peucker, keeping a shortcut only if its straight line stays clear (and, with keep_mm, loses at
    most keep_mm of the replaced points' clearance cl)."""
    if len(pts) <= 2:
        return pts
    a, b = np.asarray(pts[0]), np.asarray(pts[-1])
    d = b - a
    L = float(np.hypot(*d))
    P = np.asarray(pts)
    dist = (np.abs(d[0] * (P[:, 1] - a[1]) - d[1] * (P[:, 0] - a[0])) / L if L > 1e-9
            else np.hypot(P[:, 0] - a[0], P[:, 1] - a[1]))
    i = int(np.argmax(dist))
    if dist[i] <= tol:
        if g.segment_min(pts[0], pts[-1]) >= _kept(need_mm, min(cl) if cl else need_mm, keep_mm):
            return [pts[0], pts[-1]]
        i = len(pts) // 2
    return (_rdp(g, pts[:i + 1], tol, need_mm, cl[:i + 1] if cl else None, keep_mm)[:-1]
            + _rdp(g, pts[i:], tol, need_mm, cl[i:] if cl else None, keep_mm))


def _shortcut(g: ClearanceGrid, pts: list[XY], need_mm: int, keep_mm: int | None = None) -> list[XY]:
    """Drop corners where the straight line keeps need_mm: open areas only, corridors keep their centred path.
    With keep_mm, the line must also lose at most keep_mm of the dropped stretch's own clearance."""
    seg = [g.segment_min(a, b) for a, b in zip(pts, pts[1:])] if keep_mm is not None else []
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and g.segment_min(pts[i], pts[j]) < _kept(need_mm, min(seg[i:j]) if seg else need_mm,
                                                                  keep_mm):
            j -= 1
        out.append(pts[j])
        i = j
    return out


# ------------------------------------------------------------------------- goals
def _unit(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / float(np.hypot(*v))


def _pick_option(options: list[dict[str, Any]], wanted: str, cid: str) -> dict[str, Any]:
    norm = wanted.strip().lower().replace("_", " ")
    for o in options:
        if str(o["name"]).strip().lower() == norm:
            return o
    raise ValueError(f"{cid} has no option {wanted!r} (have {[o['name'] for o in options]})")


def _line_goals(g: ClearanceGrid, line: Sequence[Sequence[float]], direction: Sequence[float],
                p: PlannerParams, clear_mm: int, label: str, warnings: list[str]) -> list[tuple[np.ndarray, str]]:
    """An approach point before the line and an exit point past it, both at the line's most open point."""
    a, b = np.asarray(line[0], dtype=float), np.asarray(line[1], dtype=float)
    d = _unit(direction)
    n = max(2, int(math.dist(a, b) / (g.c / 2)) + 1)
    samples = a + (b - a) * np.linspace(0.0, 1.0, n)[:, None]
    clear = [g.at(*q) for q in samples]
    gap = samples[int(np.argmax(clear))]                 # first most-open point along the line
    if max(clear) < clear_mm:
        raise ValueError(f"{label}: the line's most open point has {max(clear) / MM:.3f} m of clearance, "
                         f"the drone needs {clear_mm / MM:.3f} m")
    need = p.drone_radius_m + p.cross_extra_m
    steps = [float(s) for s in np.arange(0.0, need, g.c / 2)] + [need]    # always test `need` itself
    reach = 0.0
    for s in steps:
        if g.at(*(gap + d * s)) < clear_mm:
            break
        reach = s
    if reach < p.drone_radius_m - 1e-9:
        warnings.append(f"{label}: the drone's centre can only get {reach:.2f} m past the line with "
                        f"{clear_mm / MM:.2f} m clearance; the whole drone needs {p.drone_radius_m:.2f} m")
    return [(gap - d * p.approach_m, f"{label} approach"), (gap + d * reach, label)]


def _crossings(map_xy: list[XY], z: float, checkpoints: list[dict[str, Any]], chosen: dict[str, str],
               radius: float) -> list[dict[str, Any]]:
    """How far past each flown checkpoint line the route's centre gets (nav.path_geometry's check)."""
    from nav.path_geometry import checkpoint_crossings
    P = np.asarray(map_xy, dtype=float)
    dense = [P[0]]
    for a, b in zip(P, P[1:]):
        n = max(2, int(math.dist(a, b) / 0.01) + 1)
        dense.extend(a + (b - a) * t for t in np.linspace(0.0, 1.0, n)[1:])
    xyz = np.c_[np.asarray(dense), np.full(len(dense), z)]
    flown = []
    for cp in checkpoints:
        if "options" in cp:
            cp = dict(cp, options=[o for o in cp["options"] if o["name"] == chosen.get(cp.get("id", ""))])
        flown.append(cp)
    return checkpoint_crossings(SimpleNamespace(xyz=xyz), flown, radius)


# -------------------------------------------------------------------------- plan
def plan_route(cm: CourseMap, checkpoints: list[dict[str, Any]], launch: dict[str, float], z: float,
               params: PlannerParams | None = None, cp4: str = "window") -> RoutePlan:
    """The route from the launch pose through every checkpoint, in course order, to the pad's centre."""
    p = params or PlannerParams()
    g = ClearanceGrid(cm, z)
    clear_mm = int(math.floor(p.clearance_m * MM + 0.5))
    warnings: list[str] = []

    chosen: dict[str, str] = {}
    for cp in checkpoints:                                # close the options not flown, before any search
        if "options" in cp:
            pick = _pick_option(cp["options"], cp4, cp.get("id", "?"))
            chosen[cp.get("id", "")] = pick["name"]
            for o in cp["options"]:
                if o is not pick and "line" in o:
                    g.close_segment(o["line"][0], o["line"][1], g.c)
            if "z" in pick and not float(pick["z"][0]) < z < float(pick["z"][1]):
                raise ValueError(f"{cp.get('id')} {pick['name']} spans z {pick['z'][0]}-{pick['z'][1]} m but the "
                                 f"route is planned at z {z:.2f} m: fly at the opening's height")

    goals: list[tuple[np.ndarray, str]] = []
    for cp in checkpoints:
        cid = str(cp.get("id", "?"))
        spec = next((o for o in cp.get("options", []) if o["name"] == chosen.get(cid)), cp)
        label = f"{cid} {spec['name']}" if spec is not cp else cid
        if "line" in spec:
            goals += _line_goals(g, spec["line"], spec["direction"], p, clear_mm, label, warnings)
        elif "centre" in spec:
            goals.append((np.asarray(spec["centre"], dtype=float), label))
        else:
            raise ValueError(f"{cid}: a checkpoint needs a line or a centre")

    free = (g.mm >= clear_mm).tolist()
    pref = p.centre_pref_m * MM
    pen = (1000 + np.floor(p.centre_weight * 1000 * np.clip((pref - g.mm) / pref, 0.0, 1.0) + 0.5)
           ).astype(np.int64).tolist()

    launch_xy = (float(launch["x"]), float(launch["y"]))
    cur = _snap(g, free, *launch_xy, p.snap_m)
    roomy_mm = clear_mm + int(math.floor(p.roomy_m * MM + 0.5))
    keep_mm = None if p.keep_centre_m is None else int(math.floor(p.keep_centre_m * MM + 0.5))
    pts: list[XY] = [g.xy(cur)]
    labels = ["launch"]
    for xy, label in goals:
        goal = _snap(g, free, float(xy[0]), float(xy[1]), p.snap_m)
        cells = _astar(g, free, pen, cur, goal)
        leg = [g.xy(c) for c in cells]
        cl = [int(g.mm[c]) for c in cells] if keep_mm is not None else None
        leg = _shortcut(g, _rdp(g, leg, p.simplify_tol_m, clear_mm, cl, keep_mm), roomy_mm, keep_mm)
        for q in leg[1:]:
            pts.append(q)
            labels.append("")
        labels[-1] = label
        cur = goal

    # merge points closer than min_spacing (a goal wins over a plain corner), then drop the launch point
    keep_pts, keep_lbl = [pts[0]], [labels[0]]
    for q, lb in zip(pts[1:], labels[1:]):
        if math.dist(q, keep_pts[-1]) < p.min_spacing_m:
            if lb and not keep_lbl[-1]:
                keep_pts[-1], keep_lbl[-1] = q, lb
            continue
        keep_pts.append(q)
        keep_lbl.append(lb)
    map_xy, lbls = keep_pts[1:], keep_lbl[1:]

    a = math.radians(float(launch["yaw_deg"]))
    c, s = math.cos(a), math.sin(a)
    route = [(round(c * (x - launch_xy[0]) + s * (y - launch_xy[1]), 3),
              round(-s * (x - launch_xy[0]) + c * (y - launch_xy[1]), 3)) for x, y in map_xy]
    plan = RoutePlan(route, map_xy, lbls, launch_xy, z, cp4, warnings=warnings)
    plan.crossings = _crossings([launch_xy] + map_xy, z, checkpoints, chosen, p.drone_radius_m)
    return plan


def gate_problems(plan: RoutePlan) -> list[str]:
    """Reasons to refuse the route before arming: a checkpoint line its centre never crosses, or an end off the pad.
    A line crossed but not by the whole drone (CP1, as drawn) is a warning, not a refusal."""
    out = []
    for c in plan.crossings:
        if "past_m" in c and not c["past_m"] > 0.0:
            out.append(f"{c['checkpoint']}: the route never crosses the line (centre {c['past_m']} m past)")
        if "whole_drone_inside" in c and not c["whole_drone_inside"]:
            out.append(f"{c['checkpoint']}: the route ends {c['end_from_centre_m']} m from the pad's centre")
    return out


def plan_from_files(sdf_path: str | Path, yaml_path: str | Path, launch_key: str, height_m: float,
                    cp4: str = "window", keep_centre_m: float | None = None) -> RoutePlan:
    """Load the map and course.yaml and plan. Height is above the launch point, as refnav's height_m."""
    import yaml

    from nav.course_map import CourseMap
    from nav.frames import load_launch
    cfg = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    lim = cfg["limits"]
    params = PlannerParams(drone_radius_m=float(lim["drone_radius"]),
                           clearance_m=float(lim["drone_radius"]) + float(lim["clearance_margin"]) + 0.01,
                           keep_centre_m=keep_centre_m)
    launch = load_launch(yaml_path, launch_key)
    return plan_route(CourseMap.load(sdf_path), cfg["checkpoints"], launch, height_m + launch["z"], params, cp4)


def _plot(plan: RoutePlan, sdf_path: str | Path, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nav.course_map import CourseMap
    g = ClearanceGrid(CourseMap.load(sdf_path), plan.z)
    ext = (g.ox - g.c / 2, g.ox + (g.nx - 0.5) * g.c, g.oy - g.c / 2, g.oy + (g.ny - 0.5) * g.c)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(g.mm.T / MM, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=0.8)
    P = np.asarray([plan.launch_xy] + plan.map_xy)
    ax.plot(P[:, 0], P[:, 1], "-o", color="w", ms=3, lw=1.5)
    for (x, y), lb in zip(plan.map_xy, plan.labels):
        if lb:
            ax.annotate(lb, (x, y), color="w", fontsize=7)
    ax.set_title(f"{plan.length_m:.1f} m, CP4 {plan.cp4}, z {plan.z:.2f} m, digest {plan.digest}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json
    import time
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdf", default=str(ROOT / "maps" / "course_sdf.npz"))
    ap.add_argument("--yaml", default=str(ROOT / "maps" / "course.yaml"))
    ap.add_argument("--launch-key", default="launch_sim", help="launch (the real mark) or launch_sim (sim spawn)")
    ap.add_argument("--height", type=float, default=0.7, help="flight height above the launch point (m)")
    ap.add_argument("--cp4", default="window", help="window or 'Gate B'")
    ap.add_argument("--out", help="write the route as a route file (for devsim / comparisons)")
    ap.add_argument("--plot", help="PNG of the route over the clearance slice")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    plan = plan_from_files(a.sdf, a.yaml, a.launch_key, a.height, a.cp4)
    print(f"planned in {time.perf_counter() - t0:.2f} s: {plan.summary()}")
    for w in plan.warnings:
        print("  WARNING:", w)
    for prob in gate_problems(plan):
        print("  REFUSE:", prob)
    if a.out:
        Path(a.out).write_text(json.dumps({"route": [list(q) for q in plan.route]}, indent=1) + "\n")
    if a.plot:
        try:
            _plot(plan, a.sdf, a.plot)
        except ImportError:
            print("matplotlib is not installed: skipped the plot")
    return 1 if gate_problems(plan) else 0


if __name__ == "__main__":
    raise SystemExit(main())
