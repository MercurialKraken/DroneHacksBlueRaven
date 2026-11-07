"""Map-based route planner for the DroneHacks course.

The planner knows nothing about this particular course: it reads walls,
openings, the tunnel, boulders, the pad and an ordered checkpoint list from the
map file, then
  1. builds a clearance field (distance from every cell to the nearest obstacle),
  2. runs A* between successive checkpoint goals, preferring corridor centres,
  3. simplifies the path without losing clearance,
  4. assigns a height, speed and heading to every waypoint,
  5. adds the optional return leg over the walls,
  6. exports setpoints in PX4 local NED relative to the launch mark.
Change the map and the route changes with it; nothing here is a fixed waypoint.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .geometry import Grid, rasterise


@dataclass
class Params:
    drone_radius_m: float = 0.29      # circle covering the drone incl. prop tips (7 in props, confirmed)
    margin_m: float = 0.05            # extra wall clearance on top of the radius
    half_length_m: float = 0.205      # front-to-back half extent, used to fully cross checkpoint lines
    cross_extra_m: float = 0.05       # go this much past "whole drone across the line"
    grid_res_m: float = 0.025
    centre_weight: float = 8.0        # how strongly A* prefers the middle of a corridor
    centre_pref_m: float = 0.55       # clearance beyond which a cell counts as "centred"
    cruise_z_m: float = 0.90
    cruise_v: float = 0.50
    tight_v: float = 0.30             # where clearance is under safe radius + 0.15 m
    turn_v: float = 0.20              # at corners sharper than turn_deg
    gate_v: float = 0.30              # through checkpoint lines, window and tunnel
    turn_deg: float = 40.0
    return_above_walls_m: float = 0.45
    ceiling_margin_m: float = 0.60
    simplify_tol_m: float = 0.03

    @property
    def safe_r(self) -> float:
        return self.drone_radius_m + self.margin_m


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    speed: float
    phase: str
    label: str = ""
    hover_s: float = 0.0
    yaw_enu: float | None = None


@dataclass
class Plan:
    route: str
    params: Params
    waypoints: list[Waypoint]
    warnings: list[str] = field(default_factory=list)
    checkpoint_order: list[str] = field(default_factory=list)

    def length_m(self, include_return=False) -> float:
        wps = [w for w in self.waypoints if include_return or not w.phase.startswith("return")]
        return sum(math.dist((a.x, a.y, a.z), (b.x, b.y, b.z)) for a, b in zip(wps, wps[1:]))

    def time_s(self, include_return=False) -> float:
        wps = [w for w in self.waypoints if include_return or not w.phase.startswith("return")]
        t = sum(w.hover_s for w in wps)
        for a, b in zip(wps, wps[1:]):
            t += math.dist((a.x, a.y, a.z), (b.x, b.y, b.z)) / max(min(a.speed, b.speed), 0.05)
        return t


# ---------------------------------------------------------------- A* search

_NEIGH = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _astar(grid: Grid, start, goal, p: Params):
    free = grid.clearance >= p.safe_r
    s, g = start, goal
    if not (free[s] and free[g]):
        raise ValueError("start or goal is not free")
    pen = 1.0 + p.centre_weight * np.clip((p.centre_pref_m - grid.clearance) / p.centre_pref_m, 0, 1)
    h = lambda a: math.hypot(a[0] - g[0], a[1] - g[1])
    best = {s: 0.0}
    came = {}
    q = [(h(s), 0.0, s)]
    while q:
        _, cost, cur = heapq.heappop(q)
        if cur == g:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if cost > best.get(cur, math.inf):
            continue
        for dr, dc in _NEIGH:
            nb = (cur[0] + dr, cur[1] + dc)
            if not grid.inside(*nb) or not free[nb]:
                continue
            nc = cost + math.hypot(dr, dc) * pen[nb]
            if nc < best.get(nb, math.inf):
                best[nb] = nc
                came[nb] = cur
                heapq.heappush(q, (nc + h(nb), nc, nb))
    raise ValueError("no path between goals: the gap may be narrower than the drone plus margin")


def _snap(grid: Grid, x, y, p: Params, max_dist=0.25):
    """Nearest free cell to (x, y), within max_dist."""
    r0, c0 = grid.to_cell(x, y)
    n = int(max_dist / grid.res)
    best = None
    for dr in range(-n, n + 1):
        for dc in range(-n, n + 1):
            r, c = r0 + dr, c0 + dc
            if grid.inside(r, c) and grid.clearance[r, c] >= p.safe_r:
                d = dr * dr + dc * dc
                if best is None or d < best[0]:
                    best = (d, (r, c))
    if best is None:
        raise ValueError(f"no free cell within {max_dist} m of ({x:.2f}, {y:.2f})")
    return best[1]


def _segment_clear(grid: Grid, a, b, p: Params) -> bool:
    n = max(2, int(math.dist(a, b) / (grid.res / 2)))
    for t in np.linspace(0, 1, n):
        if grid.clearance_at(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])) < p.safe_r - grid.res:
            return False
    return True


def _simplify(grid: Grid, pts, p: Params):
    """Ramer-Douglas-Peucker, but a shortcut is only kept if it stays clear."""
    if len(pts) <= 2:
        return pts
    a, b = np.asarray(pts[0]), np.asarray(pts[-1])
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-9:
        dists = [np.linalg.norm(np.asarray(q) - a) for q in pts]
    else:
        dists = [abs(d[0] * (q[1] - a[1]) - d[1] * (q[0] - a[0])) / L for q in pts]
    i = int(np.argmax(dists))
    if dists[i] <= p.simplify_tol_m and _segment_clear(grid, pts[0], pts[-1], p):
        return [pts[0], pts[-1]]
    if dists[i] <= p.simplify_tol_m:
        i = len(pts) // 2
    left = _simplify(grid, pts[: i + 1], p)
    right = _simplify(grid, pts[i:], p)
    return left[:-1] + right


# ---------------------------------------------------------------- goals


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _crossing_goals(grid, p, mid, along, cur, label, warnings, z=None):
    """Points before and after a line/opening so the whole drone crosses it squarely."""
    n = np.array([-along[1], along[0]])
    if np.dot(np.asarray(cur) - mid, n) > 0:
        n = -n                                   # point from the drone's side to the far side
    need = p.half_length_m + p.cross_extra_m
    pre = mid - n * max(need, 0.30)
    best = 0.0
    for s in np.arange(0.0, need + 0.15 + 1e-9, grid.res):
        q = mid + n * s
        if grid.clearance_at(*q) >= p.safe_r:
            best = s
        elif s > 0.05:
            break
    if best < need:
        warnings.append(
            f"{label}: the drone centre can only get {best:.2f} m past the line with the planned "
            f"clearance, but {need:.2f} m is needed for the whole drone to cross. Fly this part "
            f"slowly with less margin, or ask organizers how the crossing is judged.")
    post = mid + n * max(best, 0.0)
    goals = [(tuple(pre), f"{label} (approach)", z), (tuple(post), label, z)]
    return goals, tuple(post)


def _goals(m, grid, p, route, warnings):
    cur = tuple(m["launch_mark_xy"])
    goals = []
    order = []
    openings = {o["name"]: o for o in m.get("openings", [])}
    for cp in m["checkpoints"]:
        spec = cp
        name = f"CP{cp['id']} {cp['name']}"
        if cp["kind"] == "choice":
            if route not in cp["options"]:
                raise ValueError(f"route '{route}' not offered for {name}")
            spec = cp["options"][route]
            name = f"CP{cp['id']} {route.replace('_', ' ')}"
        order.append(name)
        if spec["kind"] == "line":
            a, b = (np.asarray(q, float) for q in spec["line"])
            g, cur = _crossing_goals(grid, p, (a + b) / 2, _unit(b - a), cur, name, warnings)
        elif spec["kind"] == "opening":
            op = openings[spec["opening"]]
            z = (op["bottom_m"] + op["top_m"]) / 2
            g, cur = _crossing_goals(grid, p, np.asarray(op["centre_xy"], float),
                                     _unit(op["along_wall_unit"]), cur, name, warnings, z=z)
            g.insert(1, (tuple(op["centre_xy"]), f"{name} (centre)", z))
        elif spec["kind"] == "pad":
            cur = tuple(m["landing_pad"]["centre_xy"])
            g = [(cur, name, None)]
        else:
            raise ValueError(f"unknown checkpoint kind {spec['kind']}")
        goals += g
    return goals, order


# ---------------------------------------------------------------- main entry


def _in_tunnel(m, x, y):
    t = m.get("tunnel")
    if not t:
        return False
    a, e = np.asarray(t["entry_xy"]), np.asarray(t["exit_xy"])
    ax = e - a
    L = np.linalg.norm(ax)
    u = ax / L
    s = np.dot([x - a[0], y - a[1]], u)
    off = abs(-(x - a[0]) * u[1] + (y - a[1]) * u[0])
    return -0.05 <= s <= L + 0.05 and off <= t["inner_width_m"] / 2


def _leg(m, grid, a_cell, b_cell, p):
    """A* between two cells; if the result runs through the tunnel, fly it straight down the axis."""
    pts = _simplify(grid, [grid.to_xy(*c) for c in _astar(grid, a_cell, b_cell, p)], p)
    t = m.get("tunnel")
    if not t:
        return _shortcut(grid, pts, p)
    dense = [grid.to_xy(*c) for c in _astar(grid, a_cell, b_cell, p)]
    inside = [i for i, q in enumerate(dense) if _in_tunnel(m, *q)]
    if not inside:
        return _shortcut(grid, pts, p)
    ends = [np.asarray(t["entry_xy"], float), np.asarray(t["exit_xy"], float)]
    first = np.asarray(dense[inside[0]])
    if np.linalg.norm(first - ends[1]) < np.linalg.norm(first - ends[0]):
        ends = ends[::-1]                       # entered from the other end
    u = _unit(ends[1] - ends[0])
    align, out = ends[0] - u * 0.40, ends[1] + u * 0.30
    c_align, c_out = _snap(grid, *align, p), _snap(grid, *out, p)
    before = _simplify(grid, [grid.to_xy(*c) for c in _astar(grid, a_cell, c_align, p)], p)
    after = _simplify(grid, [grid.to_xy(*c) for c in _astar(grid, c_out, b_cell, p)], p)
    straight = [tuple(ends[0] + u * 0.05), tuple((ends[0] + ends[1]) / 2), tuple(ends[1] - u * 0.05)]
    return _shortcut(grid, before, p) + straight + _shortcut(grid, after, p)


def _segment_min_clearance(grid, a, b):
    n = max(2, int(math.dist(a, b) / (grid.res / 2)))
    return min(grid.clearance_at(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])) for t in np.linspace(0, 1, n))


def _shortcut(grid, pts, p, roomy=0.10):
    """Remove kinks, but only where the straight line stays comfortably clear (open areas,
    not corridors, where the centred A* path is what we want)."""
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and _segment_min_clearance(grid, pts[i], pts[j]) < p.safe_r + roomy:
            j -= 1
        out.append(pts[j])
        i = j
    return out


def plan(m: dict, route: str = "gate_b", params: Params | None = None, with_return: bool = True) -> Plan:
    p = params or Params()
    opened = tuple(o["name"] for o in m.get("openings", []) if o["name"] == route)
    grid = rasterise(m, p.grid_res_m, open_openings=opened)
    warnings: list[str] = []
    goals, order = _goals(m, grid, p, route, warnings)

    launch = tuple(m["launch_mark_xy"])
    tun = m.get("tunnel")
    tunnel_z = tun["floor_m"] + tun["inner_height_m"] / 2 if tun else p.cruise_z_m

    wps = [Waypoint(*launch, 0.0, p.gate_v, "takeoff", "launch"),
           Waypoint(*launch, p.cruise_z_m, p.gate_v, "takeoff", "climb", hover_s=2.0)]
    cur_cell = _snap(grid, *launch, p)
    labels_seen = set()
    for (gx, gy), label, gz in goals:
        if "(approach)" in label and _in_tunnel(m, gx, gy):
            continue                       # the tunnel already lines the drone up
        gcell = _snap(grid, gx, gy, p)
        phase = "to " + label.split(" (")[0]
        for x, y in _leg(m, grid, cur_cell, gcell, p)[1:]:
            wps.append(Waypoint(x, y, p.cruise_z_m, p.cruise_v, phase))
        wps[-1].label = label
        if gz is not None:
            wps[-1].z = gz
        labels_seen.add(label)
        cur_cell = gcell

    # heights and speeds
    for i, w in enumerate(wps[2:], start=2):
        if _in_tunnel(m, w.x, w.y):
            w.z, w.speed, w.phase = tunnel_z, p.gate_v, "tunnel"
        if grid.clearance_at(w.x, w.y) < p.safe_r + 0.15:
            w.speed = min(w.speed, p.tight_v)
        if w.label and "approach" not in w.label:
            w.speed = min(w.speed, p.gate_v)
        if 0 < i < len(wps) - 1:
            a, b = wps[i - 1], wps[i + 1]
            h1 = math.atan2(w.y - a.y, w.x - a.x)
            h2 = math.atan2(b.y - w.y, b.x - w.x)
            turn = abs((math.degrees(h2 - h1) + 180) % 360 - 180)
            if turn > p.turn_deg and math.dist((a.x, a.y), (w.x, w.y)) > 1e-3:
                w.speed = min(w.speed, p.turn_v)
                w.hover_s = max(w.hover_s, 0.5)
    # hover at the tunnel mouth so vision settles before the dark section
    for i, w in enumerate(wps):
        if w.phase == "tunnel":
            prev = wps[i - 1]
            prev.hover_s = max(prev.hover_s, 1.5)
            prev.z = tunnel_z
            break
    # descend to the window height before reaching it
    for i, w in enumerate(wps):
        if "(approach)" in w.label and i + 1 < len(wps) and wps[i + 1].label.endswith("(centre)"):
            w.z = wps[i + 1].z
            w.hover_s = max(w.hover_s, 1.0)

    pad = tuple(m["landing_pad"]["centre_xy"])
    wps.append(Waypoint(*pad, 0.30, p.gate_v, "landing", "pad (slow final)", hover_s=1.0))
    wps.append(Waypoint(*pad, 0.0, 0.15, "landing", "touchdown + disarm"))

    if with_return:
        top = max([w["height_m"] for w in m["walls"]] + ([tun["floor_m"] + tun["inner_height_m"]] if tun else []))
        z_ret = min(top + p.return_above_walls_m, m["bounds"]["height_m"] - p.ceiling_margin_m)
        if z_ret <= top + 0.2:
            warnings.append("return: not enough height under the net to clear the walls safely")
        if grid.clearance_at(*launch) < p.safe_r:
            warnings.append("return: launch mark is closer to a wall than the safe radius")
        wps += [Waypoint(*pad, z_ret, 0.40, "return", "re-arm, climb over walls"),
                Waypoint(*launch, z_ret, 0.40, "return", "over launch mark", hover_s=1.0),
                Waypoint(*launch, 0.30, 0.25, "return", "descend into corridor"),
                Waypoint(*launch, 0.0, 0.15, "return", "touchdown + disarm")]

    # heading: face the direction of travel (camera looks forward); keep it on vertical moves
    yaw = math.pi / 2  # start facing north, along the first corridor
    for i, w in enumerate(wps):
        nxt = wps[i + 1] if i + 1 < len(wps) else None
        if nxt and math.dist((w.x, w.y), (nxt.x, nxt.y)) > 0.05:
            yaw = math.atan2(nxt.y - w.y, nxt.x - w.x)
        w.yaw_enu = yaw

    return Plan(route, p, wps, warnings, order)


def to_px4_ned(plan_: Plan, origin_xy, origin_heading_enu: float = math.pi / 2):
    """Course ENU -> PX4 local NED, with the origin at the launch mark.

    origin_heading_enu is the direction (ENU radians) that PX4's local +x (north) axis
    points in the course. If the estimator's frame is referenced to true north this is
    pi/2; if it is referenced to the drone's initial heading, pass that heading.
    Check onboard/vio/frames.py and ev_bridge.py before trusting this.
    """
    ox, oy = origin_xy
    c, s = math.cos(origin_heading_enu), math.sin(origin_heading_enu)
    out = []
    for w in plan_.waypoints:
        dx, dy = w.x - ox, w.y - oy
        north = c * dx + s * dy          # component along PX4 +x
        east = s * dx - c * dy           # component along PX4 +y (90 deg clockwise from +x)
        yaw_ned = (origin_heading_enu - w.yaw_enu + math.pi) % (2 * math.pi) - math.pi
        out.append(dict(x=round(north, 3), y=round(east, 3), z=round(-w.z, 3), yaw=round(yaw_ned, 3),
                        speed=w.speed, hover_s=w.hover_s, phase=w.phase, label=w.label))
    return out


def to_refnav_route(plan_: Plan, origin_xy, start_heading_enu: float = math.pi / 2, min_leg_m: float = 0.05):
    """Waypoints in the starter's route.json format: [(forward_m, left_m)] from the take-off point,
    in the drone's heading at take-off (refnav.core rotates them by PX4's heading at arming).
    start_heading_enu = the direction the drone faces on the launch mark, ENU radians (pi/2 = north).
    Only the horizontal route from take-off to the pad: refnav climbs first and lands after the last leg."""
    ox, oy = origin_xy
    c, s = math.cos(start_heading_enu), math.sin(start_heading_enu)
    pts = []
    for w in plan_.waypoints:
        if w.phase in ("takeoff", "landing") or w.phase.startswith("return"):
            continue
        dx, dy = w.x - ox, w.y - oy
        fwd, left = c * dx + s * dy, -s * dx + c * dy
        if not pts or math.dist(pts[-1], (fwd, left)) >= min_leg_m:
            pts.append((round(fwd, 3), round(left, 3)))
    return pts


def plan_to_dict(plan_: Plan, m: dict) -> dict:
    return dict(
        route=plan_.route, checkpoint_order=plan_.checkpoint_order, warnings=plan_.warnings,
        length_m=round(plan_.length_m(), 2), est_time_s=round(plan_.time_s(), 1),
        length_with_return_m=round(plan_.length_m(True), 2), est_time_with_return_s=round(plan_.time_s(True), 1),
        params=asdict(plan_.params),
        course_enu=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(w).items()}
                    for w in plan_.waypoints],
        px4_ned_from_launch=to_px4_ned(plan_, m["launch_mark_xy"]),
    )
