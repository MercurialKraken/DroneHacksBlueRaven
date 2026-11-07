"""python3 -m course_planner [--map course_map.json] [--route gate_b|window] [--out plan.json] [--plot plan.png]"""
import argparse
import json
from pathlib import Path

from .geometry import load_map, rasterise
from .planner import Params, plan, plan_to_dict

HERE = Path(__file__).resolve().parent


def draw(m, plans, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    p0 = plans[0].params
    grid = rasterise(m, p0.grid_res_m, open_openings=("window",))
    fig, ax = plt.subplots(figsize=(10, 10))
    b = m["bounds"]
    ext = (b["x_min"], b["x_max"], b["y_min"], b["y_max"])
    ax.imshow(grid.clearance < p0.safe_r, origin="lower", extent=ext, cmap="Greys", alpha=0.18)
    for w in m["walls"]:
        ax.add_patch(Polygon(w["footprint_xy"], closed=True, fc="#222", ec="#222"))
    for op in m.get("openings", []):
        c, u = op["centre_xy"], op["along_wall_unit"]
        h = op["width_m"] / 2
        ax.plot([c[0] - u[0] * h, c[0] + u[0] * h], [c[1] - u[1] * h, c[1] + u[1] * h], color="#3fa7ff", lw=5)
    t = m["tunnel"]
    ax.plot([t["entry_xy"][0], t["exit_xy"][0]], [t["entry_xy"][1], t["exit_xy"][1]], color="#888", lw=40, alpha=.35)
    for bo in m["boulders"]:
        ax.add_patch(Circle(bo["centre_xy"], bo["radius_m"], fc="#b08d57"))
    ax.add_patch(Circle(m["landing_pad"]["centre_xy"], m["landing_pad"]["outer_ring_radius_m"], fill=False, ec="#2e7d32", lw=2))
    for cp in m["checkpoints"]:
        lines = [cp["line"]] if cp["kind"] == "line" else [o["line"] for o in cp.get("options", {}).values() if o["kind"] == "line"]
        for ln in lines:
            ax.plot(*zip(*ln), color="#8b1a1a", lw=2, ls="--")
    colours = {"gate_b": "#8e44ad", "window": "#16a085"}
    for pl in plans:
        ws = [w for w in pl.waypoints if w.phase != "return"]
        ax.plot([w.x for w in ws], [w.y for w in ws], "-o", ms=3, color=colours.get(pl.route, "k"),
                label=f"{pl.route}: {pl.length_m():.1f} m, ~{pl.time_s():.0f} s")
        for w in ws:
            if w.speed <= p0.turn_v + 1e-9:
                ax.add_patch(Circle((w.x, w.y), 0.04, color="red", zorder=5))
    ret = [w for w in plans[0].waypoints if w.phase == "return"]
    if ret:
        ax.plot([w.x for w in ret], [w.y for w in ret], "-.", color="#e67e22", lw=2, label="return (over walls)")
    for x, y in [m["launch_mark_xy"]]:
        ax.add_patch(Circle((x, y), 0.12, color="#1565c0"))
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_title(f"Planned routes. Grey = too close to a wall for the drone centre "
                 f"(radius {p0.drone_radius_m} m + margin {p0.margin_m} m). Red dots = slow turns.", fontsize=10)
    ax.legend(loc="lower right")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(HERE / "course_map.json"))
    ap.add_argument("--route", default="gate_b", help="gate_b, window, or both")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--no-return", action="store_true")
    ap.add_argument("--radius", type=float, default=None, help="override drone radius (m)")
    a = ap.parse_args(argv)
    m = load_map(a.map)
    prm = Params()
    if a.radius:
        prm.drone_radius_m = a.radius
    routes = ["gate_b", "window"] if a.route == "both" else [a.route]
    plans = [plan(m, r, prm, with_return=not a.no_return) for r in routes]
    for pl in plans:
        print(f"{pl.route}: {pl.length_m():.1f} m, ~{pl.time_s():.0f} s "
              f"(with return {pl.length_m(True):.1f} m, ~{pl.time_s(True):.0f} s), {len(pl.waypoints)} waypoints")
        for w in pl.warnings:
            print("  WARNING:", w)
    if a.out:
        out = [plan_to_dict(pl, m) for pl in plans]
        Path(a.out).write_text(json.dumps(out[0] if len(out) == 1 else out, indent=1))
    if a.plot:
        draw(m, plans, a.plot)


if __name__ == "__main__":
    main()
