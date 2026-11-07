"""Summarise one recorded run: mission status timeline + flown vs commanded path over the course map.

    python3 tools/analyze_bag.py bags/run_0613/run_0613_0.mcap --out run_0613
      -> run_0613_path.png      top-down: commanded setpoints vs PX4 position, on the course map
      -> run_0613_status.png    corridor coverage, depth validity, holds, phase over time
      -> run_0613_status.txt    every /refnav/status message, one line each

Only needs numpy + matplotlib (no ROS, no mcap package): tools/mcap_read.py parses the file.
Assumes PX4's local NED origin = launch mark and +x = north (true for the sim: the drone starts on the
launch mark facing north, and run 1 flew the maze correctly on that assumption).
"""
import argparse, json, struct, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from mcap_read import read  # noqa: E402


def cdr_floats(b, off, n):  # b includes the 4-byte CDR encapsulation header
    return struct.unpack_from("<%df" % n, b, 4 + off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mcap")
    ap.add_argument("--out", default="run")
    ap.add_argument("--map", default=str(HERE.parent / "course_planner" / "course_map.json"))
    a = ap.parse_args()
    schemas, chans, msgs = read(a.mcap)
    topic = {cid: t for cid, (t, _) in chans.items()}
    t0 = min(lt for _, lt, _ in msgs)
    pos, sp, st = [], [], []
    for cid, lt, b in msgs:
        t = (lt - t0) / 1e9
        name = topic[cid]
        if name == "/fmu/out/vehicle_local_position_v1":
            x, y, z = cdr_floats(b, 20, 3)          # uint64, uint64, 4 x bool, then x y z
            pos.append((t, x, y, z))
        elif name == "/fmu/in/trajectory_setpoint":
            x, y, z = cdr_floats(b, 8, 3)           # uint64 timestamp, then position[3]
            sp.append((t, x, y, z))
        elif name == "/refnav/status":
            n = struct.unpack_from("<I", b, 4)[0]
            st.append((t, json.loads(b[8:8 + n - 1].decode())))

    with open(f"{a.out}_status.txt", "w") as f:
        for t, s in st:
            f.write(f"{t:7.1f} {json.dumps(s)}\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon
    m = json.load(open(a.map))
    L = m["launch_mark_xy"]
    fig, ax = plt.subplots(figsize=(9, 9))
    for w in m["walls"]:
        ax.add_patch(Polygon(w["footprint_xy"], closed=True, fc="#222"))
    for bo in m["boulders"]:
        ax.add_patch(Circle(bo["centre_xy"], bo["radius_m"], fc="#b08d57"))
    ax.add_patch(Circle(m["landing_pad"]["centre_xy"], m["landing_pad"]["outer_ring_radius_m"], fill=False, ec="g"))
    te, ts = m["tunnel"]["entry_xy"], m["tunnel"]["exit_xy"]
    ax.plot([te[0], ts[0]], [te[1], ts[1]], color="#888", lw=30, alpha=.3)
    if sp:
        ax.plot([L[0] + s[2] for s in sp], [L[1] + s[1] for s in sp], "--", color="#8e44ad", label="commanded (setpoint)")
    if pos:
        ax.plot([L[0] + p[2] for p in pos], [L[1] + p[1] for p in pos], color="#d35400", lw=2, label="flown (PX4 estimate)")
        ax.plot(L[0] + pos[-1][2], L[1] + pos[-1][1], "x", color="#d35400", ms=12)
    ax.set_xlim(-3.1, 3.1); ax.set_ylim(-3.1, 3.1); ax.set_aspect("equal"); ax.grid(alpha=.2)
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)"); ax.legend(loc="lower right")
    ax.set_title(f"{Path(a.mcap).stem}: commanded vs flown (x = last position)")
    fig.tight_layout(); fig.savefig(f"{a.out}_path.png", dpi=120)

    if st:
        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        T = [t for t, _ in st]
        g = lambda k: [s.get(k) if s.get(k) is not None else float("nan") for _, s in st]
        axs[0].plot(T, g("corridor_coverage"), label="corridor_coverage")
        axs[0].plot(T, g("depth_valid_frac"), label="depth_valid_frac")
        axs[0].axhline(0.6, ls=":", c="r", label="starter min_corridor_coverage 0.6")
        axs[0].legend(fontsize=8); axs[0].set_ylim(0, 1)
        axs[1].plot(T, g("depth_holds"), label="depth_holds"); axs[1].plot(T, g("corridor_unobserved"), label="corridor_unobserved")
        axs[1].legend(fontsize=8)
        phases = sorted({s["phase"] for _, s in st})
        axs[2].step(T, [phases.index(s["phase"]) for _, s in st], where="post")
        axs[2].set_yticks(range(len(phases))); axs[2].set_yticklabels(phases); axs[2].set_xlabel("seconds from recording start")
        fig.tight_layout(); fig.savefig(f"{a.out}_status.png", dpi=110)
    print(f"{len(pos)} positions, {len(sp)} setpoints, {len(st)} status msgs -> {a.out}_*.png/.txt")


if __name__ == "__main__":
    main()
