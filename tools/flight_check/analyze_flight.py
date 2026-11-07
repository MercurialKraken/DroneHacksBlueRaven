#!/usr/bin/env python3
"""Summarise one REAL-DRONE recording: topic dropouts, VIO jumps/freezes, setpoint tracking.

    python3 tools/flight_check/analyze_flight.py bags/flight_03 --out flight_03
      -> flight_03_report.txt   the numbers below, readable
      -> flight_03_report.json  the same, for comparing flights
      -> flight_03.png          top-down path + tracking error + speed (if matplotlib is present)

Accepts one .mcap file or a ros2 bag folder (all .mcap files inside are read). Offline and
read-only: no ROS, no node, no publisher. Uses tools/mcap_read.py (zstd/lz4 chunks need the
optional zstandard/lz4 packages, laptop only).

tools/analyze_bag.py does the same job for SIM runs flown by refnav (PX4 trajectory setpoints). On the
drone, setpoints go through /skybeta/setpoint/* instead, which is what this script reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import flight_log as F  # noqa: E402
from mcap_read import read  # noqa: E402

DEFAULTS = {
    "odom": "/visual_slam/tracking/odometry",
    "pose_sp": "/skybeta/setpoint/pose",
    "twist_sp": "/skybeta/setpoint/twist",
    "state": "/skybeta/state",
}


def load(path: Path) -> dict[str, dict]:
    """-> {topic: {"schema": name, "t": [log s], "msgs": [decoded or None]}}, time-ordered."""
    files = sorted(path.glob("*.mcap")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"no .mcap file in {path}")
    out: dict[str, dict] = {}
    for f in files:
        schemas, chans, msgs = read(str(f))
        for cid, lt, data in msgs:
            topic, sid = chans[cid]
            schema = schemas.get(sid, ("?", ""))[0]
            e = out.setdefault(topic, {"schema": schema, "t": [], "raw": []})
            e["t"].append(lt / 1e9)
            e["raw"].append(data)
    for e in out.values():
        order = np.argsort(e["t"], kind="stable")
        e["t"] = np.asarray(e["t"])[order]
        e["msgs"] = []
        for i in order:
            try:
                e["msgs"].append(F.decode(e["schema"], e["raw"][i]))
            except ValueError:
                e["msgs"].append(None)
        del e["raw"]
    return out


def _times(e: dict, use_header: bool) -> np.ndarray:
    """Header stamps when every message has one, else recording (arrival) time."""
    if use_header and e["msgs"] and all(m and m.get("stamp", 0) > 0 for m in e["msgs"]):
        return np.array([m["stamp"] for m in e["msgs"]])
    return e["t"]


def analyze(topics: dict[str, dict], names: dict, th: F.Thresholds) -> dict:
    t0 = min(float(e["t"][0]) for e in topics.values() if len(e["t"]))
    rep: dict = {"thresholds": th.__dict__, "topics": {}, "frames": {}, "notes": []}
    for name, e in sorted(topics.items()):
        rep["topics"][name] = {"schema": e["schema"], **F.topic_stats(e["t"] - t0, th.gap_s)}
    for key, name in names.items():
        if name not in topics:
            rep["notes"].append(f"{key}: topic {name} was not recorded")

    # data age: how stale each message already was when recorded (header topics), or when PX4
    # stamped it (px4 VehicleOdometry, e.g. the vision input). Jetson overload shows up here.
    rep["latency"] = {}
    rep.setdefault("_series", {})["latency"] = {}
    for name, e in sorted(topics.items()):
        ms = e["msgs"]
        if ms and all(m and "px4_sample_us" in m for m in ms):
            age = np.array([(m["px4_timestamp_us"] - m["px4_sample_us"]) / 1e6 for m in ms])
            kind = "px4 timestamp - timestamp_sample"
        elif ms and all(m and m.get("stamp", 0) > 0 for m in ms):
            age = e["t"] - np.array([m["stamp"] for m in ms])
            kind = "recorded - header stamp"
            if abs(float(np.median(age))) > 10:
                rep["notes"].append(
                    f"{name}: header stamps and recording clock differ by "
                    f"{float(np.median(age)):.0f} s (different clocks?); "
                    "latency not reported"
                )
                continue
        else:
            continue
        rep["latency"][name] = {"kind": kind, **F.latency(e["t"] - t0, age, th.latency_s)}
        rep["_series"]["latency"][name] = (e["t"] - t0, age)

    odo = topics.get(names["odom"])
    good = [m for m in (odo["msgs"] if odo else []) if m and "pos" in m]
    if not good:
        rep["notes"].append("no decodable odometry: VIO and tracking checks skipped")
        return rep
    ot = _times(odo, True)
    ot = ot[[i for i, m in enumerate(odo["msgs"]) if m and "pos" in m]]
    tz = ot[0] - (odo["t"][0] - t0)  # header-time origin, so all reported times share one zero
    ot = ot - tz
    op = np.array([m["pos"] for m in good])
    rep["frames"]["odom"] = sorted({m["frame"] for m in good})
    rep["vio"] = {
        "events": F.vio_events(ot, op, th),
        "path_length_m": round(float(np.nansum(np.linalg.norm(np.diff(op, axis=0), axis=1))), 3),
        "z_range_m": [round(float(np.nanmin(op[:, 2])), 3), round(float(np.nanmax(op[:, 2])), 3)],
    }
    rep["_series"].update(odom_t=ot, odom_p=op)

    sp = topics.get(names["pose_sp"])
    if sp:
        ms = [(t, m) for t, m in zip(_times(sp, True) - tz, sp["msgs"]) if m and "pos" in m]
        if ms:
            st, spos = np.array([a for a, _ in ms]), np.array([m["pos"] for _, m in ms])
            rep["frames"]["pose_setpoint"] = sorted({m["frame"] for _, m in ms})
            pt = F.pose_tracking(ot, op, st, spos, th.hold_s)
            rep["_series"].update(sp_p=spos, pose=pt.pop("series", None))
            rep["pose_tracking"] = pt
    tw = topics.get(names["twist_sp"])
    if tw:
        ms = [(t, m) for t, m in zip(_times(tw, True) - tz, tw["msgs"]) if m and "lin" in m]
        if ms:
            tt, lin = np.array([a for a, _ in ms]), np.array([m["lin"] for _, m in ms])
            rep["frames"]["twist_setpoint"] = sorted({m["frame"] for _, m in ms})
            tr = F.twist_tracking(ot, op, tt, lin)
            rep["_series"]["twist"] = tr.pop("series", None)
            rep["twist_tracking"] = tr
    if len(rep["frames"].get("pose_setpoint", [])) and (
        rep["frames"]["pose_setpoint"] != rep["frames"]["odom"]
    ):
        rep["notes"].append(
            "pose setpoints and odometry carry different frame ids: the tracking "
            "error assumes they are the same frame; check before trusting it"
        )

    stt = topics.get(names["state"])
    if stt:
        if all(m and "data" in m for m in stt["msgs"]):
            rep["state_changes"] = F.state_changes(stt["t"] - t0, [m["data"] for m in stt["msgs"]])
        else:
            rep["notes"].append(f"{names['state']} is {stt['schema']}: only its rate is reported")
    return rep


def text(rep: dict) -> str:
    L = ["Topics (count, rate, longest gap, gaps > %.2f s)" % rep["thresholds"]["gap_s"]]
    for n, s in rep["topics"].items():
        L.append(
            f"  {n:42s} {s['count']:6d}  {s['rate_hz']:7.2f} Hz  max gap "
            f"{s['max_gap_s']} s  gaps {len(s['gaps'])}   [{s['schema']}]"
        )
    if "vio" in rep:
        v = rep["vio"]
        kinds: dict[str, int] = {}
        for e in v["events"]:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        L += ["", f"VIO: path {v['path_length_m']} m, height {v['z_range_m']} m, events {kinds}"]
        L += [
            f"  t={e['t']:8.3f}  {json.dumps({k: x for k, x in e.items() if k != 't'})}"
            for e in v["events"][:40]
        ]
        if len(v["events"]) > 40:
            L.append(f"  ... {len(v['events']) - 40} more in the JSON")
    if rep.get("latency"):
        L += [
            "",
            "Data age (PX4 fuses vision only if younger than %.0f ms)"
            % (rep["thresholds"]["latency_s"] * 1e3),
        ]
        for n, x in rep["latency"].items():
            L.append(
                f"  {n:42s} median {x.get('median_ms')} ms  p90 {x.get('p90_ms')} ms  "
                f"over limit {x.get('over_limit_fraction')}  first slow second t="
                f"{x.get('first_slow_second_t')}   [{x['kind']}]"
            )
    for key in ("pose_tracking", "twist_tracking"):
        if key in rep:
            L += ["", f"{key}: {json.dumps(rep[key])}"]
    if rep.get("state_changes"):
        L += ["", "State changes:"] + [
            f"  t={s['t']:8.3f}  {s['state']}" for s in rep["state_changes"]
        ]
    L += ["", f"Frames: {json.dumps(rep['frames'])}"]
    L += ["Notes:"] + [f"  - {n}" for n in rep["notes"]] if rep["notes"] else []
    return "\n".join(L) + "\n"


def plot(rep: dict, png: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    s = rep.get("_series")
    if not s or "odom_p" not in s:
        return False
    fig, ax = plt.subplots(1, 4, figsize=(21, 5))
    ax[0].plot(s["odom_p"][:, 0], s["odom_p"][:, 1], lw=1, label="VIO odometry")
    if "sp_p" in s:
        ax[0].plot(s["sp_p"][:, 0], s["sp_p"][:, 1], ".", ms=2, label="pose setpoints")
    for e in rep["vio"]["events"]:
        if e["kind"] == "jump":
            i = int(np.argmin(np.abs(s["odom_t"] - e["t"])))
            ax[0].plot(*s["odom_p"][i, :2], "rx", ms=9)
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[0].set(
        title="top-down, odom frame (red x = VIO jump)",
        xlabel="x m (forward at power-up)",
        ylabel="y m",
    )
    ax[0].legend()
    if s.get("pose"):
        ax[1].plot(s["pose"]["t"], s["pose"]["horizontal_m"], lw=1, label="horizontal")
        ax[1].plot(s["pose"]["t"], s["pose"]["vertical_m"], lw=1, label="vertical (signed)")
        ax[1].legend()
    ax[1].set(title="distance from pose setpoint", xlabel="s", ylabel="m")
    if s.get("twist"):
        ax[2].plot(s["twist"]["t"], s["twist"]["cmd_h"], lw=1, label="commanded")
        ax[2].plot(s["twist"]["t"], s["twist"]["fly_h"], lw=1, label="flown")
        ax[2].legend()
    ax[2].set(title="horizontal speed", xlabel="s", ylabel="m/s")
    for n, (lt, age) in s.get("latency", {}).items():
        ax[3].plot(lt, np.asarray(age) * 1e3, lw=1, label=n)
    ax[3].axhline(rep["thresholds"]["latency_s"] * 1e3, color="r", ls="--", lw=1)
    if s.get("latency"):
        ax[3].legend(fontsize=7)
    ax[3].set(title="data age (red = PX4 fusion limit)", xlabel="s", ylabel="ms")
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bag", type=Path, help=".mcap file or ros2 bag folder")
    ap.add_argument("--out", default="flight")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), default=v, help=f"topic (default {v})")
    d = F.Thresholds()
    for k in d.__dict__:
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=getattr(d, k))
    a = ap.parse_args(argv)
    th = F.Thresholds(**{k: getattr(a, k) for k in d.__dict__})
    rep = analyze(load(a.bag), {k: getattr(a, k) for k in DEFAULTS}, th)
    t = text(rep)
    Path(f"{a.out}_report.txt").write_text(t)
    drawn = plot(rep, f"{a.out}.png")
    rep.pop("_series", None)
    Path(f"{a.out}_report.json").write_text(json.dumps(rep, indent=1))
    print(
        t + f"wrote {a.out}_report.txt, {a.out}_report.json" + (f", {a.out}.png" if drawn else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
