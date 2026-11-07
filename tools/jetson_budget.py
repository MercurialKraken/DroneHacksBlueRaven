"""Summarise Jetson tegrastats logs: CPU per core, GPU, RAM and power. Pure Python + numpy.

    python3 tools/jetson_budget.py idle.log depth.log [more.log ...]

Each log is `tegrastats --interval 500 --logfile X` output (one line per sample). Prints one column per log,
so "idle" and "with the depth node" sit side by side. Capture with tools/jetson_budget.sh.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

_RAM = re.compile(r"\bRAM (\d+)/(\d+)MB")
_SWAP = re.compile(r"\bSWAP (\d+)/(\d+)MB")
_CPU = re.compile(r"\bCPU \[([^\]]*)\]")
_GPU = re.compile(r"\bGR3D_FREQ (\d+)%")
_RAIL = re.compile(r"\b(VDD_[A-Z0-9_]+|POM_5V_IN|VIN_SYS_5V0) (\d+)mW(?:/(\d+)mW)?")
_TEMP = re.compile(r"\b(cpu|gpu|tj|soc\d?)@(-?[\d.]+)C")


def parse(line: str) -> dict | None:
    """One tegrastats line -> dict, or None when it isn't one."""
    m = _RAM.search(line)
    if not m:
        return None
    out: dict = {"ram_mb": int(m.group(1)), "ram_total_mb": int(m.group(2))}
    if (s := _SWAP.search(line)):
        out["swap_mb"] = int(s.group(1))
    cores = []
    if (c := _CPU.search(line)):
        for tok in c.group(1).split(","):
            tok = tok.strip()
            mm = re.match(r"(\d+)%(?:@(\d+))?", tok)
            cores.append(None if not mm else (int(mm.group(1)), int(mm.group(2)) if mm.group(2) else None))
    out["cpu"] = cores                                     # None = core offline
    if (g := _GPU.search(line)):
        out["gpu_pct"] = int(g.group(1))
    out["power_mw"] = {k: int(v) for k, v, _ in _RAIL.findall(line)}
    out["temp_c"] = {k: float(v) for k, v in _TEMP.findall(line)}
    return out


def summarize(lines) -> dict:
    S = [s for s in (parse(l) for l in lines) if s]
    if not S:
        raise ValueError("no tegrastats samples")
    n_cores = max(len(s["cpu"]) for s in S)
    per_core = []
    for i in range(n_cores):
        v = [s["cpu"][i][0] for s in S if i < len(s["cpu"]) and s["cpu"][i] is not None]
        per_core.append((float(np.mean(v)), float(np.percentile(v, 95))) if v else None)
    online = [c for c in per_core if c is not None]
    allcpu = [np.mean([c[0] for c in s["cpu"] if c is not None]) for s in S if any(c is not None for c in s["cpu"])]
    gpu = [s["gpu_pct"] for s in S if "gpu_pct" in s]
    rails = sorted({k for s in S for k in s["power_mw"]})
    return {
        "samples": len(S),
        "cpu_per_core": per_core,
        "cpu_mean": float(np.mean(allcpu)) if allcpu else None,
        "cpu_p95": float(np.percentile(allcpu, 95)) if allcpu else None,
        "cores_online": len(online),
        "gpu_mean": float(np.mean(gpu)) if gpu else None,
        "gpu_p95": float(np.percentile(gpu, 95)) if gpu else None,
        "ram_mb_mean": float(np.mean([s["ram_mb"] for s in S])),
        "ram_mb_max": max(s["ram_mb"] for s in S),
        "ram_total_mb": S[0]["ram_total_mb"],
        "power_mw_mean": {k: float(np.mean([s["power_mw"][k] for s in S if k in s["power_mw"]])) for k in rails},
        "temp_c_max": {k: max(s["temp_c"][k] for s in S if k in s["temp_c"]) for k in sorted({k for s in S for k in s["temp_c"]})},
    }


def table(named: dict[str, dict]) -> str:
    names = list(named)
    w = max(12, *(len(n) for n in names)) + 2
    rows = [("samples", lambda s: f"{s['samples']}")]
    n_cores = max(len(s["cpu_per_core"]) for s in named.values())
    for i in range(n_cores):
        rows.append((f"cpu{i} % mean/p95", lambda s, i=i: "off" if i >= len(s["cpu_per_core"]) or s["cpu_per_core"][i] is None
                     else f"{s['cpu_per_core'][i][0]:.0f}/{s['cpu_per_core'][i][1]:.0f}"))
    rows += [("cpu all % mean/p95", lambda s: "-" if s["cpu_mean"] is None else f"{s['cpu_mean']:.0f}/{s['cpu_p95']:.0f}"),
             ("gpu % mean/p95", lambda s: "-" if s["gpu_mean"] is None else f"{s['gpu_mean']:.0f}/{s['gpu_p95']:.0f}"),
             ("ram MB mean/max", lambda s: f"{s['ram_mb_mean']:.0f}/{s['ram_mb_max']} of {s['ram_total_mb']}")]
    for k in sorted({k for s in named.values() for k in s["power_mw_mean"]}):
        rows.append((f"{k} mW", lambda s, k=k: f"{s['power_mw_mean'][k]:.0f}" if k in s["power_mw_mean"] else "-"))
    for k in sorted({k for s in named.values() for k in s["temp_c_max"]}):
        rows.append((f"{k} max C", lambda s, k=k: f"{s['temp_c_max'][k]:.1f}" if k in s["temp_c_max"] else "-"))
    out = [f"{'':22s}" + "".join(f"{n:>{w}s}" for n in names)]
    for label, f in rows:
        out.append(f"{label:22s}" + "".join(f"{f(named[n]):>{w}s}" for n in names))
    return "\n".join(out)


def main(argv=None) -> None:
    paths = argv or sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    named = {Path(p).stem: summarize(Path(p).read_text(errors="replace").splitlines()) for p in paths}
    print(table(named))


if __name__ == "__main__":
    main()
