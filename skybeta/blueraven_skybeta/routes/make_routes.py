#!/usr/bin/env python3
"""Regenerate the hardcoded route files (mission "route") from the startup planner.

    python3 skybeta/blueraven_skybeta/routes/make_routes.py          # from the repository root

One planner run per launch entry (course.yaml 'launch' = the real mark -> *_real.json, 'launch_sim'
= the sim spawn -> *_sim.json), with the node's planner settings (0.7 m, CP4 gate_b, keep_centre_m
0.02). Each file is a prefix of that route, cut at the planner's checkpoint label, so the drone
lands right after that checkpoint:

    run1_cp1       to CP1 (the first corridor's end)   run2a_cp2  to CP2 (Gate A)
    run1t_cp1_turn to CP1, turn, one corner east       run2b_cp3  to CP3
                                                       run2c_full through CP5 (the pad)

The files are data: correct them by hand on site (see README.md). This script overwrites them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from nav.route_planner import gate_problems, plan_from_files  # noqa: E402

HEIGHT_M, CP4, KEEP_CENTRE_M = 0.7, "gate_b", 0.02
CUTS = {"run1_cp1": ("CP1", 0), "run1t_cp1_turn": ("CP1", 1),    # (last checkpoint, corners after it)
        "run2a_cp2": ("CP2", 0), "run2b_cp3": ("CP3", 0), "run2c_full": ("CP5", 0)}
KEYS = {"real": "launch", "sim": "launch_sim"}


def main() -> int:
    for suffix, key in KEYS.items():
        plan = plan_from_files(ROOT / "maps" / "course_sdf.npz", ROOT / "maps" / "course.yaml", key, HEIGHT_M, CP4,
                               keep_centre_m=KEEP_CENTRE_M)
        if gate_problems(plan):
            print(f"{key}: planner refused: {gate_problems(plan)}")
            return 1
        for name, (label, extra) in CUTS.items():
            end = plan.labels.index(label) + 1 + extra
            out = HERE / f"{name}_{suffix}.json"
            to = label + (f" + {extra} corner{'s' * (extra > 1)}" if extra else "")
            src = (f"nav.route_planner {key} {HEIGHT_M} m CP4 {CP4} keep_centre_m {KEEP_CENTRE_M}, "
                   f"digest {plan.digest}, corners 1-{end} of {len(plan.route)} (to {to})")
            pts = ",\n".join(f"  [{f:.3f}, {l:.3f}]" for f, l in plan.route[:end])   # one corner per line
            text = (f'{{\n "from": {json.dumps(src)},\n "frame": "metres from the take-off pose: [forward, left]",\n'
                    f' "route": [\n{pts}\n ]\n}}\n')
            json.loads(text)                                                      # still valid JSON
            out.write_text(text, encoding="utf-8")
            print(f"wrote {out.name}: {end} corners, ends at {plan.route[end - 1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
