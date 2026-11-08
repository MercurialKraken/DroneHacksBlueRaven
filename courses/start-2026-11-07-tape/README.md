# Course `start-2026-11-07-tape`: the rebuilt start, from Darion's on-site tape

The course as rebuilt by the organizers on 2026-11-07, with the start measured by Darion on site (his FINAL map,
~19:48): Wall A (-2.743, 2.591) -> (-1.339, 2.591), Wall B (-1.747, 0.762) -> (-0.857, 1.715), cone (-1.431, 2.286);
the CP2 gap is 1.00 m (22 cm a side for the 0.56 m drone), centred at (-1.098, 2.153). Everything else is the
published v3 course minus maze dividers M1, M2, M3.

Same layout as hinson/real-drone's `courses/<name>/` folders. The files are byte-identical to chelsea/course-map
ea5071d's `maps/course.yaml`, `maps/course/`, `maps/course_sdf.npz` and `routes/route_v6_start2*.json`, where they
are generated: `course.yaml`'s `rebuilt_start:` numbers -> `tools/make_start_obj.py --all` (start mesh, SDF and its
checks, route v6 via `tools/make_route_v6.py`, clearance via `tools/check_route.py`).

| Route | What | Min straight-leg clearance |
|---|---|---|
| `routes/route_v6_start2.json` (`_sim`) | full course, real mark (sim spawn) | 0.398 m |
| `routes/route_v6_start2_stage1.json` (`_sim`) | lead-in, through the CP2 gap, land | 0.414 m |
| `routes/route_v7_cp2_simple.json` (`_sim`) | Darion's method: forward, 90 deg right through the CP2 gap, land | 0.414 m |

`route_v7_cp2_simple*.json` are written here directly by `tools/make_route_cp2_simple.py` (byte-identical to `routes/`).

For waypoints, sims and offline checks. Add `route.json -> routes/route_v6_start2_sim.json` as in the other folders.
