# The published course (layout v3), frozen for regression tests

`course.yaml` and `course_sdf.npz` exactly as in commit f1c3ce1, before the organizers rebuilt the start on
2026-11-07. The planner and refnav map-band tests pin their numbers to this course; `maps/` now holds the
rebuilt start, whose 0.75 m CP2 gap is below the planner's clearance rule (that route is placed by
`tools/make_route_v6.py`). Not used in flight.
