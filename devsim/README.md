# devsim: local dev simulator for refnav

Runs the real `refnav.core.Mission` (unchanged) against a kinematic PX4 and a ray-cast depth
camera in a world of boxes. Pure Python + numpy; a flight takes ~1 s on a laptop. It is for
developing routes and avoidance logic, not for qualifying anything.

```sh
uv run --with numpy --with matplotlib python -m devsim.run \
  --world devsim/worlds/example_corridor.json --route route.json \
  [--noise] [--drift 0.01] [--seed N] [--height 0.8] [--hover 2] [--plot out.png]
```

Exit code: 0 = "complete", 1 = any other result (blocked, refused, timeout), 2 = the true
position came within 0.25 m of a box (crash).

- `--route`: same format as `route.json`: `{"route": [[forward_m, left_m], ...], "params": {...}}`
  in the takeoff frame. `params` are `refnav.core.Params` fields. Params merge as on the aircraft:
  the route's, then `flight.yaml`'s, then `main.py`'s `HEIGHT_M`/`HOVER_S` (or `--height`/`--hover`).
  The depth border is scaled to this camera (`devsim/world.py` `DEPTH_BORDER_PX`).
- `--noise`: depth 1.5 % long (the sim's calibration error), noise growing with z^2,
  40 % pixel dropout, 1 % dropped frames.
- `--drift`: VIO error as a horizontal random walk in m/sqrt(s). PX4 controls on the estimate,
  so the true aircraft moves away from the plan.
- Sample routes: `devsim/routes/clear.json` (completes), `blocked.json` (stops at crate A),
  `hover.json` (empty route: take off, hover, land).

Worlds (`devsim/worlds/*.json`): boxes in PX4 local NED, `min`/`max` = `[n, e, up]` metres,
plus `start` `{n, e, yaw_deg}`. `example_corridor.json` is a PLACEHOLDER 10 x 10 m room; only
its start (0.31 m from the left wall, 0.50 m from the back) copies the real course.

Modelled: 15 Hz depth at 160x100 (the real 640x400 camera at 1/4 scale, fx 104.7, 74.7 deg
HFOV, real camera TF, 0.4-12 m range, ~33 ms latency), 20 Hz mission ticks, PX4 limits 1.5 m/s
and 1 m/s^2 horizontal, 1.0 up / 0.7 down m/s, 60 deg/s yaw, Offboard after 0.5 s of streaming.

Not modelled: attitude/tilt dynamics, wind, ground effect, stereo matching artefacts (edges,
textureless surfaces, false near matches), nets/poles/thin objects, estimator health faults,
VIO loss or resets, yaw drift, height error, PX4 failsafes, battery, real-time factor.
