# Build a custom simulated sensor

`dronehacks_sim` is an explicitly **simulation-only** Python library. It supplies
the simulated rigid body's true pose and a static collision occupancy array.
Use it to implement a virtual sensor, compare your estimate with truth, or
prototype a sensor that you plan to build. It issues no flight commands and
does not feed truth to PX4, cuVSLAM or your existing mission automatically.

The simulator must be running. In the lab notebook, editor terminal or SSH:

```python
from dronehacks_sim import Simulation

with Simulation() as sim:
    pose = sim.pose()
    grid = sim.voxels
    print(pose.sim_time_ns, pose.position_enu_m, pose.orientation_xyzw)
    print(grid.occupied.shape, grid.resolution_m, grid.origin_enu_m)
    print(grid.is_occupied(pose.position_enu_m))
    # Ideal downward beam from the body origin; direction follows attitude.
    hit = grid.raycast(pose.position_enu_m, pose.body_to_world([0, 0, -1]), 3.)
    print(None if hit is None else hit.distance_m)
```

Run `python SIM_SENSOR_EXAMPLE.py` for a bounded example. It prints a virtual
range measurement with configurable noise. It never arms or moves the drone.
In a notebook you can import its `sample` function and use the returned value.

The package is pre-exposed through the lab companion's Python path in a worker
image containing this feature. In another environment install from the toolkit:
`python -m pip install /path/to/toolkit`. NumPy is the only runtime dependency.
For a local rig, pass `Simulation('/path/to/rig-output/sim-sensors')`, or set
`DRONEHACKS_SIM_DATA` to that directory. Importing the package starts nothing.

## Coordinates and time

- Positions and map axes are **world ENU**, in metres. `occupied[ix, iy, iz]`
  is a boolean array, not image row/column order. Origin is the lower corner.
  `grid.index(point)` and `grid.center(index)` convert coordinates.
- Orientation is Hamilton quaternion **XYZW**, rotating body **FLU** into
  world ENU. Angular velocity is body FLU rad/s; linear velocity is world ENU m/s.
  These are not PX4's NED/FRD frames.
- `pose.sim_time_ns` is the physics snapshot's simulation timestamp. It is not
  wall time, a camera exposure time or a request for interpolated historical truth.
  A query returns the latest complete physics sample; repeated calls may return
  the same sequence. There is no invented intermediate pose between ticks.
- The default maximum wall age is0.5s (configurable up to5s). On pause, stop,
  contention or missing data, `Unavailable` is raised. Reopen `Simulation` after
  Reset (`WorldChanged`). Retry a busy read rather than reuse it as a new sample.

## What the occupancy means

Default cells are5cm. A cell is occupied if it intersects static collision
geometry; closed mesh interiors are also conservatively filled. The whole
prepared course and enclosing room are covered. Windows and tunnels remain
open at the voxel resolution. Resolution can merge openings smaller than a cell.
An intersecting cell extends at most one voxel diagonal (8.7cm at5cm resolution)
beyond its source surface; its ray hit is a **voxel boundary**, not an exact
laser/mesh return. Do not grade centimetre accuracy using this discretized range.
Beam range error can be much larger at grazing angles: touching a nearby cell
corner can produce a return long before the exact ray intersects a surface.
Use the grid for occupancy/prototyping, not as a photorealistic lidar substitute.

This is collision occupancy, not visual material occupancy: the course's net
has a continuous collision envelope even though cameras see mesh gaps. The drone,
moving objects, PhysX contact skin and edits made after map generation are not
in this static array. Out-of-map coordinates/rays raise `OutsideMap`, rather
than reporting free space. Unsupported collision models disable the exporter
with an explicit `[sim-sensors] UNAVAILABLE` log; no partial map is called complete.

## Keep portable flight code separate

Custom sensor **models** may use this oracle. Your navigation/flight algorithm
should consume their output through a normal Python or ROS interface; record
the sensor noise, latency, dropout and frames you assumed. Replace the sensor
producer with a real driver when moving to hardware. The oracle itself has no
real-drone equivalent. An ideal ground-truth sensor does not prove the real
estimator or sensor will work. Do not inject an ideal fix into the normal VIO
pipeline or report an oracle-assisted run as normal camera-only performance.

Existing team files are preserved; new guides and this example are seeded only
when absent. A gateway-only deployment cannot add this feature to an old worker.
