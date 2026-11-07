"""Virtual downward range sensor. Simulation-only; never commands the drone."""
from __future__ import annotations

import argparse
import math
import time
from typing import TYPE_CHECKING

import numpy as np
from dronehacks_sim import Simulation, Unavailable, OutsideMap, WorldChanged

if TYPE_CHECKING:
    from typing import Any


def sample(sim: Simulation, rng: np.random.Generator, *, sigma_m: float = .02,
           max_range_m: float = 3.) -> dict[str, Any]:
    if not math.isfinite(sigma_m) or sigma_m < 0:
        raise ValueError("sigma_m must be finite and nonnegative")
    pose = sim.pose()
    direction = pose.body_to_world([0., 0., -1.])
    hit = sim.voxels.raycast(pose.position_enu_m, direction, max_range_m)
    if hit is None:
        return {"sim_time_ns": pose.sim_time_ns, "sequence": pose.sequence,
                "range_m": None, "status": "no_return", "simulation_only": True}
    return {"sim_time_ns": pose.sim_time_ns, "sequence": pose.sequence,
            "range_m": float(np.clip(hit.distance_m + rng.normal(0., sigma_m), 0., max_range_m)),
            "status": "ok", "simulation_only": True,
            "model": "voxel ray + Gaussian range error; no reflectance/dropout model"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", help="local rig sensor directory; cloud default otherwise")
    parser.add_argument("--seconds", type=float, default=10.)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 3600:
        parser.error("seconds must be positive and at most3600")
    rng = np.random.default_rng(42)
    with Simulation(args.directory) as sim:
        until = time.monotonic() + args.seconds
        while time.monotonic() < until:
            try:
                print(sample(sim, rng), flush=True)
            except WorldChanged:
                print("World reset: reopen the sensor model.")
                break
            except (Unavailable, OutsideMap) as error:
                print({"status": "unavailable", "reason": str(error)}, flush=True)
            time.sleep(.1)


if __name__ == "__main__":
    main()
