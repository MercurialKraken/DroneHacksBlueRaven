"""SIM (scratch, not a pytest file): the full MPPI pipeline (refnav Mission + CmdLink + MppiFollow, rendered depth,
the T4-fitted plant 0.9 s + 0.6 s) flown STRAIGHT THROUGH A WINDOW at a given height, with stereo-like depth errors
(range noise k*r^2, dropout, outliers) and VIO jitter.
The window: a full-height wall (1.73 m) across the path with a 0.914 m wide opening; in the opening a sill box
from the floor up to SILL. Above the sill it's open (the real opening reaches ~2 m > the 1.73 m wall).
usage (from the repo root): py -3 tests/window_sim.py HEIGHT SILL BELOW_DRONE_M|- SEED [CRUISE] [CLIMB_MAX]
BELOW_DRONE_M "-" keeps the code default; CLIMB_MAX > 0 turns on refnav's ext climb (no landing on BLOCKED/STUCK)."""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import test_mppi_follow as T  # noqa: E402
from nav import local_grid as LG  # noqa: E402
from sim_world import WALL, World  # noqa: E402

HEIGHT, SILL, SEED = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[4])
BELOW = None if sys.argv[3] == "-" else float(sys.argv[3])
CRUISE = float(sys.argv[5]) if len(sys.argv) > 5 else 0.25
CLIMB = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
DEPTH_K, DROPOUT, OUTLIERS, JITTER = 0.015, 0.65, 0.02, 0.005      # mppi_course_sim 'realistic' depth + VIO jitter
OPEN_W = 0.914                                                     # opening width (CAD)
WALL_AHEAD = 1.8                                                   # m north of the spawn

_init = LG.LocalGridParams.__init__


def _init_band(self, *a, **k):
    _init(self, *a, **k)
    if BELOW is not None:
        object.__setattr__(self, "below_drone_m", BELOW)


LG.LocalGridParams.__init__ = _init_band
ZLOG = []
_odom = T.Stack.odom


def _odom_log(self):
    r = _odom(self)
    ZLOG.append((float(r[0][0]), float(r[0][2])))                  # (forward, height) in odometry
    return r


T.Stack.odom = _odom_log
_MF = T.M.MppiFollow
T.M.MppiFollow = lambda v, tau, dead=0.0: _MF(CRUISE, tau, dead)  # noqa: E731  (flight() hard-codes 0.15)
T.LAUNCH_PARAMS = {**T.LAUNCH_PARAMS, "height_m": HEIGHT}
T.STAGE1 = {"route": [[1.0, 0.0], [3.2, 0.0]], "params": {**T.STAGE1.get("params", {}), "height_m": HEIGHT,
                                                         "ext_climb_max_m": CLIMB}}

_, start, yaw, _, _ = T.tape_course()
assert abs(math.degrees(yaw) - 90.0) < 1.0, yaw                    # the spawn faces north (+y): the wall runs along x
x0, y0 = start
yw = y0 + WALL_AHEAD
boxes = [(x0 - 3.0, x0 - OPEN_W / 2, yw, yw + 0.10, WALL),        # wall west of the opening
         (x0 + OPEN_W / 2, x0 + 3.0, yw, yw + 0.10, WALL),        # wall east of it
         (x0 - OPEN_W / 2, x0 + OPEN_W / 2, yw, yw + 0.10, SILL)]  # the sill: floor -> SILL in the opening
rng = np.random.default_rng(SEED)


class NoisyWorld(World):
    def depth_image(self, R, cam, K, w, h, max_range=5.0):
        z = super().depth_image(R, cam, K, w, h, max_range)
        z = z + rng.normal(0.0, 1.0, z.shape) * DEPTH_K * z ** 2
        out = rng.random(z.shape) < OUTLIERS
        z = np.where(out & np.isfinite(z), rng.uniform(0.2, max_range, z.shape), z)
        z[rng.random(z.shape) < DROPOUT] = np.nan
        return z


w = NoisyWorld(x_range=(x0 - 3.5, x0 + 3.5), y_range=(y0 - 1.5, y0 + 4.5), boxes=boxes)
m, sk, traj, states, _, mf = T.flight(world=w, plant=(0.9, 0.6), jitter_m=JITTER, limit_s=150.0)
Z = np.array(ZLOG)
zc = Z[np.argmin(np.abs(Z[:, 0] - (WALL_AHEAD + 0.05))), 1] if len(Z) and Z[:, 0].max() > WALL_AHEAD else None
through = bool(len(traj) and traj[:, 1].max() > yw + 0.10 + 0.3)
k = int(np.argmin(np.abs(traj[:, 1] - (yw + 0.05)))) if len(traj) else None
clr_walls = float(w.clearance(traj, min_height=HEIGHT - 0.10).min()) if len(traj) else float("nan")  # sill excluded
print(json.dumps({"climb_max": CLIMB, "z_at_wall": None if zc is None else round(float(zc), 2),
                  "max_z": round(float(Z[:, 1].max()), 2) if len(Z) else None, "height": HEIGHT, "sill": SILL, "below_drone_m": BELOW, 
                  "seed": SEED, "cruise": CRUISE, "result": m.result, "why": m.why, "through_window": through,
                  "x_at_wall": None if k is None else round(float(traj[k, 0] - x0), 3),
                  "min_clear_side_walls": round(clr_walls, 3), "r_drone": mf.mppi_p.r_drone,
                  "states_seen": sorted(set(states)), "max_y_past_start": round(float(traj[:, 1].max() - y0), 2) if len(traj) else None}))
