"""SIM ONLY (scratch): mppi_course_sim with the drone PLACED OFF THE MARK: its real start heading rotated by --yawerr
deg (+ = counter-clockwise, nose toward west) and its real start shifted by --dx,--dy m (map). The route and the fence
are built from that real take-off pose (as on the drone), the world/nets don't move. Prints the true distance to the
arena net (the 6.096 m square, +-3.048 m) on top of the usual summary.
usage: py -3 sim_start_err.py <course_dir> <route_sim.json> --errors=realfit --model=0.9,0.6 --caps=0.30,0.45
       --speed=0.10 --seed=1 --yawerr=5 --dx=0 --dy=0.15 [--qlat=4]"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, "tools")
import mppi_course_sim as M  # noqa: E402

arg = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)  # noqa: E731
yawerr, dx, dy = float(arg("yawerr", 0)), float(arg("dx", 0)), float(arg("dy", 0))
qlat = arg("qlat", None)
_orig = M.load_launch


def load_launch(course_dir, key):
    p, yaw = _orig(course_dir, key)
    return p + np.array([dx, dy]), yaw + math.radians(yawerr)


M.load_launch = load_launch
if qlat is not None:
    from nav import mppi as NM
    _init = NM.MppiParams.__init__

    def _init_q(self, *a, **k):
        _init(self, *a, **k)
        object.__setattr__(self, "q_lat", float(qlat))

    NM.MppiParams.__init__ = _init_q

rot = float(arg("rot", 0))          # deg: a WRONG odom->NED yaw offset: every velocity command is flown rotated by this
if rot:
    import dataclasses
    _step = M.LocalNav.step
    _c, _s = math.cos(math.radians(rot)), math.sin(math.radians(rot))

    def _step_rot(self, *a, **k):
        c = _step(self, *a, **k)
        vx, vy = _c * c.vx - _s * c.vy, _s * c.vx + _c * c.vy
        try:
            return dataclasses.replace(c, vx=vx, vy=vy)
        except TypeError:
            object.__setattr__(c, "vx", vx)
            object.__setattr__(c, "vy", vy)
            return c

    M.LocalNav.step = _step_rot

args = [a for a in sys.argv[1:] if not a.startswith("--")]
seed = int(arg("seed", 1))
preset = arg("errors", "realfit")
speed = float(arg("speed", 0.10))
model = tuple(float(v) for v in arg("model", "0.9,0.6").split(","))
caps = tuple(float(v) for v in arg("caps", "0.30,0.45").split(","))
import dataclasses as _dc
ERR = M.PRESETS[preset]
if arg("drift", None) is not None:
    ERR = _dc.replace(ERR, drift_frac=float(arg("drift", 0)), drift_dir_deg=float(arg("driftdir", -90)))
log = M.fly(args[0], args[1], err=ERR, seed=seed, speed=speed, model=model, caps=caps, keep_frames=True)
# the true trajectory: the first (N, >=2) float array in the log that moves
traj = np.asarray(log["track"])[:, :2]
H = 3.048
net = {"N": round(H - float(traj[:, 1].max()), 2), "E": round(H - float(traj[:, 0].max()), 2),
       "S": round(H + float(traj[:, 1].min()), 2), "W_after_1m": None}
far = np.hypot(*(traj - traj[0]).T) > 1.0                       # the west net is 0.53 m from the start mark itself
if far.any():
    net["W_after_1m"] = round(H + float(traj[far, 0].min()), 2)
keep = ("ended", "follow_s", "avg_speed", "min_clearance", "min_clearance_at", "touched", "land_err_to_last_wp")
print(json.dumps({"drift": arg("drift", None), "driftdir": arg("driftdir", None), "rot": rot, "yawerr": yawerr, "dx": dx, "dy": dy, "qlat": qlat, "errors": preset, "seed": seed,
                  "net_clear_m": net, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in log.items() if k in keep}}), flush=True)
