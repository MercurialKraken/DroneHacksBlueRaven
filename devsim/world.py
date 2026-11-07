"""A world of axis-aligned boxes in PX4 local NED (n, e) with heights measured up from the floor,
and a ray-cast forward depth camera (same conventions as tests/test_refnav_core.py::render_depth)."""
from __future__ import annotations

import json
import math

import numpy as np

# The real OAK-D depth is 640x400 with fx ~419 (74.7 deg HFOV); this is the same camera at 1/4 scale.
W, H, FX = 160, 100, 418.9 / 4
K = (FX, FX, W / 2, H / 2)
DEPTH_BORDER_PX = 6        # the real camera's 24 px stereo border at this scale
R_BODY_CAM = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # FLU <- optical
T_BODY_CAM = np.array([0.116, 0.0, 0.029])                                     # TF base_link -> oak_left
FLOOR_D = 0.066            # floor below the body origin of a landed aircraft (sim truth)
MIN_RANGE, MAX_RANGE = 0.4, 12.0

_vs, _us = np.mgrid[0:H, 0:W].astype(np.float64)
_RAYS_FRD = (np.stack([(_us - K[2]) / K[0], (_vs - K[3]) / K[1], np.ones_like(_us)], -1) @ R_BODY_CAM.T) * [1, -1, -1]


def q_yaw(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


class World:
    def __init__(self, doc):
        self.name = doc.get("name", "world")
        self.names = [b.get("name", "") for b in doc["boxes"]]
        # each box as NED bounds: (n0, e0, d0), (n1, e1, d1) with d = -up
        self.lo = np.array([[b["min"][0], b["min"][1], -b["max"][2]] for b in doc["boxes"]], float)
        self.hi = np.array([[b["max"][0], b["max"][1], -b["min"][2]] for b in doc["boxes"]], float)
        s = doc["start"]
        self.start = (float(s["n"]), float(s["e"]), math.radians(float(s.get("yaw_deg", 0.0))))

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def render(self, pos_ned, yaw):
        """Optical-z depth (m, NaN = no return) from the forward camera of a level aircraft."""
        c, s = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        d = _RAYS_FRD @ Rz.T
        o = np.asarray(pos_ned, float) + Rz @ (T_BODY_CAM * [1, -1, -1])
        with np.errstate(divide="ignore", invalid="ignore"):
            tf = (FLOOR_D - o[2]) / d[..., 2]
            best = np.where(tf > 0, tf, np.inf)
            inv = 1.0 / d
            for lo, hi in zip(self.lo, self.hi):
                t1, t2 = (lo - o) * inv, (hi - o) * inv
                tmin = np.nanmax(np.minimum(t1, t2), -1)
                tmax = np.nanmin(np.maximum(t1, t2), -1)
                hit = (tmax >= tmin) & (tmin > 0)
                best = np.where(hit, np.minimum(best, tmin), best)
        best[(best < MIN_RANGE) | (best > MAX_RANGE)] = np.nan
        return best.astype(np.float32)

    def collision(self, pos_ned, radius):
        """Name of a box within radius of the point, else None."""
        p = np.asarray(pos_ned, float)
        gap = np.linalg.norm(np.maximum(np.maximum(self.lo - p, p - self.hi), 0.0), axis=1)
        i = int(np.argmin(gap))
        return (self.names[i] or f"box {i}") if gap[i] < radius else None
