"""Signed distance fields of the published course, in the `map` frame. Runtime: numpy only.

    cm = CourseMap.load("maps/course_sdf.npz")
    d = cm.distance(points, field="planning")        # (N,3) metres -> (N,) signed metres
    g = cm.gradient(points, field="planning")        # (N,3) -> (N,3) unit vectors
    d, g = cm.query(points, field="localization")   # both at once (one lookup)

Frame `map`: the published course file's coordinates, metres, Z up. For the DroneHacks 2026 OBJ
that is x east, y north, origin at the floor centre (see maps/README.md).

Sign: positive in free space, negative inside an object, about 0 on a surface. The gradient is
the direction of increasing distance, i.e. AWAY from the nearest surface. Midway between two
surfaces (a corridor's centre line) it is unreliable and can be short or zero; it is returned
normalized where it is not zero.

Values are trilinear between voxel centres. Voxel [i, j, k] is centred at origin + (i, j, k) *
cell. Anything outside the box spanned by the voxel centres, or a non-finite point, returns nan
for distance and gradient alike: the caller decides (the planner treats nan as blocked,
localization ignores the point).

The file is built offline by tools/build_course_sdf.py from maps/course.yaml.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_RESERVED = frozenset({"field_names", "origin", "cell", "meta_json"})
_O = np.arange(2)


class CourseMap:
    def __init__(self, fields: dict[str, np.ndarray], origin: Any, cell: float,
                 meta: dict[str, Any] | None = None) -> None:
        if not fields:
            raise ValueError("a CourseMap needs at least one field")
        bad = _RESERVED.intersection(fields)
        if bad:
            raise ValueError(f"field names clash with stored keys: {sorted(bad)}")
        shapes = {np.shape(f) for f in fields.values()}
        if len(shapes) != 1:
            raise ValueError(f"all fields must share one grid shape, got {sorted(shapes)}")
        shape = shapes.pop()
        if len(shape) != 3 or min(shape) < 2:
            raise ValueError(f"fields must be 3D with at least 2 voxels per axis, got {shape}")
        cell = float(cell)
        if not (np.isfinite(cell) and cell > 0):
            raise ValueError(f"cell must be finite and positive, got {cell}")
        origin = np.asarray(origin, dtype=float).reshape(3)
        if not np.isfinite(origin).all():
            raise ValueError("origin must be finite")
        self.fields: dict[str, np.ndarray] = {
            str(k): np.ascontiguousarray(v, dtype=np.float32) for k, v in fields.items()}
        self.shape: tuple[int, int, int] = (int(shape[0]), int(shape[1]), int(shape[2]))
        self.origin: np.ndarray = origin
        self.cell: float = cell
        self.meta: dict[str, Any] = dict(meta or {})
        self._hi = np.array(self.shape, dtype=float) - 1.0

    # ------------------------------------------------------------------ files
    @classmethod
    def load(cls, path: str | Path) -> CourseMap:
        with np.load(path, allow_pickle=False) as z:
            names = [str(n) for n in z["field_names"]]
            fields = {n: z[n] for n in names}
            meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
            return cls(fields, z["origin"], float(z["cell"]), meta)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, field_names=np.array(list(self.fields)), origin=self.origin,
                            cell=np.float64(self.cell), meta_json=np.array(json.dumps(self.meta)),
                            **self.fields)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(lowest, highest) voxel centre: the box where queries return numbers."""
        return self.origin.copy(), self.origin + self._hi * self.cell

    # ---------------------------------------------------------------- lookups
    def distance(self, points: Any, field: str = "planning") -> np.ndarray:
        corners, w, _, inside = self._lookup(points, field)
        d = np.einsum("nabc,na,nb,nc->n", corners, *w)
        d[~inside] = np.nan
        return d

    def gradient(self, points: Any, field: str = "planning") -> np.ndarray:
        return self.query(points, field)[1]

    def query(self, points: Any, field: str = "planning") -> tuple[np.ndarray, np.ndarray]:
        """Signed distance (N,) and unit gradient (N,3) from one lookup."""
        corners, (wx, wy, wz), dw, inside = self._lookup(points, field)
        d = np.einsum("nabc,na,nb,nc->n", corners, wx, wy, wz)
        g = np.stack([np.einsum("nabc,a,nb,nc->n", corners, dw, wy, wz),
                      np.einsum("nabc,na,b,nc->n", corners, wx, dw, wz),
                      np.einsum("nabc,na,nb,c->n", corners, wx, wy, dw)], axis=1) / self.cell
        norm = np.linalg.norm(g, axis=1, keepdims=True)
        g = np.divide(g, norm, out=np.zeros_like(g), where=norm > 1e-9)
        d[~inside] = np.nan
        g[~inside] = np.nan
        return d, g

    def _lookup(self, points: Any, field: str):
        try:
            F = self.fields[field]
        except KeyError:
            raise KeyError(f"unknown field {field!r}; this map has {sorted(self.fields)}") from None
        p = np.atleast_2d(np.asarray(points, dtype=float))
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {np.shape(points)}")
        u = (p - self.origin) / self.cell
        finite = np.isfinite(u).all(axis=1)
        inside = finite & np.all((u >= -1e-9) & (u <= self._hi + 1e-9), axis=1)
        u = np.clip(np.where(finite[:, None], u, 0.0), 0.0, self._hi)
        i0 = np.minimum(np.floor(u), self._hi - 1).astype(np.intp)
        f = u - i0
        x, y, z = (i0[:, a, None, None, None] for a in range(3))
        corners = F[x + _O[None, :, None, None], y + _O[None, None, :, None], z + _O[None, None, None, :]]
        w = tuple(np.stack([1.0 - f[:, a], f[:, a]], axis=1) for a in range(3))
        return corners.astype(float), w, np.array([-1.0, 1.0]), inside
