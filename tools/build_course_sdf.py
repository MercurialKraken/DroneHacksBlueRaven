#!/usr/bin/env python3
"""OFFLINE (laptop) builder: the published course mesh -> the signed distance fields in an .npz.

    python tools/build_course_sdf.py maps/course.yaml

Every course-specific fact comes from the YAML: the mesh file, the grid, which named pieces go
into which field, and the sanity points the result must satisfy. A different or updated course is
a different YAML, not different code. Needs numpy, scipy and pyyaml; the runtime loader
(nav/course_map.py) needs numpy only.

Method:
  1. Parse the OBJ by named group (`g` lines), fan-triangulating every face.
  2. Per piece, on its own bounding box of fine voxels (`oversample` per output cell):
       a. sample its triangles at a quarter of the fine pitch and mark every voxel a sample
          lands in: the surface shell;
       b. fill what the shell encloses (scipy binary_fill_holes). A solid wall or boulder fills;
          an open-ended tube's passage or a window opening touches the outside and stays free.
  3. Per field, take the union of its pieces and run two Euclidean distance transforms:
       free voxel      -> +distance to the nearest occupied voxel centre
       occupied voxel  -> -(distance to the nearest free voxel centre - 1)
     so 0 sits at the centre of the surface voxels, where the surface is on average. The error
     is at most half a fine voxel (0.8 cm for 5 cm cells oversampled 3x).
  4. Keep every `oversample`-th fine voxel: with an odd oversample their centres are exactly the
     output grid's centres.
  5. Check the YAML's sanity points through the runtime loader. A failing build is written to
     <output>.failed.npz for inspection, never over the good file, and exits 1.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nav.course_map import CourseMap  # noqa: E402

BUILDER_VERSION = 1


# ---------------------------------------------------------------------- mesh
def parse_obj_groups(path: str | Path) -> dict[str, np.ndarray]:
    """{group name: (M, 3, 3) triangles}. Faces before any `g` line go to "(ungrouped)"."""
    verts: list[list[float]] = []
    faces: dict[str, list[list[int]]] = {}
    group = "(ungrouped)"
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("g "):
                group = line[2:].strip() or "(ungrouped)"
            elif line.startswith("f "):
                n = len(verts)
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                idx = [i - 1 if i > 0 else n + i for i in idx]
                tris = faces.setdefault(group, [])
                tris.extend([idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1))
    V = np.asarray(verts, dtype=float)
    return {g: V[np.asarray(f, dtype=np.intp)] for g, f in faces.items() if f}


def select_groups(names: list[str], include: list[str], exclude: list[str]) -> list[str]:
    """Case-sensitive glob selection. An include pattern that matches nothing is an error: after
    a re-download it means a piece was renamed, and silently losing a wall is the worst outcome."""
    for pat in include:
        if not any(fnmatch.fnmatchcase(n, pat) for n in names):
            raise ValueError(f"include pattern {pat!r} matches no mesh group; groups: {sorted(names)}")
    return [n for n in names if any(fnmatch.fnmatchcase(n, p) for p in include)
            and not any(fnmatch.fnmatchcase(n, p) for p in exclude)]


def sample_triangles(tris: np.ndarray, step: float) -> np.ndarray:
    """Points on every triangle no more than `step` apart along two of its edges, plus its third
    edge and vertices. Triangles are bucketed by sample counts so the work is vectorized."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    u, v = b - a, c - a
    ns = np.maximum(1, np.ceil(np.linalg.norm(u, axis=1) / step)).astype(int)
    nt = np.maximum(1, np.ceil(np.linalg.norm(v, axis=1) / step)).astype(int)
    nbc = np.maximum(1, np.ceil(np.linalg.norm(c - b, axis=1) / step)).astype(int)
    out = [tris.reshape(-1, 3)]
    for key in np.unique(np.stack([ns, nt], axis=1), axis=0):
        sel = (ns == key[0]) & (nt == key[1])
        S, T = np.meshgrid(np.arange(key[0] + 1) / key[0], np.arange(key[1] + 1) / key[1], indexing="ij")
        keep = S + T <= 1.0 + 1e-9
        S, T = S[keep], T[keep]
        out.append((a[sel, None] + S[None, :, None] * u[sel, None] + T[None, :, None] * v[sel, None]).reshape(-1, 3))
    for n in np.unique(nbc):
        sel = nbc == n
        s = np.arange(n + 1) / n
        out.append((b[sel, None] + s[None, :, None] * (c - b)[sel, None]).reshape(-1, 3))
    return np.concatenate(out)


# ---------------------------------------------------------------------- grid
class Grid:
    """Output voxel centres lo..hi (inclusive) every `cell`; fine voxels `oversample` per cell."""

    def __init__(self, lo: Any, hi: Any, cell: float, oversample: int) -> None:
        if oversample < 1 or oversample % 2 == 0:
            raise ValueError("oversample must be a positive odd integer, so fine and output centres line up")
        self.lo = np.asarray(lo, dtype=float).reshape(3)
        hi = np.asarray(hi, dtype=float).reshape(3)
        self.cell = float(cell)
        self.k = int(oversample)
        self.shape = tuple(int(n) for n in np.rint((hi - self.lo) / self.cell).astype(int) + 1)
        self.fine = self.cell / self.k
        self.fine_lo = self.lo - (self.k // 2) * self.fine          # centre of fine voxel [0, 0, 0]
        self.fine_shape = tuple(n * self.k for n in self.shape)

    def fine_index(self, points: np.ndarray) -> np.ndarray:
        return np.rint((points - self.fine_lo) / self.fine).astype(np.intp)


def voxelize_piece(tris: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray] | None:
    """(index of the sub-box's first fine voxel, filled bool sub-box), or None if off the grid."""
    pts = sample_triangles(tris, grid.fine / 4.0)
    idx = grid.fine_index(pts)
    shape = np.array(grid.fine_shape)
    on = np.all((idx >= 0) & (idx < shape), axis=1)
    if not on.any():
        return None
    if not on.all():
        print(f"    warning: {np.count_nonzero(~on)} surface samples fall outside the grid", file=sys.stderr)
    idx = idx[on]
    lo = np.maximum(idx.min(axis=0) - 1, 0)
    hi = np.minimum(idx.max(axis=0) + 1, shape - 1)
    sub = np.zeros(tuple(hi - lo + 1), dtype=bool)
    sub[tuple((idx - lo).T)] = True
    return lo, ndimage.binary_fill_holes(sub)


def signed_distance(occ: np.ndarray, grid: Grid) -> np.ndarray:
    if not occ.any():
        raise ValueError("field has no occupied voxels")
    s = (slice(grid.k // 2, None, grid.k),) * 3
    d_out = ndimage.distance_transform_edt(~occ)[s].copy()
    d_in = ndimage.distance_transform_edt(occ)[s].copy()
    return (np.where(occ[s], 1.0 - d_in, d_out) * grid.fine).astype(np.float32)


# --------------------------------------------------------------------- build
def build(build_cfg: dict[str, Any], mesh_path: str | Path, log=print) -> CourseMap:
    t0 = time.perf_counter()
    groups = parse_obj_groups(mesh_path)
    log(f"mesh: {len(groups)} groups, {sum(len(t) for t in groups.values())} triangles")
    b = build_cfg["bounds"]
    grid = Grid(b["min"], b["max"], build_cfg["cell"], build_cfg.get("oversample", 3))
    log(f"grid: {grid.shape} cells of {grid.cell} m; fine {grid.fine_shape} of {grid.fine * 100:.2f} cm")

    selection = {name: select_groups(list(groups), spec.get("include", []), spec.get("exclude", []))
                 for name, spec in build_cfg["fields"].items()}
    unused = sorted(set(groups) - {g for sel in selection.values() for g in sel})
    if unused:
        log(f"in no field: {unused}")

    pieces: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    for g in sorted({g for sel in selection.values() for g in sel}):
        pieces[g] = voxelize_piece(groups[g], grid)
        if pieces[g] is None:
            log(f"    warning: {g!r} lies entirely outside the grid")
    log(f"voxelized {len(pieces)} pieces in {time.perf_counter() - t0:.1f} s")

    fields: dict[str, np.ndarray] = {}
    for name, sel in selection.items():
        occ = np.zeros(grid.fine_shape, dtype=bool)
        for g in sel:
            if pieces[g] is not None:
                lo, sub = pieces[g]
                occ[tuple(slice(l, l + n) for l, n in zip(lo, sub.shape))] |= sub
        fields[name] = signed_distance(occ, grid)
        log(f"field {name}: {len(sel)} pieces, {occ.mean() * 100:.2f}% occupied, "
            f"distance {fields[name].min():+.3f}..{fields[name].max():+.3f} m")
        del occ

    mesh_path = Path(mesh_path)
    meta = {
        "builder": "tools/build_course_sdf.py", "builder_version": BUILDER_VERSION,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mesh": mesh_path.name, "mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "cell": grid.cell, "oversample": grid.k, "bounds": {"min": grid.lo.tolist(), "max": b["max"]},
        "fields": {name: {"pieces": sel, **build_cfg["fields"][name]} for name, sel in selection.items()},
        "sign": "positive free, negative inside, ~0 on a surface; metres",
    }
    log(f"built in {time.perf_counter() - t0:.1f} s")
    return CourseMap(fields, grid.lo, grid.cell, meta)


def check_sanity(cmap: CourseMap, checks: list[dict[str, Any]]) -> list[tuple[bool, str]]:
    """Each check: {name, field, at: [x, y, z], min?, max?} on the signed distance in metres."""
    results = []
    for ch in checks:
        d = float(cmap.distance(np.asarray(ch["at"], dtype=float), ch.get("field", "planning"))[0])
        lo, hi = ch.get("min"), ch.get("max")
        ok = np.isfinite(d) and (lo is None or d >= lo) and (hi is None or d <= hi)
        want = " and ".join(s for s in (f">= {lo}" if lo is not None else "", f"<= {hi}" if hi is not None else "") if s)
        results.append((ok, f"{'ok  ' if ok else 'FAIL'} {ch['name']:<40s} {ch.get('field', 'planning'):<12s} "
                            f"{d:+.3f} m (want {want or 'finite'})"))
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", nargs="?", default=str(ROOT / "maps" / "course.yaml"))
    args = ap.parse_args(argv)
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    here = cfg_path.parent
    mesh = here / cfg["course"]["mesh"]
    out = here / cfg["build"]["output"]

    cmap = build(cfg["build"], mesh)
    cmap.meta["course"] = cfg["course"].get("name", "")
    results = check_sanity(cmap, cfg.get("sanity", []))
    for _, line in results:
        print(line)
    if all(ok for ok, _ in results):
        cmap.save(out)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return 0
    failed = out.with_suffix(".failed.npz")
    cmap.save(failed)
    print(f"SANITY FAILED: wrote {failed} for inspection; {out} left unchanged", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
