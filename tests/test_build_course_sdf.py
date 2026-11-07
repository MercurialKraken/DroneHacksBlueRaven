"""tools/build_course_sdf.py on a tiny synthetic OBJ: a solid wall, an open-ended hollow tube, a wall
with a window and a thin floor slab. The builder is offline-only, so these tests skip themselves
when scipy or pyyaml is missing (as in a minimal flight container)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_course_sdf as B  # noqa: E402

AXES = {"x": (0, 1, 2), "y": (1, 0, 2), "z": (2, 0, 1)}   # (prism axis, u axis, v axis)


def frame_prism(axis: str, a0: float, a1: float, outer, inner=None):
    """A closed prism along `axis` from a0 to a1. Its cross-section is the rectangle `outer`
    (u0, u1, v0, v1), minus the rectangle `inner` if given (a hollow frame: a tube or a window).
    Returns (vertices, triangles) with outward-facing normals."""
    ax, au, av = AXES[axis]

    def rect(r):
        u0, u1, v0, v1 = r
        return [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]

    def vert(a, u, v):
        p = [0.0, 0.0, 0.0]
        p[ax], p[au], p[av] = a, u, v
        return p

    loops = [rect(outer)] + ([rect(inner)] if inner else [])
    V, F = [], []
    rings = []
    for loop in loops:
        r0 = len(V)
        V += [vert(a0, u, v) for u, v in loop] + [vert(a1, u, v) for u, v in loop]
        rings.append(r0)
    for li, r0 in enumerate(rings):                       # side walls of each loop
        for i in range(4):
            j = (i + 1) % 4
            q = [r0 + i, r0 + j, r0 + 4 + j, r0 + 4 + i]
            F += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] if li == 0 else [[q[0], q[2], q[1]], [q[0], q[3], q[2]]]
    if inner is None:                                     # plain box: two end caps
        r0 = rings[0]
        F += [[r0, r0 + 2, r0 + 1], [r0, r0 + 3, r0 + 2], [r0 + 4, r0 + 5, r0 + 6], [r0 + 4, r0 + 6, r0 + 7]]
    else:                                                 # annular end caps
        o, n = rings
        for end, flip in ((0, True), (4, False)):
            for i in range(4):
                j = (i + 1) % 4
                q = [o + end + i, o + end + j, n + end + j, n + end + i]
                F += [[q[0], q[2], q[1]], [q[0], q[3], q[2]]] if flip else [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    return np.array(V, dtype=float), np.array(F)


def write_obj(path: Path, pieces: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    lines, base = [], 1
    for name, (V, F) in pieces.items():
        lines.append(f"g {name}")
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in V]
        lines += [f"f {a + base} {b + base} {c + base}" for a, b, c in F]
        base += len(V)
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    obj = tmp_path_factory.mktemp("course") / "toy.obj"
    write_obj(obj, {
        "Wall A": frame_prism("x", 1.0, 1.1, (0.0, 2.0, 0.0, 1.5)),                         # x 1.0-1.1
        "Tube": frame_prism("y", 0.0, 1.0, (2.5, 3.5, 0.0, 1.0), (2.6, 3.4, 0.1, 0.9)),     # open ends
        "Window wall": frame_prism("x", 4.5, 4.6, (0.0, 2.0, 0.0, 1.5), (0.6, 1.4, 0.4, 1.0)),
        "Floor": frame_prism("z", -0.006, 0.0, (-0.2, 5.0, -0.2, 2.2)),
    })
    cfg = {
        "bounds": {"min": [-0.25, -0.25, -0.10], "max": [5.0, 2.25, 1.60]},
        "cell": 0.05, "oversample": 3,
        "fields": {"planning": {"include": ["*"]},
                   "localization": {"include": ["Wall*", "Tube", "Window*"]}},
    }
    return B.build(cfg, obj, log=lambda *_: None)


def d(m, p, field="planning"):
    return float(m.distance(np.asarray(p, dtype=float), field)[0])


def test_wall_face_and_interior(built):
    assert d(built, [0.9, 1.0, 0.75]) == pytest.approx(0.10, abs=0.012)     # 10 cm in front of the face
    assert d(built, [1.0, 1.0, 0.75]) == pytest.approx(0.0, abs=0.012)      # on the face
    assert d(built, [1.05, 1.0, 0.75]) <= 0.0                              # inside the solid wall


def test_open_tube_passage_stays_free(built):
    assert d(built, [3.0, 0.5, 0.5]) > 0.35                                # inner half-width is 0.4
    assert d(built, [2.55, 0.5, 0.5]) <= 0.01                              # inside the tube's shell


def test_window_stays_open(built):
    # Window: y 0.6-1.4, z 0.4-1.0, so the opening's centre is 0.3 m from its top and bottom edges.
    assert d(built, [4.55, 1.0, 0.7]) == pytest.approx(0.3, abs=0.03)
    assert d(built, [4.55, 0.3, 0.7]) <= 0.01                              # solid beside the window


def test_localization_field_leaves_out_the_floor(built):
    p = [0.5, 1.0, 0.2]
    assert d(built, p, "planning") == pytest.approx(0.2, abs=0.012)        # floor below
    assert d(built, p, "localization") == pytest.approx(0.5, abs=0.012)    # wall face at x = 1.0


def test_meta_records_the_build(built):
    assert built.meta["fields"]["localization"]["pieces"] == ["Wall A", "Tube", "Window wall"]
    assert len(built.meta["mesh_sha256"]) == 64 and built.meta["oversample"] == 3


def test_a_renamed_piece_fails_loudly():
    with pytest.raises(ValueError, match="matches no mesh group"):
        B.select_groups(["Wall A", "Tube"], ["Wall*", "Boulder*"], [])


def test_even_oversample_is_rejected():
    with pytest.raises(ValueError):
        B.Grid([0, 0, 0], [1, 1, 1], 0.05, 2)


def test_sanity_checks_report_failures(built):
    res = B.check_sanity(built, [
        {"name": "in front of wall", "at": [0.9, 1.0, 0.75], "min": 0.05},
        {"name": "inside wall", "at": [1.05, 1.0, 0.75], "min": 0.05},
        {"name": "off the grid", "at": [9.0, 9.0, 9.0], "max": 1.0},
    ])
    assert [ok for ok, _ in res] == [True, False, False]
