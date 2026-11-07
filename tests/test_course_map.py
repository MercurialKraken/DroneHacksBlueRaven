"""nav.course_map without the builder: analytic signed distance fields sampled on a grid.

numpy only, like the runtime loader. Nothing here knows the competition course: every world is
synthetic, so these tests also pin the loader's behaviour on a different course."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nav.course_map import CourseMap  # noqa: E402

CELL = 0.05
ORIGIN = np.array([-0.5, -0.5, -0.5])
SHAPE = (61, 41, 31)                      # centres span x -0.5..2.5, y -0.5..1.5, z -0.5..1.0


def centres() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = [ORIGIN[a] + CELL * np.arange(SHAPE[a]) for a in range(3)]
    return np.meshgrid(*axes, indexing="ij")


def make_map(**fields: np.ndarray) -> CourseMap:
    return CourseMap(fields, ORIGIN, CELL, meta={"test": True})


def wall_field() -> np.ndarray:
    """Solid half-space x >= 1.0: free (positive) for x < 1."""
    X, _, _ = centres()
    return 1.0 - X


def corridor_field() -> np.ndarray:
    """Solids y <= 0 and y >= 1: a 1 m corridor along x."""
    _, Y, _ = centres()
    return np.minimum(Y, 1.0 - Y)


def ball_field(c=(1.0, 0.5, 0.25), r=0.3) -> np.ndarray:
    X, Y, Z = centres()
    return np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) - r


def test_linear_field_is_exact_and_zero_on_the_face():
    m = make_map(planning=wall_field())
    rng = np.random.default_rng(1)
    p = rng.uniform([-0.4, -0.4, -0.4], [2.4, 1.4, 0.9], size=(500, 3))
    assert np.allclose(m.distance(p), 1.0 - p[:, 0], atol=1e-5)
    assert abs(m.distance(np.array([[1.0, 0.3, 0.2]]))[0]) < 1e-6        # on the wall face
    assert m.distance(np.array([[1.2, 0.3, 0.2]]))[0] < 0                # inside the solid


def test_gradient_points_away_from_the_wall():
    m = make_map(planning=wall_field())
    g = m.gradient(np.array([[0.3, 0.2, 0.1], [0.77, 0.9, 0.6]]))
    assert np.allclose(g, [[-1.0, 0.0, 0.0]] * 2, atol=1e-6)


def test_corridor_centre_is_half_the_width():
    m = make_map(planning=corridor_field())
    assert m.distance(np.array([[1.0, 0.5, 0.3]]))[0] == pytest.approx(0.5, abs=1e-6)


def test_corridor_gradients_point_to_the_middle():
    m = make_map(planning=corridor_field())
    d, g = m.query(np.array([[1.0, 0.2, 0.3], [1.0, 0.83, 0.3]]))
    assert d == pytest.approx([0.2, 0.17], abs=1e-6)
    assert np.allclose(g, [[0, 1, 0], [0, -1, 0]], atol=1e-6)


def test_ball_is_negative_inside_and_gradient_is_radial():
    c = np.array([1.0, 0.5, 0.25])
    m = make_map(planning=ball_field(tuple(c)))
    assert m.distance(c[None])[0] == pytest.approx(-0.3, abs=0.02)
    rng = np.random.default_rng(2)
    dirs = rng.normal(size=(200, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    p = c + dirs * rng.uniform(0.4, 0.6, size=(200, 1))
    p = p[np.all((p > ORIGIN + 0.01) & (p < ORIGIN + CELL * (np.array(SHAPE) - 1) - 0.01), axis=1)]
    d, g = m.query(p)
    assert np.allclose(d, np.linalg.norm(p - c, axis=1) - 0.3, atol=0.01)
    radial = (p - c) / np.linalg.norm(p - c, axis=1, keepdims=True)
    assert np.all(np.einsum("ij,ij->i", g, radial) > 0.99)
    assert np.allclose(np.linalg.norm(g, axis=1), 1.0)


def test_outside_the_grid_is_nan_and_the_edge_is_finite():
    m = make_map(planning=wall_field())
    hi = ORIGIN + CELL * (np.array(SHAPE) - 1)
    p = np.array([ORIGIN, hi, [hi[0] + 0.01, 0, 0], [0, ORIGIN[1] - 0.01, 0], [0, 0, np.nan]])
    d, g = m.query(p)
    assert np.isfinite(d[:2]).all() and np.isfinite(g[:2]).all()
    assert np.isnan(d[2:]).all() and np.isnan(g[2:]).all()


def test_fields_are_selected_by_name():
    m = make_map(planning=wall_field(), localization=corridor_field())
    p = np.array([[0.5, 0.25, 0.0]])
    assert m.distance(p, "planning")[0] == pytest.approx(0.5, abs=1e-6)
    assert m.distance(p, "localization")[0] == pytest.approx(0.25, abs=1e-6)
    with pytest.raises(KeyError):
        m.distance(p, "nope")


def test_save_and_load_round_trip(tmp_path):
    m = make_map(planning=wall_field(), localization=ball_field())
    path = tmp_path / "sdf.npz"
    m.save(path)
    back = CourseMap.load(path)
    assert set(back.fields) == {"planning", "localization"}
    assert back.cell == CELL and np.allclose(back.origin, ORIGIN) and back.shape == SHAPE
    assert back.meta == {"test": True}
    p = np.random.default_rng(3).uniform([-0.4, -0.4, -0.4], [2.4, 1.4, 0.9], size=(50, 3))
    for f in ("planning", "localization"):
        assert np.allclose(back.distance(p, f), m.distance(p, f))


def test_rejects_bad_input():
    m = make_map(planning=wall_field())
    with pytest.raises(ValueError):
        m.distance(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        CourseMap({}, ORIGIN, CELL)
    with pytest.raises(ValueError):
        CourseMap({"a": np.zeros((3, 3, 3)), "b": np.zeros((3, 3, 4))}, ORIGIN, CELL)
    with pytest.raises(ValueError):
        CourseMap({"origin": np.zeros((3, 3, 3))}, ORIGIN, CELL)        # clashes with a stored key
    with pytest.raises(ValueError):
        CourseMap({"a": np.zeros((3, 3, 3))}, ORIGIN, 0.0)


def test_single_point_is_accepted():
    m = make_map(planning=wall_field())
    assert m.distance(np.array([0.5, 0.0, 0.0])).shape == (1,)
