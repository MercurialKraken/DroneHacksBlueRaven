"""tools/make_start_obj.py: the start rebuilt on 2026-11-07, from course.yaml's numbers (tape measurements -> map)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_start_obj as M  # noqa: E402
from build_course_sdf import parse_obj_groups  # noqa: E402

CFG = yaml.safe_load(M.COURSE.read_text(encoding="utf-8"))
LAUNCH = CFG["launch"]


def _to_mark(P) -> list:
    """Course points -> [forward, left] from the launch mark (the inverse of frame: mark)."""
    a = math.radians(LAUNCH["yaw_deg"])
    d = np.asarray(P, dtype=float).reshape(-1, 2) - [LAUNCH["x"], LAUNCH["y"]]
    return np.c_[d[:, 0] * math.cos(a) + d[:, 1] * math.sin(a), -d[:, 0] * math.sin(a) + d[:, 1] * math.cos(a)].tolist()


def test_the_committed_start_mesh_is_what_the_numbers_make(tmp_path):
    p = tmp_path / "start.obj"
    p.write_text(M.obj_text(M.start_geometry(CFG)), encoding="utf-8")
    made, committed = parse_obj_groups(p), parse_obj_groups(M.START_OBJ)
    assert set(made) == set(committed) == {"Start wall A", "Start wall B diagonal", "Boulder 6 CP2 cone"}
    for k in made:
        assert np.allclose(made[k], committed[k], atol=1e-4), k


def test_mark_frame_points_are_forward_and_left_of_the_launch_mark():
    P = M.to_course([[1.0, 0.0], [0.0, 1.0]], "mark", LAUNCH)          # the mark faces north (yaw 90)
    assert np.allclose(P[0], [LAUNCH["x"], LAUNCH["y"] + 1.0])          # 1 m forward = 1 m north
    assert np.allclose(P[1], [LAUNCH["x"] - 1.0, LAUNCH["y"]])          # 1 m left = 1 m west
    with pytest.raises(ValueError, match="frame"):
        M.to_course([[0.0, 0.0]], "feet", LAUNCH)


def test_the_same_tape_numbers_in_either_frame_give_the_same_start():
    g0 = M.start_geometry(CFG)
    rs = {**CFG["rebuilt_start"], "frame": "mark", "wall_a": _to_mark(g0["A"]), "wall_b": _to_mark(g0["B"]),
          "cone": _to_mark(g0["cone"])[0]}
    assert M.obj_text(M.start_geometry({**CFG, "rebuilt_start": rs})) == M.obj_text(g0)


def test_cp2_follows_the_wall_ends_and_points_away_from_the_launch_mark():
    text = M.COURSE.read_text(encoding="utf-8")
    assert M.sync_cp2(text, M.start_geometry(CFG)) == text              # in sync as committed
    rs = {**CFG["rebuilt_start"], "wall_a": [[-2.743, 2.700], [-1.200, 2.700]]}   # a taped Wall A
    new = yaml.safe_load(M.sync_cp2(text, M.start_geometry({**CFG, "rebuilt_start": rs})))
    cp2 = next(c for c in new["checkpoints"] if c["id"] == "CP2")
    b_end = [round(v, 3) for v in M.start_geometry(CFG)["B"][1].tolist()]     # Wall B's gap end, unchanged
    assert cp2["line"] == [[-1.2, 2.7], b_end]
    n, line = np.asarray(cp2["direction"]), np.asarray(cp2["line"])
    assert abs(np.linalg.norm(n) - 1.0) < 2e-3 and abs(n @ (line[1] - line[0])) < 2e-3   # the gap's unit normal
    assert n @ (line.mean(axis=0) - [LAUNCH["x"], LAUNCH["y"]]) > 0                       # away from the mark
    old = yaml.safe_load(text)
    assert {k: v for k, v in new.items() if k != "checkpoints"} == {k: v for k, v in old.items() if k != "checkpoints"}
