"""tools/solve_launch.py: the launch pose from two tape measurements taken from the drone on its start position."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import solve_launch as S  # noqa: E402
from make_start_obj import COURSE, start_geometry  # noqa: E402

TEXT = COURSE.read_text(encoding="utf-8")
CFG = yaml.safe_load(TEXT)
ENDS = np.array([start_geometry(CFG)["A"][1], start_geometry(CFG)["B"][1]])


def _tape(x0: float, y0: float, yaw_deg: float) -> np.ndarray:
    """What the tape reads (forward, left from the drone) for each gap end, from a given true start."""
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    d = ENDS - [x0, y0]
    return np.c_[d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c]


@pytest.mark.parametrize("pose", [(-2.515, 0.914, 90.0), (-2.475, 0.884, 88.0), (-2.56, 0.95, 93.5)])
def test_two_taped_gap_ends_give_the_start_position_and_heading(pose):
    x, y, yaw = S.solve(_tape(*pose), ENDS)
    assert (x, y) == pytest.approx(pose[:2], abs=1e-6) and yaw == pytest.approx(pose[2], abs=1e-6)


def test_writing_the_launch_changes_only_the_launch_entry():
    new = yaml.safe_load(S.write_launch(TEXT, -2.475, 0.884, 88.0, "test"))
    assert {k: new["launch"][k] for k in ("x", "y", "yaw_deg", "confirmed")} == \
        {"x": -2.475, "y": 0.884, "yaw_deg": 88.0, "confirmed": True}
    assert new["launch"]["base_link_z"] == CFG["launch"]["base_link_z"]
    assert {k: v for k, v in new.items() if k != "launch"} == {k: v for k, v in CFG.items() if k != "launch"}
    twice = S.write_launch(S.write_launch(TEXT, 1.0, 2.0, 3.0, "a"), -2.475, 0.884, 88.0, "b")
    assert twice.count("# taped:") == 1 and "# taped: b" in twice
