"""courses/<name>/ holds everything one course is (map, meshes, distance field, routes); courses/ACTIVE names one."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from nav.course_map import COURSES, course_dir

NAMES = sorted(p.name for p in COURSES.iterdir() if p.is_dir())


def test_active_names_a_course():
    assert course_dir().name in NAMES and course_dir("v3-published") == COURSES / "v3-published"


@pytest.mark.parametrize("name", NAMES)
def test_a_course_is_complete(name):
    d = course_dir(name)
    cfg = yaml.safe_load((d / "course.yaml").read_text())
    meshes = cfg["course"]["mesh"]
    assert all((d / m).is_file() for m in ([meshes] if isinstance(meshes, str) else meshes))
    assert (d / cfg["build"]["output"]).is_file()
    assert json.loads((d / "route.json").read_text())["route"]
    assert list((d / "routes").glob("*.json"))


def test_the_drone_launch_default_route_is_in_the_active_course():
    text = (Path(__file__).resolve().parents[1] / "skybeta" / "blueraven_skybeta" / "launch" / "refnav.launch.py").read_text()
    route = re.search(r"DeclareLaunchArgument\('route', default_value='([^']+)'", text).group(1)
    assert (course_dir() / "routes" / f"{route}.json").is_file()
