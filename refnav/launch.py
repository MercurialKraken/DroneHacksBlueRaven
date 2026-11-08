"""Ordinary ROS launch arguments shared by hardware and simulation projects."""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


def params_files(project: str | os.PathLike[str]) -> list[str]:
    """flight.yaml, then the deployment's file (it sets the clock): every node of the flight reads the same two."""
    args = ["--params-file", str(Path(project).resolve() / "flight.yaml")]
    deployment = Path(os.environ.get("DRONE_PARAMS_FILE", "/etc/flyx/mission.yaml"))
    if deployment.is_file():
        args += ["--params-file", str(deployment)]
    return args


def flight_params(project: str | os.PathLike[str]) -> dict[str, Any]:
    """flight.yaml's parameters from its flat `key: scalar` lines, read without pyyaml so devsim needs only numpy."""
    out: dict[str, Any] = {}
    for line in (Path(project) / "flight.yaml").read_text(encoding="utf-8").splitlines():
        key, sep, val = line.split("#", 1)[0].strip().partition(":")
        val = val.strip().strip("'\"")
        if not sep or not val:
            continue
        out[key] = {"true": True, "false": False}.get(val, val)
        for cast in (int, float):
            try:
                out[key] = cast(val)
                break
            except ValueError:
                pass
    return out


def arguments(project: str | os.PathLike[str], *, height_m: float, hover_s: float,
              ros_args: Sequence[str] = ()) -> list[str]:
    project = Path(project).resolve()
    args = ["refnav", "--ros-args", *params_files(project)]
    course = (project / "courses" / "ACTIVE").read_text(encoding="utf-8").strip()    # the course's default route
    # ROS infers CLI parameter types from their text: Python's 4 must reach
    # our DOUBLE declarations as 4.0, just like a fractional height does.
    return args + ["-p", f"route_file:={project / 'courses' / course / 'route.json'}", "-p", f"height_m:={float(height_m)}",
                   "-p", f"hover_s:={float(hover_s)}", *ros_args]
