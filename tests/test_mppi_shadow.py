"""MPPI shadow mode (blueraven_skybeta/mppi_shadow.py) can never command the drone: no publisher of its own, and the
only publisher it builds (nav.mppi_ros.MppiPublisher) publishes under /blueraven/mppi."""
from __future__ import annotations

import ast
from pathlib import Path

from nav import mppi_ros

ROOT = Path(__file__).resolve().parents[1]
SHADOW = ROOT / "skybeta" / "blueraven_skybeta" / "blueraven_skybeta" / "mppi_shadow.py"


def _publisher_topics(path: Path) -> list[ast.expr]:
    return [c.args[1] for c in ast.walk(ast.parse(path.read_text()))
            if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "create_publisher"]


def test_the_shadow_node_creates_no_publisher_and_never_names_a_skybeta_topic():
    src = SHADOW.read_text()
    assert not _publisher_topics(SHADOW) and "/skybeta/setpoint" not in src and "TwistStamped" not in src
    assert "MppiPublisher(" in src


def test_mppi_publisher_topics_are_all_under_blueraven_mppi():
    topics = _publisher_topics(Path(mppi_ros.__file__))
    assert mppi_ros.PREFIX == "/blueraven/mppi" and len(topics) == 6
    for t in topics:
        assert isinstance(t, ast.JoinedStr) and isinstance(t.values[0], ast.FormattedValue)
        assert t.values[0].value.id == "PREFIX"


def test_the_launch_runs_the_shadow_only_when_asked_and_at_low_priority():
    src = (ROOT / "skybeta" / "blueraven_skybeta" / "launch" / "refnav.launch.py").read_text()
    assert "DeclareLaunchArgument('mppi_shadow', default_value='false'" in src
    assert "executable='mppi_shadow'" in src and "prefix='nice -n 10'" in src
