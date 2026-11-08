"""Your companion program's entrypoint. The actual flight code is here in refnav/.

Read/edit refnav/core.py for climb, hover, navigation and landing decisions;
refnav/node.py publishes those decisions as native ROS 2/PX4 messages, and
refnav/depth.py processes camera depth. See FLIGHT_CODE.md and tests/.

The lab supplies this complete project; copy its source and dependencies for deployment.
`python3 main.py` directly runs your ROS node, like on a hardware companion.
The optional workbench Run button captures this project for managed
recording and grading. Plain Python does not call the lab API or create a Run.
Read onboard/README.md for perception bringup and native ROS examples.
"""
HEIGHT_M = 0.7      # metres above the take-off point
HOVER_S = 2.0       # seconds to hold at that height before landing
ROUTE = "full"      # full (route.json), cp1 (past checkpoint 1) or maze (through Gate A); the launch_key parameter picks the mark
MAP_ALIGN = True    # run nav/map_align_node beside the flight: corrects drift by aligning depth to the course map

from importlib.util import find_spec  # noqa: E402
from pathlib import Path              # noqa: E402

PROJECT = Path(__file__).resolve().parent
flight_package = find_spec("refnav")
if (flight_package is None or flight_package.origin is None
        or Path(flight_package.origin).resolve().parent != PROJECT / "refnav"):
    raise SystemExit("Missing local refnav package: keep the complete refnav/ source beside main.py. "
                     "No hidden flight code is used.")

if __name__ == "__main__":
    import sys
    import signal
    import subprocess
    from refnav import node  # the editable package in this run's snapshot
    from refnav.launch import arguments, params_files
    print(f"Flight source: {Path(node.__file__).parent} (this run's project)", flush=True)
    helper = None
    if MAP_ALIGN:   # one start command starts both; the flight never waits on it (no correction = PX4's estimate)
        helper = subprocess.Popen([sys.executable, "-m", "nav.map_align_node", "--ros-args", *params_files(PROJECT)],
                                  cwd=PROJECT)
    try:
        # Standard ROS parameter files. This project runs without runtime.program,
        # FLYX_PROFILE or a cloud CLI on a normally configured physical companion.
        node.main(arguments(PROJECT, height_m=HEIGHT_M, hover_s=HOVER_S, route=ROUTE, ros_args=sys.argv[1:]))
    finally:
        if helper is not None and helper.poll() is None:
            helper.send_signal(signal.SIGINT)
            try:
                helper.wait(10)
            except subprocess.TimeoutExpired:
                helper.kill()
