"""READ-ONLY: ask PX4 (USB MAVLink shell) to run its arming checks WITHOUT arming: `commander check`.
Also prints STATUSTEXT for a few seconds. Never sends an arm command, never sets a param."""
import os
import sys
import time

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil  # noqa: E402

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM8"
CMDS = sys.argv[2:] or ["ver all", "commander check"]


def main():
    m = mavutil.mavlink_connection(PORT, baud=115200, source_system=255)
    hb = m.wait_heartbeat(timeout=10)
    if hb is None:
        sys.exit("no heartbeat on " + PORT)
    want = os.environ.get("EXPECT_SYSID")
    if want and m.target_system != int(want):
        sys.exit(f"REFUSED: {PORT} is MAVLink system {m.target_system}, not {want} (not the drone we expect)")
    print(f"heartbeat from sys {m.target_system} comp {m.target_component} autopilot {hb.autopilot} "
          f"base_mode {hb.base_mode} system_status {hb.system_status}")
    dev = mavutil.mavlink.SERIAL_CONTROL_DEV_SHELL
    flags = mavutil.mavlink.SERIAL_CONTROL_FLAG_RESPOND | mavutil.mavlink.SERIAL_CONTROL_FLAG_EXCLUSIVE

    def send(text):
        b = text.encode()
        while b:
            chunk, b = b[:70], b[70:]
            data = list(chunk) + [0] * (70 - len(chunk))
            m.mav.serial_control_send(dev, flags, 0, 0, len(chunk), data)

    send("\n")
    for cmd in CMDS:
        print(f"===== nsh> {cmd}")
        send(cmd + "\n")
        out, t_end = "", time.time() + 6.0
        while time.time() < t_end:
            msg = m.recv_match(type=["SERIAL_CONTROL", "STATUSTEXT"], blocking=True, timeout=0.3)
            if msg is None:
                send("")  # poll for more output
                continue
            if msg.get_type() == "STATUSTEXT":
                print(f"[STATUSTEXT sev {msg.severity}] {msg.text}")
            elif msg.count > 0:
                out += bytes(msg.data[:msg.count]).decode(errors="replace")
        print(out)
    # release the exclusive shell
    m.mav.serial_control_send(dev, 0, 0, 0, 0, [0] * 70)
    m.close()


if __name__ == "__main__":
    main()
