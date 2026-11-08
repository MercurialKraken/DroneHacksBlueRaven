#!/usr/bin/env python3
"""SIM-ONLY stand-in for the safety pilot. Does what the pilot does on the drone and nothing more:
once the drone program streams setpoints, arm in Altitude, then switch to Position (the flight interface's go
signal). Its transmitter is /fmu/in/manual_control_input: sticks centred, throttle low until armed. Then it only watches. It takes over (Land) if Offboard never comes, ends in the air, or the run times out.

    python3 pilot.py TIMEOUT_S      exit 0: PX4 disarmed itself after an Offboard flight; 2: the pilot took over
"""
import signal
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from px4_msgs.msg import ManualControlSetpoint, VehicleCommand, VehicleLandDetected, VehicleStatus
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

POSCTL, OFFBOARD = VehicleStatus.NAVIGATION_STATE_POSCTL, VehicleStatus.NAVIGATION_STATE_OFFBOARD
MAIN_ALTCTL, MAIN_POSCTL, MAIN_AUTO, SUB_LAND = 2, 3, 4, 6     # PX4 custom modes for VEHICLE_CMD_DO_SET_MODE
OFFBOARD_WAIT_S = 5.0                           # the interface tries for 2 s after the switch
OFF_OFFBOARD_S = 2.0                            # armed out of Offboard this long: the pilot lands it


class Pilot(Node):
    def __init__(self):
        super().__init__('sim_pilot')
        self.status = self.landed = None
        self.setpoint_wall = None
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self._on_status, qos_profile_sensor_data)
        self.create_subscription(VehicleLandDetected, '/fmu/out/vehicle_land_detected',
                                 lambda m: setattr(self, 'landed', m.landed), qos_profile_sensor_data)
        for kind, topic in ((PoseStamped, '/skybeta/setpoint/pose'), (TwistStamped, '/skybeta/setpoint/twist')):
            self.create_subscription(kind, topic, lambda _: setattr(self, 'setpoint_wall', time.monotonic()), 10)
        self.cmd = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile_sensor_data)
        self.sticks = self.create_publisher(ManualControlSetpoint, '/fmu/in/manual_control_input', qos_profile_sensor_data)
        self.create_timer(0.05, self._transmit)

    def _transmit(self):
        m = ManualControlSetpoint(valid=True, data_source=ManualControlSetpoint.SOURCE_MAVLINK_0,
                                  throttle=0.0 if self.armed else -1.0)
        m.timestamp = m.timestamp_sample = self.get_clock().now().nanoseconds // 1000
        self.sticks.publish(m)

    def _on_status(self, m):
        if self.status is None or (m.arming_state, m.nav_state) != (self.status.arming_state, self.status.nav_state):
            self.say(f'armed {m.arming_state == VehicleStatus.ARMING_STATE_ARMED} nav_state {m.nav_state}')
        self.status = m

    @property
    def armed(self):
        return self.status is not None and self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED

    def say(self, text):
        print(f'[pilot sim {self.get_clock().now().nanoseconds * 1e-9:9.2f}] {text}', flush=True)

    def command(self, command, p1=0.0, p2=0.0, p3=0.0):
        m = VehicleCommand(command=command, param1=float(p1), param2=float(p2), param3=float(p3),
                           target_system=1, target_component=1, source_system=1, source_component=1, from_external=True)
        m.timestamp = self.get_clock().now().nanoseconds // 1000
        self.cmd.publish(m)


def main():
    timeout_s = float(sys.argv[1])
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(2))
    rclpy.init()
    p = Pilot()
    t0 = time.monotonic()
    step, mark, offboard_seen, off_since = 'wait_setpoints', t0, False, None

    def spin(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(p, timeout_sec=0.05)

    while True:
        spin(0.1)
        now = time.monotonic()
        if step == 'wait_setpoints':
            if p.status is not None and p.setpoint_wall is not None and now - p.setpoint_wall < 0.5:
                p.say('setpoints streaming: arming')
                step, mark = 'arming', 0.0
        elif step == 'arming':
            if p.armed:
                p.say('armed: switching to Position')
                p.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, MAIN_POSCTL)
                step, mark = 'wait_offboard', now
            elif now - mark > 1.0:
                p.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, MAIN_ALTCTL)
                p.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1)
                mark = now
        elif step == 'wait_offboard':
            if p.status.nav_state == OFFBOARD:
                p.say('Offboard: the drone program flies')
                step, offboard_seen = 'watch', True
            elif now - mark > OFFBOARD_WAIT_S or not p.armed:
                p.say('no Offboard: taking over')
                step = 'take_over'
        elif step == 'watch':
            if not p.armed:
                p.say(f'disarmed by PX4 (landed {p.landed}): self-disarm')
                break
            off_since = None if p.status.nav_state == OFFBOARD else (off_since or now)
            if off_since is not None and now - off_since > OFF_OFFBOARD_S and not p.landed:
                p.say(f'out of Offboard in the air (nav_state {p.status.nav_state}): taking over')
                step = 'take_over'
        if step == 'take_over' or (step != 'landing' and now - t0 > timeout_s):
            if step != 'take_over':
                p.say(f'timeout {timeout_s:.0f} s: taking over')
            if not p.armed:
                break
            p.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, MAIN_AUTO, SUB_LAND)
            step, mark = 'landing', now
        if step == 'landing':
            if not p.armed:
                p.say('disarmed after the pilot took over')
                break
            if now - mark > 60.0:
                p.say('still armed 60 s after Land: disarming')
                p.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0)
                spin(1.0)
                break
    ok = offboard_seen and step == 'watch'
    p.say('RESULT ' + ('self-disarm' if ok else 'pilot took over'))
    sys.exit(0 if ok else 2)


if __name__ == '__main__':
    main()
