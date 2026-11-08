#!/usr/bin/env python3
"""SIM-ONLY scorer: Gazebo truth against what the drone did. Never import this from flight code.

    python3 score.py OUT.json ROUTE_JSON|none COURSE_YAML      runs until SIGTERM, then writes OUT.json and prints it

Route points are [forward, left] from the course's sim launch mark (launch_sim); reached = truth within 0.30 m.
Clearance is the truth centre's distance to the nearest occupied 5 cm voxel of the sim world (floor excluded), airborne only.
"""
import json
import math
import signal
import sys
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleLocalPosition, VehicleStatus
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from scipy.ndimage import distance_transform_edt

OCCUPANCY = '/opt/devenv/occupancy-5cm.npz'
REACH_M = 0.30
AIRBORNE_M = 0.15       # truth height above the take-off height


class Score(Node):
    def __init__(self, route_file, course_yaml):
        super().__init__('sim_score', parameter_overrides=[Parameter('use_sim_time', value=True)])
        occ = np.load(OCCUPANCY)
        grid = occ['occupied'].copy()
        self.res, self.origin = float(occ['resolution_m']), occ['origin_enu_m']
        grid[:, :, :int(round(-self.origin[2] / self.res))] = False          # the floor slab below z = 0
        self.clear = distance_transform_edt(~grid) * self.res
        L = yaml.safe_load(Path(course_yaml).read_text())['launch_sim']
        a = math.radians(L['yaw_deg'])
        fwd, left = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        pts = [] if route_file == 'none' else json.loads(Path(route_file).read_text())['route']
        self.route = np.array([[L['x'], L['y']] + f * fwd + l * left for f, l in pts]).reshape(-1, 2)
        self.reached = np.zeros(len(self.route), bool)
        self.z0 = self.takeoff_xy = self.t_takeoff = self.t_disarm = self.disarm_xy = None
        self.truth = self.truth_px4 = None
        self.armed, self.min_clear, self.errors = False, math.inf, []
        self.create_subscription(Odometry, '/sim/truth/odometry', self._on_truth, 50)
        self.create_subscription(PoseStamped, '/sim/truth/pose_px4', lambda m: setattr(self, 'truth_px4', m.pose.position), 50)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self._on_est, qos_profile_sensor_data)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v1', self._on_status, qos_profile_sensor_data)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_truth(self, m):
        p = m.pose.pose.position
        self.truth = np.array([p.x, p.y, p.z])
        if self.z0 is None:
            self.z0 = p.z
        if self.armed and self.t_takeoff is None and p.z > self.z0 + AIRBORNE_M:
            self.t_takeoff = self.now()
        if self.armed and p.z > self.z0 + AIRBORNE_M:
            i = np.clip(((self.truth - self.origin) / self.res).astype(int), 0, np.array(self.clear.shape) - 1)
            self.min_clear = min(self.min_clear, float(self.clear[tuple(i)]))
            if len(self.route):
                self.reached |= np.hypot(*(self.route - self.truth[:2]).T) < REACH_M

    def _on_est(self, m):
        if self.armed and self.truth_px4 is not None:
            self.errors.append(math.hypot(m.x - self.truth_px4.x, m.y - self.truth_px4.y))

    def _on_status(self, m):
        armed = m.arming_state == VehicleStatus.ARMING_STATE_ARMED
        if armed and self.takeoff_xy is None and self.truth is not None:
            self.takeoff_xy = self.truth[:2].copy()
        if self.armed and not armed and self.t_takeoff is not None and self.t_disarm is None:
            self.t_disarm, self.disarm_xy = self.now(), self.truth[:2].copy()
        self.armed = armed

    def result(self):
        end = self.route[-1] if len(self.route) else self.takeoff_xy
        landed = self.disarm_xy is not None
        return {
            'took_off': self.t_takeoff is not None,
            'disarmed_after_flight': landed,
            'route_points_reached': f'{int(self.reached.sum())}/{len(self.route)}',
            'flight_time_s': round(self.t_disarm - self.t_takeoff, 1) if landed else None,
            'min_truth_clearance_m': round(self.min_clear, 3) if math.isfinite(self.min_clear) else None,
            'est_error_xy_p95_m': round(float(np.percentile(self.errors, 95)), 3) if self.errors else None,
            'est_error_xy_max_m': round(max(self.errors), 3) if self.errors else None,
            'landing_distance_m': round(float(np.hypot(*(self.disarm_xy - end))), 3) if landed else None,
            'landing_distance_to': 'last route point' if len(self.route) else 'take-off point',
        }


def main():
    out, route, course_yaml = sys.argv[1:4]
    rclpy.init()
    s = Score(route, course_yaml)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    print('score: ready', flush=True)
    while not stop:
        rclpy.spin_once(s, timeout_sec=0.1)
    r = s.result()
    Path(out).write_text(json.dumps(r, indent=1) + '\n')
    print(json.dumps(r, indent=1), flush=True)


if __name__ == '__main__':
    main()
