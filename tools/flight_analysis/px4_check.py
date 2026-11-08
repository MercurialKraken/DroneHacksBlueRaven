"""READ-ONLY PX4 pre-arm check (no publishers): vehicle_status, health_report (which arming checks fail),
failsafe_flags, estimator flags, local position, battery. Run inside the stack container."""
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data as QOS

# PX4 events health_component_t bit order (health_report flags)
COMP = ["sensor_imu", "sensor_absolute_pressure", "sensor_differential_pressure", "sensor_gps", "sensor_optical_flow",
        "sensor_vision_position", "sensor_distance", "manual_control_input", "motors_escs", "utm", "logging", "battery",
        "communication_links", "rate_controller", "attitude_controller", "position_controller", "attitude_estimate",
        "local_position_estimate", "mission", "avoidance", "system", "camera", "gimbal", "payload",
        "global_position_estimate", "storage", "parachute", "magnetometer", "accel", "gyro", "open_drone_id",
        "traffic_avoidance"]


def bits(v):
    return [COMP[i] if i < len(COMP) else f"bit{i}" for i in range(64) if (int(v) >> i) & 1] or ["-"]


class C(Node):
    def __init__(self):
        super().__init__("arnav_px4_check")
        self.last = {}
        import px4_msgs.msg as M
        for topic, name in (("/fmu/out/vehicle_status_v1", "VehicleStatus"), ("/fmu/out/vehicle_status", "VehicleStatus"),
                            ("/fmu/out/health_report", "HealthReport"), ("/fmu/out/failsafe_flags", "FailsafeFlags"),
                            ("/fmu/out/estimator_status_flags", "EstimatorStatusFlags"),
                            ("/fmu/out/vehicle_local_position", "VehicleLocalPosition"),
                            ("/fmu/out/vehicle_local_position_v1", "VehicleLocalPosition"),
                            ("/fmu/out/battery_status", "BatteryStatus"), ("/fmu/out/battery_status_v1", "BatteryStatus")):
            T = getattr(M, name, None)
            if T is not None:
                self.create_subscription(T, topic, lambda m, t=topic: self.last.__setitem__(t, m), QOS)


def main():
    rclpy.init()
    n = C()
    t0 = time.time()
    while time.time() - t0 < 6.0:
        rclpy.spin_once(n, timeout_sec=0.1)
    L = n.last
    print("topics seen:", sorted(L))
    vs = L.get("/fmu/out/vehicle_status_v1") or L.get("/fmu/out/vehicle_status")
    if vs:
        f = lambda k: getattr(vs, k, "n/a")
        print(f"VEHICLE_STATUS arming_state={f('arming_state')} nav_state={f('nav_state')} "
              f"PRE_FLIGHT_CHECKS_PASS={f('pre_flight_checks_pass')} usb_connected={f('usb_connected')} "
              f"safety_off={f('safety_off')} rc_signal_lost={f('rc_signal_lost')} gcs_connection_lost={f('gcs_connection_lost')} "
              f"latest_arming_reason={f('latest_arming_reason')} failsafe={f('failsafe')}")
    hr = L.get("/fmu/out/health_report")
    if hr:
        print("HEALTH arming_check_ERROR:", bits(hr.arming_check_error_flags))
        print("HEALTH arming_check_WARNING:", bits(hr.arming_check_warning_flags))
        print("HEALTH health_ERROR:", bits(hr.health_error_flags), " health_WARNING:", bits(hr.health_warning_flags))
        print(f"HEALTH can_arm_mode_flags=0x{int(hr.can_arm_mode_flags):x} can_run_mode_flags=0x{int(hr.can_run_mode_flags):x}")
    else:
        print("HEALTH: no /fmu/out/health_report seen")
    ff = L.get("/fmu/out/failsafe_flags")
    if ff:
        on = [k for k in ff.get_fields_and_field_types() if k != "timestamp" and isinstance(getattr(ff, k), bool) and getattr(ff, k)]
        print("FAILSAFE_FLAGS true:", on or "-")
    es = L.get("/fmu/out/estimator_status_flags")
    if es:
        keys = ["cs_tilt_align", "cs_yaw_align", "cs_ev_pos", "cs_ev_yaw", "cs_ev_hgt", "cs_ev_vel", "cs_baro_hgt",
                "cs_rng_hgt", "cs_mag_hdg", "cs_mag_3d", "cs_mag_field_disturbed", "cs_mag_heading_consistent",
                "cs_in_air", "cs_gps"]
        print("ESTIMATOR:", {k: getattr(es, k) for k in keys if hasattr(es, k)})
        print(f"ESTIMATOR fault_status_changes={getattr(es, 'fault_status_changes', 'n/a')} "
              f"reject_ev_pos={getattr(es, 'reject_ev_pos', 'n/a')} reject_ev_yaw={getattr(es, 'reject_ev_yaw', 'n/a')}")
    lp = L.get("/fmu/out/vehicle_local_position") or L.get("/fmu/out/vehicle_local_position_v1")
    if lp:
        print(f"LOCAL_POSITION xy_valid={lp.xy_valid} z_valid={lp.z_valid} v_xy_valid={lp.v_xy_valid} "
              f"heading_good_for_control={getattr(lp, 'heading_good_for_control', 'n/a')} heading={lp.heading:.3f} "
              f"eph={lp.eph:.3f} epv={lp.epv:.3f}")
    b = L.get("/fmu/out/battery_status") or L.get("/fmu/out/battery_status_v1")
    if b:
        print(f"BATTERY voltage_v={b.voltage_v:.2f} remaining={b.remaining:.2f} connected={getattr(b, 'connected', 'n/a')} "
              f"warning={getattr(b, 'warning', 'n/a')}")
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
