"""READ-ONLY: list and get every parameter of /visual_slam_node (and /oak) through the parameter services."""
import sys, time
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import ListParameters, GetParameters

rclpy.init()
n = Node("arnav_params")
for target in sys.argv[1:]:
    lc = n.create_client(ListParameters, f"{target}/list_parameters")
    gc = n.create_client(GetParameters, f"{target}/get_parameters")
    if not lc.wait_for_service(timeout_sec=5.0):
        print(target, "no parameter service"); continue
    f = lc.call_async(ListParameters.Request(depth=0))
    rclpy.spin_until_future_complete(n, f, timeout_sec=5.0)
    names = sorted(f.result().result.names) if f.result() else []
    keep = [x for x in names if any(k in x.lower() for k in ("imu", "frame", "rectif", "baseline", "image", "camera", "enable", "gyro", "accel", "jitter", "sync", "denois", "planar", "ground", "num_cameras", "verbos", "landmark", "map", "path", "publish", "input"))] if target.endswith("oak") else names
    g = gc.call_async(GetParameters.Request(names=keep))
    rclpy.spin_until_future_complete(n, g, timeout_sec=5.0)
    vals = g.result().values if g.result() else []
    print(f"=== {target}: {len(names)} params, showing {len(keep)}")
    for k, v in zip(keep, vals):
        val = [v.bool_value, v.integer_value, v.double_value, v.string_value, list(v.byte_array_value), list(v.bool_array_value),
               list(v.integer_array_value), list(v.double_array_value), list(v.string_array_value)][v.type - 1] if v.type else None
        print(f"  {k} = {val}")
n.destroy_node(); rclpy.try_shutdown()
