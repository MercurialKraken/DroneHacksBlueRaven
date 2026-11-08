"""READ-ONLY: camera_info of every camera topic, the image sizes actually published, static TFs, and node names."""
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data as Q
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage

rclpy.init()
n = Node("arnav_calib")
t0 = time.time()
while time.time() - t0 < 5.0:
    rclpy.spin_once(n, timeout_sec=0.1)
topics = n.get_topic_names_and_types()
infos = [t for t, ty in topics if "CameraInfo" in ty[0]]
imgs = [t for t, ty in topics if ty[0].endswith("/Image")]
print("nodes:", sorted(f"{ns.rstrip('/')}/{nm}" for nm, ns in n.get_node_names_and_namespaces()))
got = {}
for t in infos:
    n.create_subscription(CameraInfo, t, lambda m, t=t: got.setdefault(t, m), Q)
sizes = {}
for t in imgs:
    n.create_subscription(Image, t, lambda m, t=t: sizes.setdefault(t, (m.width, m.height, m.encoding, m.header.frame_id)), Q)
tfs = []
latched = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
n.create_subscription(TFMessage, "/tf_static", lambda m: tfs.extend(m.transforms), latched)
t0 = time.time()
while time.time() - t0 < 6.0:
    rclpy.spin_once(n, timeout_sec=0.05)
for t, (w, h, e, f) in sorted(sizes.items()):
    print(f"IMAGE {t}: {w}x{h} {e} frame {f}")
for t, m in sorted(got.items()):
    K, P = list(m.k), list(m.p)
    base = -P[3] / P[0] if P[0] else float("nan")
    print(f"INFO {t}: {m.width}x{m.height} frame {m.header.frame_id} model {m.distortion_model}\n"
          f"   K fx {K[0]:.2f} fy {K[4]:.2f} cx {K[2]:.2f} cy {K[5]:.2f} | P fx {P[0]:.2f} cx {P[2]:.2f} cy {P[6]:.2f} Tx {P[3]:.3f} -> baseline {base:.4f} m"
          f" | D {[round(x, 4) for x in m.d][:5]}")
for tr in tfs:
    q, p = tr.transform.rotation, tr.transform.translation
    print(f"TF {tr.header.frame_id} -> {tr.child_frame_id}: t ({p.x:.3f},{p.y:.3f},{p.z:.3f}) q ({q.x:.3f},{q.y:.3f},{q.z:.3f},{q.w:.3f})")
n.destroy_node()
rclpy.try_shutdown()
