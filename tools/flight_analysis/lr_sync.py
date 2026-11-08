"""READ-ONLY: 15 s of the camera topics cuVSLAM uses: per-topic rate and gaps, and left/right header-stamp sync."""
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data as Q
from sensor_msgs.msg import Image


class S(Node):
    def __init__(self, topics):
        super().__init__("arnav_lr_sync")
        self.st = {t: [] for t in topics}
        for t in topics:
            self.create_subscription(Image, t, lambda m, t=t: self.st[t].append((time.time(), m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m.width, m.height, m.encoding)), Q)


rclpy.init()
probe = Node("arnav_lr_list")
time.sleep(2.0)
imgs = sorted(n for n, ty in probe.get_topic_names_and_types() if "sensor_msgs/msg/Image" in ty and ("oak" in n or "left" in n or "right" in n))
print("image topics:", imgs)
probe.destroy_node()
pick = [t for t in imgs if any(k in t for k in ("left", "right")) and "depth" not in t][:4]
n = S(pick)
t0 = time.time()
while time.time() - t0 < 15.0:
    rclpy.spin_once(n, timeout_sec=0.05)
for t, v in n.st.items():
    if len(v) > 2:
        st = [s for _, s, *_ in v]
        gaps = sorted(b - a for a, b in zip(st, st[1:]))
        print(f"{t}: {len(v) / 15.0:.1f} Hz received, {v[0][2]}x{v[0][3]} {v[0][4]}, stamp gaps p50 {gaps[len(gaps)//2]*1e3:.1f} "
              f"p95 {gaps[int(.95*len(gaps))]*1e3:.1f} max {gaps[-1]*1e3:.1f} ms, frames missing (gap > 50 ms): {sum(g > 0.05 for g in gaps)}")
    else:
        print(f"{t}: {len(v)} msgs")
L = [t for t in pick if "left" in t]
R = [t for t in pick if "right" in t]
if L and R:
    ls = [s for _, s, *_ in n.st[L[0]]]
    rs = sorted(s for _, s, *_ in n.st[R[0]])
    import bisect
    d = []
    for s in ls:
        i = bisect.bisect_left(rs, s)
        c = [abs(rs[j] - s) for j in (i - 1, i) if 0 <= j < len(rs)]
        if c:
            d.append(min(c))
    d.sort()
    if d:
        print(f"L/R sync {L[0]} vs {R[0]}: |dt| p50 {d[len(d)//2]*1e3:.2f} p95 {d[int(.95*len(d))]*1e3:.2f} max {d[-1]*1e3:.2f} ms, "
              f"pairs off by > 5 ms: {sum(x > 0.005 for x in d)} of {len(d)}")
n.destroy_node()
rclpy.try_shutdown()
