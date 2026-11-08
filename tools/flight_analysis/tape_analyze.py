import sys, numpy as np, math
d = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
t = d["stamp"] - d["stamp"][0]; P = np.c_[d["x"], d["y"], d["z"]]; yaw = d["yaw_deg"]
print(f"{len(t)} samples over {t[-1]:.1f} s; start ({P[0,0]:.3f},{P[0,1]:.3f},{P[0,2]:.3f}) yaw {yaw[0]:.1f}")
# stills: windows of >= 2.0 s where the position spread is < 2 cm
w = int(2.0 * len(t) / max(t[-1], 1e-9))
still = np.zeros(len(t), bool)
for i in range(0, len(t) - w):
    seg = P[i:i + w]
    if np.ptp(seg[:, 0]) < 0.02 and np.ptp(seg[:, 1]) < 0.02 and np.ptp(seg[:, 2]) < 0.02:
        still[i:i + w] = True
segs, i = [], 0
while i < len(t):
    if still[i]:
        j = i
        while j < len(t) and still[j]: j += 1
        if t[j - 1] - t[i] >= 1.5: segs.append((i, j))
        i = j
    else:
        i += 1
ref = P[segs[0][0]:segs[0][1]].mean(0) if segs else P[0]
yaw0 = yaw[segs[0][0]] if segs else yaw[0]
print("STILL SEGMENTS (position relative to the first still, in the VIO frame; and distance from the previous still):")
prev = None
for k, (a, b) in enumerate(segs):
    p = P[a:b].mean(0) - ref
    dprev = np.linalg.norm(p - prev) if prev is not None else 0.0
    dh = np.hypot(*(p - prev)[:2]) if prev is not None else 0.0
    dz = (p - prev)[2] if prev is not None else 0.0
    print(f" #{k:2d} t {t[a]:6.1f}-{t[b-1]:6.1f}s  rel ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})  yaw {np.mean(yaw[a:b]) - yaw0:+6.1f}  "
          f"| from prev: horiz {dh:.3f} m, vert {dz:+.3f} m")
    prev = p
