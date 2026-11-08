"""SIM matrix: refnav (blind) on a route through tests/test_refnav_skybeta.flight() (skybeta adapter + lag Stack).
Unlike refnav_speed.py it can pass push_ff_follow from the route to S.command, as refnav_node does on the drone
(the harness's flight() omits it, so S.command's default True leans in follow too).

usage (from hfr): py -3 ../refnav_matrix.py [--route R] [--speeds 0.15,0.2,0.25] [--push 0,0.5,1] [--ffzero]
                  [--harness-ff] [--adapter-lag 1.5] [--tau 0.9] [--delay 0.6] [--trace]
"""
import argparse, functools, json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "tests")
import test_refnav_skybeta as TR
sys.path.insert(0, str(Path(__file__).resolve().parent))
import route_clear as RC
from refnav import core as C

ap = argparse.ArgumentParser()
ap.add_argument("--route", default="courses/start-2026-11-07-tape/routes/route_full_hold.json")
ap.add_argument("--speeds", default="0.15,0.2,0.25")
ap.add_argument("--push", default="0,0.5,1")            # x skybeta-09's body push (-0.056 fwd, 0.100 left)
ap.add_argument("--ffzero", action="store_true")        # push_ff 0/0
ap.add_argument("--harness-ff", action="store_true")    # the harness as is: push_ff_follow ignored (leans in follow)
ap.add_argument("--adapter-lag", type=float, default=1.5)
ap.add_argument("--tau", type=float, default=0.9)
ap.add_argument("--delay", type=float, default=0.6)
ap.add_argument("--height", type=float, default=1.1)
ap.add_argument("--trace", action="store_true")
ap.add_argument("--over", default="{}")             # extra params (JSON), e.g. {"push_tau_s": 3.0}
ap.add_argument("--climb", action="store_true")      # + skybeta-09's climb push (-0.30, 0.06) x k while climbing
a = ap.parse_args()

doc0 = json.loads(Path(a.route).read_text())
ORIG = TR.S.command


def run(speed, k):
    doc = json.loads(json.dumps(doc0))
    doc["params"]["speed_mps"] = speed
    over = {"height_m": a.height, **json.loads(a.over)}
    if a.ffzero:
        over.update(push_ff_fwd_mps=0.0, push_ff_right_mps=0.0)
    prm, _ = C.params_from_doc(doc, {**TR.LAUNCH_PARAMS, **over})
    TR.S.command = ORIG if a.harness_ff else functools.partial(ORIG, push_ff_follow=prm.push_ff_follow)
    trail = []
    m, sk, sent = TR.flight(doc=doc, prm_over=over, tau=a.tau, lag_s=a.adapter_lag, delay=a.delay, trail=trail,
                            push_body=(-0.056 * k, 0.100 * k), climb_push=(-0.30 * k, 0.06 * k) if a.climb else (0.0, 0.0), limit_s=400.0)
    TR.S.command = ORIG
    P = np.stack([m.follower.path.x, m.follower.path.y], 1) if m.follower else None
    fol = [(t, p) for t, ph, p in trail if ph == "follow"]
    xy = np.array([p[:2] for _, p in fol])
    seg_a, seg_b = P[:-1], P[1:]
    ab = seg_b - seg_a
    ab2 = np.maximum((ab * ab).sum(1), 1e-12)

    def d(p):
        t = np.clip(((p - seg_a) * ab).sum(1) / ab2, 0, 1)
        return float(np.min(np.hypot(*(seg_a + ab * t[:, None] - p).T)))
    xt = np.array([d(p) for p in xy])
    hov = [p for t, ph, p in trail if ph == "hover"]
    hov_off = math.hypot(*hov[-1][:2]) if hov else None
    hov_max = max(math.hypot(*p[:2]) for p in hov) if hov else None
    end = P[-1]
    land = [p for t, ph, p in trail if ph in ("land", "done")]
    r = {"speed": speed, "push": k, "result": m.result, "follow_s": round(fol[-1][0] - fol[0][0], 1) if fol else None,
         "xt50": round(float(np.median(xt)), 3), "xt95": round(float(np.percentile(xt, 95)), 3),
         "xtmax": round(float(xt.max()), 3), "pts": m.follower.points_passed,
         "hover_end_off": None if hov_off is None else round(hov_off, 3),
         "hover_max_off": None if hov_max is None else round(hov_max, 3),
         "land_from_end_m": round(math.dist(land[-1][:2], end), 3) if land else None,
         "path_len": round(m.follower.path.length, 2)}
    air = np.array([p[:2] for t, ph, p in trail if ph in ("hover", "follow")])
    dc, win, dw, _, _ = RC.clear_xy(air[:, 0], air[:, 1], a.height)
    r["clr_min"] = round(float(np.nanmin(np.where(win, np.inf, dc))), 3)
    r["jamb_min"] = round(float(np.nanmin(np.where(win, dw, np.inf))), 3) if win.any() else None
    if a.trace:
        r["trace"] = []
        for t, p in fol[-int(60 / TR.DT)::int(2 / TR.DT)]:
            r["trace"].append([round(t, 1), round(p[0], 3), round(p[1], 3), round(math.dist(p[:2], end), 3)])
    return r


for s in [float(x) for x in a.speeds.split(",")]:
    for k in [float(x) for x in a.push.split(",")]:
        print(json.dumps(run(s, k)), flush=True)
