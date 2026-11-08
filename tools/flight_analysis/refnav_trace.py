"""SIM trace of the first seconds of follow (or the last): refnav_matrix's setup, per-tick twist and errors.
usage (from hfr): py -3 ../refnav_trace.py SPEED PUSHK [tau delay lag] [--end]"""
import functools, json, math, sys
from pathlib import Path
sys.path.insert(0, "tests")
import test_refnav_skybeta as TR
from refnav import core as C
speed, k = float(sys.argv[1]), float(sys.argv[2])
nums = [float(x) for x in sys.argv[3:6]] if len(sys.argv) > 5 and not sys.argv[3].startswith("-") else [0.9, 0.6, 1.5]
tau, delay, lag = nums
doc = json.loads(Path("courses/start-2026-11-07-tape/routes/route_full_hold.json").read_text())
doc["params"]["speed_mps"] = speed
over = {"height_m": 1.1}
if "--ffzero" in sys.argv:
    over.update(push_ff_fwd_mps=0.0, push_ff_right_mps=0.0)
ORIG = TR.S.command
rows = []
def wrap(out, v, *a, **kw):
    tw, why = ORIG(out, v, *a, **kw) if "--harness-ff" in sys.argv else ORIG(out, v, *a, push_ff_follow=False, **kw)
    if out.setpoint is not None and v.pos is not None:
        err = math.dist(out.setpoint[:2], v.pos[:2])
        rows.append((a[5] if len(a) > 5 else 0.0, out.phase, v.pos[:2], err, None if tw is None else math.hypot(tw[0], tw[1]), why,
                     math.degrees(v.heading or 0), math.hypot(*v.vel), out.velocity))
    return tw, why
TR.S.command = wrap
mm = {}
m, sk, sent = TR.flight(doc=doc, prm_over=over, tau=tau, lag_s=lag, delay=delay, push_body=(-0.056 * k, 0.100 * k), limit_s=400.0)
print(m.result)
fol = [r for r in rows if r[1] == "follow"]
sel = fol[-200:] if "--end" in sys.argv else fol[:120]
for r in sel[::5]:
    t, ph, p, err, twm, why, hd, vm, ff = r
    print(f"t {t:6.1f} pos ({p[0]:+.3f},{p[1]:+.3f}) sp_err {err:.3f} twist {twm if twm is None else round(twm,3)} vel {vm:.3f} hdg {hd:6.1f} ff {None if ff is None else tuple(round(x,3) for x in ff)} {why or ''}")
