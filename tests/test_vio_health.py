"""onboard.vio.vio_health: one bad pose must not cost the flight; a broken source must still be cut."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onboard.vio.vio_health import VioHealth  # noqa: E402

DT = 1 / 15


def walk(h, t0, x0, n, v=0.3):
    out = []
    for k in range(n):
        t = t0 + k * DT
        out.append(h.update(t, (x0 + v * (t - t0), 0.0, 0.8)))
    return out


def test_steady_motion_is_ok():
    h = VioHealth()
    assert all(ok for ok, _ in walk(h, 0.0, 0.0, 60))


def test_a_one_frame_spike_is_withheld_alone_and_the_return_is_fine():
    # the T4 failure: +0.53 m for one frame, then back
    h = VioHealth()
    walk(h, 0.0, 0.0, 30)
    t = 30 * DT
    ok, why = h.update(t, (0.3 * t + 0.53, 0.0, 0.8))
    assert not ok and why.startswith("outlier withheld")
    results = walk(h, t + DT, 0.3 * (t + DT), 30)
    assert all(ok for ok, _ in results)                  # vision never drops after the spike
    assert h.n_withheld == 1


def test_a_persistent_offset_is_accepted_as_a_real_jump():
    h = VioHealth()
    walk(h, 0.0, 0.0, 30)
    t = 30 * DT
    oks = [h.update(t + k * DT, (0.3 * (t + k * DT) + 1.5, 0.0, 0.8))[0] for k in range(10)]
    assert oks[0] is False                                # the first sample of a jump is held
    assert any(oks)                                       # but after persist_s it is taken as the new truth
    assert h.n_rebased == 1
    assert all(ok for ok, _ in walk(h, t + 10 * DT, 0.3 * (t + 10 * DT) + 1.5, 20))


def test_a_broken_source_is_cut_and_needs_quiet_to_recover():
    h = VioHealth(quiet_s=2.0)
    walk(h, 0.0, 0.0, 30)
    t = 30 * DT
    for k in range(6):                                    # garbage that never settles: +-1.5 m alternating
        assert not h.update(t + k * DT, (0.3 * t + (1.5 if k % 2 else -1.5), 0.0, 0.8))[0]
    assert h.reason.startswith("broken")
    t2 = t + 6 * DT
    res = walk(h, t2, 0.3 * t, 45, v=0.0)               # back to sane, near the last good position
    assert not res[0][0] and res[0][1].startswith("recovering")
    first_ok = next(i for i, (ok, _) in enumerate(res) if ok)
    assert abs(first_ok * DT - 2.0) < 2 * DT              # healthy again after quiet_s


def test_non_finite_and_non_advancing_samples_are_withheld():
    h = VioHealth()
    walk(h, 0.0, 0.0, 10)
    assert not h.update(float("nan"), (0.0, 0.0, 0.8))[0]
    assert not h.update(9 * DT, (0.3 * 9 * DT, 0.0, 0.8))[0]   # same time as the last good sample
    assert h.update(10 * DT, (0.3 * 10 * DT, 0.0, 0.8))[0]


def test_a_high_variance_jump_is_withheld_never_re_anchored_and_cut_as_broken():
    # T4 touchdowns: cuVSLAM's variance went 4e-4 -> 3 m^2 and its poses moved off together; that
    # self-consistent garbage must not be accepted as a real jump (variance vetoes re-anchoring)
    h = VioHealth()
    walk(h, 0.0, 0.0, 30)
    t = 30 * DT
    oks = [h.update(t + k * DT, (0.3 * t + 15.0 + 0.01 * k, 0.0, 0.8), var=3.0)[0] for k in range(15)]
    assert not any(oks) and h.n_rebased == 0
    assert h.reason.startswith("broken")
    assert not h.update(t + 15 * DT, (0.3 * t, 0.0, 0.8), var=3.0)[0]        # broken: needs quiet_s to recover


def test_run2_accurate_samples_with_high_variance_are_passed_and_never_break_vision():
    """T4 darion-wholestack run 2 (2026-11-07): cuVSLAM's variance was 0.053-0.071 m^2 for 7 samples in 0.4 s
    while its positions stayed accurate. Withholding them tripped 'broken', PX4 got no vision for 0.5 s and
    failsafe-landed. Now they pass (PX4 weighs the variance) and vision never stops."""
    h = VioHealth()
    walk(h, 0.0, 0.0, 30)
    t0 = 30 * DT
    variances = (0.053, 0.071, 0.067, 0.07, 0.059, 0.053, 0.062)
    res = [h.update(t0 + k * DT, (0.3 * (t0 + k * DT), 0.0, 0.8), var=v) for k, v in enumerate(variances)]
    assert all(ok for ok, _ in res)
    assert all("passed for PX4 to weigh" in why for _, why in res)
    assert h.n_high_var == 7 and h.n_withheld == 0 and not h.reason.startswith("broken")
    assert all(ok for ok, _ in walk(h, t0 + 7 * DT, 0.3 * (t0 + 7 * DT), 20))


def test_low_variance_passes_and_unknown_variance_is_allowed():
    h = VioHealth()
    assert h.update(0.0, (0, 0, 0.8), var=4e-4)[0] and h.update(DT, (0.02, 0, 0.8), var=None)[0]
    assert h.update(2 * DT, (0.04, 0, 0.8), var=3.0)[0]                     # high variance alone: passed


def test_a_persistent_jump_sets_rebased_on_the_accepting_sample_only():
    h = VioHealth()
    walk(h, 0.0, 0.0, 30)
    t = 30 * DT
    flags = []
    for k in range(10):
        h.update(t + k * DT, (0.3 * (t + k * DT) + 1.5, 0.0, 0.8), var=4e-4)
        flags.append(h.rebased)
    assert sum(flags) == 1


def test_the_bridge_constructs_it_with_its_arguments():
    VioHealth(max_speed_mps=5.0, jump_m=1.0, quiet_s=5.0, max_var_m2=0.05)   # as onboard/vio/ev_bridge.py does
