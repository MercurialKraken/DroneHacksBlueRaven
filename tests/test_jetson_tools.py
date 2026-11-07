"""tools/jetson_budget.py (tegrastats) and tools/depth_bench.py (tape-measure stats)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import depth_bench  # noqa: E402
import jetson_budget as jb  # noqa: E402

# Orin Nano (JetPack 6) tegrastats lines; two cores offline in the first one
L1 = ("11-08-2026 03:00:00 RAM 2861/7620MB (lfb 3x4MB) SWAP 0/3810MB (cached 0MB) CPU [2%@729,1%@729,0%@729,3%@729,off,off] "
      "EMC_FREQ 0%@2133 GR3D_FREQ 0%@[305] NVENC off NVDEC off VIC off APE 200 cpu@47.5C soc2@45.8C gpu@46.6C tj@47.5C "
      "VDD_IN 4436mW/4436mW VDD_CPU_GPU_CV 483mW/483mW VDD_SOC 1374mW/1374mW")
L2 = ("11-08-2026 03:00:01 RAM 3400/7620MB (lfb 3x4MB) SWAP 12/3810MB (cached 0MB) CPU [60%@1728,40%@1728,20%@1728,80%@1728,10%@1728,30%@1728] "
      "GR3D_FREQ 50% cpu@55.0C gpu@52.0C tj@56.0C VDD_IN 9000mW/6000mW VDD_CPU_GPU_CV 3000mW/1500mW VDD_SOC 2000mW/1600mW")


def test_parse_reads_every_field():
    s = jb.parse(L1)
    assert s["ram_mb"] == 2861 and s["ram_total_mb"] == 7620 and s["swap_mb"] == 0
    assert s["cpu"][:4] == [(2, 729), (1, 729), (0, 729), (3, 729)] and s["cpu"][4:] == [None, None]
    assert s["gpu_pct"] == 0
    assert s["power_mw"] == {"VDD_IN": 4436, "VDD_CPU_GPU_CV": 483, "VDD_SOC": 1374}
    assert s["temp_c"]["tj"] == 47.5
    assert jb.parse("not a tegrastats line") is None


def test_summary_averages_and_handles_offline_cores():
    S = jb.summarize([L1, "", L2])
    assert S["samples"] == 2 and S["cores_online"] == 6
    assert S["cpu_per_core"][0][0] == pytest.approx(31.0)               # (2 + 60) / 2
    assert S["cpu_per_core"][4][0] == pytest.approx(10.0)               # online in one sample only
    assert S["gpu_mean"] == pytest.approx(25.0)
    assert S["ram_mb_max"] == 3400 and S["power_mw_mean"]["VDD_IN"] == pytest.approx(6718.0)
    text = jb.table({"idle": jb.summarize([L1]), "depth": jb.summarize([L2])})
    assert "cpu5 % mean/p95" in text and "off" in text and "VDD_IN mW" in text


def test_roi_stats_and_report():
    z = np.full((100, 200), 2.0, np.float32)
    z[45:55, 90:110:2] = np.nan                                         # half the pixels of a central patch
    med, vf = depth_bench.roi_stats(z, 0.1)
    assert med == pytest.approx(2.0) and vf == pytest.approx(0.5)
    assert math.isnan(depth_bench.roi_stats(np.full((10, 10), np.nan), 0.5)[0])
    r = depth_bench.report([2.02, 2.04], [0.9, 0.95], [0.0, 0.2], [80.0, 90.0], distance=2.0)
    assert "error vs tape +0.030 m (+1.5%)" in r and "rate 5.00 Hz" in r and "latency" in r
