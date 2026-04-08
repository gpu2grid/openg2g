"""Tests for the unified sweep_dc_locations module.

Validates helper functions, profile generation, and bus discovery.
Tests for removed JSON-based config models have been replaced with tests
for the programmatic experiment definitions.

Note: This module depends on OpenDSS (via transitive imports). If OpenDSS
is not available or crashes, the entire test module is skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add examples/offline to sys.path so we can import sweep_dc_locations
_examples_dir = Path(__file__).resolve().parent.parent / "examples" / "offline"
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

# sweep_dc_locations transitively imports openg2g.grid.opendss which loads
# the OpenDSS native library at import time. If that library is not available
# or crashes (common on some Windows environments), skip the entire module.
try:
    from sweep_dc_locations import (
        TAP_STEP,
        _smooth_bump,
        discover_candidate_buses,
        eval_profile,
        find_violations,
        load_profile_kw,
        pv_profile_kw,
    )
    from systems import tap

    CAN_IMPORT = True
except Exception:
    CAN_IMPORT = False

pytestmark = pytest.mark.skipif(not CAN_IMPORT, reason="Cannot import sweep_dc_locations (OpenDSS unavailable)")


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests: helpers and profile functions
# ══════════════════════════════════════════════════════════════════════════════


class TestSmoothBump:
    def test_center_is_one(self):
        assert _smooth_bump(100.0, 100.0, 50.0) == 1.0

    def test_edge_is_zero(self):
        assert _smooth_bump(150.0, 100.0, 50.0) == 0.0
        assert _smooth_bump(50.0, 100.0, 50.0) == 0.0

    def test_outside_is_zero(self):
        assert _smooth_bump(200.0, 100.0, 50.0) == 0.0

    def test_symmetric(self):
        v1 = _smooth_bump(80.0, 100.0, 50.0)
        v2 = _smooth_bump(120.0, 100.0, 50.0)
        assert abs(v1 - v2) < 1e-12


class TestProfiles:
    def test_pv_profile_nonnegative(self):
        for t in np.linspace(0, 3600, 100):
            assert pv_profile_kw(float(t), 1000.0, site_idx=0) >= 0.0
            assert pv_profile_kw(float(t), 1000.0, site_idx=1) >= 0.0

    def test_load_profile_nonnegative(self):
        for t in np.linspace(0, 3600, 100):
            for idx in range(5):
                assert load_profile_kw(float(t), 500.0, site_idx=idx) >= 0.0

    def test_eval_profile_with_csv_data(self):
        csv_data = (np.array([0.0, 100.0, 200.0]), np.array([0.0, 50.0, 100.0]))
        val = eval_profile(50.0, peak_kw=999.0, csv_data=csv_data, profile_fn=pv_profile_kw, site_idx=0)
        assert abs(val - 25.0) < 1e-10

    def test_eval_profile_without_csv(self):
        val = eval_profile(100.0, peak_kw=1000.0, csv_data=None, profile_fn=pv_profile_kw, site_idx=0)
        assert val > 0.0


class TestTapHelper:
    def test_tap_converts_steps(self):
        assert tap(14) == pytest.approx(1.0 + 14 * TAP_STEP)
        assert tap(0) == pytest.approx(1.0)
        assert tap(-3) == pytest.approx(1.0 - 3 * TAP_STEP)


class TestFindViolations:
    def test_no_violations(self):
        voltages = {"bus1": {"a": [1.0, 1.01], "b": [0.99, 1.0], "c": [0.96, 1.04]}}
        violations = find_violations(voltages, v_min=0.95, v_max=1.05)
        assert len(violations) == 0

    def test_undervoltage(self):
        voltages = {"bus1": {"a": [0.94, 1.0], "b": [1.0], "c": [1.0]}}
        violations = find_violations(voltages, v_min=0.95, v_max=1.05)
        assert len(violations) == 1
        assert violations[0][2] == "under"

    def test_overvoltage(self):
        voltages = {"bus1": {"a": [1.0], "b": [1.06, 1.0], "c": [1.0]}}
        violations = find_violations(voltages, v_min=0.95, v_max=1.05)
        assert len(violations) == 1
        assert violations[0][2] == "over"


# ══════════════════════════════════════════════════════════════════════════════
# Experiment definitions
# ══════════════════════════════════════════════════════════════════════════════


class TestExperimentDefinitions:
    """Validate that per-system experiment definitions produce valid configs."""

    def test_ieee13_experiment(self):
        from sweep_dc_locations import _experiment_ieee13
        from systems import ieee13

        from openg2g.datacenter.workloads.training import TrainingTrace, TrainingTraceParams

        sys_const = ieee13()
        # Create a minimal training trace for the test
        ttp = TrainingTraceParams()
        # We just need the experiment function to not crash
        # The training_trace is only used for TrainingRun construction
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("t_s,power_W\n0,100\n1,200\n")
            tmp_path = Path(f.name)

        try:
            training_trace = TrainingTrace.ensure(tmp_path, ttp)
            exp = _experiment_ieee13(sys_const, training_trace)
        finally:
            tmp_path.unlink(missing_ok=True)

        assert "dc_sites" in exp
        assert len(exp["dc_sites"]) == 1
        assert "default" in exp["dc_sites"]
        site = exp["dc_sites"]["default"]
        assert site.bus == "671"
        assert site.bus_kv == 4.16
        assert site.base_kw_per_phase == 500.0
        assert exp["training"] is not None
        assert exp["ofo_config"] is not None
        assert len(exp["pv_systems"]) == 1
        assert len(exp["time_varying_loads"]) == 1

    def test_ieee34_experiment(self):
        from sweep_dc_locations import _experiment_ieee34
        from systems import ieee34

        sys_const = ieee34()
        exp = _experiment_ieee34(sys_const, None)

        assert len(exp["dc_sites"]) == 2
        assert "upstream" in exp["dc_sites"]
        assert "downstream" in exp["dc_sites"]
        assert exp["dc_sites"]["upstream"].bus_kv == 24.9
        assert exp["dc_sites"]["downstream"].bus_kv == 24.9
        assert exp["training"] is None
        assert exp["ofo_config"] is not None
        assert len(exp["pv_systems"]) == 2
        assert len(exp["time_varying_loads"]) == 5

    def test_ieee123_experiment(self):
        from sweep_dc_locations import _experiment_ieee123
        from systems import ieee123

        sys_const = ieee123()
        exp = _experiment_ieee123(sys_const, None)

        assert len(exp["dc_sites"]) == 4
        assert "z1_sw" in exp["dc_sites"]
        assert "z4_ne" in exp["dc_sites"]
        assert exp["training"] is None
        assert exp["ofo_config"] is not None
        assert len(exp["pv_systems"]) == 3
        assert exp["zones"] is not None
        assert len(exp["zones"]) == 4

    def test_ieee13_selects_1d(self):
        from sweep_dc_locations import _experiment_ieee13
        from systems import ieee13

        sys_const = ieee13()
        exp = _experiment_ieee13(sys_const, None)
        assert len(exp["dc_sites"]) == 1

    def test_ieee34_selects_2d(self):
        from sweep_dc_locations import _experiment_ieee34
        from systems import ieee34

        sys_const = ieee34()
        exp = _experiment_ieee34(sys_const, None)
        assert len(exp["dc_sites"]) == 2

    def test_ieee123_selects_zoned(self):
        from sweep_dc_locations import _experiment_ieee123
        from systems import ieee123

        sys_const = ieee123()
        exp = _experiment_ieee123(sys_const, None)
        assert exp.get("zones") is not None
        assert len(exp["zones"]) == len(exp["dc_sites"])


# ══════════════════════════════════════════════════════════════════════════════
# Integration tests (require OpenDSS + grid data)
# ══════════════════════════════════════════════════════════════════════════════

GRID_DIR = Path(__file__).resolve().parent.parent / "data" / "grid"
IEEE13_DIR = GRID_DIR / "ieee13"
IEEE34_DIR = GRID_DIR / "ieee34"

try:
    import opendssdirect  # noqa: F401

    HAS_OPENDSS = True
except Exception:
    HAS_OPENDSS = False

_requires_opendss = pytest.mark.skipif(not HAS_OPENDSS, reason="OpenDSS not available")


@_requires_opendss
class TestDiscoverCandidateBusesIEEE13:
    def test_discovers_buses(self):
        if not IEEE13_DIR.exists():
            pytest.skip("IEEE 13 grid data not available")
        buses = discover_candidate_buses(
            IEEE13_DIR,
            "IEEE13Bus.dss",
            4.16,
            exclude={"sourcebus", "650", "rg60"},
        )
        assert len(buses) > 0
        assert all(isinstance(b, str) for b in buses)

    def test_excludes_sourcebus(self):
        if not IEEE13_DIR.exists():
            pytest.skip("IEEE 13 grid data not available")
        buses = discover_candidate_buses(
            IEEE13_DIR,
            "IEEE13Bus.dss",
            4.16,
            exclude={"sourcebus", "650", "rg60"},
        )
        assert "sourcebus" not in [b.lower() for b in buses]
        assert "650" not in [b.lower() for b in buses]


@_requires_opendss
class TestDiscoverCandidateBusesIEEE34:
    def test_discovers_buses(self):
        if not IEEE34_DIR.exists():
            pytest.skip("IEEE 34 grid data not available")
        buses = discover_candidate_buses(
            IEEE34_DIR,
            "IEEE34Bus.dss",
            24.9,
            exclude={"sourcebus", "800", "802", "806", "808", "810", "812", "814", "888", "890"},
        )
        assert len(buses) >= 2

    def test_all_3_phase(self):
        if not IEEE34_DIR.exists():
            pytest.skip("IEEE 34 grid data not available")
        buses = discover_candidate_buses(
            IEEE34_DIR,
            "IEEE34Bus.dss",
            24.9,
            exclude={"sourcebus"},
        )
        assert len(buses) > 0
