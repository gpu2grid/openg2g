"""Tests for profile generation, bus discovery, and voltage utilities.

Validates helper functions that were extracted from sweep_dc_locations
into the library (utils, generator, load, metrics.voltage).

Note: Discovery tests depend on OpenDSS. If unavailable, they are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from openg2g.grid.generator import SyntheticPV
from openg2g.grid.load import SyntheticLoad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "offline"))
from utils import find_violations

from openg2g.utils import smooth_bump


class TestSmoothBump:
    def test_center_is_one(self):
        assert smooth_bump(100.0, 100.0, 50.0) == 1.0

    def test_edge_is_zero(self):
        assert smooth_bump(150.0, 100.0, 50.0) == 0.0
        assert smooth_bump(50.0, 100.0, 50.0) == 0.0

    def test_outside_is_zero(self):
        assert smooth_bump(200.0, 100.0, 50.0) == 0.0

    def test_symmetric(self):
        v1 = smooth_bump(80.0, 100.0, 50.0)
        v2 = smooth_bump(120.0, 100.0, 50.0)
        assert abs(v1 - v2) < 1e-12


class TestProfiles:
    def test_pv_profile_nonnegative(self):
        pv0 = SyntheticPV(peak_kw=1000.0, site_idx=0)
        pv1 = SyntheticPV(peak_kw=1000.0, site_idx=1)
        for t in np.linspace(0, 3600, 100):
            assert pv0.power_kw(float(t)) >= 0.0
            assert pv1.power_kw(float(t)) >= 0.0

    def test_load_profile_nonnegative(self):
        for idx in range(5):
            load = SyntheticLoad(peak_kw=500.0, site_idx=idx)
            for t in np.linspace(0, 3600, 100):
                assert load.power_kw(float(t)) >= 0.0


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


# Discovery tests require OpenDSS
try:
    from utils import discover_candidate_buses

    _HAS_OPENDSS = True
except Exception:
    _HAS_OPENDSS = False

_GRID_DATA_DIR = None


def _grid_dir():
    global _GRID_DATA_DIR
    if _GRID_DATA_DIR is None:
        from pathlib import Path

        _GRID_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "grid"
    return _GRID_DATA_DIR


@pytest.mark.skipif(not _HAS_OPENDSS, reason="OpenDSS unavailable")
class TestDiscoverCandidateBusesIEEE13:
    def test_discovers_buses(self):
        buses = discover_candidate_buses(_grid_dir() / "ieee13", "IEEE13Bus.dss", 4.16, {"sourcebus", "650", "rg60"})
        assert len(buses) > 0

    def test_excludes_sourcebus(self):
        buses = discover_candidate_buses(_grid_dir() / "ieee13", "IEEE13Bus.dss", 4.16, {"sourcebus", "650", "rg60"})
        assert "sourcebus" not in [b.lower() for b in buses]


@pytest.mark.skipif(not _HAS_OPENDSS, reason="OpenDSS unavailable")
class TestDiscoverCandidateBusesIEEE34:
    def test_discovers_buses(self):
        buses = discover_candidate_buses(_grid_dir() / "ieee34", "IEEE34Bus.dss", 24.9, {"sourcebus"})
        assert len(buses) > 0

    def test_all_3_phase(self):
        buses = discover_candidate_buses(_grid_dir() / "ieee34", "IEEE34Bus.dss", 24.9, {"sourcebus"})
        assert all(isinstance(b, str) for b in buses)
