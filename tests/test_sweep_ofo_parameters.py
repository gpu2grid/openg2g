"""Tests for profile generation utilities used by OFO parameter sweeps."""

from __future__ import annotations

import numpy as np

from openg2g.grid.generator import SyntheticPV
from openg2g.grid.load import SyntheticLoad
from openg2g.utils import smooth_bump


class TestSmoothBump:
    def test_center_is_one(self):
        assert smooth_bump(100.0, 100.0, 50.0) == 1.0

    def test_edge_is_zero(self):
        assert smooth_bump(150.0, 100.0, 50.0) == 0.0

    def test_outside_is_zero(self):
        assert smooth_bump(200.0, 100.0, 50.0) == 0.0

    def test_symmetric(self):
        v1 = smooth_bump(80.0, 100.0, 50.0)
        v2 = smooth_bump(120.0, 100.0, 50.0)
        assert abs(v1 - v2) < 1e-12


class TestProfiles:
    def test_pv_nonnegative(self):
        pv0 = SyntheticPV(peak_kw=1000.0, site_idx=0)
        pv1 = SyntheticPV(peak_kw=1000.0, site_idx=1)
        for t in np.linspace(0, 3600, 50):
            assert pv0.power_kw(float(t)) >= 0.0
            assert pv1.power_kw(float(t)) >= 0.0

    def test_load_nonnegative(self):
        for idx in range(5):
            load = SyntheticLoad(peak_kw=500.0, site_idx=idx)
            for t in np.linspace(0, 3600, 50):
                assert load.power_kw(float(t)) >= 0.0
