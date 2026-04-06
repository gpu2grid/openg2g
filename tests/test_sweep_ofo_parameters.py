"""Tests for the sweep_ofo_parameters module.

Validates sweep grid construction (1-D and 2-D), config parsing, and helper
functions for both single-DC and multi-DC systems.

Note: This module depends on OpenDSS (via transitive imports). If OpenDSS
is not available or crashes, the entire test module is skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add examples/offline to sys.path so we can import sweep_ofo_parameters
_examples_dir = Path(__file__).resolve().parent.parent / "examples" / "offline"
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

try:
    from sweep_ofo_parameters import (
        TAP_STEP,
        DCSiteConfig,
        SweepConfig,
        _compute_sweep_values,
        _parse_fraction,
        _parse_tap,
        _resolve_models_for_site,
        _run_id,
        _run_id_2d,
        _smooth_bump,
        build_sweep_grid,
        build_sweep_grid_2d,
        eval_profile,
        load_profile_kw,
        pv_profile_kw,
    )

    from openg2g.controller.ofo import OFOConfig

    CAN_IMPORT = True
except Exception:
    CAN_IMPORT = False

pytestmark = pytest.mark.skipif(not CAN_IMPORT, reason="Cannot import sweep_ofo_parameters (OpenDSS unavailable)")


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests: helpers
# ══════════════════════════════════════════════════════════════════════════════


class TestSmoothBump:
    def test_center_is_one(self):
        assert _smooth_bump(100.0, 100.0, 50.0) == 1.0

    def test_edge_is_zero(self):
        assert _smooth_bump(150.0, 100.0, 50.0) == 0.0

    def test_outside_is_zero(self):
        assert _smooth_bump(200.0, 100.0, 50.0) == 0.0

    def test_symmetric(self):
        v1 = _smooth_bump(80.0, 100.0, 50.0)
        v2 = _smooth_bump(120.0, 100.0, 50.0)
        assert abs(v1 - v2) < 1e-12


class TestProfiles:
    def test_pv_nonnegative(self):
        for t in np.linspace(0, 3600, 50):
            assert pv_profile_kw(float(t), 1000.0, site_idx=0) >= 0.0
            assert pv_profile_kw(float(t), 1000.0, site_idx=1) >= 0.0

    def test_load_nonnegative(self):
        for t in np.linspace(0, 3600, 50):
            for idx in range(5):
                assert load_profile_kw(float(t), 500.0, site_idx=idx) >= 0.0

    def test_eval_profile_with_csv(self):
        csv_data = (np.array([0.0, 100.0, 200.0]), np.array([0.0, 50.0, 100.0]))
        val = eval_profile(50.0, peak_kw=999.0, csv_data=csv_data, profile_fn=pv_profile_kw, site_idx=0)
        assert abs(val - 25.0) < 1e-10


class TestParseFraction:
    def test_integer(self):
        assert _parse_fraction("60") == 60

    def test_fraction(self):
        assert _parse_fraction("1/10") == pytest.approx(0.1)


class TestParseTap:
    def test_none(self):
        assert _parse_tap(None) is None

    def test_positive(self):
        assert _parse_tap("+14") == pytest.approx(1.0 + 14 * TAP_STEP)

    def test_negative(self):
        assert _parse_tap("-3") == pytest.approx(1.0 - 3 * TAP_STEP)

    def test_float(self):
        assert _parse_tap(1.05) == pytest.approx(1.05)


class TestRunId:
    def test_1d(self):
        assert _run_id("primal_step_size", 0.05) == "primal_step_size__0.05"

    def test_2d(self):
        rid = _run_id_2d("voltage_dual_step_size", {"upstream": 10.0, "downstream": 20.0})
        assert rid == "voltage_dual_step_size__upstream_10__downstream_20"

    def test_2d_float(self):
        rid = _run_id_2d("w_switch", {"site_a": 0.5, "site_b": 2.5})
        assert "site_a_0.5" in rid
        assert "site_b_2.5" in rid


# ══════════════════════════════════════════════════════════════════════════════
# Sweep grid construction
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildSweepGrid1D:
    def test_returns_list(self):
        baseline = OFOConfig()
        runs = build_sweep_grid(baseline)
        assert isinstance(runs, list)
        assert len(runs) > 0

    def test_each_run_has_three_elements(self):
        baseline = OFOConfig()
        for param_name, param_value, ofo_cfg in build_sweep_grid(baseline):
            assert isinstance(param_name, str)
            assert isinstance(param_value, (int, float))
            assert isinstance(ofo_cfg, OFOConfig)

    def test_multiplier_1_is_baseline(self):
        baseline = OFOConfig(primal_step_size=0.1)
        runs = build_sweep_grid(baseline, {"primal_step_size": [1.0]})
        assert len(runs) == 1
        assert runs[0][1] == pytest.approx(0.1)
        assert runs[0][2].primal_step_size == pytest.approx(0.1)

    def test_deduplication(self):
        baseline = OFOConfig(primal_step_size=0.1)
        runs = build_sweep_grid(baseline, {"primal_step_size": [1.0, 1.0, 2.0]})
        values = [v for _, v, _ in runs]
        assert len(values) == len(set(f"{v:.10f}" for v in values))

    def test_zero_baseline_uses_absolute(self):
        baseline = OFOConfig(w_throughput=0.0)
        runs = build_sweep_grid(baseline, {"w_throughput": [0.0, 0.5, 1.0]})
        values = sorted(v for _, v, _ in runs)
        assert values == [0.0, 0.5, 1.0]


class TestComputeSweepValues:
    def test_returns_dict(self):
        baseline = OFOConfig()
        result = _compute_sweep_values(baseline)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_custom_multipliers(self):
        baseline = OFOConfig(primal_step_size=0.1)
        result = _compute_sweep_values(baseline, {"primal_step_size": [0.5, 1.0, 2.0]})
        assert "primal_step_size" in result
        assert len(result["primal_step_size"]) == 3
        assert result["primal_step_size"][1] == pytest.approx(0.1)  # 1.0 * 0.1


class TestBuildSweepGrid2D:
    def test_two_sites(self):
        baseline = OFOConfig(primal_step_size=0.1)
        runs = build_sweep_grid_2d(
            baseline,
            ["upstream", "downstream"],
            {"primal_step_size": [0.5, 1.0, 2.0]},
        )
        # 3 values × 3 values = 9 combinations
        assert len(runs) == 9

    def test_returns_per_site_configs(self):
        baseline = OFOConfig(primal_step_size=0.1)
        runs = build_sweep_grid_2d(
            baseline,
            ["site_a", "site_b"],
            {"primal_step_size": [1.0, 2.0]},
        )
        for param_name, site_values, site_configs in runs:
            assert param_name == "primal_step_size"
            assert "site_a" in site_values
            assert "site_b" in site_values
            assert "site_a" in site_configs
            assert "site_b" in site_configs
            assert isinstance(site_configs["site_a"], OFOConfig)
            assert isinstance(site_configs["site_b"], OFOConfig)

    def test_independent_values(self):
        baseline = OFOConfig(primal_step_size=0.1)
        runs = build_sweep_grid_2d(
            baseline,
            ["A", "B"],
            {"primal_step_size": [1.0, 2.0]},
        )
        # Should have (0.1, 0.1), (0.1, 0.2), (0.2, 0.1), (0.2, 0.2)
        combos = [(sv["A"], sv["B"]) for _, sv, _ in runs]
        assert (0.1, 0.1) in [(pytest.approx(a), pytest.approx(b)) for a, b in combos]
        assert len(combos) == 4

    def test_configs_match_values(self):
        baseline = OFOConfig(voltage_dual_step_size=10.0)
        runs = build_sweep_grid_2d(
            baseline,
            ["up", "down"],
            {"voltage_dual_step_size": [0.5, 2.0]},
        )
        for _, site_values, site_configs in runs:
            for sid in ["up", "down"]:
                assert site_configs[sid].voltage_dual_step_size == pytest.approx(site_values[sid])

    def test_multiple_params(self):
        baseline = OFOConfig(primal_step_size=0.1, w_switch=1.0)
        runs = build_sweep_grid_2d(
            baseline,
            ["A", "B"],
            {"primal_step_size": [1.0, 2.0], "w_switch": [0.5, 1.0]},
        )
        # primal: 2×2=4, w_switch: 2×2=4, total=8
        assert len(runs) == 8
        param_names = [r[0] for r in runs]
        assert param_names.count("primal_step_size") == 4
        assert param_names.count("w_switch") == 4

    def test_three_sites(self):
        baseline = OFOConfig(primal_step_size=0.1)
        runs = build_sweep_grid_2d(
            baseline,
            ["A", "B", "C"],
            {"primal_step_size": [1.0, 2.0]},
        )
        # 2^3 = 8 combinations
        assert len(runs) == 8


# ══════════════════════════════════════════════════════════════════════════════
# Config parsing
# ══════════════════════════════════════════════════════════════════════════════


CONFIG_DIR = Path(__file__).resolve().parent.parent / "examples" / "offline"
CONFIG_IEEE13 = CONFIG_DIR / "config_ieee13.json"
CONFIG_IEEE34 = CONFIG_DIR / "config_ieee34.json"


class TestSweepConfigMode:
    def test_ieee13_single_dc(self):
        if not CONFIG_IEEE13.exists():
            pytest.skip("config_ieee13.json not found")
        config = SweepConfig.model_validate_json(CONFIG_IEEE13.read_bytes())
        assert len(config.dc_sites) == 1

    def test_ieee34_multi_dc(self):
        if not CONFIG_IEEE34.exists():
            pytest.skip("config_ieee34.json not found")
        config = SweepConfig.model_validate_json(CONFIG_IEEE34.read_bytes())
        assert len(config.dc_sites) == 2

    def test_default_dc_site(self):
        minimal = {"models": [], "data_sources": [], "ieee_case_dir": "."}
        config = SweepConfig.model_validate(minimal)
        assert "default" in config.dc_sites
        assert len(config.dc_sites) == 1

    def test_ieee34_site_ids(self):
        if not CONFIG_IEEE34.exists():
            pytest.skip("config_ieee34.json not found")
        config = SweepConfig.model_validate_json(CONFIG_IEEE34.read_bytes())
        assert "upstream" in config.dc_sites
        assert "downstream" in config.dc_sites


class TestResolveModels:
    def test_none_returns_all(self):
        from openg2g.datacenter.config import InferenceModelSpec

        models = (
            InferenceModelSpec(
                model_label="A",
                model_id="a",
                gpus_per_replica=1,
                itl_deadline_s=0.1,
                feasible_batch_sizes=[8],
            ),
            InferenceModelSpec(
                model_label="B",
                model_id="b",
                gpus_per_replica=1,
                itl_deadline_s=0.1,
                feasible_batch_sizes=[8],
            ),
        )
        site = DCSiteConfig(bus="x", models=None)
        assert _resolve_models_for_site(site, models) == models

    def test_filter(self):
        from openg2g.datacenter.config import InferenceModelSpec

        models = (
            InferenceModelSpec(
                model_label="A",
                model_id="a",
                gpus_per_replica=1,
                itl_deadline_s=0.1,
                feasible_batch_sizes=[8],
            ),
            InferenceModelSpec(
                model_label="B",
                model_id="b",
                gpus_per_replica=1,
                itl_deadline_s=0.1,
                feasible_batch_sizes=[8],
            ),
        )
        site = DCSiteConfig(bus="x", models=["B"])
        result = _resolve_models_for_site(site, models)
        assert len(result) == 1
        assert result[0].model_label == "B"

    def test_missing_raises(self):
        from openg2g.datacenter.config import InferenceModelSpec

        models = (
            InferenceModelSpec(
                model_label="A",
                model_id="a",
                gpus_per_replica=1,
                itl_deadline_s=0.1,
                feasible_batch_sizes=[8],
            ),
        )
        site = DCSiteConfig(bus="x", models=["Z"])
        with pytest.raises(ValueError, match="unknown model labels"):
            _resolve_models_for_site(site, models)
