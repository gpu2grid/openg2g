"""Tests for the unified sweep_dc_locations module.

Validates config parsing, sweep mode selection (1-D vs 2-D), and shared
helper functions for both IEEE 13 and IEEE 34 systems.

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
        DCSiteConfig,
        SimulationParams,
        SweepConfig,
        _build_ofo_config,
        _extract_scenario_base_taps,
        _parse_fraction,
        _parse_tap,
        _resolve_models_for_site,
        _smooth_bump,
        _taps_dict_to_position,
        discover_candidate_buses,
        eval_profile,
        find_violations,
        load_profile_kw,
        pv_profile_kw,
    )

    from openg2g.grid.config import TapPosition

    CAN_IMPORT = True
except Exception:
    CAN_IMPORT = False

pytestmark = pytest.mark.skipif(not CAN_IMPORT, reason="Cannot import sweep_dc_locations (OpenDSS unavailable)")


# ── Config file paths ────────────────────────────────────────────────────────

CONFIG_DIR = Path(__file__).resolve().parent.parent / "examples" / "offline"
CONFIG_IEEE13 = CONFIG_DIR / "config_ieee13.json"
CONFIG_IEEE34 = CONFIG_DIR / "config_ieee34.json"


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests: helpers and config parsing
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


class TestParseFraction:
    def test_integer(self):
        assert _parse_fraction("60") == 60

    def test_fraction(self):
        assert _parse_fraction("1/10") == pytest.approx(0.1)

    def test_fraction_value(self):
        from fractions import Fraction

        assert _parse_fraction("1/10") == Fraction(1, 10)


class TestParseTap:
    def test_none(self):
        assert _parse_tap(None) is None

    def test_string_positive(self):
        assert _parse_tap("+14") == pytest.approx(1.0 + 14 * TAP_STEP)

    def test_string_negative(self):
        assert _parse_tap("-3") == pytest.approx(1.0 - 3 * TAP_STEP)

    def test_string_zero(self):
        assert _parse_tap("+0") == pytest.approx(1.0)

    def test_float(self):
        assert _parse_tap(1.05) == pytest.approx(1.05)


class TestTapsDictToPosition:
    def test_none(self):
        assert _taps_dict_to_position(None) is None

    def test_empty(self):
        assert _taps_dict_to_position({}) is None

    def test_valid(self):
        tp = _taps_dict_to_position({"reg1": "+14", "reg2": "+6", "reg3": "+15"})
        assert tp is not None
        assert len(tp.regulators) == 3
        assert tp.regulators["reg1"] == pytest.approx(1.0 + 14 * TAP_STEP)


class TestResolveModels:
    def test_none_returns_all(self):
        from openg2g.datacenter.config import InferenceModelSpec

        models = (
            InferenceModelSpec(
                model_label="A",
                model_id="a",
                gpus_per_replica=1,
                initial_num_replicas=1,
                initial_batch_size=8,
                itl_deadline_s=0.1,
                feasible_batch_sizes=[8],
            ),
            InferenceModelSpec(
                model_label="B",
                model_id="b",
                gpus_per_replica=1,
                initial_num_replicas=1,
                initial_batch_size=8,
                itl_deadline_s=0.1,
                feasible_batch_sizes=[8],
            ),
        )
        site = DCSiteConfig(bus="x", models=None)
        result = _resolve_models_for_site(site, models)
        assert result == models

    def test_filter(self):
        from openg2g.datacenter.config import InferenceModelSpec

        models = (
            InferenceModelSpec(
                model_label="A",
                model_id="a",
                gpus_per_replica=1,
                initial_num_replicas=1,
                initial_batch_size=8,
                itl_deadline_s=0.1,
                feasible_batch_sizes=[8],
            ),
            InferenceModelSpec(
                model_label="B",
                model_id="b",
                gpus_per_replica=1,
                initial_num_replicas=1,
                initial_batch_size=8,
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
                initial_num_replicas=1,
                initial_batch_size=8,
                itl_deadline_s=0.1,
                feasible_batch_sizes=[8],
            ),
        )
        site = DCSiteConfig(bus="x", models=["Z"])
        with pytest.raises(ValueError, match="unknown model labels"):
            _resolve_models_for_site(site, models)


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
# Config parsing and mode selection
# ══════════════════════════════════════════════════════════════════════════════


class TestSweepConfig:
    def test_ieee13_parses(self):
        config = SweepConfig.model_validate_json(CONFIG_IEEE13.read_bytes())
        assert config.num_dc_sites == 1
        assert "default" in config.dc_sites

    def test_ieee34_parses(self):
        config = SweepConfig.model_validate_json(CONFIG_IEEE34.read_bytes())
        assert config.num_dc_sites == 2
        assert "upstream" in config.dc_sites
        assert "downstream" in config.dc_sites

    def test_ieee13_selects_1d(self):
        config = SweepConfig.model_validate_json(CONFIG_IEEE13.read_bytes())
        assert config.num_dc_sites <= 1

    def test_ieee34_selects_2d(self):
        config = SweepConfig.model_validate_json(CONFIG_IEEE34.read_bytes())
        assert config.num_dc_sites >= 2

    def test_default_dc_site(self):
        minimal = {
            "models": [],
            "data_sources": [],
            "ieee_case_dir": ".",
        }
        config = SweepConfig.model_validate(minimal)
        assert config.dc_sites is not None
        assert "default" in config.dc_sites
        assert config.num_dc_sites == 1

    def test_data_hash_deterministic(self):
        config = SweepConfig.model_validate_json(CONFIG_IEEE13.read_bytes())
        h1 = config.data_hash
        h2 = config.data_hash
        assert h1 == h2
        assert len(h1) == 16

    def test_ieee13_dc_site_properties(self):
        config = SweepConfig.model_validate_json(CONFIG_IEEE13.read_bytes())
        site = config.dc_sites["default"]
        assert site.bus == "671"
        assert site.bus_kv == 4.16
        assert site.base_kw_per_phase == 500.0

    def test_ieee34_dc_site_properties(self):
        config = SweepConfig.model_validate_json(CONFIG_IEEE34.read_bytes())
        upstream = config.dc_sites["upstream"]
        downstream = config.dc_sites["downstream"]
        assert upstream.bus_kv == 24.9
        assert downstream.bus_kv == 24.9
        assert upstream.models is not None
        assert downstream.models is not None
        assert set(upstream.models) != set(downstream.models)

    def test_ieee13_has_tap_schedule(self):
        config = SweepConfig.model_validate_json(CONFIG_IEEE13.read_bytes())
        assert len(config.tap_schedule) == 2
        assert config.tap_schedule[0].t == 1500
        assert config.tap_schedule[1].t == 3300

    def test_ieee34_has_tap_schedule(self):
        config = SweepConfig.model_validate_json(CONFIG_IEEE34.read_bytes())
        assert len(config.tap_schedule) == 1
        assert config.tap_schedule[0].t == 1800

    def test_ieee13_has_training_and_site_ramps(self):
        config = SweepConfig.model_validate_json(CONFIG_IEEE13.read_bytes())
        assert config.training is not None
        # Inference ramps are per-site, not top-level
        site = config.dc_sites["default"]
        assert len(site.inference_ramps) > 0

    def test_ieee34_no_training(self):
        config = SweepConfig.model_validate_json(CONFIG_IEEE34.read_bytes())
        assert config.training is None
        assert config.inference_ramp is None

    def test_ieee34_has_regulator_zones(self):
        config = SweepConfig.model_validate_json(CONFIG_IEEE34.read_bytes())
        assert config.regulator_zones is not None
        assert "creg1" in config.regulator_zones
        assert "creg2" in config.regulator_zones


class TestExtractScenarioBaseTaps:
    def test_no_schedule(self):
        config = SweepConfig.model_validate_json(CONFIG_IEEE34.read_bytes())
        initial = TapPosition(regulators={"reg1": 1.0})
        bases = _extract_scenario_base_taps(config, initial)
        assert "inference" in bases
        assert len(bases) == 1

    def test_with_schedule(self):
        config = SweepConfig.model_validate_json(CONFIG_IEEE13.read_bytes())
        initial = _taps_dict_to_position(config.initial_taps)
        bases = _extract_scenario_base_taps(config, initial)
        assert "inference" in bases
        assert "training" in bases


class TestBuildOFOConfig:
    def test_builds(self):
        from sweep_dc_locations import OFOParams

        ofo_params = OFOParams()
        sim = SimulationParams()
        ofo_config = _build_ofo_config(ofo_params, sim)
        assert ofo_config.v_min == 0.95
        assert ofo_config.v_max == 1.05


# ══════════════════════════════════════════════════════════════════════════════
# Integration tests (require OpenDSS + grid data)
# ══════════════════════════════════════════════════════════════════════════════

GRID_DIR = Path(__file__).resolve().parent.parent / "data" / "grid"
IEEE13_DIR = GRID_DIR / "ieee13"
IEEE34_DIR = GRID_DIR / "ieee34"


class TestDiscoverCandidateBusesIEEE13:
    def test_discovers_buses(self):
        if not IEEE13_DIR.exists():
            pytest.skip("IEEE 13 grid data not available")
        buses = discover_candidate_buses(
            IEEE13_DIR,
            "IEEE13Nodeckt.dss",
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
            "IEEE13Nodeckt.dss",
            4.16,
            exclude={"sourcebus", "650", "rg60"},
        )
        assert "sourcebus" not in [b.lower() for b in buses]
        assert "650" not in [b.lower() for b in buses]


class TestDiscoverCandidateBusesIEEE34:
    def test_discovers_buses(self):
        if not IEEE34_DIR.exists():
            pytest.skip("IEEE 34 grid data not available")
        buses = discover_candidate_buses(
            IEEE34_DIR,
            "ieee34Mod1_halfline.dss",
            24.9,
            exclude={"sourcebus", "800", "802", "806", "808", "810", "812", "814", "888", "890"},
        )
        assert len(buses) >= 2

    def test_all_3_phase(self):
        if not IEEE34_DIR.exists():
            pytest.skip("IEEE 34 grid data not available")
        buses = discover_candidate_buses(
            IEEE34_DIR,
            "ieee34Mod1_halfline.dss",
            24.9,
            exclude={"sourcebus"},
        )
        assert len(buses) > 0
