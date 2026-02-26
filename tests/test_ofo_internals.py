"""Tests for OFO controller internals: PrimalBatchOptimizer, VoltageDualVariables.

Focuses on edge cases, projections, NaN handling, and shape validation,
not trivial arithmetic."""

from __future__ import annotations

import math

import numpy as np
import pytest
from mlenergy_data.modeling import LogisticModel

from openg2g.controller.ofo import (
    PrimalBatchOptimizer,
    VoltageDualVariables,
    _PrimalConfig,
    _VoltageDualConfig,
)
from openg2g.models.spec import LLMInferenceModelSpec


def _trivial_logistic(L: float = 100.0, x0: float = 5.0, k: float = 1.0, b0: float = 0.0):
    """Build a LogisticModel with known parameters."""
    return LogisticModel.from_dict({"L": L, "x0": x0, "k": k, "b0": b0})


def _make_primal(
    *,
    feasible_batch_sizes: list[int] | None = None,
    config: _PrimalConfig | None = None,
) -> PrimalBatchOptimizer:
    """Build a PrimalBatchOptimizer with a single model and trivial fits."""
    if feasible_batch_sizes is None:
        feasible_batch_sizes = [8, 16, 32, 64, 128]
    if config is None:
        config = _PrimalConfig()
    model = LLMInferenceModelSpec("M", num_replicas=10, gpus_per_replica=1, initial_batch_size=128, itl_deadline_s=0.1)
    fit = _trivial_logistic()
    return PrimalBatchOptimizer(
        models=[model],
        feasible_batch_sizes=feasible_batch_sizes,
        power_fits={"M": fit},
        latency_fits={"M": fit},
        throughput_fits={"M": fit},
        config=config,
    )


def _step_no_gradient(p: PrimalBatchOptimizer) -> dict[str, int]:
    """Run a step with zero gradient (all weights and duals zero)."""
    n = 6
    return p.step(
        voltage_dual_diff=np.zeros(n),
        sensitivity_matrix=np.zeros((n, 3)),
        phase_share_by_model={},
        latency_dual_by_model={"M": 0.0},
        replica_count_by_model={"M": 0.0},
    )


class TestBatchDiscretization:
    def test_each_batch_size_round_trips(self) -> None:
        """Initializing to any feasible batch size and taking a zero-gradient
        step should return that same batch size (log2 -> discrete round-trip)."""
        for b in [8, 16, 32, 64, 128]:
            p = _make_primal(feasible_batch_sizes=[8, 16, 32, 64, 128])
            p.init_from_batches({"M": b})
            assert _step_no_gradient(p)["M"] == b

    def test_single_batch_set(self) -> None:
        """With only one feasible batch size, discretization should always
        return it regardless of continuous x."""
        p = _make_primal(feasible_batch_sizes=[64])
        p.init_from_batches({"M": 64})
        assert _step_no_gradient(p)["M"] == 64


class TestInitFromBatches:
    def test_missing_model_defaults_to_max(self) -> None:
        """Models not in the init dict should default to the max batch size."""
        p = _make_primal(feasible_batch_sizes=[8, 16, 32, 64, 128])
        p.init_from_batches({})
        assert p.log_batch_size_by_model["M"] == pytest.approx(math.log2(128))

    def test_x_prev_matches_x(self) -> None:
        """After init, x_prev should equal x (no switching cost on first step)."""
        p = _make_primal(feasible_batch_sizes=[8, 16, 32, 64, 128])
        p.init_from_batches({"M": 64})
        assert p.prev_log_batch_size_by_model["M"] == p.log_batch_size_by_model["M"]


class TestVoltageDualUpdate:
    def test_no_violation_stays_zero(self) -> None:
        """Voltages within bounds should leave both duals at zero."""
        vd = VoltageDualVariables(3, _VoltageDualConfig(v_min=0.95, v_max=1.05, ascent_step_size=1.0))
        vd.update(np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(vd.dual_undervoltage, np.zeros(3))
        np.testing.assert_array_equal(vd.dual_overvoltage, np.zeros(3))

    def test_dual_projects_to_nonneg(self) -> None:
        """After a violation clears, the dual should be clamped to zero
        (not go negative), preserving the [.]+ projection."""
        vd = VoltageDualVariables(1, _VoltageDualConfig(v_min=0.95, v_max=1.05, ascent_step_size=1.0))
        vd.update(np.array([0.93]))
        vd.update(np.array([1.0]))
        assert vd.dual_undervoltage[0] == 0.0

    def test_eta_sign_convention(self) -> None:
        """eta = dual_overvoltage - dual_undervoltage should be negative for undervoltage
        (drives power down) and positive for overvoltage."""
        vd = VoltageDualVariables(2, _VoltageDualConfig(v_min=0.95, v_max=1.05, ascent_step_size=1.0))
        vd.update(np.array([0.93, 1.07]))
        eta = vd.dual_difference()
        assert eta[0] < 0
        assert eta[1] > 0

    def test_shape_mismatch_raises(self) -> None:
        """Passing a v_hat with wrong length should raise ValueError."""
        vd = VoltageDualVariables(3, _VoltageDualConfig())
        with pytest.raises(ValueError, match="len 2 but duals have len 3"):
            vd.update(np.array([1.0, 1.0]))


class TestPrimalStep:
    def test_zero_gradient_no_change(self) -> None:
        """With all gradient weights and duals set to zero, the batch size
        should remain unchanged after a step."""
        p = _make_primal(
            feasible_batch_sizes=[8, 16, 32, 64, 128],
            config=_PrimalConfig(
                descent_step_size=0.1,
                w_throughput=0.0,
                w_switch=0.0,
                voltage_gradient_scale=0.0,
            ),
        )
        p.init_from_batches({"M": 32})
        batch = _step_no_gradient(p)
        assert batch["M"] == 32

    def test_latency_dual_pushes_batch_down(self) -> None:
        """A positive latency dual (mu > 0) should decrease x, since the
        logistic latency fit has dL/dx > 0 (latency increases with batch)."""
        p = _make_primal(
            feasible_batch_sizes=[8, 16, 32, 64, 128],
            config=_PrimalConfig(
                descent_step_size=0.5,
                w_throughput=0.0,
                w_switch=0.0,
                voltage_gradient_scale=0.0,
            ),
        )
        p.init_from_batches({"M": 64})
        x_before = p.log_batch_size_by_model["M"]
        p.step(
            voltage_dual_diff=np.zeros(6),
            sensitivity_matrix=np.zeros((6, 3)),
            phase_share_by_model={},
            latency_dual_by_model={"M": 5.0},
            replica_count_by_model={"M": 0.0},
        )
        assert p.log_batch_size_by_model["M"] < x_before

    def test_nan_mu_treated_as_zero(self) -> None:
        """NaN mu (from zero-replica models) should be treated as zero,
        producing no latency gradient contribution."""
        p = _make_primal(
            feasible_batch_sizes=[8, 16, 32, 64, 128],
            config=_PrimalConfig(
                descent_step_size=0.1,
                w_throughput=0.0,
                w_switch=0.0,
                voltage_gradient_scale=0.0,
            ),
        )
        p.init_from_batches({"M": 32})
        batch = p.step(
            voltage_dual_diff=np.zeros(6),
            sensitivity_matrix=np.zeros((6, 3)),
            phase_share_by_model={},
            latency_dual_by_model={"M": float("nan")},
            replica_count_by_model={"M": 0.0},
        )
        assert batch["M"] == 32

    def test_projection_clamps_with_huge_gradient(self) -> None:
        """Even with an extreme gradient, x should stay within
        [log2(min_batch), log2(max_batch)] after projection."""
        p = _make_primal(
            feasible_batch_sizes=[8, 128],
            config=_PrimalConfig(
                descent_step_size=100.0,
                w_throughput=0.0,
                w_switch=0.0,
                voltage_gradient_scale=1e9,
            ),
        )
        p.init_from_batches({"M": 64})
        p.step(
            voltage_dual_diff=np.ones(6) * 100.0,
            sensitivity_matrix=np.ones((6, 3)),
            phase_share_by_model={"M": np.array([0.5, 0.3, 0.2])},
            latency_dual_by_model={"M": 0.0},
            replica_count_by_model={"M": 100.0},
        )
        assert p.log_batch_size_by_model["M"] >= math.log2(8)
        assert p.log_batch_size_by_model["M"] <= math.log2(128)
