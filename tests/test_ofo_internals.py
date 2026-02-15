"""Tests for OFO controller internals — PerModelPrimalX, FullNetworkVoltageDual,
phase_share_from_placement.

Focuses on edge cases, projections, NaN handling, and shape validation —
not trivial arithmetic."""

from __future__ import annotations

import math

import numpy as np
import pytest
from mlenergy_data.modeling import LogisticModel

from openg2g.controller.ofo import (
    FullNetworkVoltageDual,
    PerModelPrimalX,
    PrimalCfg,
    VoltageDualCfg,
    phase_share_from_placement,
)
from openg2g.models.spec import LLMInferenceModelSpec


def _trivial_logistic(L: float = 100.0, x0: float = 5.0, k: float = 1.0, b0: float = 0.0):
    """Build a LogisticModel with known parameters."""
    return LogisticModel.from_dict({"L": L, "x0": x0, "k": k, "b0": b0})


def _make_primal(
    *,
    batch_set: list[int] | None = None,
    cfg: PrimalCfg | None = None,
) -> PerModelPrimalX:
    """Build a PerModelPrimalX with a single model and trivial fits."""
    if batch_set is None:
        batch_set = [8, 16, 32, 64, 128]
    if cfg is None:
        cfg = PrimalCfg()
    model = LLMInferenceModelSpec("M", num_replicas=10, gpus_per_replica=1)
    fit = _trivial_logistic()
    return PerModelPrimalX(
        models=[model],
        batch_set=batch_set,
        power_fits={"M": fit},
        latency_fits={"M": fit},
        throughput_fits={"M": fit},
        cfg=cfg,
    )


def _step_no_gradient(p: PerModelPrimalX) -> dict[str, int]:
    """Run a step with zero gradient (all weights and duals zero)."""
    n = 6
    return p.step(
        eta_vec=np.zeros(n),
        H=np.zeros((n, 3)),
        phase_share_by_model={},
        mu_by_model={"M": 0.0},
        w_by_model={"M": 0.0},
    )


class TestBatchDiscretization:
    def test_each_batch_size_round_trips(self) -> None:
        """Initializing to any feasible batch size and taking a zero-gradient
        step should return that same batch size (log2 -> discrete round-trip)."""
        for b in [8, 16, 32, 64, 128]:
            p = _make_primal(batch_set=[8, 16, 32, 64, 128])
            p.init_from_batches({"M": b})
            assert _step_no_gradient(p)["M"] == b

    def test_single_batch_set(self) -> None:
        """With only one feasible batch size, discretization should always
        return it regardless of continuous x."""
        p = _make_primal(batch_set=[64])
        p.init_from_batches({"M": 64})
        assert _step_no_gradient(p)["M"] == 64


class TestInitFromBatches:
    def test_missing_model_defaults_to_max(self) -> None:
        """Models not in the init dict should default to the max batch size."""
        p = _make_primal(batch_set=[8, 16, 32, 64, 128])
        p.init_from_batches({})
        assert p.x_by_model["M"] == pytest.approx(math.log2(128))

    def test_x_prev_matches_x(self) -> None:
        """After init, x_prev should equal x (no switching cost on first step)."""
        p = _make_primal(batch_set=[8, 16, 32, 64, 128])
        p.init_from_batches({"M": 64})
        assert p.x_prev_by_model["M"] == p.x_by_model["M"]


class TestVoltageDualUpdate:
    def test_no_violation_stays_zero(self) -> None:
        """Voltages within bounds should leave both duals at zero."""
        vd = FullNetworkVoltageDual(3, VoltageDualCfg(v_min=0.95, v_max=1.05, rho_v=1.0))
        vd.update(np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(vd.lam_low, np.zeros(3))
        np.testing.assert_array_equal(vd.lam_high, np.zeros(3))

    def test_dual_projects_to_nonneg(self) -> None:
        """After a violation clears, the dual should be clamped to zero
        (not go negative), preserving the [.]+ projection."""
        vd = FullNetworkVoltageDual(1, VoltageDualCfg(v_min=0.95, v_max=1.05, rho_v=1.0))
        vd.update(np.array([0.93]))
        vd.update(np.array([1.0]))
        assert vd.lam_low[0] == 0.0

    def test_eta_sign_convention(self) -> None:
        """eta = lam_high - lam_low should be negative for undervoltage
        (drives power down) and positive for overvoltage."""
        vd = FullNetworkVoltageDual(2, VoltageDualCfg(v_min=0.95, v_max=1.05, rho_v=1.0))
        vd.update(np.array([0.93, 1.07]))
        eta = vd.eta()
        assert eta[0] < 0
        assert eta[1] > 0

    def test_shape_mismatch_raises(self) -> None:
        """Passing a v_hat with wrong length should raise ValueError."""
        vd = FullNetworkVoltageDual(3, VoltageDualCfg())
        with pytest.raises(ValueError, match="len 2 but duals have len 3"):
            vd.update(np.array([1.0, 1.0]))


class TestPhaseShareFromPlacement:
    def test_unbalanced(self) -> None:
        """Unequal server counts should produce proportional phase shares."""
        share = phase_share_from_placement({"servers_A": 6, "servers_B": 3, "servers_C": 1})
        np.testing.assert_allclose(share, [0.6, 0.3, 0.1])

    def test_all_zero_returns_uniform(self) -> None:
        """Zero servers on all phases should fall back to uniform 1/3 share."""
        share = phase_share_from_placement({"servers_A": 0, "servers_B": 0, "servers_C": 0})
        np.testing.assert_allclose(share, [1 / 3, 1 / 3, 1 / 3])

    def test_empty_dict_returns_uniform(self) -> None:
        """Missing placement keys should fall back to uniform 1/3 share."""
        share = phase_share_from_placement({})
        np.testing.assert_allclose(share, [1 / 3, 1 / 3, 1 / 3])


class TestPrimalStep:
    def test_zero_gradient_no_change(self) -> None:
        """With all gradient weights and duals set to zero, the batch size
        should remain unchanged after a step."""
        p = _make_primal(
            batch_set=[8, 16, 32, 64, 128],
            cfg=PrimalCfg(eta_primal=0.1, w_latency=0.0, w_throughput=0.0, w_switch=0.0, k_v=0.0),
        )
        p.init_from_batches({"M": 32})
        batch = _step_no_gradient(p)
        assert batch["M"] == 32

    def test_latency_dual_pushes_batch_down(self) -> None:
        """A positive latency dual (mu > 0) should decrease x, since the
        logistic latency fit has dL/dx > 0 (latency increases with batch)."""
        p = _make_primal(
            batch_set=[8, 16, 32, 64, 128],
            cfg=PrimalCfg(eta_primal=0.5, w_latency=0.0, w_throughput=0.0, w_switch=0.0, k_v=0.0),
        )
        p.init_from_batches({"M": 64})
        x_before = p.x_by_model["M"]
        p.step(
            eta_vec=np.zeros(6),
            H=np.zeros((6, 3)),
            phase_share_by_model={},
            mu_by_model={"M": 5.0},
            w_by_model={"M": 0.0},
        )
        assert p.x_by_model["M"] < x_before

    def test_nan_mu_treated_as_zero(self) -> None:
        """NaN mu (from zero-replica models) should be treated as zero,
        producing no latency gradient contribution."""
        p = _make_primal(
            batch_set=[8, 16, 32, 64, 128],
            cfg=PrimalCfg(eta_primal=0.1, w_latency=0.0, w_throughput=0.0, w_switch=0.0, k_v=0.0),
        )
        p.init_from_batches({"M": 32})
        batch = p.step(
            eta_vec=np.zeros(6),
            H=np.zeros((6, 3)),
            phase_share_by_model={},
            mu_by_model={"M": float("nan")},
            w_by_model={"M": 0.0},
        )
        assert batch["M"] == 32

    def test_projection_clamps_with_huge_gradient(self) -> None:
        """Even with an extreme gradient, x should stay within
        [log2(min_batch), log2(max_batch)] after projection."""
        p = _make_primal(
            batch_set=[8, 128],
            cfg=PrimalCfg(
                eta_primal=100.0, w_latency=0.0, w_throughput=0.0, w_switch=0.0, k_v=1e9
            ),
        )
        p.init_from_batches({"M": 64})
        p.step(
            eta_vec=np.ones(6) * 100.0,
            H=np.ones((6, 3)),
            phase_share_by_model={"M": np.array([0.5, 0.3, 0.2])},
            mu_by_model={"M": 0.0},
            w_by_model={"M": 100.0},
        )
        assert p.x_by_model["M"] >= math.log2(8)
        assert p.x_by_model["M"] <= math.log2(128)
