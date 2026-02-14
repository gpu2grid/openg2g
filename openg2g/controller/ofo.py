"""Online Feedback Optimization (OFO) batch-size controller.

Implements the primal-dual algorithm for joint voltage regulation and
latency management via GPU batch size control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.datacenter.base import LLMBatchSizeControlledDatacenter
from openg2g.events import EventEmitter
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.models.logistic import LogisticFit
from openg2g.models.spec import ModelSpec
from openg2g.types import (
    Command,
    ControlAction,
)


@dataclass
class VoltageDualCfg:
    """Configuration for the voltage dual variable update.

    Attributes:
        v_min: Lower voltage limit (pu).
        v_max: Upper voltage limit (pu).
        rho_v: Step size for the voltage dual ascent.
    """

    v_min: float = 0.95
    v_max: float = 1.05
    rho_v: float = 0.5


@dataclass
class PrimalCfg:
    r"""Configuration for the primal batch-size optimizer.

    Attributes:
        eta_primal: Primal gradient descent step size.
        w_latency: Weight on the direct latency penalty term
            :math:`w_L \, \partial L / \partial x`.  This is an implementation
            extension beyond the paper's Eq. 18, which only has the dual term.
        w_throughput: Weight on the (negative) throughput gradient.
        w_switch: Weight on the switching cost regularizer
            :math:`\gamma \|x - x_{\mathrm{prev}}\|^2`.
        k_v: Scaling factor applied to the voltage gradient term.  Multiplies
            :math:`\eta^\top H \, e_i \, \partial P / \partial x`.  Not present
            in the paper's equations; used here as a tuning knob.
    """

    eta_primal: float = 0.05
    w_latency: float = 0.0
    w_throughput: float = 0.1
    w_switch: float = 0.0
    k_v: float = 1e6


class FullNetworkVoltageDual:
    r"""Full-network duals for voltage box constraints.

    .. math::
        \lambda_\text{low}  \leftarrow [\lambda_\text{low}  + \rho_v (v_\min - \hat v)]_+
        \lambda_\text{high} \leftarrow [\lambda_\text{high} + \rho_v (\hat v - v_\max)]_+
    """

    def __init__(self, M3: int, cfg: VoltageDualCfg):
        self.cfg = cfg
        self.lam_low = np.zeros(int(M3), dtype=float)
        self.lam_high = np.zeros(int(M3), dtype=float)

    def update(self, v_hat: np.ndarray) -> None:
        v_hat = np.asarray(v_hat, float).reshape(-1)
        if v_hat.shape[0] != self.lam_low.shape[0]:
            raise ValueError(
                f"v_hat has len {v_hat.shape[0]} but duals have len {self.lam_low.shape[0]}"
            )
        vmin = float(self.cfg.v_min)
        vmax = float(self.cfg.v_max)
        rho = float(self.cfg.rho_v)
        self.lam_low = np.maximum(self.lam_low + rho * (vmin - v_hat), 0.0)
        self.lam_high = np.maximum(self.lam_high + rho * (v_hat - vmax), 0.0)

    def eta(self) -> np.ndarray:
        r"""Return :math:`\eta = \bar\lambda - \underline\lambda`."""
        return self.lam_high - self.lam_low


def phase_share_from_placement(placement: dict[str, int]) -> np.ndarray:
    """Convert server placement counts to a normalized 3-phase share vector."""
    a = float(placement.get("servers_A", 0))
    b = float(placement.get("servers_B", 0))
    c = float(placement.get("servers_C", 0))
    s = a + b + c
    if s <= 0:
        return np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    return np.array([a / s, b / s, c / s], dtype=float)


class PerModelPrimalX:
    """Primal batch-size optimizer operating in log2 space.

    Maintains continuous state ``x_i = log2(batch_i)`` per model and applies
    a gradient descent step using voltage duals, latency duals, and fitted
    power/latency/throughput curves.
    """

    def __init__(
        self,
        *,
        models: list[ModelSpec],
        batch_set: list[int],
        fits: dict[str, LogisticFit],
        cfg: PrimalCfg,
    ):
        self.models = list(models)
        self.batch_set = sorted({int(b) for b in batch_set})
        if not self.batch_set:
            raise ValueError("batch_set cannot be empty.")

        self.fits = fits
        self.cfg = cfg

        self.x_min = math.log2(min(self.batch_set))
        self.x_max = math.log2(max(self.batch_set))

        self.x_by_model: dict[str, float] = {
            ms.model_label: float(self.x_max) for ms in self.models
        }
        self.x_prev_by_model: dict[str, float] = dict(self.x_by_model)

        # Per-model throughput normalization: r_i(x_max) for a single replica
        self.th_max_by_model: dict[str, float] = {}
        b_max = int(max(self.batch_set))
        for ms in self.models:
            label = ms.model_label
            try:
                th_max = float(self.fits[label].throughput.eval(b_max))
            except Exception:
                th_max = float("nan")
            if (not np.isfinite(th_max)) or (th_max <= 0.0):
                th_max = 1.0
            self.th_max_by_model[label] = th_max

    def _project_x(self, x: float) -> float:
        return float(min(max(float(x), self.x_min), self.x_max))

    def _discretize_batch(self, x: float) -> int:
        b_cont = 2.0 ** float(x)
        b = min(self.batch_set, key=lambda bb: abs(float(bb) - b_cont))
        return int(b)

    def init_from_batches(self, batch_init: dict[str, int]) -> None:
        """Initialize x state from discrete batch sizes."""
        for ms in self.models:
            label = ms.model_label
            b = int(batch_init.get(label, max(self.batch_set)))
            x = math.log2(max(b, 1))
            x = self._project_x(x)
            self.x_by_model[label] = float(x)
            self.x_prev_by_model[label] = float(x)

    def step(
        self,
        *,
        eta_vec: np.ndarray,
        H: np.ndarray,
        phase_share_by_model: dict[str, np.ndarray],
        mu_by_model: dict[str, float] | None = None,
        w_by_model: dict[str, float] | None = None,
    ) -> dict[str, int]:
        """Primal gradient descent step.

        Args:
            eta_vec: Voltage dual difference vector
                (lambda_high - lambda_low), shape ``(M3,)``.
            H: Voltage sensitivity matrix, shape ``(M3, 3)``.
            phase_share_by_model: Per-model normalized phase share vectors,
                shape ``(3,)`` each.
            mu_by_model: Per-model latency dual variables.
            w_by_model: Per-model replica weights (active replicas count).

        Returns:
            Next batch sizes per model.
        """
        eta_vec = np.asarray(eta_vec, float).reshape(-1)
        H = np.asarray(H, float)
        mu_by_model = {} if mu_by_model is None else dict(mu_by_model)
        w_by_model = {} if w_by_model is None else dict(w_by_model)

        eta_pr = float(self.cfg.eta_primal)
        wL = float(self.cfg.w_latency)
        wT = float(self.cfg.w_throughput)
        wS = float(self.cfg.w_switch)
        k_v = float(self.cfg.k_v)

        batch_next: dict[str, int] = {}

        for ms in self.models:
            label = ms.model_label
            x = float(self.x_by_model[label])
            x_prev = float(self.x_prev_by_model.get(label, x))

            w_i = float(w_by_model.get(label, 0.0))
            if (not np.isfinite(w_i)) or (w_i < 0.0):
                w_i = 0.0

            e = np.asarray(
                phase_share_by_model.get(label, np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)),
                float,
            ).reshape(3)
            s = float(np.sum(e))
            if (not np.isfinite(s)) or s <= 0.0:
                e = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
            else:
                e = e / s

            He = H @ e
            g_voltage = float(eta_vec @ He)

            p_fit = self.fits[label].power
            l_fit = self.fits[label].latency
            th_fit = self.fits[label].throughput

            dPdx_1 = float(p_fit.deriv_wrt_x(x))
            dLdx_1 = float(l_fit.deriv_wrt_x(x))
            dThdx_1 = float(th_fit.deriv_wrt_x(x))

            dPdx_1_kw = dPdx_1 / 1000.0

            th_max = float(self.th_max_by_model.get(label, 1.0))
            if (not np.isfinite(th_max)) or (th_max <= 0.0):
                th_max = 1.0
            dThdx_norm_1 = dThdx_1 / th_max

            dPdx = w_i * dPdx_1_kw
            dThdx = w_i * dThdx_norm_1
            dLdx = dLdx_1

            mu_i = float(mu_by_model.get(label, 0.0))
            if (not np.isfinite(mu_i)) or (mu_i < 0.0):
                mu_i = 0.0

            # Gradient of the Lagrangian w.r.t. x_i = log2(batch_i).
            # Paper Eq. 18 has:  nabla_x L = mu_i * dL/dx  (latency dual)
            #                               + eta^T H e_i dP/dx  (voltage dual)
            # Implementation extensions (tuning knobs not in the paper):
            #   wL * dL/dx      : direct latency penalty (separate from dual mu_i)
            #   -wT * dTh/dx    : throughput incentive
            #   k_v * (...)     : scaling factor on the voltage term
            #   wS * (x-x_prev) : switching cost regularizer
            grad = 0.0
            grad += wL * dLdx
            grad -= wT * dThdx
            grad += k_v * g_voltage * dPdx
            grad += mu_i * dLdx
            if wS > 0.0:
                grad += wS * (x - x_prev)

            x_new = self._project_x(x - eta_pr * grad)
            self.x_prev_by_model[label] = x
            self.x_by_model[label] = x_new
            batch_next[label] = self._discretize_batch(x_new)

        return batch_next


class OFOBatchController(Controller[LLMBatchSizeControlledDatacenter, OpenDSSGrid]):
    """Online Feedback Optimization controller for batch-size regulation.

    Reads grid voltage and datacenter state, updates voltage and latency
    duals, runs the primal batch-size optimizer, and returns new batch sizes.
    Latency dual updates use ``dc_state.observed_itl_s_by_model``.

    Args:
        models: Model specifications.
        fits: Per-model logistic fits for power/latency/throughput curves.
        Lth_by_model: Per-model latency threshold (seconds).
        primal_cfg: Primal optimizer configuration.
        voltage_dual_cfg: Voltage dual configuration (v_min, v_max, rho_v).
        batch_set: Allowed batch sizes.
        batch_init: Initial batch size.
        rho_l: Latency dual step size.
        dt_s: Control interval (seconds).
        estimate_H_every: Re-estimate H every N control steps
            (0 = only once at init).
        estimate_H_dp_kw: Perturbation size for H estimation.
    """

    def __init__(
        self,
        *,
        models: list[ModelSpec],
        fits: dict[str, LogisticFit],
        Lth_by_model: dict[str, float],
        primal_cfg: PrimalCfg,
        voltage_dual_cfg: VoltageDualCfg,
        batch_set: list[int],
        batch_init: int = 128,
        rho_l: float = 1.0,
        dt_s: float = 1.0,
        estimate_H_every: int = 0,
        estimate_H_dp_kw: float = 100.0,
    ):
        self._dt_s = float(dt_s)
        self._models = list(models)
        self._fits = fits
        self._Lth_by_model = dict(Lth_by_model)
        self._rho_l = float(rho_l)
        self._estimate_H_every = int(estimate_H_every)
        self._estimate_H_dp_kw = float(estimate_H_dp_kw)

        # Voltage duals are initialized lazily once voltage feature is available.
        self._voltage_dual: FullNetworkVoltageDual | None = None
        self._voltage_dual_cfg = voltage_dual_cfg

        # Latency duals (μ_i per model)
        self._mu_by_model: dict[str, float] = {ms.model_label: 0.0 for ms in models}

        # Primal optimizer
        self._primal = PerModelPrimalX(
            models=models,
            batch_set=batch_set,
            fits=fits,
            cfg=primal_cfg,
        )
        self._primal.init_from_batches({ms.model_label: batch_init for ms in models})

        # H estimation state
        self._H: np.ndarray | None = None
        self._v0: np.ndarray | None = None
        self._ctrl_step_count: int = 0

        # Phase share cache (from datacenter placement)
        self._phase_share_by_model: dict[str, np.ndarray] = {
            ms.model_label: np.array([1 / 3, 1 / 3, 1 / 3], dtype=float) for ms in models
        }

    @property
    def dt_s(self) -> float:
        return self._dt_s

    @property
    def voltage_dual(self) -> FullNetworkVoltageDual:
        if self._voltage_dual is None:
            raise RuntimeError("Voltage dual not initialized yet.")
        return self._voltage_dual

    @property
    def mu_by_model(self) -> dict[str, float]:
        return dict(self._mu_by_model)

    def set_phase_share(self, phase_share_by_model: dict[str, np.ndarray]) -> None:
        """Update per-model phase share vectors (from datacenter placement)."""
        self._phase_share_by_model.update(phase_share_by_model)

    def step(
        self,
        clock: SimulationClock,
        datacenter: LLMBatchSizeControlledDatacenter,
        grid: OpenDSSGrid,
        events: EventEmitter,
    ) -> ControlAction:

        if self._voltage_dual is None:
            self._voltage_dual = FullNetworkVoltageDual(len(grid.v_index), self._voltage_dual_cfg)

        # 1. Re-estimate H if needed
        if self._H is None or (
            self._estimate_H_every > 0 and self._ctrl_step_count % self._estimate_H_every == 0
        ):
            self._H, self._v0 = grid.estimate_H(self._estimate_H_dp_kw)

        # 2. Update voltage duals from grid state
        if grid.state is not None:
            v_hat = grid.voltages_vector()
            self._voltage_dual.update(v_hat)

        eta_vec = self._voltage_dual.eta()

        # 3. Read observed latency from datacenter and update latency duals
        dc_state = datacenter.state
        if dc_state is not None:
            missing_replicas = [
                ms.model_label
                for ms in self._models
                if ms.model_label not in dc_state.active_replicas_by_model
            ]
            if missing_replicas:
                miss = ", ".join(sorted(missing_replicas))
                raise RuntimeError(
                    "OFOBatchController requires active_replicas_by_model for all models. "
                    f"Missing: {miss}."
                )
            missing_itl = [
                ms.model_label
                for ms in self._models
                if ms.model_label not in dc_state.observed_itl_s_by_model
            ]
            if missing_itl:
                miss = ", ".join(sorted(missing_itl))
                raise RuntimeError(
                    "OFOBatchController requires observed_itl_s_by_model for all models. "
                    f"Missing: {miss}."
                )
            for ms in self._models:
                label = ms.model_label
                n_rep = max(int(dc_state.active_replicas_by_model.get(label, 0)), 0)
                l_hat = float(dc_state.observed_itl_s_by_model.get(label, float("nan")))
                if n_rep <= 0:
                    l_hat = float("nan")

                Lth = float(self._Lth_by_model.get(label, 0.1))
                if np.isfinite(l_hat):
                    self._mu_by_model[label] = max(
                        self._mu_by_model[label] + self._rho_l * (l_hat - Lth),
                        0.0,
                    )
                else:
                    self._mu_by_model[label] = max(self._mu_by_model[label], 0.0)

        # 4. Compute replica weights
        w_by_model: dict[str, float] = {}
        if dc_state is not None:
            for ms in self._models:
                label = ms.model_label
                w_by_model[label] = float(dc_state.active_replicas_by_model.get(label, 0))

        # 5. Primal update -> next batch sizes
        assert self._H is not None
        batch_next = self._primal.step(
            eta_vec=eta_vec,
            H=self._H,
            phase_share_by_model=self._phase_share_by_model,
            mu_by_model=self._mu_by_model,
            w_by_model=w_by_model,
        )

        self._ctrl_step_count += 1
        return ControlAction(
            commands=[
                Command(
                    target="datacenter",
                    kind="set_batch_size",
                    payload={"batch_size_by_model": batch_next},
                )
            ]
        )
