"""Online Feedback Optimization (OFO) batch-size controller.

Implements the primal-dual algorithm for joint voltage regulation and
latency management via GPU batch size control.
"""

from __future__ import annotations

import bisect
import logging
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mlenergy_data.modeling import LogisticModel
from mlenergy_data.records import LLMRuns
from pydantic import BaseModel, ConfigDict

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.datacenter.base import LLMBatchSizeControlledDatacenter, LLMDatacenterState
from openg2g.datacenter.command import DatacenterCommand, SetBatchSize
from openg2g.datacenter.config import InferenceModelSpec
from openg2g.events import EventEmitter
from openg2g.grid.command import GridCommand
from openg2g.grid.opendss import OpenDSSGrid

logger = logging.getLogger(__name__)


class OFOConfig(BaseModel):
    """Online Feedback Optimization tuning parameters.

    Attributes:
        primal_step_size: Primal descent step size ρ_x (Eq. 8).
        w_throughput: Throughput weight in primal gradient.
        w_switch: Switching cost regularizer weight γ (Eq. 4a).
        voltage_gradient_scale: Scaling factor k_v for voltage dual term
            in the primal gradient.
        v_min: Lower voltage bound (pu).
        v_max: Upper voltage bound (pu).
        voltage_dual_step_size: Voltage dual ascent step size ρ_v (Eqs. 5-6).
        latency_dual_step_size: Latency dual ascent step size ρ_l (Eq. 7).
        sensitivity_update_interval: Steps between H-matrix re-estimation
            (0 = only once at init).
        sensitivity_perturbation_kw: Perturbation magnitude (kW) for
            finite-difference sensitivity estimation.
    """

    model_config = ConfigDict(frozen=True)

    # Primal
    primal_step_size: float = 0.05
    w_throughput: float = 0.1
    w_switch: float = 0.0
    voltage_gradient_scale: float = 1e6

    # Dual
    v_min: float = 0.95
    v_max: float = 1.05
    voltage_dual_step_size: float = 0.5
    latency_dual_step_size: float = 1.0

    # Sensitivity
    sensitivity_update_interval: int = 0
    sensitivity_perturbation_kw: float = 100.0


class LogisticModelStore:
    """Per-model logistic models for power, latency, and throughput.

    Used by
    [`OFOBatchSizeController`][openg2g.controller.ofo.OFOBatchSizeController]
    to compute gradients of the Lagrangian with respect to batch size.

    Attributes:
        COL_MODEL_LABEL: Column name for model label in the CSV.
        COL_METRIC: Column name for metric type in the CSV.
    """

    COL_MODEL_LABEL = "model_label"
    COL_METRIC = "metric"

    def __init__(
        self,
        power: dict[str, LogisticModel],
        latency: dict[str, LogisticModel],
        throughput: dict[str, LogisticModel],
    ) -> None:
        self._power = dict(power)
        self._latency = dict(latency)
        self._throughput = dict(throughput)
        self._by_batch: dict[str, dict[int, list[tuple[float, float, float]]]] | None = None

    def power(self, model: str) -> LogisticModel:
        """Return the power logistic model for a model label."""
        return self._power[model]

    def latency(self, model: str) -> LogisticModel:
        """Return the latency logistic model for a model label."""
        return self._latency[model]

    def throughput(self, model: str) -> LogisticModel:
        """Return the throughput logistic model for a model label."""
        return self._throughput[model]

    @property
    def power_fits(self) -> dict[str, LogisticModel]:
        return dict(self._power)

    @property
    def latency_fits(self) -> dict[str, LogisticModel]:
        return dict(self._latency)

    @property
    def throughput_fits(self) -> dict[str, LogisticModel]:
        return dict(self._throughput)

    @classmethod
    def generate(
        cls,
        models: tuple[InferenceModelSpec, ...],
        data_sources: dict[str, Any],
        *,
        runs: Any = None,
        mlenergy_data_dir: Path | None = None,
    ) -> LogisticModelStore:
        """Generate logistic fits from ML.ENERGY benchmark data.

        Args:
            models: Model specifications.
            data_sources: Per-model `MLEnergySource` instances, keyed by
                `model_label`.
            runs: Pre-loaded `LLMRuns` object. If `None`, loads from
                `mlenergy_data_dir` or the HuggingFace Hub.
            mlenergy_data_dir: Path to compiled mlenergy-data directory.
                Ignored if `runs` is provided.

        Returns:
            A new `LogisticModelStore` with fitted logistic models.
        """
        if runs is None:
            unique_tasks = {src.task for src in data_sources.values()}
            if mlenergy_data_dir:
                runs = LLMRuns.from_directory(str(mlenergy_data_dir), stable_only=False).task(*unique_tasks)
            else:
                runs = LLMRuns.from_hf(stable_only=False).task(*unique_tasks)
        if not runs:
            raise ValueError("No runs found for the specified tasks")

        subsets_by_label: dict[str, Any] = {}
        for ms in models:
            src = data_sources.get(ms.model_label)
            if src is None:
                raise ValueError(f"No data source for model {ms.model_label!r}")
            model_id = ms.model_id
            if not model_id:
                raise ValueError(f"model_id is required for data generation (model={ms.model_label!r})")

            subset = (
                runs.model_id(model_id).gpu_model(src.gpu).num_gpus(ms.gpus_per_replica).max_num_seqs(*src.batch_sizes)
            )
            if not subset:
                raise ValueError(
                    f"Config matched zero runs for logistic fits: model_id={model_id!r}, "
                    f"gpu={src.gpu!r}, num_gpus={ms.gpus_per_replica}, "
                    f"batch_sizes={src.batch_sizes}"
                )
            subsets_by_label[ms.model_label] = subset

        all_by_batch: dict[str, dict[int, list[tuple[float, float, float]]]] = {}
        power: dict[str, LogisticModel] = {}
        latency: dict[str, LogisticModel] = {}
        throughput: dict[str, LogisticModel] = {}
        for model_label, group in subsets_by_label.items():
            exclude = set(data_sources[model_label].fit_exclude_batch_sizes)
            by_batch: dict[int, list[tuple[float, float, float]]] = {}
            for r in group:
                if r.max_num_seqs in exclude:
                    continue
                by_batch.setdefault(r.max_num_seqs, []).append(
                    (r.avg_power_watts, r.mean_itl_ms / 1000.0, r.output_throughput_tokens_per_sec)
                )
            all_by_batch[model_label] = by_batch

            batches = sorted(by_batch.keys())
            if not batches:
                continue

            x = np.log2(np.array(batches, dtype=float).clip(min=1))
            for _metric_name, idx, target in [
                ("power", 0, power),
                ("latency", 1, latency),
                ("throughput", 2, throughput),
            ]:
                y = np.array([float(np.median([t[idx] for t in by_batch[b]])) for b in batches])
                fit = LogisticModel.fit(x, y)
                target[model_label] = fit

        if not power and not latency and not throughput:
            raise ValueError("No logistic fit rows produced")
        store = cls(power=power, latency=latency, throughput=throughput)
        store._by_batch = all_by_batch
        return store

    def save(self, csv_path: Path, *, plot: bool = False) -> None:
        """Save logistic fits to a CSV.

        Args:
            csv_path: Output CSV path.
            plot: If `True`, also write a logistic fits plot to the
                same directory.
        """
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for metric_name, fits in [("power", self._power), ("latency", self._latency), ("throughput", self._throughput)]:
            for label in sorted(fits):
                model = fits[label]
                rows.append(
                    {
                        self.COL_MODEL_LABEL: label,
                        self.COL_METRIC: metric_name,
                        "L": model.L,
                        "x0": model.x0,
                        "k": model.k,
                        "b0": model.b0,
                    }
                )
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        by_batch = getattr(self, "_by_batch", None)
        if plot and by_batch is not None:
            model_labels = sorted(self._power.keys())
            _plot_logistic_fits(
                by_batch,
                self._power,
                self._latency,
                self._throughput,
                model_labels,
                csv_path.parent,
            )

    @classmethod
    def load(cls, csv_path: Path | str) -> LogisticModelStore:
        """Load power, latency, and throughput fits from a merged CSV.

        Expected columns: `model_label`, `metric`, plus the logistic
        model parameter columns (`L`, `x0`, `k`, `b0`).

        The `metric` column must contain `power`, `latency`, or
        `throughput` (case-insensitive).

        Args:
            csv_path: Path to the logistic fits CSV.
        """
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)

        required_cols = [cls.COL_MODEL_LABEL, cls.COL_METRIC]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} missing columns: {missing}. Got: {list(df.columns)}")

        power: dict[str, LogisticModel] = {}
        latency: dict[str, LogisticModel] = {}
        throughput: dict[str, LogisticModel] = {}
        targets = {"power": power, "latency": latency, "throughput": throughput}
        for row in df.to_dict(orient="records"):
            metric = str(row[cls.COL_METRIC]).strip().lower()
            if metric in targets:
                targets[metric][str(row[cls.COL_MODEL_LABEL])] = LogisticModel.from_dict(row)

        if not power and not latency and not throughput:
            raise ValueError(f"No logistic model rows loaded from {csv_path}")
        return cls(power=power, latency=latency, throughput=throughput)

    @classmethod
    def ensure(
        cls,
        csv_path: Path,
        models: tuple[InferenceModelSpec, ...] | None = None,
        data_sources: dict[str, Any] | None = None,
        *,
        mlenergy_data_dir: Path | None = None,
        plot: bool = False,
    ) -> LogisticModelStore:
        """Load from `csv_path`, generating first if needed.

        Args:
            csv_path: Path to the logistic fits CSV.
            models: Model specifications. Required when no cached file exists.
            data_sources: Per-model `MLEnergySource` instances, keyed by
                `model_label`. Required when no cached file exists.
            mlenergy_data_dir: Path to compiled mlenergy-data directory.
            plot: If `True`, generate a logistic fits plot on generation.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            if models is None or data_sources is None:
                raise ValueError("models and data_sources required for LogisticModelStore generation (no cached data)")
            logger.info("Generating logistic fits to %s ...", csv_path)
            cls.generate(models, data_sources, mlenergy_data_dir=mlenergy_data_dir).save(csv_path, plot=plot)
        return cls.load(csv_path)


class VoltageDualVariables:
    """Full-network duals for voltage box constraints.

    Maintains per-bus dual variables for under- and overvoltage and updates
    them via projected gradient ascent:

        dual_undervoltage  <- [dual_undervoltage  + ρ_v * (v_min - v̂)]+
        dual_overvoltage   <- [dual_overvoltage   + ρ_v * (v̂ - v_max)]+

    Args:
        n_bus_phases: Number of bus-phase pairs in the voltage vector (3M).
        config: OFO configuration (voltage bounds and dual step size).
    """

    def __init__(self, n_bus_phases: int, config: OFOConfig) -> None:
        self.config = config
        self.dual_undervoltage = np.zeros(int(n_bus_phases), dtype=float)  # λ in G2G paper Eq. 5
        self.dual_overvoltage = np.zeros(int(n_bus_phases), dtype=float)  # λ̄ in G2G paper Eq. 6

    def update(self, observed_voltages: np.ndarray) -> None:
        """Update duals given observed voltage vector.

        Args:
            observed_voltages: Observed voltage magnitudes (pu), shape
                `(n_bus_phases,)`.

        Raises:
            ValueError: If `observed_voltages` length does not match the dual
                dimension.
        """
        observed_voltages = np.asarray(observed_voltages, float).reshape(-1)
        if observed_voltages.shape[0] != self.dual_undervoltage.shape[0]:
            raise ValueError(
                f"observed_voltages has len {observed_voltages.shape[0]} "
                f"but duals have len {self.dual_undervoltage.shape[0]}"
            )
        vmin = float(self.config.v_min)
        vmax = float(self.config.v_max)
        rho = float(self.config.voltage_dual_step_size)
        self.dual_undervoltage = np.maximum(self.dual_undervoltage + rho * (vmin - observed_voltages), 0.0)
        self.dual_overvoltage = np.maximum(self.dual_overvoltage + rho * (observed_voltages - vmax), 0.0)

    def dual_difference(self) -> np.ndarray:
        """Return the voltage dual difference (η = λ̄ − λ, Appendix B)."""
        return self.dual_overvoltage - self.dual_undervoltage


class PrimalBatchOptimizer:
    """Primal batch-size optimizer operating in log2 space.

    Maintains continuous state `x_i = log2(batch_i)` per model and applies
    a gradient descent step using voltage duals, latency duals, and fitted
    power/latency/throughput curves.

    Args:
        models: Model specifications for each served model.
        feasible_batch_sizes: Allowed batch sizes (union across all models).
        power_fits: Per-model logistic fit for power vs log2(batch_size).
        latency_fits: Per-model logistic fit for latency vs log2(batch_size).
        throughput_fits: Per-model logistic fit for throughput vs
            log2(batch_size).
        config: OFO configuration (step size, throughput/switch weights,
            voltage gradient scale).
    """

    def __init__(
        self,
        *,
        models: list[InferenceModelSpec],
        feasible_batch_sizes: list[int],
        power_fits: dict[str, LogisticModel],
        latency_fits: dict[str, LogisticModel],
        throughput_fits: dict[str, LogisticModel],
        config: OFOConfig,
    ) -> None:
        self.models = list(models)
        self.feasible_batch_sizes = sorted({int(b) for b in feasible_batch_sizes})
        if not self.feasible_batch_sizes:
            raise ValueError("feasible_batch_sizes cannot be empty.")

        self.power_fits = power_fits
        self.latency_fits = latency_fits
        self.throughput_fits = throughput_fits
        self.config = config

        self.log_batch_size_min = math.log2(min(self.feasible_batch_sizes))
        self.log_batch_size_max = math.log2(max(self.feasible_batch_sizes))

        self.log_batch_size_by_model: dict[str, float] = {
            ms.model_label: float(self.log_batch_size_max) for ms in self.models
        }
        self.prev_log_batch_size_by_model: dict[str, float] = dict(self.log_batch_size_by_model)

        # Per-model throughput normalization: r_i(x_max) for a single replica
        self.throughput_max_by_model: dict[str, float] = {}
        b_max = int(max(self.feasible_batch_sizes))
        for ms in self.models:
            label = ms.model_label
            try:
                th_max = float(self.throughput_fits[label].eval(b_max))
            except Exception:
                th_max = float("nan")
            if (not np.isfinite(th_max)) or (th_max <= 0.0):
                th_max = 1.0
            self.throughput_max_by_model[label] = th_max

    def _clamp_log_batch_size(self, log_batch_size: float) -> float:
        return float(min(max(float(log_batch_size), self.log_batch_size_min), self.log_batch_size_max))

    def _discretize_batch(self, log_batch_size: float) -> int:
        b_cont = 2.0 ** float(log_batch_size)
        idx = bisect.bisect_left(self.feasible_batch_sizes, b_cont)
        candidates = []
        if idx > 0:
            candidates.append(self.feasible_batch_sizes[idx - 1])
        if idx < len(self.feasible_batch_sizes):
            candidates.append(self.feasible_batch_sizes[idx])
        return int(min(candidates, key=lambda bb: abs(bb - b_cont)))

    def init_from_batches(self, batch_init: dict[str, int]) -> None:
        """Initialize log-batch-size state from discrete batch sizes."""
        for ms in self.models:
            label = ms.model_label
            b = int(batch_init.get(label, max(self.feasible_batch_sizes)))
            log_batch_size = math.log2(max(b, 1))
            log_batch_size = self._clamp_log_batch_size(log_batch_size)
            self.log_batch_size_by_model[label] = float(log_batch_size)
            self.prev_log_batch_size_by_model[label] = float(log_batch_size)

    def step(
        self,
        *,
        voltage_dual_diff: np.ndarray,
        sensitivity_matrix: np.ndarray,
        phase_share_by_model: dict[str, np.ndarray],
        latency_dual_by_model: dict[str, float] | None = None,
        replica_count_by_model: dict[str, float] | None = None,
    ) -> dict[str, int]:
        """Primal gradient descent step.

        Args:
            voltage_dual_diff: Voltage dual difference vector
                (η = λ̄ − λ), shape `(n_bus_phases,)`.
            sensitivity_matrix: Voltage sensitivity matrix (H = dv/dp),
                shape `(n_bus_phases, 3)`.
            phase_share_by_model: Per-model normalized phase share vectors,
                shape `(3,)` each.
            latency_dual_by_model: Per-model latency dual variables (μ_i).
            replica_count_by_model: Per-model active replica counts (w_i).

        Returns:
            Next batch sizes per model.
        """
        voltage_dual_diff = np.asarray(voltage_dual_diff, float).reshape(-1)
        sensitivity_matrix = np.asarray(sensitivity_matrix, float)
        latency_dual_by_model = {} if latency_dual_by_model is None else dict(latency_dual_by_model)
        replica_count_by_model = {} if replica_count_by_model is None else dict(replica_count_by_model)

        step_size = float(self.config.primal_step_size)  # ρ_x
        w_throughput = float(self.config.w_throughput)
        w_switch = float(self.config.w_switch)
        voltage_gradient_scale = float(self.config.voltage_gradient_scale)

        batch_next: dict[str, int] = {}

        for ms in self.models:
            label = ms.model_label
            log_batch_size = float(self.log_batch_size_by_model[label])
            prev_log_batch_size = float(self.prev_log_batch_size_by_model.get(label, log_batch_size))

            replica_count = float(replica_count_by_model.get(label, 0.0))  # w_i
            if (not np.isfinite(replica_count)) or (replica_count < 0.0):
                replica_count = 0.0

            phase_share = np.asarray(  # e_i (phase-allocation weight, p.7)
                phase_share_by_model.get(label, np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)),
                float,
            ).reshape(3)
            s = float(np.sum(phase_share))
            if (not np.isfinite(s)) or s <= 0.0:
                phase_share = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
            else:
                phase_share = phase_share / s

            weighted_sensitivity = sensitivity_matrix @ phase_share  # H @ e_i
            voltage_gradient = float(voltage_dual_diff @ weighted_sensitivity)

            dPdx_1 = float(self.power_fits[label].deriv_wrt_x(log_batch_size))
            dLdx_1 = float(self.latency_fits[label].deriv_wrt_x(log_batch_size))
            dThdx_1 = float(self.throughput_fits[label].deriv_wrt_x(log_batch_size))

            dPdx_1_kw = dPdx_1 / 1000.0

            th_max = float(self.throughput_max_by_model.get(label, 1.0))
            if (not np.isfinite(th_max)) or (th_max <= 0.0):
                th_max = 1.0
            dThdx_norm_1 = dThdx_1 / th_max

            dPdx = replica_count * dPdx_1_kw
            dThdx = replica_count * dThdx_norm_1
            dLdx = dLdx_1

            latency_dual = float(latency_dual_by_model.get(label, 0.0))  # μ_i
            if (not np.isfinite(latency_dual)) or (latency_dual < 0.0):
                latency_dual = 0.0

            # Gradient of the Lagrangian w.r.t. x_i = log2(batch_i).
            # G2G paper Eq. 18: nabla_x L = -dR/dx (throughput)
            #                              + 2*gamma*(x - x_prev) (switching)
            #                              + eta^T H e_i dP/dx (voltage dual)
            #                              + mu_i * dL/dx (latency dual)
            # Implementation extensions: wT scaling on throughput,
            #                            k_v scaling on voltage term
            grad = 0.0
            grad -= w_throughput * dThdx
            grad += voltage_gradient_scale * voltage_gradient * dPdx
            grad += latency_dual * dLdx
            grad += w_switch * (log_batch_size - prev_log_batch_size)

            new_log_batch_size = self._clamp_log_batch_size(log_batch_size - step_size * grad)
            self.prev_log_batch_size_by_model[label] = log_batch_size
            self.log_batch_size_by_model[label] = new_log_batch_size
            batch_next[label] = self._discretize_batch(new_log_batch_size)

        return batch_next


class OFOBatchSizeController(Controller[LLMBatchSizeControlledDatacenter[LLMDatacenterState], OpenDSSGrid]):
    """Online Feedback Optimization controller for batch-size regulation.

    Reads grid voltage and datacenter state, updates voltage and latency
    duals, runs the primal batch-size optimizer, and returns new batch
    sizes. Latency dual updates use [`dc_state.observed_itl_s_by_model`
    ][openg2g.datacenter.base.LLMDatacenterState.observed_itl_s_by_model].

    Args:
        inference_models: Model specifications served in the datacenter.
        models: Per-model logistic models for power, latency, and
            throughput used in gradient computation.
        config: Unified OFO tuning parameters.
        dt_s: Control interval (seconds).
    """

    def __init__(
        self,
        inference_models: tuple[InferenceModelSpec, ...],
        *,
        datacenter: LLMBatchSizeControlledDatacenter[LLMDatacenterState],
        models: LogisticModelStore,
        config: OFOConfig | None = None,
        dt_s: Fraction = Fraction(1),
        initial_batch_sizes: dict[str, int] | None = None,
    ) -> None:
        if config is None:
            config = OFOConfig()

        if not inference_models:
            raise ValueError("inference_models must not be empty.")
        labels = [ms.model_label for ms in inference_models]
        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate model labels: {labels}")

        model_specs = list(inference_models)
        self._initial_batch_sizes = initial_batch_sizes or {}

        for ms in model_specs:
            label = ms.model_label
            for metric_name, accessor in [
                ("power", models.power),
                ("latency", models.latency),
                ("throughput", models.throughput),
            ]:
                try:
                    accessor(label)
                except KeyError:
                    raise ValueError(f"LogisticModelStore missing {metric_name} model for {label!r}.") from None

        self._dt_s = dt_s
        self._models = model_specs
        self._config = config
        self._datacenter = datacenter
        self._itl_deadline_by_model = {ms.model_label: ms.itl_deadline_s for ms in model_specs}

        self._voltage_dual: VoltageDualVariables | None = None
        self._latency_dual_by_model: dict[str, float] = {ms.model_label: 0.0 for ms in model_specs}

        all_bs: set[int] = set()
        for ms in model_specs:
            all_bs.update(ms.feasible_batch_sizes)
        feasible_batch_sizes = sorted(all_bs)

        self._optimizer = PrimalBatchOptimizer(
            models=model_specs,
            feasible_batch_sizes=feasible_batch_sizes,
            power_fits=models.power_fits,
            latency_fits=models.latency_fits,
            throughput_fits=models.throughput_fits,
            config=config,
        )
        self._optimizer.init_from_batches(
            {
                ms.model_label: self._initial_batch_sizes.get(ms.model_label, ms.feasible_batch_sizes[0])
                for ms in model_specs
            }
        )

        self._sensitivity_matrix: np.ndarray | None = None
        self._control_step_count: int = 0

        logger.info(
            "OFOBatchSizeController: %d models, dt=%s s, feasible_batches=%s",
            len(model_specs),
            dt_s,
            feasible_batch_sizes,
        )

    def reset(self) -> None:
        self._voltage_dual = None
        self._latency_dual_by_model = {ms.model_label: 0.0 for ms in self._models}
        self._optimizer.init_from_batches(
            {
                ms.model_label: self._initial_batch_sizes.get(ms.model_label, ms.feasible_batch_sizes[0])
                for ms in self._models
            }
        )
        self._sensitivity_matrix = None
        self._control_step_count = 0

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    @property
    def datacenters(self) -> list:
        return [self._datacenter]

    def step(
        self,
        clock: SimulationClock,
        grid: OpenDSSGrid,
        events: EventEmitter,
    ) -> list[DatacenterCommand | GridCommand]:
        datacenter = self._datacenter

        if self._voltage_dual is None:
            self._voltage_dual = VoltageDualVariables(len(grid.v_index), self._config)

        # 1. Re-estimate sensitivity if needed
        if self._sensitivity_matrix is None or (
            self._config.sensitivity_update_interval > 0
            and self._control_step_count % self._config.sensitivity_update_interval == 0
        ):
            self._sensitivity_matrix, _ = grid.estimate_sensitivity(
                perturbation_kw=self._config.sensitivity_perturbation_kw,
                dc=self._datacenter,
            )

        # 2. Update voltage duals from grid state
        observed_voltages = grid.voltages_vector()
        self._voltage_dual.update(observed_voltages)

        voltage_dual_diff = self._voltage_dual.dual_difference()  # η = λ̄ − λ

        # 3. Read observed latency from datacenter and update latency duals
        dc_state = datacenter.state
        missing_replicas = [
            ms.model_label for ms in self._models if ms.model_label not in dc_state.active_replicas_by_model
        ]
        if missing_replicas:
            miss = ", ".join(sorted(missing_replicas))
            raise RuntimeError(
                f"OFOBatchSizeController requires active_replicas_by_model for all models. Missing: {miss}."
            )
        missing_itl = [ms.model_label for ms in self._models if ms.model_label not in dc_state.observed_itl_s_by_model]
        if missing_itl:
            miss = ", ".join(sorted(missing_itl))
            raise RuntimeError(
                f"OFOBatchSizeController requires observed_itl_s_by_model for all models. Missing: {miss}."
            )
        for ms in self._models:
            label = ms.model_label
            num_replicas = max(int(dc_state.active_replicas_by_model[label]), 0)
            observed_itl = float(dc_state.observed_itl_s_by_model[label])
            if num_replicas <= 0:
                logger.debug("Model %s has 0 replicas, skipping latency dual update", label)
                observed_itl = float("nan")

            deadline = float(self._itl_deadline_by_model[label])
            if np.isfinite(observed_itl):
                self._latency_dual_by_model[label] = max(
                    self._latency_dual_by_model[label]
                    + self._config.latency_dual_step_size * (observed_itl - deadline),
                    0.0,
                )
            else:
                self._latency_dual_by_model[label] = max(self._latency_dual_by_model[label], 0.0)

        # 4. Compute replica counts
        replica_count_by_model: dict[str, float] = {}
        for ms in self._models:
            label = ms.model_label
            replica_count_by_model[label] = float(dc_state.active_replicas_by_model[label])

        # 5. Primal update -> next batch sizes
        batch_next = self._optimizer.step(
            voltage_dual_diff=voltage_dual_diff,
            sensitivity_matrix=self._sensitivity_matrix,
            phase_share_by_model=datacenter.phase_share_by_model,
            latency_dual_by_model=self._latency_dual_by_model,
            replica_count_by_model=replica_count_by_model,
        )

        self._control_step_count += 1
        logger.debug(
            "OFO step %d (t=%.1f s): batch=%s",
            self._control_step_count,
            clock.time_s,
            batch_next,
        )
        events.emit(
            "controller.ofo.step",
            {
                "batch_size_by_model": batch_next,
                "latency_dual_by_model": dict(self._latency_dual_by_model),
            },
        )
        return [SetBatchSize(batch_size_by_model=batch_next)]


def _plot_logistic_fits(
    by_batch: dict[str, dict[int, list[tuple[float, float, float]]]],
    power: dict[str, LogisticModel],
    latency_fits: dict[str, LogisticModel],
    throughput_fits: dict[str, LogisticModel],
    model_labels: list[str],
    out_dir: Path,
) -> None:
    """Plot 3x1 stacked logistic fits: power, latency, throughput.

    Scatter dots for measured medians, smooth fitted curves from
    LogisticModel parameters. Saves to `out_dir / "logistic_fits.png"`.
    """
    import matplotlib.pyplot as plt

    metric_specs: list[tuple[str, int, dict[str, LogisticModel], str, str]] = [
        ("power", 0, power, "W", "(a) Average GPU power consumption vs batch size"),
        ("latency", 1, latency_fits, "s/token", "(b) Average inter-token latency vs batch size"),
        ("throughput", 2, throughput_fits, "tokens/s", "(c) Average token throughput vs batch size"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(6.45, 5.2), dpi=300, sharex=True)

    for ax_idx, (ax, (_metric_name, val_idx, fits, ylabel, title)) in enumerate(zip(axes, metric_specs, strict=True)):
        xmins: list[float] = []
        xmaxs: list[float] = []

        for label in model_labels:
            model_by_batch = by_batch.get(label, {})
            batches = sorted(model_by_batch.keys())
            if not batches:
                continue
            x = np.log2(np.array(batches, dtype=float).clip(min=1))
            if len(x) > 0:
                xmins.append(float(np.min(x)))
                xmaxs.append(float(np.max(x)))

        if not xmins:
            ax.set_title(title, fontsize=12, loc="center")
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, alpha=0.25)
            continue

        xs = np.linspace(min(xmins), max(xmaxs), 400)

        for label in model_labels:
            model_by_batch = by_batch.get(label, {})
            batches = sorted(model_by_batch.keys())
            if not batches or label not in fits:
                continue

            x = np.log2(np.array(batches, dtype=float).clip(min=1))
            y = np.array([float(np.median([t[val_idx] for t in model_by_batch[b]])) for b in batches])

            fit = fits[label]
            ys_fit = np.array([fit.eval_x(float(xi)) for xi in xs])

            (line,) = ax.plot(xs, ys_fit, lw=1.8, label=label, zorder=2)
            ax.scatter(x, y, s=16.0, color=line.get_color(), zorder=3)

        ax.set_title(title, fontsize=12, loc="center")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=10)

        if ax_idx == 2:
            ax.legend(frameon=True, fontsize=9, loc="best")

    axes[-1].set_xlabel(r"$\log_2(\mathrm{batch\ size})$", fontsize=10)
    fig.tight_layout(pad=0.35, h_pad=0.6)

    save_path = out_dir / "logistic_fits.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    logger.info("Saved logistic fits plot to %s", save_path)
