"""Offline (trace-based) datacenter backend.

Loads power-trace CSVs and serves per-timestep OfflineDatacenterState objects via step().
"""

from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from mlenergy_data.modeling import ITLMixtureModel

from openg2g.clock import SimulationClock
from openg2g.datacenter.base import LLMBatchSizeControlledDatacenter
from openg2g.datacenter.config import DatacenterConfig, WorkloadConfig
from openg2g.datacenter.training_overlay import TrainingOverlayCache
from openg2g.events import EventEmitter
from openg2g.models.spec import LLMInferenceModelSpec
from openg2g.types import (
    DatacenterCommand,
    OfflineDatacenterState,
    ServerRamp,
    ServerRampSchedule,
    SetBatchSize,
    ThreePhase,
    TrainingRun,
    TrainingSchedule,
)
from openg2g.utils import split_integer_evenly

logger = logging.getLogger(__name__)


def build_periodic_per_gpu_template(
    trace_t: np.ndarray,
    trace_p_total: np.ndarray,
    measured_gpus: int,
    *,
    timestep_s: Fraction | float,
    duration_s: Fraction | float,
    is_total: bool = True,
    steady_skip_s: float = 0.0,
) -> np.ndarray:
    """Build a per-GPU power template over [0, duration_s] by periodic repetition."""
    trace_t = np.asarray(trace_t, float)
    trace_p_total = np.asarray(trace_p_total, float)
    if trace_t.size < 5:
        raise ValueError("Trace too short to build a periodic template.")

    mg = max(int(measured_gpus), 1)
    p_per_gpu = (trace_p_total / mg) if is_total else trace_p_total.copy()
    p_per_gpu = np.clip(p_per_gpu, 0.0, None)

    if steady_skip_s > 0.0:
        idx0 = np.searchsorted(trace_t, trace_t[0] + float(steady_skip_s))
        if idx0 < trace_t.size - 5:
            trace_t = trace_t[idx0:] - trace_t[idx0]
            p_per_gpu = p_per_gpu[idx0:]

    trace_t = trace_t - trace_t[0]
    period = float(trace_t[-1] - trace_t[0])
    if period <= 0:
        raise ValueError("Non-positive trace duration.")

    n_steps = int(np.ceil(float(duration_s) / float(timestep_s))) + 1
    t_grid = np.arange(n_steps, dtype=float) * float(timestep_s)
    t_mod = np.mod(t_grid, period)

    template = np.interp(t_mod, trace_t, p_per_gpu, left=p_per_gpu[0], right=p_per_gpu[-1])
    return np.clip(template, 0.0, None)


def load_traces_by_batch_from_dir(
    *,
    base_dir: Path,
    batch_set: list[int],
    required_measured_gpus: dict[str, int],
    file_pattern: str = "{model_label}_num_gpus_{measured_gpus}_max_num_seqs_{batch}",
    allow_missing_csv_suffix: bool = True,
    time_col: str = "relative_time_s",
    power_col: str = "power_total_W",
    is_total_default: bool = True,
    amp_jitter_default: tuple[float, float] = (1.0, 1.0),
    noise_std_frac_default: float = 0.0,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Load per-(batch, model) power-trace CSVs into memory.

    Returns ``traces_by_batch[batch][model_label] -> {t, p, measured_gpus, ...}``.
    """
    import pandas as pd

    base_dir = Path(base_dir)
    traces_by_batch: dict[int, dict[str, dict[str, Any]]] = {}

    for batch in batch_set:
        batch = int(batch)
        per_model: dict[str, dict[str, Any]] = {}

        for model_label, mg in required_measured_gpus.items():
            mg = int(mg)
            stem = file_pattern.format(model_label=model_label, measured_gpus=mg, batch=batch)
            fpath = base_dir / stem
            if not fpath.exists() and allow_missing_csv_suffix:
                fpath2 = base_dir / (stem + ".csv")
                if fpath2.exists():
                    fpath = fpath2

            if not fpath.exists():
                raise FileNotFoundError(
                    f"Missing power trace CSV for model={model_label}, batch={batch}, "
                    f"measured_gpus={mg}. Tried: {base_dir / stem} and "
                    f"{base_dir / (stem + '.csv')}"
                )

            df = pd.read_csv(fpath)
            if time_col not in df.columns or power_col not in df.columns:
                raise ValueError(f"{fpath} must contain {time_col!r} and {power_col!r}. Got columns={list(df.columns)}")

            t = df[time_col].to_numpy(float)
            p = df[power_col].to_numpy(float)

            if np.any(np.diff(t) < 0):
                idx = np.argsort(t)
                t = t[idx]
                p = p[idx]

            per_model[model_label] = {
                "t": t,
                "p": p,
                "measured_gpus": mg,
                "is_total": bool(is_total_default),
                "amp_jitter": (float(amp_jitter_default[0]), float(amp_jitter_default[1])),
                "noise_std_frac": float(noise_std_frac_default),
                "source_path": str(fpath),
            }

        traces_by_batch[batch] = per_model

    return traces_by_batch


class TraceByBatchCache:
    """Pre-built per-GPU templates keyed by (batch, model_label)."""

    def __init__(self, traces_by_batch: dict[int, dict[str, dict[str, Any]]]) -> None:
        self.traces_by_batch = {int(b): v for b, v in traces_by_batch.items()}
        self._templates: dict[tuple[int, str], np.ndarray] = {}
        self._built = False

    def build_templates(
        self,
        *,
        duration_s: Fraction | float,
        timestep_s: Fraction | float,
        steady_skip_s: float = 0.0,
    ) -> None:
        self._templates.clear()
        for b, per_model in self.traces_by_batch.items():
            for label, tr in per_model.items():
                if "measured_gpus" not in tr:
                    raise KeyError(f"Missing required trace field 'measured_gpus' for model={label}")
                if "is_total" not in tr:
                    raise KeyError(f"Missing required trace field 'is_total' for model={label}")
                tpl = build_periodic_per_gpu_template(
                    trace_t=np.asarray(tr["t"], float),
                    trace_p_total=np.asarray(tr["p"], float),
                    measured_gpus=int(tr["measured_gpus"]),
                    timestep_s=float(timestep_s),
                    duration_s=float(duration_s),
                    is_total=bool(tr["is_total"]),
                    steady_skip_s=float(steady_skip_s),
                )
                self._templates[(int(b), str(label))] = tpl
        self._built = True

    @classmethod
    def from_traces(
        cls,
        traces_by_batch: dict[int, dict[str, dict[str, Any]]],
        *,
        duration_s: Fraction | float,
        timestep_s: Fraction | float,
        steady_skip_s: float = 0.0,
    ) -> TraceByBatchCache:
        """Create a cache and build all templates in one step."""
        cache = cls(traces_by_batch)
        cache.build_templates(duration_s=duration_s, timestep_s=timestep_s, steady_skip_s=steady_skip_s)
        return cache

    def template(self, model_label: str, batch_size: int) -> np.ndarray:
        if not self._built:
            raise RuntimeError("Call cache.build_templates(...) first.")
        key = (int(batch_size), str(model_label))
        if key not in self._templates:
            raise KeyError(f"Missing template for (batch={batch_size}, model={model_label}).")
        return self._templates[key]


def _server_count_at_time(
    t: float,
    *,
    initial_server_count: int,
    t_start: float,
    t_end: float,
    floor: float,
) -> int:
    """Number of active servers at time *t* (monotone non-increasing ramp)."""
    s0 = int(initial_server_count)
    if s0 < 0:
        raise ValueError("initial_server_count must be >= 0.")
    floor_n = int(math.ceil(max(float(floor), 0.0) * s0))
    floor_n = max(min(floor_n, s0), 0)

    if t < t_start:
        return s0
    if t_end <= t_start or t >= t_end:
        return floor_n
    alpha = (t - t_start) / (t_end - t_start)
    return max(floor_n, min(s0, round(s0 + (floor_n - s0) * alpha)))


@dataclass
class ServerLayout:
    """Per-model server layout for the offline datacenter."""

    num_servers: int
    total_gpus: int
    gpus_per_replica: int
    gpus_per_server_list: np.ndarray
    phase_list: np.ndarray
    shutdown_order: np.ndarray
    template_offsets: np.ndarray
    amplitude_scales: np.ndarray
    noise_std_frac: float


class OfflineDatacenter(LLMBatchSizeControlledDatacenter[OfflineDatacenterState]):
    """Trace-based datacenter simulation with step-by-step interface.

    Each ``step()`` call computes one timestep of power output by indexing
    into pre-built per-GPU templates, applying per-server amplitude jitter
    and noise, and summing across active servers per phase.

    Batch size changes via ``apply_control()`` take effect on the next
    ``step()`` call.

    Args:
        trace_cache: Pre-built ``TraceByBatchCache`` with templates for all
            (batch, model) pairs.
        models: List of model specs describing the served models.
        timestep_s: Simulation timestep (seconds).
        gpus_per_server: Number of GPUs per physical server rack.
        seed: Random seed for layout generation and noise.
        ramp_t_start: Server shutoff ramp start time (global time).
        ramp_t_end: Server shutoff ramp end time (global time).
        ramp_floor: Fraction of servers remaining after ramp.
        base_kW_per_phase: Constant base load per phase (kW).
        training_overlay: Optional ``TrainingOverlayCache`` for training
            workload.
        training_t_add_start: Global time when training overlay starts.
        training_t_add_end: Global time when training overlay ends.
        training_n_train_gpus: Number of GPUs running training.
        itl_distributions: Optional per-model ITL mixture distributions:
            ``model_label -> batch_size -> ITLMixtureModel``.
        latency_exact_threshold: Exact-sampling threshold for latency averaging.
        latency_seed: Optional seed for latency RNG. Defaults to ``seed + 54321``.
    """

    def __init__(
        self,
        *,
        trace_cache: TraceByBatchCache,
        models: list[LLMInferenceModelSpec],
        timestep_s: Fraction,
        gpus_per_server: int = 8,
        seed: int = 0,
        ramp_t_start: float = 2500.0,
        ramp_t_end: float = 3000.0,
        ramp_floor: float = 0.2,
        base_kW_per_phase: float = 0.0,
        training_overlay: TrainingOverlayCache | None = None,
        training_t_add_start: float = 1000.0,
        training_t_add_end: float = 2000.0,
        training_n_train_gpus: int = 2400,
        itl_distributions: dict[str, dict[int, ITLMixtureModel]] | None = None,
        latency_exact_threshold: int = 30,
        latency_seed: int | None = None,
    ) -> None:
        self._timestep_s = timestep_s
        self._trace_cache = trace_cache
        self._models = list(models)
        self._gpus_per_server = int(gpus_per_server)
        self._seed = int(seed)

        self._layout_rng = np.random.default_rng(self._seed)
        self._rng = np.random.default_rng(self._seed + 12345)

        self._ramp_t_start = float(ramp_t_start)
        self._ramp_t_end = float(ramp_t_end)
        self._ramp_floor = float(ramp_floor)
        self._base_W_per_phase = float(base_kW_per_phase) * 1e3

        self._training_overlay = training_overlay
        self._training_t_add_start = float(training_t_add_start)
        self._training_t_add_end = float(training_t_add_end)
        self._training_n_train_gpus = int(training_n_train_gpus)
        self._itl_distributions = itl_distributions
        self._latency_exact_threshold = int(latency_exact_threshold)

        # Multi-training and multi-ramp support (set by from_config)
        self._training_overlays: list[tuple[TrainingOverlayCache, TrainingRun]] = []
        self._ramp_schedule: ServerRampSchedule | None = None

        # Current batch sizes
        self._batch_by_model: dict[str, int] = {ms.model_label: ms.initial_batch_size for ms in models}

        # Persistent per-model layout (frozen after first build)
        self._layout: dict[str, ServerLayout] = {}

        # Step counter for global time tracking
        self._global_step: int = 0
        self._latency_seed = int(seed) + 54321 if latency_seed is None else int(latency_seed)
        self._latency_rng = np.random.default_rng(self._latency_seed)
        self._events: EventEmitter | None = None
        self._state: OfflineDatacenterState | None = None
        self._history: list[OfflineDatacenterState] = []

        logger.info(
            "OfflineDatacenter: %d models, dt=%s s, seed=%d",
            len(models),
            timestep_s,
            seed,
        )
        for ms in models:
            logger.info(
                "  %s: %d replicas, %d GPUs/replica, batch=%d",
                ms.model_label,
                ms.num_replicas,
                ms.gpus_per_replica,
                ms.initial_batch_size,
            )

    @property
    def dt_s(self) -> Fraction:
        return self._timestep_s

    @property
    def models(self) -> list[LLMInferenceModelSpec]:
        return list(self._models)

    @property
    def batch_by_model(self) -> dict[str, int]:
        return dict(self._batch_by_model)

    @property
    def rng(self) -> np.random.Generator:
        """Noise/latency RNG (seed + 12345)."""
        return self._rng

    @property
    def state(self) -> OfflineDatacenterState:
        if self._state is None:
            raise RuntimeError("OfflineDatacenter.state accessed before first step().")
        return self._state

    def history(self, n: int | None = None) -> list[OfflineDatacenterState]:
        if n is None:
            return list(self._history)
        if n <= 0:
            return []
        return list(self._history[-int(n) :])

    def step(self, clock: SimulationClock) -> OfflineDatacenterState:
        t_now = clock.time_s

        phase_power = np.zeros(3, dtype=float)
        if self._base_W_per_phase > 0:
            phase_power[:] += self._base_W_per_phase

        power_by_model: dict[str, float] = {}
        active_replicas_by_model: dict[str, int] = {}

        for ms in self._models:
            label = ms.model_label
            num_replicas = int(ms.num_replicas)
            gpus_per_replica = int(ms.gpus_per_replica)
            if label not in self._batch_by_model:
                raise KeyError(f"Missing required batch size for model {label!r}")
            batch = int(self._batch_by_model[label])

            if num_replicas <= 0:
                power_by_model[label] = 0.0
                active_replicas_by_model[label] = 0
                continue

            layout = self._get_or_build_layout(ms)
            template = self._trace_cache.template(label, batch)
            L = len(template)

            # Vectorized per-server power: index into template with per-server offsets
            indices = (self._global_step + layout.template_offsets) % L
            per_gpu_values = template[indices]
            server_powers = per_gpu_values * layout.gpus_per_server_list * layout.amplitude_scales
            if layout.noise_std_frac > 0:
                levels = np.maximum(server_powers, 1.0)
                server_powers = (
                    server_powers + self._rng.normal(0.0, 1.0, size=layout.num_servers) * layout.noise_std_frac * levels
                )
            server_powers = np.maximum(server_powers, 0.0)

            # Active server count at this timestep
            if self._ramp_schedule is not None:
                frac = float(self._ramp_schedule.fraction_at(t_now))
                k = int(round(frac * layout.num_servers))
                k = max(0, min(k, layout.num_servers))
            else:
                k = _server_count_at_time(
                    t_now,
                    initial_server_count=layout.num_servers,
                    t_start=self._ramp_t_start,
                    t_end=self._ramp_t_end,
                    floor=self._ramp_floor,
                )

            # Sum power by phase for the first k servers in shutdown order
            active_indices = layout.shutdown_order[:k]
            active_powers = server_powers[active_indices]
            active_phases = layout.phase_list[active_indices]
            model_phase_power = np.zeros(3, dtype=float)
            np.add.at(model_phase_power, active_phases, active_powers)

            phase_power += model_phase_power
            power_by_model[label] = float(np.sum(active_powers))

            active_gpus = int(np.sum(layout.gpus_per_server_list[active_indices]))
            active_replicas_by_model[label] = active_gpus // gpus_per_replica

        # Training overlay
        t_arr = np.asarray(t_now, dtype=float)
        if self._training_overlays:
            for overlay, tr in self._training_overlays:
                training_power_w = float(
                    overlay.eval_total_on_grid(
                        t_arr,
                        t_add_start=tr.t_start,
                        t_add_end=tr.t_end,
                        n_train_gpus=tr.n_gpus,
                    )
                )
                phase_power += training_power_w / 3.0
        elif self._training_overlay is not None:
            training_power_w = float(
                self._training_overlay.eval_total_on_grid(
                    t_arr,
                    t_add_start=self._training_t_add_start,
                    t_add_end=self._training_t_add_end,
                    n_train_gpus=self._training_n_train_gpus,
                )
            )
            phase_power += training_power_w / 3.0

        # ITL sampling
        observed_itl_s_by_model: dict[str, float] = {}
        for ms in self._models:
            label = ms.model_label
            n_rep = active_replicas_by_model.get(label, 0)
            if self._itl_distributions is None or n_rep <= 0:
                observed_itl_s_by_model[label] = float("nan")
                continue
            batch = int(self._batch_by_model[label])
            model_dists = self._itl_distributions.get(label)
            if model_dists is None:
                raise KeyError(f"No ITL distributions for model={label!r}")
            params = model_dists.get(batch)
            if params is None:
                raise KeyError(
                    f"No ITL distributions for model={label!r}, batch={batch}. Available={sorted(model_dists.keys())}"
                )
            observed_itl_s_by_model[label] = params.sample_avg(
                n_replicas=n_rep,
                rng=self._latency_rng,
                exact_threshold=self._latency_exact_threshold,
            )

        state = OfflineDatacenterState(
            time_s=float(t_now),
            power_w=ThreePhase(
                a=float(phase_power[0]),
                b=float(phase_power[1]),
                c=float(phase_power[2]),
            ),
            power_by_model_w=power_by_model,
            active_replicas_by_model=active_replicas_by_model,
            batch_size_by_model=dict(self._batch_by_model),
            observed_itl_s_by_model=observed_itl_s_by_model,
        )
        self._global_step += 1
        self._state = state
        self._history.append(state)
        return state

    @functools.singledispatchmethod
    def apply_control(self, command: DatacenterCommand) -> None:
        """Apply a control command. Dispatches on command type."""
        raise TypeError(f"OfflineDatacenter does not support {type(command).__name__}")

    @apply_control.register
    def _(self, command: SetBatchSize) -> None:
        """Record new batch sizes. Changes take effect on the next step."""
        for label, b in command.batch_size_by_model.items():
            b_int = int(b)
            if b_int <= 0:
                raise ValueError(f"Batch size must be positive for model {label!r}, got {b_int}.")
            old = self._batch_by_model.get(str(label))
            self._batch_by_model[str(label)] = b_int
            if old != b_int:
                logger.info("Batch size %s: %s -> %d", label, old, b_int)
        if self._events is not None:
            self._events.emit(
                "datacenter.batch_size.updated",
                {"batch_size_by_model": dict(self._batch_by_model)},
            )

    def reset(self) -> None:
        self._state = None
        self._history = []
        self._global_step = 0
        self._batch_by_model = {ms.model_label: ms.initial_batch_size for ms in self._models}
        self._layout = {}
        self._layout_rng = np.random.default_rng(self._seed)
        self._rng = np.random.default_rng(self._seed + 12345)
        self._latency_rng = np.random.default_rng(self._latency_seed)

    def bind_event_emitter(self, emitter: EventEmitter) -> None:
        self._events = emitter

    @classmethod
    def from_config(
        cls,
        datacenter: DatacenterConfig,
        workload: WorkloadConfig,
        *,
        trace_cache: TraceByBatchCache,
        timestep_s: Fraction,
        seed: int = 0,
        itl_distributions: dict[str, dict[int, ITLMixtureModel]] | None = None,
        latency_seed: int | None = None,
        latency_exact_threshold: int = 30,
    ) -> OfflineDatacenter:
        """Create an OfflineDatacenter from config objects.

        Args:
            datacenter: Facility configuration (GPUs per server, base load).
            workload: Workload configuration (inference models, training, ramps).
            trace_cache: Pre-built trace cache.
            timestep_s: Simulation timestep (seconds).
            seed: Random seed for layout generation and noise.
            itl_distributions: Optional per-model ITL mixture distributions.
            latency_seed: Optional seed for latency RNG.
            latency_exact_threshold: Exact-sampling threshold for latency averaging.
        """
        inference = workload.inference
        models = list(inference.models)

        # Resolve training config
        training_runs: list[TrainingRun] = []
        if isinstance(workload.training, TrainingRun):
            training_runs = [workload.training]
        elif isinstance(workload.training, TrainingSchedule):
            training_runs = list(workload.training)

        # Resolve ramp config
        ramp_schedule: ServerRampSchedule | None = None
        if isinstance(workload.server_ramps, ServerRamp):
            ramp_schedule = ServerRampSchedule(entries=(workload.server_ramps,))
        elif isinstance(workload.server_ramps, ServerRampSchedule):
            ramp_schedule = workload.server_ramps

        # Use the first training run's params for the legacy fields, or defaults
        if training_runs:
            first_tr = training_runs[0]
            training_overlay = TrainingOverlayCache(
                first_tr.trace_csv,
                target_peak_W_per_gpu=first_tr.target_peak_W_per_gpu,
            )
            t_add_start = first_tr.t_start
            t_add_end = first_tr.t_end
            n_train_gpus = first_tr.n_gpus
        else:
            training_overlay = None
            t_add_start = 0.0
            t_add_end = 0.0
            n_train_gpus = 0

        # Use the first ramp's params for legacy fields, or defaults
        if ramp_schedule and len(ramp_schedule) > 0:
            first_ramp = next(iter(ramp_schedule))
            ramp_t_start = first_ramp.t_start
            ramp_t_end = first_ramp.t_end
            ramp_floor = first_ramp.target
        else:
            ramp_t_start = float("inf")
            ramp_t_end = float("inf")
            ramp_floor = 1.0

        instance = cls(
            trace_cache=trace_cache,
            models=models,
            timestep_s=timestep_s,
            gpus_per_server=datacenter.gpus_per_server,
            seed=seed,
            ramp_t_start=ramp_t_start,
            ramp_t_end=ramp_t_end,
            ramp_floor=ramp_floor,
            base_kW_per_phase=datacenter.base_kW_per_phase,
            training_overlay=training_overlay,
            training_t_add_start=t_add_start,
            training_t_add_end=t_add_end,
            training_n_train_gpus=n_train_gpus,
            itl_distributions=itl_distributions,
            latency_exact_threshold=latency_exact_threshold,
            latency_seed=latency_seed,
        )

        # Set up multi-training overlays
        instance._training_overlays = []
        for tr in training_runs:
            overlay = TrainingOverlayCache(
                tr.trace_csv,
                target_peak_W_per_gpu=tr.target_peak_W_per_gpu,
            )
            instance._training_overlays.append((overlay, tr))

        # Set up multi-ramp schedule
        if ramp_schedule and len(ramp_schedule) > 1:
            instance._ramp_schedule = ramp_schedule

        return instance

    def _get_or_build_layout(self, ms: LLMInferenceModelSpec) -> ServerLayout:
        """Get or build a frozen per-model server layout."""
        label = ms.model_label
        num_replicas = int(ms.num_replicas)
        gpus_per_replica = int(ms.gpus_per_replica)
        total_gpus = num_replicas * gpus_per_replica
        num_servers = int(math.ceil(total_gpus / self._gpus_per_server))

        if label in self._layout:
            existing = self._layout[label]
            if existing.num_servers == num_servers and existing.total_gpus == total_gpus:
                return existing

        gpus_per_server_list = np.full(num_servers, self._gpus_per_server, dtype=int)
        tail = total_gpus - (num_servers - 1) * self._gpus_per_server
        gpus_per_server_list[-1] = int(tail) if tail > 0 else self._gpus_per_server

        sA, sB, sC = split_integer_evenly(num_servers, 3)
        phase_list = np.asarray(([0] * sA) + ([1] * sB) + ([2] * sC), dtype=int)
        self._layout_rng.shuffle(phase_list)

        shutdown_order = np.arange(num_servers, dtype=int)
        self._layout_rng.shuffle(shutdown_order)

        # Read jitter/noise params from any available trace
        tr0 = None
        for b in self._trace_cache.traces_by_batch:
            if label in self._trace_cache.traces_by_batch[int(b)]:
                tr0 = self._trace_cache.traces_by_batch[int(b)][label]
                break
        if tr0 is None:
            raise KeyError(f"Model {label!r} not found in cache.traces_by_batch.")

        if "amp_jitter" not in tr0:
            raise KeyError(f"Missing required trace field 'amp_jitter' for model {label!r}")
        if "noise_std_frac" not in tr0:
            raise KeyError(f"Missing required trace field 'noise_std_frac' for model {label!r}")
        amp_lo, amp_hi = tr0["amp_jitter"]
        noise_std_frac = float(tr0["noise_std_frac"])

        # Use any template's length for offset range (length is invariant to batch size)
        any_batch = next(iter(self._trace_cache.traces_by_batch))
        tpl_len = len(self._trace_cache.template(label, int(any_batch)))
        template_offsets = self._layout_rng.integers(low=0, high=max(tpl_len, 1), size=num_servers)
        amplitude_scales = self._layout_rng.uniform(float(amp_lo), float(amp_hi), size=num_servers)

        layout = ServerLayout(
            num_servers=num_servers,
            total_gpus=total_gpus,
            gpus_per_replica=gpus_per_replica,
            gpus_per_server_list=gpus_per_server_list,
            phase_list=phase_list,
            shutdown_order=shutdown_order,
            template_offsets=template_offsets,
            amplitude_scales=amplitude_scales,
            noise_std_frac=noise_std_frac,
        )
        self._layout[label] = layout
        return layout
