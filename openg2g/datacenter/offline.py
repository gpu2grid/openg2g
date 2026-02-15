"""Offline (trace-based) datacenter backend.

Loads power-trace CSVs and serves per-timestep OfflineDatacenterState objects via step().
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mlenergy_data.modeling import ITLMixtureModel

from openg2g.clock import SimulationClock
from openg2g.datacenter.base import LLMBatchSizeControlledDatacenter
from openg2g.datacenter.config import DatacenterConfig, WorkloadConfig
from openg2g.datacenter.training_overlay import TrainingOverlayCache
from openg2g.events import EventEmitter
from openg2g.models.spec import LLMInferenceModelSpec, ModelSpec
from openg2g.types import (
    Command,
    OfflineDatacenterState,
    ServerRamp,
    ServerRampSchedule,
    ThreePhase,
    TrainingRun,
    TrainingSchedule,
)
from openg2g.utils import split_integer_evenly


def build_periodic_per_gpu_template(
    trace_t: np.ndarray,
    trace_p_total: np.ndarray,
    measured_gpus: int,
    *,
    dt: float,
    T: float,
    is_total: bool = True,
    steady_skip_s: float = 0.0,
) -> np.ndarray:
    """Build a per-GPU power template over [0, T] by periodic repetition."""
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

    n_steps = int(np.ceil(float(T) / float(dt))) + 1
    t_grid = np.arange(n_steps, dtype=float) * float(dt)
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
                raise ValueError(
                    f"{fpath} must contain {time_col!r} and {power_col!r}. "
                    f"Got columns={list(df.columns)}"
                )

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

    def build_templates(self, *, T: float, dt: float, steady_skip_s: float = 0.0) -> None:
        self._templates.clear()
        for b, per_model in self.traces_by_batch.items():
            for label, tr in per_model.items():
                if "measured_gpus" not in tr:
                    raise KeyError(
                        f"Missing required trace field 'measured_gpus' for model={label}"
                    )
                if "is_total" not in tr:
                    raise KeyError(f"Missing required trace field 'is_total' for model={label}")
                tpl = build_periodic_per_gpu_template(
                    trace_t=np.asarray(tr["t"], float),
                    trace_p_total=np.asarray(tr["p"], float),
                    measured_gpus=int(tr["measured_gpus"]),
                    dt=float(dt),
                    T=float(T),
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
        T: float,
        dt: float,
        steady_skip_s: float = 0.0,
    ) -> TraceByBatchCache:
        """Create a cache and build all templates in one step."""
        cache = cls(traces_by_batch)
        cache.build_templates(T=T, dt=dt, steady_skip_s=steady_skip_s)
        return cache

    def template(self, model_label: str, batch_size: int) -> np.ndarray:
        if not self._built:
            raise RuntimeError("Call cache.build_templates(...) first.")
        key = (int(batch_size), str(model_label))
        if key not in self._templates:
            raise KeyError(f"Missing template for (batch={batch_size}, model={model_label}).")
        return self._templates[key]


def _servers_active_over_time(
    time_s: np.ndarray,
    *,
    servers0: int,
    t_start: float,
    t_end: float,
    floor: float,
) -> np.ndarray:
    """Discrete shutoff schedule (monotone non-increasing)."""
    t = np.asarray(time_s, float)
    s0 = int(servers0)
    if s0 < 0:
        raise ValueError("servers0 must be >= 0.")
    if t_end < t_start:
        raise ValueError("t_end must be >= t_start.")

    floor_n = int(math.ceil(max(float(floor), 0.0) * s0))
    floor_n = max(min(floor_n, s0), 0)

    S = np.full(t.shape, float(s0), dtype=float)

    if t_end == t_start:
        S[t >= t_start] = float(floor_n)
    else:
        mask = (t >= t_start) & (t <= t_end)
        alpha = (t[mask] - float(t_start)) / (float(t_end) - float(t_start))
        S[mask] = float(s0) + (float(floor_n) - float(s0)) * alpha
        S[t > t_end] = float(floor_n)

    S_int = np.rint(S).astype(int)
    S_int = np.clip(S_int, floor_n, s0)

    for i in range(1, len(S_int)):
        if S_int[i] > S_int[i - 1]:
            S_int[i] = S_int[i - 1]
    return S_int


def _random_restart_profile(
    tpl: np.ndarray,
    *,
    rng: np.random.Generator,
    n_steps: int,
) -> np.ndarray:
    """Walk through *tpl* and restart from a random index at the end.

    Fills the output in segments: each segment copies ``tpl[idx:L]``,
    then draws a new random start index.  The RNG call sequence is
    identical to the scalar loop it replaces.
    """
    tpl = np.asarray(tpl, float)
    L = int(tpl.size)
    if L <= 0:
        return np.zeros(int(n_steps), dtype=float)
    if L == 1:
        return np.full(int(n_steps), float(tpl[0]), dtype=float)

    out = np.empty(int(n_steps), dtype=float)
    pos = 0
    idx = int(rng.integers(0, L))

    while pos < n_steps:
        seg_len = min(L - idx, n_steps - pos)
        out[pos : pos + seg_len] = tpl[idx : idx + seg_len]
        pos += seg_len
        if pos < n_steps:
            idx = int(rng.integers(0, L))
    return out


@dataclass
class ServerLayout:
    """Per-model server layout for the offline datacenter."""

    S_total: int
    total_gpus: int
    g: int
    gpus_per_server_list: np.ndarray
    phase_list: np.ndarray
    server_order: np.ndarray
    server_shifts: np.ndarray
    server_amps: np.ndarray
    noise_std_frac: float


class OfflineDatacenter(LLMBatchSizeControlledDatacenter):
    """Trace-based datacenter simulation with step-by-step interface.

    Generates power-trace chunks and serves one sample per ``step()`` call.
    When the chunk buffer is depleted or batch sizes change, a new chunk is generated.

    Args:
        trace_cache: Pre-built ``TraceByBatchCache`` with templates for all
            (batch, model) pairs.
        models: List of ``ModelSpec`` describing the served models.
        dt: Simulation timestep (seconds).
        batch_init: Initial batch size for all models.
        gpus_per_server: Number of GPUs per physical server rack.
        seed: Random seed for layout generation and noise.
        chunk_steps: Number of timesteps to pre-generate per chunk.
        ramp_t_start: Server shutoff ramp start time (global time).
        ramp_t_end: Server shutoff ramp end time (global time).
        ramp_floor: Fraction of servers remaining after ramp.
        base_kW_per_phase: Constant base load per phase (kW).
        training_overlay: Optional ``TrainingOverlayCache`` for training workload.
        training_t_add_start: Global time when training overlay starts.
        training_t_add_end: Global time when training overlay ends.
        training_n_train_gpus: Number of GPUs running training.
        latency_fits: Optional per-model latency fits:
            ``model_label -> batch_size -> ITLMixtureModel``.
        latency_exact_threshold: Exact-sampling threshold for latency averaging.
        latency_seed: Optional seed for latency RNG. Defaults to ``seed + 54321``.
    """

    def __init__(
        self,
        *,
        trace_cache: TraceByBatchCache,
        models: list[ModelSpec],
        dt: float,
        batch_init: int,
        gpus_per_server: int = 8,
        seed: int = 0,
        chunk_steps: int = 600,
        ramp_t_start: float = 2500.0,
        ramp_t_end: float = 3000.0,
        ramp_floor: float = 0.2,
        base_kW_per_phase: float = 0.0,
        training_overlay: TrainingOverlayCache | None = None,
        training_t_add_start: float = 1000.0,
        training_t_add_end: float = 2000.0,
        training_n_train_gpus: int = 2400,
        latency_fits: dict[str, dict[int, ITLMixtureModel]] | None = None,
        latency_exact_threshold: int = 30,
        latency_seed: int | None = None,
    ) -> None:
        self._dt = float(dt)
        self._cache = trace_cache
        self._models = list(models)
        self._gpus_per_server = int(gpus_per_server)
        self._chunk_steps = int(chunk_steps)

        self._layout_rng = np.random.default_rng(int(seed))
        self._rng = np.random.default_rng(int(seed) + 12345)

        self._ramp_t_start = float(ramp_t_start)
        self._ramp_t_end = float(ramp_t_end)
        self._ramp_floor = float(ramp_floor)
        self._base_W_per_phase = float(base_kW_per_phase) * 1e3

        self._training_overlay = training_overlay
        self._training_t_add_start = float(training_t_add_start)
        self._training_t_add_end = float(training_t_add_end)
        self._training_n_train_gpus = int(training_n_train_gpus)
        self._latency_fits = latency_fits
        self._latency_exact_threshold = int(latency_exact_threshold)

        # Multi-training and multi-ramp support (set by from_config)
        self._training_overlays: list[tuple[TrainingOverlayCache, TrainingRun]] = []
        self._ramp_schedule: ServerRampSchedule | None = None

        # Current batch sizes
        self._batch_by_model: dict[str, int] = {ms.model_label: int(batch_init) for ms in models}

        # Persistent per-model layout (frozen after first build)
        self._layout: dict[str, ServerLayout] = {}

        # Chunk buffer
        self._chunk: list[OfflineDatacenterState] = []
        self._chunk_idx: int = 0

        # Step counter for global time tracking
        self._global_step: int = 0
        seed_latency = int(seed) + 54321 if latency_seed is None else int(latency_seed)
        self._latency_rng = np.random.default_rng(seed_latency)
        self._events: EventEmitter | None = None
        self._state: OfflineDatacenterState | None = None
        self._history: list[OfflineDatacenterState] = []

    @property
    def dt_s(self) -> float:
        return self._dt

    @property
    def models(self) -> list[ModelSpec]:
        return list(self._models)

    @property
    def batch_by_model(self) -> dict[str, int]:
        return dict(self._batch_by_model)

    @property
    def rng(self) -> np.random.Generator:
        """Noise/latency RNG (seed + 12345)."""
        return self._rng

    @property
    def state(self) -> OfflineDatacenterState | None:
        return self._state

    def history(self, n: int | None = None) -> list[OfflineDatacenterState]:
        if n is None:
            return list(self._history)
        if n <= 0:
            return []
        return list(self._history[-int(n) :])

    def step(self, clock: SimulationClock) -> OfflineDatacenterState:
        if self._chunk_idx >= len(self._chunk):
            self._generate_chunk(clock.time_s)
        state = self._chunk[self._chunk_idx]
        self._chunk_idx += 1
        self._global_step += 1
        self._state = state
        self._history.append(state)
        return state

    def apply_control(self, command: Command) -> None:
        """Record new batch sizes. Changes take effect at the next chunk."""
        if command.kind != "set_batch_size":
            raise ValueError(f"OfflineDatacenter does not support command kind={command.kind!r}")
        if "batch_size_by_model" not in command.payload:
            raise ValueError("set_batch_size requires payload['batch_size_by_model'].")
        batch_map = command.payload["batch_size_by_model"]
        if not isinstance(batch_map, dict):
            raise ValueError("set_batch_size requires payload['batch_size_by_model'] as a dict.")
        for label, b in batch_map.items():
            b_int = int(b)
            if b_int <= 0:
                raise ValueError(f"Batch size must be positive for model {label!r}, got {b_int}.")
            self._batch_by_model[str(label)] = b_int
        if self._events is not None:
            self._events.emit(
                "datacenter.batch_size.updated",
                {
                    "kind": command.kind,
                    "batch_size_by_model": dict(self._batch_by_model),
                },
            )

    def bind_event_emitter(self, emitter: EventEmitter) -> None:
        self._events = emitter

    @classmethod
    def from_config(
        cls,
        datacenter: DatacenterConfig,
        workload: WorkloadConfig,
        *,
        trace_cache: TraceByBatchCache,
        dt: float,
        seed: int = 0,
        chunk_steps: int = 600,
        latency_fits: dict[str, dict[int, ITLMixtureModel]] | None = None,
        latency_seed: int | None = None,
        latency_exact_threshold: int = 30,
    ) -> OfflineDatacenter:
        """Create an OfflineDatacenter from config objects.

        Args:
            datacenter: Facility configuration (GPUs per server, base load).
            workload: Workload configuration (inference models, training, ramps).
            trace_cache: Pre-built trace cache.
            dt: Simulation timestep (seconds).
            seed: Random seed for layout generation and noise.
            chunk_steps: Number of timesteps to pre-generate per chunk.
            latency_fits: Optional per-model latency fits.
            latency_seed: Optional seed for latency RNG.
            latency_exact_threshold: Exact-sampling threshold for latency averaging.
        """
        inference = workload.inference
        models = list(inference.models)
        batch_init_by_model = inference.initial_batch_size_by_model

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

        # Build a dummy batch_init for the legacy constructor (overridden below)
        first_batch = next(iter(batch_init_by_model.values()), 128)

        instance = cls(
            trace_cache=trace_cache,
            models=models,
            dt=dt,
            batch_init=first_batch,
            gpus_per_server=datacenter.gpus_per_server,
            seed=seed,
            chunk_steps=chunk_steps,
            ramp_t_start=ramp_t_start,
            ramp_t_end=ramp_t_end,
            ramp_floor=ramp_floor,
            base_kW_per_phase=datacenter.base_kW_per_phase,
            training_overlay=training_overlay,
            training_t_add_start=t_add_start,
            training_t_add_end=t_add_end,
            training_n_train_gpus=n_train_gpus,
            latency_fits=latency_fits,
            latency_exact_threshold=latency_exact_threshold,
            latency_seed=latency_seed,
        )

        # Override per-model initial batch sizes
        instance._batch_by_model = dict(batch_init_by_model)

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

    def _generate_chunk(self, t0_s: float) -> None:
        """Generate a chunk of power-trace samples starting at *t0_s*."""
        dt = self._dt
        n = self._chunk_steps
        # Generate n+1 internal steps (endpoint-inclusive), serving the first n.
        n_internal = n + 1
        time_global = np.arange(n_internal, dtype=float) * dt + t0_s

        P_phase = np.zeros((3, n_internal), dtype=float)
        if self._base_W_per_phase > 0:
            P_phase[:] += self._base_W_per_phase

        power_by_model: dict[str, np.ndarray] = {}
        active_replicas_arr: dict[str, np.ndarray] = {}
        active_gpus_arr: dict[str, np.ndarray] = {}

        for ms in self._models:
            label = ms.model_label
            R = int(ms.num_replicas)
            g = int(ms.gpus_per_replica)
            if label not in self._batch_by_model:
                raise KeyError(f"Missing required batch size for model {label!r}")
            batch = int(self._batch_by_model[label])

            if R <= 0:
                power_by_model[label] = np.zeros(n_internal, dtype=float)
                active_replicas_arr[label] = np.zeros(n_internal, dtype=int)
                active_gpus_arr[label] = np.zeros(n_internal, dtype=int)
                continue

            layout = self._get_or_build_layout(ms, n_internal)
            S_total = layout.S_total
            gpus_per_server_list = layout.gpus_per_server_list
            phase_list = layout.phase_list
            server_order = layout.server_order
            server_shifts = layout.server_shifts
            server_amps = layout.server_amps
            noise_std_frac = layout.noise_std_frac

            base_per_gpu = self._cache.template(label, batch)[:n_internal]

            # Per-server power profiles
            server_power = np.empty((S_total, n_internal), dtype=float)
            for s in range(S_total):
                gpus_s = float(gpus_per_server_list[s])
                tpl_shifted = np.roll(base_per_gpu, int(server_shifts[s]))
                prof_1gpu = _random_restart_profile(tpl_shifted, rng=self._rng, n_steps=n_internal)
                prof = prof_1gpu * (gpus_s * float(server_amps[s]))
                if noise_std_frac > 0:
                    level = max(float(np.mean(prof)), 1.0)
                    prof = prof + self._rng.normal(0.0, noise_std_frac * level, size=prof.shape)
                server_power[s] = np.clip(prof, 0.0, None)

            # Prefix sums per phase in shutoff order
            prefix_phase = np.zeros((3, S_total + 1, n_internal), dtype=float)
            for kk in range(1, S_total + 1):
                s_idx = int(server_order[kk - 1])
                ph = int(phase_list[s_idx])
                prefix_phase[:, kk, :] = prefix_phase[:, kk - 1, :]
                prefix_phase[ph, kk, :] += server_power[s_idx]

            if self._ramp_schedule is not None:
                frac = self._ramp_schedule.fraction_at(time_global)
                S_active_f = np.rint(frac * S_total).astype(int)
                S_active = np.clip(S_active_f, 0, S_total)
            else:
                S_active = _servers_active_over_time(
                    time_global,
                    servers0=S_total,
                    t_start=self._ramp_t_start,
                    t_end=self._ramp_t_end,
                    floor=self._ramp_floor,
                )

            P_model_phase = np.zeros((3, n_internal), dtype=float)
            for i in range(n_internal):
                kk = int(S_active[i])
                P_model_phase[0, i] = prefix_phase[0, kk, i]
                P_model_phase[1, i] = prefix_phase[1, kk, i]
                P_model_phase[2, i] = prefix_phase[2, kk, i]

            P_phase += P_model_phase
            power_by_model[label] = P_model_phase.sum(axis=0)

            # Active GPUs + replicas
            gpus_in_order = np.array(
                [gpus_per_server_list[int(s)] for s in server_order], dtype=int
            )
            gpus_prefix = np.zeros(S_total + 1, dtype=int)
            gpus_prefix[1:] = np.cumsum(gpus_in_order)

            ag = np.array(
                [int(gpus_prefix[int(S_active[i])]) for i in range(n_internal)], dtype=int
            )
            active_gpus_arr[label] = ag
            active_replicas_arr[label] = ag // g

        # Training overlay
        if self._training_overlays:
            for overlay, tr in self._training_overlays:
                P_training = overlay.eval_total_on_grid(
                    time_global,
                    t_add_start=tr.t_start,
                    t_add_end=tr.t_end,
                    n_train_gpus=tr.n_gpus,
                )
                P_phase[0] += P_training / 3.0
                P_phase[1] += P_training / 3.0
                P_phase[2] += P_training / 3.0
        elif self._training_overlay is not None:
            P_training = self._training_overlay.eval_total_on_grid(
                time_global,
                t_add_start=self._training_t_add_start,
                t_add_end=self._training_t_add_end,
                n_train_gpus=self._training_n_train_gpus,
            )
            P_phase[0] += P_training / 3.0
            P_phase[1] += P_training / 3.0
            P_phase[2] += P_training / 3.0

        # Build state objects
        self._chunk = []
        for i in range(n):
            observed_itl_s_by_model: dict[str, float] = {}
            for ms in self._models:
                label = ms.model_label
                n_rep = int(active_replicas_arr[label][i])
                if self._latency_fits is None or n_rep <= 0:
                    observed_itl_s_by_model[label] = float("nan")
                    continue
                if label not in self._batch_by_model:
                    raise KeyError(f"Missing required batch size for model {label!r}")
                batch = int(self._batch_by_model[label])
                model_fits = self._latency_fits.get(label)
                if model_fits is None:
                    raise KeyError(f"No latency fits for model={label!r}")
                params = model_fits.get(batch)
                if params is None:
                    raise KeyError(
                        f"No latency fits for model={label!r}, batch={batch}. "
                        f"Available={sorted(model_fits.keys())}"
                    )
                observed_itl_s_by_model[label] = params.sample_avg(
                    n_replicas=n_rep,
                    rng=self._latency_rng,
                    exact_threshold=self._latency_exact_threshold,
                )

            state = OfflineDatacenterState(
                time_s=float(time_global[i]),
                power_w=ThreePhase(
                    a=float(P_phase[0, i]),
                    b=float(P_phase[1, i]),
                    c=float(P_phase[2, i]),
                ),
                power_by_model_w={
                    label: float(power_by_model[label][i]) for label in power_by_model
                },
                active_replicas_by_model={
                    label: int(active_replicas_arr[label][i]) for label in active_replicas_arr
                },
                batch_size_by_model=dict(self._batch_by_model),
                observed_itl_s_by_model=observed_itl_s_by_model,
            )
            self._chunk.append(state)

        self._chunk_idx = 0

    def _get_or_build_layout(self, ms: LLMInferenceModelSpec, n_steps: int) -> ServerLayout:
        """Get or build a frozen per-model server layout."""
        label = ms.model_label
        R = int(ms.num_replicas)
        g = int(ms.gpus_per_replica)
        total_gpus = R * g
        S_total = int(math.ceil(total_gpus / self._gpus_per_server))

        if label in self._layout:
            existing = self._layout[label]
            if existing.S_total == S_total and existing.total_gpus == total_gpus:
                return existing

        gpus_per_server_list = np.full(S_total, self._gpus_per_server, dtype=int)
        tail = total_gpus - (S_total - 1) * self._gpus_per_server
        gpus_per_server_list[-1] = int(tail) if tail > 0 else self._gpus_per_server

        sA, sB, sC = split_integer_evenly(S_total, 3)
        phase_list = np.asarray(([0] * sA) + ([1] * sB) + ([2] * sC), dtype=int)
        self._layout_rng.shuffle(phase_list)

        server_order = np.arange(S_total, dtype=int)
        self._layout_rng.shuffle(server_order)

        # Read jitter/noise params from any available trace
        tr0 = None
        for b in self._cache.traces_by_batch:
            if label in self._cache.traces_by_batch[int(b)]:
                tr0 = self._cache.traces_by_batch[int(b)][label]
                break
        if tr0 is None:
            raise KeyError(f"Model {label!r} not found in cache.traces_by_batch.")

        if "amp_jitter" not in tr0:
            raise KeyError(f"Missing required trace field 'amp_jitter' for model {label!r}")
        if "noise_std_frac" not in tr0:
            raise KeyError(f"Missing required trace field 'noise_std_frac' for model {label!r}")
        amp_lo, amp_hi = tr0["amp_jitter"]
        noise_std_frac = float(tr0["noise_std_frac"])

        server_shifts = self._layout_rng.integers(low=0, high=max(n_steps, 1), size=S_total)
        server_amps = self._layout_rng.uniform(float(amp_lo), float(amp_hi), size=S_total)

        layout = ServerLayout(
            S_total=S_total,
            total_gpus=total_gpus,
            g=g,
            gpus_per_server_list=gpus_per_server_list,
            phase_list=phase_list,
            server_order=server_order,
            server_shifts=server_shifts,
            server_amps=server_amps,
            noise_std_frac=noise_std_frac,
        )
        self._layout[label] = layout
        return layout
