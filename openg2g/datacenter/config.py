"""Datacenter facility and workload configuration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from openg2g.datacenter.workloads.training import TrainingTrace


class InferenceModelSpec(BaseModel):
    """Specification for one LLM model served in the datacenter.

    This is a pure model-identity object describing *what* is served, not
    *how many* or *at what batch size*.  Deployment-specific parameters
    (replica count, initial batch size) are specified via
    [`ModelDeployment`][openg2g.datacenter.config.ModelDeployment].

    The fields cover (a) model identity (`model_label`, `model_id`,
    `precision`), (b) the measurement setting in ML.ENERGY v3
    (`gpu_model`, `task`, `gpus_per_replica`, `tensor_parallel`,
    `expert_parallel`, `batch_sizes`), and (c) the serving-level knobs
    the simulator needs (`itl_deadline_s`, `feasible_batch_sizes`,
    `fit_exclude_batch_sizes`). The spec is thus the single source of
    truth for "what exactly is deployed here".

    Attributes:
        model_label: Human-readable model identifier (e.g. `"Llama-3.1-70B"`).
        model_id: HuggingFace model ID (e.g. `"meta-llama/Llama-3.1-70B-Instruct"`).
            Used for benchmark data lookups and online API model fields.
        gpu_model: GPU generation (e.g. `"H100"`, `"B200"`). Used to
            filter ML.ENERGY benchmark runs.
        task: Benchmark task name (e.g. `"lm-arena-chat"`, `"gpqa"`,
            `"sourcegraph-fim"`).
        precision: Weight precision (e.g. `"bfloat16"`, `"fp8"`, `"mxfp4"`).
            Informational — the underlying precision is already encoded in
            the `model_id` of the HuggingFace checkpoint.
        gpus_per_replica: GPUs allocated to each replica.
        tensor_parallel: Tensor-parallel degree. For metadata / cache
            key; the data pipeline matches on `gpus_per_replica`.
        expert_parallel: Expert-parallel degree (MoE models). For
            metadata / cache key; the data pipeline matches on `gpus_per_replica`.
        itl_deadline_s: Per-model inter-token latency deadline for the
            OFO latency dual (seconds).
        batch_sizes: Benchmark batch sizes that were measured for this
            model. These are what the data pipeline requests from v3.
        feasible_batch_sizes: OFO- / simulator-allowed subset of
            `batch_sizes`. Must be ⊆ `batch_sizes`. When omitted at
            construction (set to `None`), defaults to `batch_sizes`.
        fit_exclude_batch_sizes: Batch sizes to exclude from logistic
            curve fitting (still included in trace extraction). Use
            when a specific batch has pathological measurements that
            would warp the fit.
    """

    model_config = ConfigDict(frozen=True)

    model_label: str
    model_id: str
    gpu_model: str
    task: str
    precision: str = "bfloat16"
    gpus_per_replica: int = 1
    tensor_parallel: int = 1
    expert_parallel: int = 1
    itl_deadline_s: float
    batch_sizes: tuple[int, ...]
    feasible_batch_sizes: tuple[int, ...] | None = None
    fit_exclude_batch_sizes: tuple[int, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _fill_feasible(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("feasible_batch_sizes") is None:
            data["feasible_batch_sizes"] = tuple(data.get("batch_sizes", ()))
        return data

    @model_validator(mode="after")
    def _validate(self) -> InferenceModelSpec:
        if self.gpus_per_replica < 1:
            raise ValueError(f"gpus_per_replica must be >= 1, got {self.gpus_per_replica}.")
        if self.tensor_parallel < 1:
            raise ValueError(f"tensor_parallel must be >= 1, got {self.tensor_parallel}.")
        if self.expert_parallel < 1:
            raise ValueError(f"expert_parallel must be >= 1, got {self.expert_parallel}.")
        if self.itl_deadline_s <= 0:
            raise ValueError(f"itl_deadline_s must be > 0, got {self.itl_deadline_s}.")
        if not self.batch_sizes:
            raise ValueError("batch_sizes must not be empty.")
        if not self.feasible_batch_sizes:
            raise ValueError("feasible_batch_sizes must not be empty.")
        if not set(self.feasible_batch_sizes).issubset(set(self.batch_sizes)):
            raise ValueError(
                f"feasible_batch_sizes {self.feasible_batch_sizes} must be a subset of batch_sizes {self.batch_sizes}."
            )
        return self

    def cache_hash(self) -> str:
        """Content-addressable SHA-256 hex digest (16-char prefix) of the
        measurement-relevant spec fields.

        Two `InferenceModelSpec` instances with matching `model_id`,
        `gpu_model`, `task`, `precision`, `gpus_per_replica`,
        `tensor_parallel`, `expert_parallel`, `batch_sizes`, and
        `fit_exclude_batch_sizes` produce the same hash. `model_label`,
        `itl_deadline_s`, and `feasible_batch_sizes` are simulation-time
        knobs that do not invalidate the on-disk measurement cache.

        Used as the per-spec cache directory name under `data/specs/`.
        """
        payload = {
            "model_id": self.model_id,
            "gpu_model": self.gpu_model,
            "task": self.task,
            "precision": self.precision,
            "gpus_per_replica": self.gpus_per_replica,
            "tensor_parallel": self.tensor_parallel,
            "expert_parallel": self.expert_parallel,
            "batch_sizes": sorted(self.batch_sizes),
            "fit_exclude_batch_sizes": sorted(self.fit_exclude_batch_sizes),
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class ModelDeployment:
    """One model's deployment at a datacenter site.

    Pairs an [`InferenceModelSpec`][openg2g.datacenter.config.InferenceModelSpec] (model identity)
    with the initial batch size. Replica counts (including runtime ramps)
    live on [`ReplicaSchedule`][openg2g.datacenter.config.ReplicaSchedule]
    and are passed separately via `OfflineWorkload.replica_schedules`.

    Attributes:
        spec: The model specification.
        initial_batch_size: Starting batch size for this deployment.
            Must be in `spec.feasible_batch_sizes`.
    """

    spec: InferenceModelSpec
    initial_batch_size: int

    def __post_init__(self) -> None:
        if self.initial_batch_size <= 0:
            raise ValueError(f"initial_batch_size must be > 0, got {self.initial_batch_size}.")
        feasible_batch_sizes = self.spec.feasible_batch_sizes
        if feasible_batch_sizes is None:
            raise ValueError("spec.feasible_batch_sizes must not be None.")
        if self.initial_batch_size not in feasible_batch_sizes:
            raise ValueError(
                f"initial_batch_size ({self.initial_batch_size}) must be in "
                f"feasible_batch_sizes ({feasible_batch_sizes})."
            )


class TrainingRun:
    """Training workload parameters.

    The trace is eagerly rescaled so its peak matches `target_peak_W_per_gpu`.
    Use `eval_power` to evaluate total training power at a given simulation time.

    Combine with [`at`][.at] and `|` to build a [`TrainingSchedule`][..TrainingSchedule]:

    ```python
    schedule = (
        TrainingRun(n_gpus=2400, trace=trace_a).at(t_start=1000, t_end=2000)
        | TrainingRun(n_gpus=1200, trace=trace_b).at(t_start=2500, t_end=3500)
    )
    ```

    Attributes:
        n_gpus: Number of GPUs running the training workload.
        trace: Single-GPU [`TrainingTrace`][openg2g.datacenter.workloads.training.TrainingTrace].
        target_peak_W_per_gpu: The trace is rescaled so its peak equals this value.
    """

    __slots__ = ("_period", "_rescaled_power", "_trace_time", "n_gpus", "target_peak_W_per_gpu", "trace")

    def __init__(self, *, n_gpus: int, trace: TrainingTrace, target_peak_W_per_gpu: float = 400.0) -> None:
        if n_gpus <= 0:
            raise ValueError(f"TrainingRun n_gpus must be > 0, got {n_gpus}.")
        self.n_gpus = n_gpus
        self.trace = trace
        self.target_peak_W_per_gpu = target_peak_W_per_gpu

        t = np.asarray(trace.t_s, float)
        p = np.asarray(trace.power_w, float)
        t = t - t[0]
        period = float(t[-1] - t[0])
        if period <= 0:
            raise ValueError("Training trace time span must be positive.")
        peak = float(np.max(p))
        if peak <= 0:
            raise ValueError("Training trace has non-positive peak; cannot scale.")
        self._rescaled_power = p * (target_peak_W_per_gpu / peak)
        self._trace_time = t
        self._period = period

    def eval_power(self, t: float, t_start: float, t_end: float) -> float:
        """Evaluate total training power at simulation time `t`.

        Returns zero if `t` is outside `[t_start, t_end]`.

        Args:
            t: Global simulation time (seconds).
            t_start: Time when training becomes active (seconds).
            t_end: Time when training stops (seconds).

        Returns:
            Total training power (W) across all `n_gpus` GPUs.
        """
        if t < t_start or t > t_end:
            return 0.0
        t_local = t - t_start
        t_mod = t_local % self._period
        p_1gpu = float(np.interp(t_mod, self._trace_time, self._rescaled_power))
        return p_1gpu * self.n_gpus

    def at(self, t_start: float, t_end: float) -> TrainingSchedule:
        """Schedule this training run over `[t_start, t_end]`.

        Args:
            t_start: Global simulation time when training becomes active (seconds).
            t_end: Global simulation time when training stops (seconds).

        Returns:
            A single-entry [`TrainingSchedule`][...TrainingSchedule].
        """
        if t_end < t_start:
            raise ValueError(f"t_end ({t_end}) must be >= t_start ({t_start}).")
        return TrainingSchedule(((self, float(t_start), float(t_end)),))


class TrainingSchedule:
    """Ordered collection of [`TrainingRun`][..TrainingRun] objects scheduled
    over time windows.

    Each entry is a `(TrainingRun, t_start, t_end)` tuple. Entries are
    sorted by `t_start`.

    Built with [`TrainingRun.at`][..TrainingRun.at] and `|`.

    Example:

    ```python
    schedule = (
        TrainingRun(n_gpus=2400, trace=trace_a).at(t_start=1000, t_end=2000)
        | TrainingRun(n_gpus=1200, trace=trace_b).at(t_start=2500, t_end=3500)
    )
    ```
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: tuple[tuple[TrainingRun, float, float], ...] = ()) -> None:
        self._entries = tuple(sorted(entries, key=lambda e: e[1]))

    def __or__(self, other: TrainingSchedule) -> TrainingSchedule:
        return TrainingSchedule((*self._entries, *other._entries))

    def __iter__(self) -> Iterator[tuple[TrainingRun, float, float]]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __repr__(self) -> str:
        parts = [f"TrainingRun(n_gpus={r.n_gpus}).at(t_start={s}, t_end={e})" for r, s, e in self._entries]
        return " | ".join(parts)


class ReplicaSchedule:
    """Per-model replica count over time.

    Specifies the initial replica count and optional linear ramps to
    new target counts over time windows. Method-chaining API:

    ```python
    ReplicaSchedule(initial=720).ramp_to(144, t_start=2500, t_end=3000)

    ReplicaSchedule(initial=720)
        .ramp_to(144, t_start=2500, t_end=3000)
        .ramp_to(200, t_start=3200, t_end=3400)
    ```

    Semantics: before the first ramp, the active count equals `initial`.
    During each `[t_start, t_end]` window the count linearly interpolates
    from the previous level to `target`. Between ramps, the count holds
    at the last target.

    Attributes:
        initial: Replica count before any ramp.
    """

    __slots__ = ("_initial", "_ramps")

    def __init__(self, *, initial: int) -> None:
        if initial < 0:
            raise ValueError(f"ReplicaSchedule initial must be >= 0, got {initial}.")
        self._initial = initial
        self._ramps: tuple[tuple[int, float, float], ...] = ()

    @property
    def initial(self) -> int:
        """Replica count before any ramp."""
        return self._initial

    def ramp_to(self, target: int, *, t_start: float, t_end: float) -> ReplicaSchedule:
        """Add a linear ramp to `target` over `[t_start, t_end]`.

        Ramps may be specified in any order; they are sorted by `t_start`
        in the returned schedule. The new ramp must not overlap any
        existing ramp's window (touching at the boundary is allowed:
        `prev.t_end == new.t_start`).

        Args:
            target: Target replica count after the ramp completes.
            t_start: Global simulation time when the ramp begins (seconds).
            t_end: Global simulation time when the ramp ends (seconds).

        Returns:
            A new `ReplicaSchedule` with the ramp added and ramps sorted
            by `t_start`.

        Raises:
            ValueError: If `target` is negative, `t_end < t_start`, or
                the new ramp overlaps any existing ramp window.
        """
        if target < 0:
            raise ValueError(f"ramp_to target must be >= 0, got {target}.")
        if t_end < t_start:
            raise ValueError(f"t_end ({t_end}) must be >= t_start ({t_start}).")
        # Reject overlap with existing ramps: [t_start, t_end] must not
        # overlap any existing [s, e]. Touching boundaries (t_end == s or
        # t_start == e) is allowed.
        for _target, s, e in self._ramps:
            if t_start < e and t_end > s:
                raise ValueError(f"ramp_to window [{t_start}, {t_end}] overlaps existing ramp window [{s}, {e}].")
        new = ReplicaSchedule(initial=self._initial)
        new._ramps = tuple(
            sorted(
                (*self._ramps, (int(target), float(t_start), float(t_end))),
                key=lambda r: r[1],
            )
        )
        return new

    def max_count(self) -> int:
        """Return the maximum replica count (initial or any ramp target)."""
        if not self._ramps:
            return self._initial
        return max(self._initial, *(target for target, _, _ in self._ramps))

    def count_at(self, t: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the active replica count at time(s) *t*.

        Piecewise-linear interpolation between ramp events.
        Before the first ramp, returns `initial`.

        Args:
            t: Scalar or array of global simulation times (seconds).

        Returns:
            Active replica count(s), same shape as *t*.
        """
        if isinstance(t, np.ndarray):
            vfunc = np.vectorize(self._count_scalar, otypes=[float])
            return vfunc(t)
        return float(self._count_scalar(float(t)))

    def _count_scalar(self, t: float) -> float:
        level = float(self._initial)
        for target, t_start, t_end in self._ramps:
            if t < t_start:
                return level
            if t <= t_end:
                if t_end == t_start:
                    return float(target)
                alpha = (t - t_start) / (t_end - t_start)
                return level + (float(target) - level) * alpha
            level = float(target)
        return level

    def __repr__(self) -> str:
        parts = [f"ReplicaSchedule(initial={self._initial})"]
        for target, t_start, t_end in self._ramps:
            parts.append(f".ramp_to({target}, t_start={t_start}, t_end={t_end})")
        return "".join(parts)

    def __len__(self) -> int:
        return len(self._ramps)

    def __bool__(self) -> bool:
        return True


class DatacenterConfig(BaseModel):
    """Physical datacenter facility configuration.

    Attributes:
        gpus_per_server: Number of GPUs per physical server rack.
        base_kw_per_phase: Constant base load per phase (kW).
        power_factor: Power factor of the datacenter loads (lagging).
    """

    model_config = ConfigDict(frozen=True)

    gpus_per_server: int = 8
    base_kw_per_phase: float = 0.0
    power_factor: float = 0.95

    @model_validator(mode="after")
    def _validate(self) -> DatacenterConfig:
        if self.gpus_per_server < 1:
            raise ValueError(f"gpus_per_server must be >= 1, got {self.gpus_per_server}.")
        if not (0.0 < self.power_factor <= 1.0):
            raise ValueError(f"power_factor must be in (0, 1], got {self.power_factor}.")
        return self


class PowerAugmentationConfig(BaseModel):
    """Power augmentation settings for virtual server scaling.

    Controls per-server amplitude jitter and additive noise applied during
    power augmentation.

    Attributes:
        amplitude_scale_range: `(low, high)` range for per-server amplitude
            scaling. Each virtual server draws a uniform multiplier from this range.
        noise_fraction: Gaussian noise standard deviation as a fraction of
            per-server power.
    """

    model_config = ConfigDict(frozen=True)

    amplitude_scale_range: tuple[float, float] = (1.0, 1.0)
    noise_fraction: float = 0.0

    @model_validator(mode="after")
    def _validate(self) -> PowerAugmentationConfig:
        lo, hi = self.amplitude_scale_range
        if lo > hi:
            raise ValueError(f"amplitude_scale_range low ({lo}) must be <= high ({hi}).")
        if lo <= 0:
            raise ValueError(f"amplitude_scale_range low ({lo}) must be > 0.")
        if self.noise_fraction < 0:
            raise ValueError(f"noise_fraction must be >= 0, got {self.noise_fraction}.")
        return self
