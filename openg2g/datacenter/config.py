"""Datacenter facility and workload configuration."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from openg2g.datacenter.workloads.training import TrainingTrace


class InferenceModelSpec(BaseModel):
    """Specification for one LLM model served in the datacenter.

    Attributes:
        model_label: Human-readable model identifier (e.g. `"Llama-3.1-70B"`).
        model_id: HuggingFace model ID (e.g. `"meta-llama/Llama-3.1-70B-Instruct"`).
            Used for benchmark data lookups and online API model fields.
        initial_num_replicas: Initial number of replicas of this model in the datacenter.
            Inference ramps with target > 1.0 can scale beyond this count.
        gpus_per_replica: GPUs allocated to each replica (determines model
            parallelism and per-replica power draw).
        initial_batch_size: Initial batch size for this model.
        itl_deadline_s: Per-model inter-token latency deadline for the OFO
            latency dual (seconds).
        feasible_batch_sizes: Allowed batch sizes. Used by the OFO
            controller for discretizing continuous batch-size updates
            and by the online datacenter for load-generator sizing.
            Defaults to `(initial_batch_size,)`.
    """

    model_config = ConfigDict(frozen=True)

    model_label: str
    model_id: str = ""
    initial_num_replicas: int
    gpus_per_replica: int
    initial_batch_size: int
    itl_deadline_s: float
    feasible_batch_sizes: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> InferenceModelSpec:
        if not self.feasible_batch_sizes:
            object.__setattr__(self, "feasible_batch_sizes", (self.initial_batch_size,))
        elif self.initial_batch_size not in self.feasible_batch_sizes:
            raise ValueError(
                f"initial_batch_size ({self.initial_batch_size}) must be in "
                f"feasible_batch_sizes ({self.feasible_batch_sizes})."
            )
        if self.initial_num_replicas < 0:
            raise ValueError(f"initial_num_replicas must be >= 0, got {self.initial_num_replicas}.")
        if self.gpus_per_replica < 1:
            raise ValueError(f"gpus_per_replica must be >= 1, got {self.gpus_per_replica}.")
        if self.initial_batch_size <= 0:
            raise ValueError(f"initial_batch_size must be > 0, got {self.initial_batch_size}.")
        if self.itl_deadline_s <= 0:
            raise ValueError(f"itl_deadline_s must be > 0, got {self.itl_deadline_s}.")
        return self


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


@dataclass(frozen=True)
class InferenceRamp:
    """Inference server ramp parameters.

    Transitions the active inference server fraction to `target`. Combine with
    [`at`][.at] and `|` to build an [`InferenceRampSchedule`][..InferenceRampSchedule]:

    ```python
    ramps = (
        InferenceRamp(target=0.2).at(t_start=2500, t_end=3000)
        | InferenceRamp(target=1.2, model="Llama-3.1-8B").at(t_start=3200, t_end=3400)
    )
    ```

    Attributes:
        target: Target active-server fraction after the ramp.
            Values below 1.0 reduce active servers (ramp down),
            values above 1.0 increase active servers beyond the
            initial count (ramp up / scale out).
        model: Model label this ramp applies to. ``None`` applies to all
            models in the datacenter.
    """

    target: float
    model: str | None = None

    def __post_init__(self) -> None:
        if self.target < 0.0:
            raise ValueError(f"InferenceRamp target must be >= 0.0, got {self.target}.")

    def at(self, t_start: float, t_end: float) -> InferenceRampSchedule:
        """Schedule this ramp over `[t_start, t_end]`.

        Args:
            t_start: Global simulation time when the ramp begins (seconds).
            t_end: Global simulation time when the ramp ends (seconds).

        Returns:
            A single-entry [`InferenceRampSchedule`][...InferenceRampSchedule].
        """
        if t_end < t_start:
            raise ValueError(f"t_end ({t_end}) must be >= t_start ({t_start}).")
        return InferenceRampSchedule(((self, float(t_start), float(t_end)),))


class InferenceRampSchedule:
    """Ordered collection of [`InferenceRamp`][..InferenceRamp] events.

    Each entry is an `(InferenceRamp, t_start, t_end)` tuple. Entries are
    sorted by `t_start`.

    Built with [`InferenceRamp.at`][..InferenceRamp.at] and `|`.

    Semantics: before the first ramp, fraction = 1.0. During each
    `[t_start, t_end]` window, the fraction linearly interpolates from
    the previous level to `target`. Between ramps, the fraction holds
    at the last target.

    An empty schedule means all servers are active (fraction = 1.0)
    at all times.

    Example:

    ```python
    ramps = (
        InferenceRamp(target=0.2).at(t_start=2500, t_end=3000)
        | InferenceRamp(target=1.0).at(t_start=3200, t_end=3400)
    )
    ```
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: tuple[tuple[InferenceRamp, float, float], ...] = ()) -> None:
        self._entries = tuple(sorted(entries, key=lambda e: e[1]))

    def __or__(self, other: InferenceRampSchedule) -> InferenceRampSchedule:
        return InferenceRampSchedule((*self._entries, *other._entries))

    def __iter__(self) -> Iterator[tuple[InferenceRamp, float, float]]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def for_model(self, model_label: str) -> InferenceRampSchedule:
        """Return a schedule containing only entries that apply to *model_label*.

        An entry applies if its ``model`` is ``None`` (all models) or matches
        *model_label* exactly.
        """
        filtered = tuple(e for e in self._entries if e[0].model is None or e[0].model == model_label)
        return InferenceRampSchedule(filtered)

    def max_target(self) -> float:
        """Return the maximum target across all entries, or 1.0 if empty."""
        if not self._entries:
            return 1.0
        return max(r.target for r, _, _ in self._entries)

    def __repr__(self) -> str:
        parts = []
        for r, s, e in self._entries:
            model_part = f", model={r.model!r}" if r.model else ""
            parts.append(f"InferenceRamp(target={r.target}{model_part}).at(t_start={s}, t_end={e})")
        return " | ".join(parts)

    def fraction_at(self, t: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the active inference server fraction at time(s) *t*.

        Piecewise-linear interpolation between ramp events.
        Before the first ramp, fraction = 1.0.

        Args:
            t: Scalar or array of global simulation times (seconds).

        Returns:
            Active-server fraction(s), same shape as *t*.
        """
        if isinstance(t, np.ndarray):
            return self._fraction_array(t)
        return float(self._fraction_scalar(float(t)))

    def _fraction_scalar(self, t: float) -> float:
        level = 1.0
        for ramp, t_start, t_end in self._entries:
            if t < t_start:
                return level
            if t <= t_end:
                if t_end == t_start:
                    return ramp.target
                alpha = (t - t_start) / (t_end - t_start)
                return level + (ramp.target - level) * alpha
            level = ramp.target
        return level

    def _fraction_array(self, t: np.ndarray) -> np.ndarray:
        vfunc = np.vectorize(self._fraction_scalar, otypes=[float])
        return vfunc(t)


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
