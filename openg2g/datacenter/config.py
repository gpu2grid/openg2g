"""Datacenter facility and workload configuration."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from openg2g.datacenter.training_overlay import TrainingTrace
from openg2g.models.spec import LLMInferenceWorkload


@dataclass(frozen=True)
class TrainingRun:
    """A single training workload window.

    Attributes:
        t_start: Global simulation time when training becomes active (seconds).
        t_end: Global simulation time when training stops (seconds).
        n_gpus: Number of GPUs running the training workload.
        trace: Single-GPU training power trace.
        target_peak_W_per_gpu: The trace is rescaled so its peak equals this value.
    """

    t_start: float
    t_end: float
    n_gpus: int
    trace: TrainingTrace
    target_peak_W_per_gpu: float = 400.0

    def __post_init__(self) -> None:
        if self.t_end < self.t_start:
            raise ValueError(f"TrainingRun t_end ({self.t_end}) must be >= t_start ({self.t_start}).")
        if self.n_gpus < 0:
            raise ValueError(f"TrainingRun n_gpus must be >= 0, got {self.n_gpus}.")

    def __or__(self, other: TrainingRun | TrainingSchedule) -> TrainingSchedule:
        if isinstance(other, TrainingRun):
            return TrainingSchedule(entries=(self, other))
        return TrainingSchedule(entries=(self, *other))


class TrainingSchedule:
    """Ordered collection of training windows, built with `|`.

    Example:

        schedule = (
            TrainingRun(t_start=500, t_end=1500, n_gpus=2400, trace=trace_a)
            | TrainingRun(t_start=2000, t_end=3000, n_gpus=1200, trace=trace_b)
        )
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: tuple[TrainingRun, ...]) -> None:
        self._entries = tuple(sorted(entries, key=lambda e: e.t_start))

    def __or__(self, other: TrainingRun | TrainingSchedule) -> TrainingSchedule:
        if isinstance(other, TrainingRun):
            return TrainingSchedule(entries=(*self._entries, other))
        return TrainingSchedule(entries=(*self._entries, *other._entries))

    def __iter__(self) -> Iterator[TrainingRun]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __repr__(self) -> str:
        parts = [f"TrainingRun(t_start={r.t_start}, t_end={r.t_end}, n_gpus={r.n_gpus})" for r in self._entries]
        return " | ".join(parts)


@dataclass(frozen=True)
class ServerRamp:
    """A single server ramp event.

    Transitions the active-server fraction to `target` linearly over
    `[t_start, t_end]`.

    Attributes:
        t_start: Global simulation time when the ramp begins (seconds).
        t_end: Global simulation time when the ramp ends (seconds).
        target: Target active-server fraction after the ramp (0.0--1.0).
    """

    t_start: float
    t_end: float
    target: float

    def __post_init__(self) -> None:
        if self.t_end < self.t_start:
            raise ValueError(f"ServerRamp t_end ({self.t_end}) must be >= t_start ({self.t_start}).")
        if not (0.0 <= self.target <= 1.0):
            raise ValueError(f"ServerRamp target must be in [0.0, 1.0], got {self.target}.")

    def __or__(self, other: ServerRamp | ServerRampSchedule) -> ServerRampSchedule:
        if isinstance(other, ServerRamp):
            return ServerRampSchedule(entries=(self, other))
        return ServerRampSchedule(entries=(self, *other))


class ServerRampSchedule:
    """Ordered collection of server ramp events, built with `|`.

    Semantics: before the first ramp, fraction = 1.0.  During each
    `[t_start, t_end]` window, the fraction linearly interpolates from
    the previous level to `target`.  Between ramps, the fraction holds
    at the last target.

    An empty schedule means all servers are active (fraction = 1.0) at all times.

    Example:

        ramps = (
            ServerRamp(t_start=2500, t_end=3000, target=0.2)
            | ServerRamp(t_start=3200, t_end=3400, target=1.0)
        )
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: tuple[ServerRamp, ...]) -> None:
        self._entries = tuple(sorted(entries, key=lambda e: e.t_start))

    def __or__(self, other: ServerRamp | ServerRampSchedule) -> ServerRampSchedule:
        if isinstance(other, ServerRamp):
            return ServerRampSchedule(entries=(*self._entries, other))
        return ServerRampSchedule(entries=(*self._entries, *other._entries))

    def __iter__(self) -> Iterator[ServerRamp]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __repr__(self) -> str:
        parts = [f"ServerRamp(t_start={r.t_start}, t_end={r.t_end}, target={r.target})" for r in self._entries]
        return " | ".join(parts)

    def fraction_at(self, t: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the active-server fraction at time(s) *t*.

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
        for ramp in self._entries:
            if t < ramp.t_start:
                return level
            if t <= ramp.t_end:
                if ramp.t_end == ramp.t_start:
                    return ramp.target
                alpha = (t - ramp.t_start) / (ramp.t_end - ramp.t_start)
                return level + (ramp.target - level) * alpha
            level = ramp.target
        return level

    def _fraction_array(self, t: np.ndarray) -> np.ndarray:
        vfunc = np.vectorize(self._fraction_scalar, otypes=[float])
        return vfunc(t)


@dataclass(frozen=True)
class DatacenterConfig:
    """Physical datacenter facility configuration.

    Attributes:
        gpus_per_server: Number of GPUs per physical server rack.
        base_kw_per_phase: Constant base load per phase (kW).
    """

    gpus_per_server: int = 8
    base_kw_per_phase: float = 0.0

    def __post_init__(self) -> None:
        if self.gpus_per_server < 1:
            raise ValueError(f"gpus_per_server must be >= 1, got {self.gpus_per_server}.")


class WorkloadConfig:
    """What runs in the datacenter: inference, training, and ramp events.

    Accepts flexible input types and normalizes them internally:
    - A single `TrainingRun` is wrapped in a `TrainingSchedule`.
    - A single `ServerRamp` is wrapped in a `ServerRampSchedule`.
    - `None` yields an empty schedule.

    Properties always return schedule types, eliminating `isinstance`
    checks at consumption sites.

    Args:
        inference: LLM inference workload specification.
        training: Training workload window(s). `None` disables training overlay.
        server_ramps: Server ramp event(s). `None` keeps all servers active.
    """

    def __init__(
        self,
        inference: LLMInferenceWorkload,
        training: TrainingRun | TrainingSchedule | None = None,
        server_ramps: ServerRamp | ServerRampSchedule | None = None,
    ) -> None:
        self._inference = inference

        if training is None:
            self._training = TrainingSchedule(entries=())
        elif isinstance(training, TrainingRun):
            self._training = TrainingSchedule(entries=(training,))
        else:
            self._training = training

        if server_ramps is None:
            self._server_ramps = ServerRampSchedule(entries=())
        elif isinstance(server_ramps, ServerRamp):
            self._server_ramps = ServerRampSchedule(entries=(server_ramps,))
        else:
            self._server_ramps = server_ramps

    @property
    def inference(self) -> LLMInferenceWorkload:
        return self._inference

    @property
    def training(self) -> TrainingSchedule:
        return self._training

    @property
    def server_ramps(self) -> ServerRampSchedule:
        return self._server_ramps
