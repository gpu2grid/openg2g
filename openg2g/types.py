"""Shared state and action dataclasses for the G2G framework."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import numpy as np


@dataclass(frozen=True)
class ThreePhase:
    """Three-phase quantity. Access via .a, .b, .c."""

    a: float
    b: float
    c: float

    def total(self) -> float:
        """Return the sum of all three phases."""
        return self.a + self.b + self.c

    def as_tuple(self) -> tuple[float, float, float]:
        """Return `(a, b, c)` as a plain tuple."""
        return (self.a, self.b, self.c)


@dataclass(frozen=True)
class BusVoltages:
    """Per-bus, per-phase voltage map.

    Access: voltages["671"].a -> Vpu for bus 671, phase A.
    Buses missing a phase have NaN for that field.
    """

    _data: dict[str, ThreePhase]

    def __getitem__(self, bus: str) -> ThreePhase:
        return self._data[bus]

    def buses(self) -> list[str]:
        """Return the list of bus names."""
        return list(self._data.keys())

    def __contains__(self, bus: str) -> bool:
        return bus in self._data

    def __len__(self) -> int:
        return len(self._data)


@dataclass(frozen=True)
class DatacenterState:
    """State emitted by a datacenter backend each timestep."""

    time_s: float
    power_w: ThreePhase
    batch_size_by_model: dict[str, int] = field(default_factory=dict)
    active_replicas_by_model: dict[str, int] = field(default_factory=dict)
    observed_itl_s_by_model: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OfflineDatacenterState(DatacenterState):
    """Extended state from the offline (trace-based) backend."""

    power_by_model_w: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OnlineDatacenterState(DatacenterState):
    """Extended state from the online (live GPU) backend.

    The base `power_w` field carries the augmented three-phase power
    (what the grid sees). This subclass adds the measured (pre-augmentation)
    breakdown for post-hoc analysis.

    Attributes:
        measured_power_w: Total measured three-phase power from real GPUs
            (before augmentation), plus base load.
        measured_power_w_by_model: Per-model total measured power from real
            GPUs (watts).
        augmented_power_w_by_model: Per-model augmented power (watts). This
            is the power fed to the grid for each model after scaling up.
        augmentation_factor_by_model: Per-model augmentation multiplier
            (virtual replicas / real replicas).
    """

    measured_power_w: ThreePhase = field(default_factory=lambda: ThreePhase(a=0.0, b=0.0, c=0.0))
    measured_power_w_by_model: dict[str, float] = field(default_factory=dict)
    augmented_power_w_by_model: dict[str, float] = field(default_factory=dict)
    augmentation_factor_by_model: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GridState:
    """State emitted by the grid simulator each timestep."""

    time_s: float
    voltages: BusVoltages
    tap_positions: TapPosition | None = None


DCStateT = TypeVar("DCStateT", bound=DatacenterState)
GridStateT = TypeVar("GridStateT", bound=GridState)


@dataclass(frozen=True)
class ControlAction:
    """Collection of control commands emitted by a controller.

    Use an empty `commands` list for a no-op action.
    """

    commands: list[Command] = field(default_factory=list)


@dataclass(frozen=True)
class Command:
    """Single command envelope routed by target and kind."""

    target: CommandTarget | str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", CommandTarget(self.target))


class CommandTarget(str, Enum):
    """Command routing target."""

    DATACENTER = "datacenter"
    GRID = "grid"


class Phase(str, Enum):
    """Electrical phase."""

    A = "a"
    B = "b"
    C = "c"


@dataclass(frozen=True)
class TapPosition:
    """Regulator tap position per phase, as per-unit tap ratios.

    Each field is the tap ratio for the corresponding phase regulator.
    Phases set to `None` are left unchanged when applied.  At least
    one phase must be specified.

    Combine with `at()` and `|` to build a `TapSchedule`:

        TAP_STEP = 0.00625  # standard 5/8% tap step
        schedule = (
            TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP).at(t=0)
            | TapPosition(a=1.1).at(t=1500)
            | TapPosition(a=1.0625, c=1.0625).at(t=3300)
        )
    """

    a: float | None = None
    b: float | None = None
    c: float | None = None

    def __post_init__(self) -> None:
        if self.a is None and self.b is None and self.c is None:
            raise ValueError("TapPosition requires at least one phase (a, b, or c).")

    def at(self, t: float) -> TapSchedule:
        """Schedule this position at time *t* seconds."""
        return TapSchedule(((t, self),))

    def as_reg_dict(self) -> dict[str, float]:
        """Return a dict mapping regulator names to tap ratios, omitting `None` phases."""
        d: dict[str, float] = {}
        if self.a is not None:
            d["reg1"] = self.a
        if self.b is not None:
            d["reg2"] = self.b
        if self.c is not None:
            d["reg3"] = self.c
        return d


class TapSchedule:
    """Ordered sequence of scheduled tap positions.

    Build using `TapPosition.at()` and the `|` operator:

        TAP_STEP = 0.00625  # standard 5/8% tap step
        schedule = (
            TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP).at(t=0)
            | TapPosition(a=1.0 + 16 * TAP_STEP).at(t=25 * 60)
        )
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: tuple[tuple[float, TapPosition], ...]) -> None:
        self._entries = tuple(sorted(entries, key=lambda e: e[0]))

    def __or__(self, other: TapSchedule) -> TapSchedule:
        return TapSchedule(self._entries + other._entries)

    def __iter__(self) -> Iterator[tuple[float, TapPosition]]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __repr__(self) -> str:
        parts: list[str] = []
        for t, p in self._entries:
            fields = []
            if p.a is not None:
                fields.append(f"a={p.a}")
            if p.b is not None:
                fields.append(f"b={p.b}")
            if p.c is not None:
                fields.append(f"c={p.c}")
            parts.append(f"TapPosition({', '.join(fields)}).at(t={t})")
        return " | ".join(parts)


@dataclass(frozen=True)
class TrainingRun:
    """A single training workload window.

    Attributes:
        t_start: Global simulation time when training becomes active (seconds).
        t_end: Global simulation time when training stops (seconds).
        n_gpus: Number of GPUs running the training workload.
        trace_csv: Path to CSV with columns `t_s` and `power_W` (1-GPU trace).
        target_peak_W_per_gpu: The trace is rescaled so its peak equals this value.
    """

    t_start: float
    t_end: float
    n_gpus: int
    trace_csv: Path
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
            TrainingRun(t_start=500, t_end=1500, n_gpus=2400, trace_csv=path_a)
            | TrainingRun(t_start=2000, t_end=3000, n_gpus=1200, trace_csv=path_b)
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
