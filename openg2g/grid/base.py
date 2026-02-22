"""Abstract base class for grid backends and grid-level types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Generic, TypeVar

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.events import EventEmitter
from openg2g.types import GridCommand, ThreePhase


class Phase(str, Enum):
    """Electrical phase."""

    A = "a"
    B = "b"
    C = "c"


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

    Raises:
        ValueError: If two entries share the same timestamp.
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: tuple[tuple[float, TapPosition], ...]) -> None:
        self._entries = tuple(sorted(entries, key=lambda e: e[0]))
        times = [t for t, _ in self._entries]
        if len(times) != len(set(times)):
            seen: set[float] = set()
            dupes = sorted({t for t in times if t in seen or seen.add(t)})
            raise ValueError(f"TapSchedule has duplicate timestamps: {dupes}")

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
class GridState:
    """State emitted by the grid simulator each timestep."""

    time_s: float
    voltages: BusVoltages
    tap_positions: TapPosition | None = None


GridStateT = TypeVar("GridStateT", bound=GridState)


class GridBackend(Generic[GridStateT], ABC):
    """Interface for grid simulation backends."""

    @property
    @abstractmethod
    def dt_s(self) -> Fraction:
        """Native timestep as a Fraction (seconds)."""

    @property
    @abstractmethod
    def state(self) -> GridStateT:
        """Latest emitted state.

        Raises:
            RuntimeError: If accessed before the first `step()` call.
        """

    @abstractmethod
    def history(self, n: int | None = None) -> Sequence[GridStateT]:
        """Return emitted state history (all, or latest `n`)."""

    @property
    @abstractmethod
    def v_index(self) -> list[tuple[str, int]]:
        """Fixed (bus, phase) ordering used by `voltages_vector`."""

    @abstractmethod
    def step(
        self,
        clock: SimulationClock,
        power_samples_w: list[ThreePhase],
    ) -> GridStateT:
        """Advance one native timestep and return state for this step."""

    @abstractmethod
    def apply_control(self, command: GridCommand) -> None:
        """Apply one control command."""

    @abstractmethod
    def voltages_vector(self) -> np.ndarray:
        """Return voltage magnitudes in `v_index` order."""

    @abstractmethod
    def estimate_sensitivity(self, perturbation_kw: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
        """Estimate voltage sensitivity matrix (H = dv/dp) and return ``(H, v0)``."""

    @abstractmethod
    def reset(self) -> None:
        """Reset simulation state to initial conditions.

        Called by the coordinator before each `start()`. Must clear all
        simulation state: history, counters, cached values.
        Configuration (dt_s, case files, tap schedules) is not affected.

        Abstract so every implementation explicitly enumerates its state.
        A forgotten field is a bug -- not clearing it silently corrupts
        the second run.
        """

    def start(self) -> None:
        """Acquire per-run resources (solver circuits, connections).

        Called after `reset()`, before the simulation loop. Override for
        backends that need resource acquisition (e.g., `OpenDSSGrid`
        compiles its DSS circuit here). No-op by default because most
        offline components have no resources to acquire.
        """

    def stop(self) -> None:
        """Release per-run resources. Simulation state is preserved.

        Called after the simulation loop in LIFO order. Override for
        backends that acquired resources in `start()`. No-op by default.
        """

    def bind_event_emitter(self, emitter: EventEmitter) -> None:
        """Attach a clock-bound emitter for backend-originated events."""
