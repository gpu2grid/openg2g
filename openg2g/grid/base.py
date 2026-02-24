"""Abstract base class for grid backends and grid-level types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Generic, TypeVar

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.events import EventEmitter
from openg2g.types import GridCommand, TapPosition, ThreePhase


class Phase(str, Enum):
    """Electrical phase."""

    A = "a"
    B = "b"
    C = "c"


@dataclass(frozen=True)
class PhaseVoltages:
    """Per-phase voltage magnitudes in per-unit.

    Phases missing from the bus have NaN for that field.
    """

    a: float
    b: float
    c: float


@dataclass(frozen=True)
class BusVoltages:
    """Per-bus, per-phase voltage map.

    Access: voltages["671"].a -> Vpu for bus 671, phase A.
    Buses missing a phase have NaN for that field.
    """

    _data: dict[str, PhaseVoltages]

    def __getitem__(self, bus: str) -> PhaseVoltages:
        return self._data[bus]

    def buses(self) -> list[str]:
        """Return the list of bus names."""
        return list(self._data.keys())

    def __contains__(self, bus: str) -> bool:
        return bus in self._data

    def __len__(self) -> int:
        return len(self._data)


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
        """Fixed (bus, phase) ordering used by
        [`voltages_vector`][..voltages_vector]."""

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
        """Estimate voltage sensitivity matrix (H = dv/dp) and return
        `(H, v0)`."""

    @abstractmethod
    def reset(self) -> None:
        """Reset simulation state to initial conditions.

        Called by the coordinator before each [`start`][..start]. Must
        clear all simulation state: history, counters, cached values.
        Configuration (dt_s, case files, tap schedules) is not
        affected.

        Abstract so every implementation explicitly enumerates its state.
        A forgotten field is a bug -- not clearing it silently corrupts
        the second run.
        """

    def start(self) -> None:
        """Acquire per-run resources (solver circuits, connections).

        Called after [`reset`][..reset], before the simulation loop.
        Override for backends that need resource acquisition (e.g.,
        [`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid] compiles its
        DSS circuit here). No-op by default because most offline
        components have no resources to acquire.
        """

    def stop(self) -> None:
        """Release per-run resources. Simulation state is preserved.

        Called after the simulation loop in LIFO order. Override for
        backends that acquired resources in [`start`][..start]. No-op
        by default.
        """

    def bind_event_emitter(self, emitter: EventEmitter) -> None:
        """Attach a clock-bound [`EventEmitter`][openg2g.events.EventEmitter]
        for backend-originated events."""
