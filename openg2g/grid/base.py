"""Abstract base class for grid backends and grid-level types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from typing import Generic, TypeVar, final

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.common import ThreePhase
from openg2g.events import EventEmitter
from openg2g.grid.command import GridCommand
from openg2g.grid.config import TapPosition


@dataclass(frozen=True)
class PhaseVoltages:
    """Per-phase voltage magnitudes in per-unit.

    Phases missing from the bus have NaN for that field.

    Attributes:
        a: Phase A voltage magnitude (pu).
        b: Phase B voltage magnitude (pu).
        c: Phase C voltage magnitude (pu).
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


@dataclass(frozen=True)
class GridState:
    """State emitted by the grid simulator each timestep.

    Attributes:
        time_s: Simulation time in seconds.
        voltages: Per-bus, per-phase voltage magnitudes.
        tap_positions: Current regulator tap positions, or `None` if
            no regulator is present.
    """

    time_s: float
    voltages: BusVoltages
    tap_positions: TapPosition | None = None


GridStateT = TypeVar("GridStateT", bound=GridState)


class GridBackend(Generic[GridStateT], ABC):
    """Interface for grid simulation backends."""

    _INIT_SENTINEL = object()

    def __init__(self) -> None:
        self._state: GridStateT | None = None
        self._history: list[GridStateT] = []
        self._grid_base_init = GridBackend._INIT_SENTINEL

    def _check_base_init(self) -> None:
        if getattr(self, "_grid_base_init", None) is not GridBackend._INIT_SENTINEL:
            raise TypeError(f"{type(self).__name__}.__init__ must call super().__init__().")

    @property
    @abstractmethod
    def dt_s(self) -> Fraction:
        """Native timestep as a Fraction (seconds)."""

    @final
    @property
    def state(self) -> GridStateT:
        """Latest emitted state.

        Raises:
            RuntimeError: If accessed before the first `step()` call.
        """
        self._check_base_init()
        if self._state is None:
            raise RuntimeError(f"{type(self).__name__}.state accessed before first step().")
        return self._state

    @final
    def history(self, n: int | None = None) -> list[GridStateT]:
        """Return emitted state history (all, or latest `n`)."""
        self._check_base_init()
        if n is None:
            return list(self._history)
        if n <= 0:
            return []
        return list(self._history[-int(n) :])

    @final
    def do_step(
        self,
        clock: SimulationClock,
        power_samples_w: dict[str, list[ThreePhase]] | list[ThreePhase],
        events: EventEmitter,
    ) -> GridStateT:
        """Call `step`, record the state, and return it.

        Called by the coordinator. Subclasses should not override this.
        """
        self._check_base_init()
        state = self.step(clock, power_samples_w, events)
        self._state = state
        self._history.append(state)
        return state

    @abstractmethod
    def step(
        self,
        clock: SimulationClock,
        power_samples_w: dict[str, list[ThreePhase]] | list[ThreePhase],
        events: EventEmitter,
    ) -> GridStateT:
        """Advance one native timestep and return state for this step."""

    @abstractmethod
    def apply_control(self, command: GridCommand, events: EventEmitter) -> None:
        """Apply one control command."""

    @abstractmethod
    def voltages_vector(self) -> np.ndarray:
        """Return voltage magnitudes in `v_index` order."""

    @abstractmethod
    def estimate_sensitivity(self, perturbation_kw: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
        """Estimate voltage sensitivity matrix (H = dv/dp) and return `(H, v0)`."""

    @property
    @abstractmethod
    def v_index(self) -> list[tuple[str, int]]:
        """Fixed (bus, phase) ordering used by [`voltages_vector`][..voltages_vector]."""

    @final
    def do_reset(self) -> None:
        """Clear history and call `reset`.

        Called by the coordinator. Subclasses should not override this.
        """
        self._check_base_init()
        self._state = None
        self._history.clear()
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Reset simulation state to initial conditions.

        Called by the coordinator (via `do_reset`) before each
        [`start`][..start]. Must clear all simulation state: counters,
        cached values. Configuration (dt_s, case files, tap schedules)
        is not affected. History is cleared automatically by
        `do_reset`.

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
