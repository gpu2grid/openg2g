"""Abstract base class for grid backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from fractions import Fraction
from typing import Generic

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.events import EventEmitter
from openg2g.types import GridCommand, GridStateT, ThreePhase


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
