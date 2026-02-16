"""Abstract base class for grid backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from fractions import Fraction

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.events import EventEmitter
from openg2g.types import Command, GridState, ThreePhase


class GridBackend(ABC):
    """Interface for grid simulation backends."""

    @property
    @abstractmethod
    def dt_s(self) -> Fraction:
        """Native timestep as a Fraction (seconds)."""

    @property
    @abstractmethod
    def state(self) -> GridState | None:
        """Latest emitted state, or `None` before the first step."""

    @abstractmethod
    def history(self, n: int | None = None) -> Sequence[GridState]:
        """Return emitted state history (all, or latest `n`)."""

    @property
    @abstractmethod
    def v_index(self) -> list[tuple[str, int]]:
        """Fixed (bus, phase) ordering used by `voltages_vector`."""

    @abstractmethod
    def step(
        self,
        clock: SimulationClock,
        load_trace_w: list[ThreePhase],
        *,
        interval_start_w: ThreePhase | None = None,
    ) -> GridState:
        """Advance one native timestep and return state for this step."""

    @abstractmethod
    def apply_control(self, command: Command) -> None:
        """Apply one control command."""

    @abstractmethod
    def voltages_vector(self) -> np.ndarray:
        """Return voltage magnitudes in `v_index` order."""

    @abstractmethod
    def estimate_H(self, dp_kw: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
        """Estimate `H = dv/dp` and return `(H, v0)`."""

    def bind_event_emitter(self, emitter: EventEmitter) -> None:  # noqa: B027
        """Attach a clock-bound emitter for backend-originated events."""
