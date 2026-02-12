"""Abstract base class for datacenter backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from openg2g.clock import SimulationClock
from openg2g.events import EventEmitter
from openg2g.types import Command, DatacenterState


class DatacenterBackend(ABC):
    """Interface for datacenter power simulation backends."""

    @property
    @abstractmethod
    def dt_s(self) -> float:
        """Native timestep in seconds."""

    @property
    @abstractmethod
    def state(self) -> DatacenterState | None:
        """Latest emitted state, or ``None`` before the first step."""

    @abstractmethod
    def history(self, n: int | None = None) -> Sequence[DatacenterState]:
        """Return emitted state history (all, or latest ``n``)."""

    @abstractmethod
    def step(self, clock: SimulationClock) -> DatacenterState:
        """Advance one native timestep. Return state for this step."""

    @abstractmethod
    def apply_control(self, command: Command) -> None:
        """Apply one command. Takes effect on next step() call."""

    def bind_event_emitter(self, emitter: EventEmitter) -> None:
        """Attach a clock-bound emitter for backend-originated events."""
        del emitter
