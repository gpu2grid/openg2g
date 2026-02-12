"""Abstract base class for datacenter backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from openg2g.clock import SimulationClock
from openg2g.types import DatacenterControlAction, DatacenterState


class DatacenterBackend(ABC):
    """Interface for datacenter power simulation backends."""

    @property
    @abstractmethod
    def dt_s(self) -> float:
        """Native timestep in seconds."""

    @abstractmethod
    def step(self, clock: SimulationClock) -> DatacenterState:
        """Advance one native timestep. Return state for this step."""

    @abstractmethod
    def apply_control(self, action: DatacenterControlAction) -> None:
        """Apply control action. Takes effect on next step() call."""
