"""Abstract base class for controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from openg2g.clock import SimulationClock
from openg2g.context import SimulationContext
from openg2g.types import ControlAction, DatacenterState, GridState


class Controller(ABC):
    """Interface for a control component in the G2G framework.

    Controllers receive datacenter and grid state and produce control actions.
    Multiple controllers compose in order within the coordinator.
    """

    @property
    @abstractmethod
    def dt_s(self) -> float:
        """Control interval in seconds."""

    @abstractmethod
    def step(
        self,
        clock: SimulationClock,
        dc_state: DatacenterState | None,
        grid_state: GridState | None,
        context: SimulationContext,
    ) -> ControlAction:
        """Compute a control action. Must complete synchronously."""

    def required_features(self) -> set[str]:
        """Names of required context features."""
        return set()
