"""No-op controller that does nothing."""

from __future__ import annotations

from openg2g.clock import SimulationClock
from openg2g.context import SimulationContext
from openg2g.controller.base import Controller
from openg2g.types import ControlAction, DatacenterState, GridState


class NoopController(Controller):
    """Controller that always returns an empty action."""

    def __init__(self, dt_s: float = 1.0):
        self._dt_s = float(dt_s)

    @property
    def dt_s(self) -> float:
        return self._dt_s

    def step(
        self,
        clock: SimulationClock,
        dc_state: DatacenterState | None,
        grid_state: GridState | None,
        context: SimulationContext,
    ) -> ControlAction:
        return ControlAction(commands=[])
