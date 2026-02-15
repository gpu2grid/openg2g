"""No-op controller that does nothing."""

from __future__ import annotations

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.datacenter.base import DatacenterBackend
from openg2g.events import EventEmitter
from openg2g.grid.base import GridBackend
from openg2g.types import ControlAction


class NoopController(Controller[DatacenterBackend, GridBackend]):
    """Controller that always returns an empty action."""

    def __init__(self, dt_s: float = 1.0) -> None:
        self._dt_s = float(dt_s)

    @property
    def dt_s(self) -> float:
        return self._dt_s

    def step(
        self,
        clock: SimulationClock,
        datacenter: DatacenterBackend,
        grid: GridBackend,
        events: EventEmitter,
    ) -> ControlAction:
        return ControlAction(commands=[])
