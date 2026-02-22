"""Tap schedule controller: applies pre-defined regulator tap changes at specified times."""

from __future__ import annotations

from fractions import Fraction

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.datacenter.base import DatacenterBackend
from openg2g.events import EventEmitter
from openg2g.grid.base import GridBackend, TapSchedule
from openg2g.types import ControlAction, SetTaps


class TapScheduleController(Controller[DatacenterBackend, GridBackend]):
    """Applies pre-defined tap changes at scheduled times.

    Args:
        schedule: Tap schedule built via `TapPosition(...).at(t=...) | ...`.
        dt_s: How often the controller checks the schedule (seconds).
    """

    def __init__(
        self,
        *,
        schedule: TapSchedule,
        dt_s: Fraction = Fraction(1),
    ) -> None:
        self._dt_s = dt_s
        self._entries = [(t, pos.as_reg_dict()) for t, pos in schedule]
        self._idx = 0

    def reset(self) -> None:
        self._idx = 0

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    def step(
        self,
        clock: SimulationClock,
        datacenter: DatacenterBackend,
        grid: GridBackend,
        events: EventEmitter,
    ) -> ControlAction:

        t_now = clock.time_s
        tap_changes: dict[str, float] = {}

        while self._idx < len(self._entries):
            t_ev, taps = self._entries[self._idx]
            if float(t_ev) <= t_now + 1e-12:
                tap_changes.update(taps)
                self._idx += 1
            else:
                break

        if tap_changes:
            return ControlAction(commands=[SetTaps(tap_changes=tap_changes)])
        return ControlAction(commands=[])
