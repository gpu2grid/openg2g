"""Tap schedule controller: applies pre-defined regulator tap changes at specified times."""

from __future__ import annotations

from fractions import Fraction

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.datacenter.base import DatacenterBackend
from openg2g.events import EventEmitter
from openg2g.grid.base import GridBackend
from openg2g.types import ControlAction, SetTaps


class TapScheduleController(Controller[DatacenterBackend, GridBackend]):
    """Applies pre-defined tap changes at scheduled times.

    Args:
        schedule: List of `(time_s, {regcontrol_name: tap_pu})` tuples,
            sorted by time.
        dt_s: How often the controller checks the schedule (seconds).
    """

    def __init__(
        self,
        *,
        schedule: list[tuple[float, dict[str, float]]],
        dt_s: Fraction = Fraction(1),
    ) -> None:
        self._dt_s = dt_s
        self._schedule = sorted(list(schedule), key=lambda x: float(x[0]))
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

        while self._idx < len(self._schedule):
            t_ev, taps = self._schedule[self._idx]
            if float(t_ev) <= t_now + 1e-12:
                tap_changes.update(taps)
                self._idx += 1
            else:
                break

        if tap_changes:
            return ControlAction(commands=[SetTaps(tap_changes=tap_changes)])
        return ControlAction(commands=[])
