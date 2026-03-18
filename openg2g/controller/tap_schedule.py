"""Tap schedule controller: applies pre-defined regulator tap changes at specified times."""

from __future__ import annotations

from fractions import Fraction

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.datacenter.base import DatacenterBackend
from openg2g.datacenter.command import DatacenterCommand
from openg2g.events import EventEmitter
from openg2g.grid.base import GridBackend
from openg2g.grid.command import GridCommand, SetTaps
from openg2g.grid.config import TapPosition, TapSchedule


class TapScheduleController(Controller[DatacenterBackend, GridBackend]):
    """Applies pre-defined tap changes at scheduled times.

    Supports both per-phase ``TapPosition(a=, b=, c=)`` and named-regulator
    ``TapPosition(regulators={...})`` modes.  When multiple schedule entries
    fire in the same step, their tap values are merged (later entries win).

    Args:
        schedule: Tap schedule built via
            [`TapPosition(...).at(t=...) | ...`][openg2g.grid.config.TapSchedule].
        dt_s: How often the controller checks the schedule (seconds).
    """

    def __init__(self, *, schedule: TapSchedule, dt_s: Fraction = Fraction(1)) -> None:
        self._dt_s = dt_s
        self._entries = list(schedule)
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
    ) -> list[DatacenterCommand | GridCommand]:

        t_now = clock.time_s
        merged_a: float | None = None
        merged_b: float | None = None
        merged_c: float | None = None
        merged_regs: dict[str, float] = {}
        any_fired = False

        while self._idx < len(self._entries):
            t_ev, pos = self._entries[self._idx]
            if float(t_ev) <= t_now + 1e-12:
                if pos.a is not None:
                    merged_a = pos.a
                if pos.b is not None:
                    merged_b = pos.b
                if pos.c is not None:
                    merged_c = pos.c
                if pos.regulators:
                    merged_regs.update(pos.regulators)
                any_fired = True
                self._idx += 1
            else:
                break

        if not any_fired:
            return []

        has_abc = merged_a is not None or merged_b is not None or merged_c is not None
        has_regs = bool(merged_regs)

        if has_abc or has_regs:
            tap = TapPosition(a=merged_a, b=merged_b, c=merged_c, regulators=merged_regs)
            events.emit("controller.tap_schedule.fired", {"tap_position": tap})
            return [SetTaps(tap_position=tap)]
        return []
