"""Batch size schedule controller: applies pre-defined batch size changes at specified times."""

from __future__ import annotations

from fractions import Fraction

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.datacenter.base import DatacenterBackend
from openg2g.events import EventEmitter
from openg2g.grid.base import GridBackend
from openg2g.types import BatchSizeSchedule, ControlAction, SetBatchSize


class BatchSizeScheduleController(Controller[DatacenterBackend, GridBackend]):
    """Applies pre-defined batch size changes at scheduled times.

    Walks each model's schedule and emits `set_batch_size` commands when
    the simulation clock reaches the scheduled time.

    Args:
        schedules: Per-model batch size schedules, keyed by model label.
        dt_s: How often the controller checks the schedule (seconds).
    """

    def __init__(
        self,
        *,
        schedules: dict[str, BatchSizeSchedule],
        dt_s: Fraction = Fraction(1),
    ) -> None:
        self._dt_s = dt_s
        self._schedules = dict(schedules)
        self._indices: dict[str, int] = {label: 0 for label in schedules}

    def reset(self) -> None:
        self._indices = {label: 0 for label in self._schedules}

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
        batch_changes: dict[str, int] = {}
        ramp_up_rate: float = 0.0

        for label, schedule in self._schedules.items():
            entries = list(schedule)
            idx = self._indices[label]

            while idx < len(entries):
                t_ev, change = entries[idx]
                if float(t_ev) <= t_now + 1e-12:
                    batch_changes[label] = change.batch_size
                    if change.ramp_up_rate > 0:
                        ramp_up_rate = max(ramp_up_rate, change.ramp_up_rate)
                    idx += 1
                else:
                    break

            self._indices[label] = idx

        if batch_changes:
            return ControlAction(
                commands=[
                    SetBatchSize(
                        batch_size_by_model=batch_changes,
                        ramp_up_rate=ramp_up_rate,
                    )
                ]
            )
        return ControlAction(commands=[])
