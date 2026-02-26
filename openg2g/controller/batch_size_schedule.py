"""Batch size schedule controller: applies pre-defined batch size changes at specified times."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.datacenter.base import DatacenterBackend
from openg2g.events import EventEmitter
from openg2g.grid.base import GridBackend
from openg2g.types import ControlAction, SetBatchSize


@dataclass(frozen=True)
class BatchSizeChange:
    """A batch size change event, optionally with gradual ramp-up.

    Attributes:
        batch_size: Target batch size (max_num_seqs).
        ramp_up_rate: Requests/second ramp-up rate. 0 means immediate.
    """

    batch_size: int
    ramp_up_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")
        if self.ramp_up_rate < 0:
            raise ValueError(f"ramp_up_rate must be >= 0, got {self.ramp_up_rate}.")

    def at(self, t: float) -> BatchSizeSchedule:
        """Schedule this change at time *t* seconds.

        Returns:
            A single-entry [`BatchSizeSchedule`][...BatchSizeSchedule].
        """
        return BatchSizeSchedule(((t, self),))


class BatchSizeSchedule:
    """Ordered sequence of batch size changes, built with `|` operator.

    Example:

        schedule = (
            BatchSizeChange(48).at(40)
            | BatchSizeChange(32).at(60)
            | BatchSizeChange(48, ramp_up_rate=4).at(280)
        )

    Raises:
        ValueError: If two entries share the same timestamp.
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: tuple[tuple[float, BatchSizeChange], ...]) -> None:
        self._entries = tuple(sorted(entries, key=lambda e: e[0]))
        times = [t for t, _ in self._entries]
        if len(times) != len(set(times)):
            seen: set[float] = set()
            dupes = sorted({t for t in times if t in seen or seen.add(t)})
            raise ValueError(f"BatchSizeSchedule has duplicate timestamps: {dupes}")

    def __or__(self, other: BatchSizeSchedule) -> BatchSizeSchedule:
        return BatchSizeSchedule(self._entries + other._entries)

    def __iter__(self) -> Iterator[tuple[float, BatchSizeChange]]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __repr__(self) -> str:
        parts: list[str] = []
        for t, c in self._entries:
            ramp = f", ramp_up_rate={c.ramp_up_rate}" if c.ramp_up_rate > 0 else ""
            parts.append(f"BatchSizeChange({c.batch_size}{ramp}).at(t={t})")
        return " | ".join(parts)


class BatchSizeScheduleController(Controller[DatacenterBackend, GridBackend]):
    """Applies pre-defined batch size changes at scheduled times.

    Walks each model's schedule and emits
    [`SetBatchSize`][openg2g.types.SetBatchSize] commands when the
    simulation clock reaches the scheduled time.

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
        ramp_rates: dict[str, float] = {}

        for label, schedule in self._schedules.items():
            entries = list(schedule)
            idx = self._indices[label]

            while idx < len(entries):
                t_ev, change = entries[idx]
                if float(t_ev) <= t_now + 1e-12:
                    batch_changes[label] = change.batch_size
                    if change.ramp_up_rate > 0:
                        ramp_rates[label] = change.ramp_up_rate
                    idx += 1
                else:
                    break

            self._indices[label] = idx

        if batch_changes:
            return ControlAction(
                commands=[
                    SetBatchSize(
                        batch_size_by_model=batch_changes,
                        ramp_up_rate_by_model=ramp_rates,
                    )
                ]
            )
        return ControlAction(commands=[])
