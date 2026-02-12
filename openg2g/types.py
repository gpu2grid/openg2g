"""Shared state and action dataclasses for the G2G framework."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ThreePhase:
    """Three-phase quantity. Access via .a, .b, .c."""

    a: float
    b: float
    c: float

    def total(self) -> float:
        return self.a + self.b + self.c

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.a, self.b, self.c)


@dataclass(frozen=True)
class BusVoltages:
    """Per-bus, per-phase voltage map.

    Access: voltages["671"].a -> Vpu for bus 671, phase A.
    Buses missing a phase have NaN for that field.
    """

    _data: dict[str, ThreePhase]

    def __getitem__(self, bus: str) -> ThreePhase:
        return self._data[bus]

    def buses(self) -> list[str]:
        return list(self._data.keys())

    def __contains__(self, bus: str) -> bool:
        return bus in self._data

    def __len__(self) -> int:
        return len(self._data)


@dataclass(frozen=True)
class DatacenterState:
    """State emitted by a datacenter backend each timestep."""

    time_s: float
    power_w: ThreePhase
    batch_size_by_model: dict[str, int] = field(default_factory=dict)
    active_replicas_by_model: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class OfflineDatacenterState(DatacenterState):
    """Extended state from the offline (trace-based) backend."""

    power_by_model_w: dict[str, float] = field(default_factory=dict)
    avg_itl_by_model: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OnlineDatacenterState(DatacenterState):
    """Extended state from the online (live GPU) backend."""

    gpu_power_readings: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GridState:
    """State emitted by the grid simulator each timestep."""

    time_s: float
    voltages: BusVoltages


@dataclass(frozen=True)
class ControlAction:
    """Base control action (no-op).

    Subclass ``DatacenterControlAction`` for batch size changes or
    ``GridControlAction`` for tap position changes.  The coordinator
    uses ``isinstance`` to route each action to the correct component.
    """


@dataclass(frozen=True)
class DatacenterControlAction(ControlAction):
    """Control action targeting the datacenter (batch size changes)."""

    batch_size_by_model: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class GridControlAction(ControlAction):
    """Control action targeting the grid (tap position changes)."""

    tap_changes: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TapPosition:
    """Regulator tap position per phase, as per-unit tap ratios.

    Each field is the tap ratio for the corresponding phase regulator.
    Combine with :meth:`at` and ``|`` to build a :class:`TapSchedule`::

        S = 0.00625  # standard 5/8% tap step
        schedule = (
            TapPosition(a=1.0 + 14 * S, b=1.0 + 6 * S, c=1.0 + 15 * S).at(t=0)
            | TapPosition(a=1.0 + 16 * S, b=1.0 + 6 * S, c=1.0 + 17 * S).at(t=25 * 60)
            | TapPosition(a=1.0 + 10 * S, b=1.0 + 6 * S, c=1.0 + 10 * S).at(t=55 * 60)
        )
    """

    a: float
    b: float
    c: float

    def at(self, t: float) -> TapSchedule:
        """Schedule this position at time *t* seconds."""
        return TapSchedule(((t, self),))


class TapSchedule:
    """Ordered sequence of scheduled tap positions.

    Build using :meth:`TapPosition.at` and the ``|`` operator::

        S = 0.00625  # standard 5/8% tap step
        schedule = (
            TapPosition(a=1.0 + 14 * S, b=1.0 + 6 * S, c=1.0 + 15 * S).at(t=0)
            | TapPosition(a=1.0 + 16 * S, b=1.0 + 6 * S, c=1.0 + 17 * S).at(t=25 * 60)
        )
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: tuple[tuple[float, TapPosition], ...]) -> None:
        self._entries = tuple(sorted(entries, key=lambda e: e[0]))

    def __or__(self, other: TapSchedule) -> TapSchedule:
        return TapSchedule(self._entries + other._entries)

    def __iter__(self) -> Iterator[tuple[float, TapPosition]]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __repr__(self) -> str:
        parts = [f"TapPosition(a={p.a}, b={p.b}, c={p.c}).at(t={t})" for t, p in self._entries]
        return " | ".join(parts)
