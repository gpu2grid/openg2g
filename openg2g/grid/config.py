"""Grid configuration and schedule types."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class DCLoadSpec:
    """Specification for a datacenter load connection point on the grid.

    Args:
        bus: Bus name where the datacenter is connected.
        bus_kv: Line-to-line voltage (kV) at the datacenter bus.
        connection_type: Connection type for DC loads (default ``"wye"``).
    """

    bus: str
    bus_kv: float
    connection_type: Literal["wye", "delta"] = "wye"


@dataclass(frozen=True)
class TapPosition:
    """Regulator tap position, supporting both per-phase and named-regulator modes.

    **Per-phase mode** (legacy, single regulator bank):

    ```python
    TapPosition(a=1.075, b=1.05, c=1.075)
    ```

    **Named-regulator mode** (multi-bank systems like IEEE 34/123):

    ```python
    TapPosition(regulators={"creg1a": 1.075, "creg1b": 1.05, "creg2a": 1.0})
    ```

    At least one of ``a``, ``b``, ``c``, or ``regulators`` must be specified.
    """

    a: float | None = None
    b: float | None = None
    c: float | None = None
    regulators: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.a is None and self.b is None and self.c is None and not self.regulators:
            raise ValueError("TapPosition requires at least one phase (a, b, or c) or regulators dict.")

    def at(self, t: float) -> TapSchedule:
        """Schedule this position at time `t` seconds."""
        return TapSchedule(((t, self),))


class TapSchedule:
    """Ordered sequence of scheduled tap positions.

    Build using [`TapPosition.at`][..TapPosition.at] and the ``|`` operator:

    ```python
    TAP_STEP = 0.00625  # standard 5/8% tap step
    schedule = (
        TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP).at(t=0)
        | TapPosition(a=1.0 + 16 * TAP_STEP).at(t=25 * 60)
    )
    ```

    Raises:
        ValueError: If two entries share the same timestamp.
    """

    __slots__ = ("_entries",)

    def __init__(self, entries: tuple[tuple[float, TapPosition], ...]) -> None:
        self._entries = tuple(sorted(entries, key=lambda e: e[0]))
        times = [t for t, _ in self._entries]
        if len(times) != len(set(times)):
            seen: set[float] = set()
            dupes = sorted({t for t in times if t in seen or seen.add(t)})
            raise ValueError(f"TapSchedule has duplicate timestamps: {dupes}")

    def __or__(self, other: TapSchedule) -> TapSchedule:
        return TapSchedule(self._entries + other._entries)

    def __iter__(self) -> Iterator[tuple[float, TapPosition]]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __repr__(self) -> str:
        parts: list[str] = []
        for t, p in self._entries:
            fields = []
            if p.a is not None:
                fields.append(f"a={p.a}")
            if p.b is not None:
                fields.append(f"b={p.b}")
            if p.c is not None:
                fields.append(f"c={p.c}")
            if p.regulators:
                fields.append(f"regulators={p.regulators}")
            parts.append(f"TapPosition({', '.join(fields)}).at(t={t})")
        return " | ".join(parts)
