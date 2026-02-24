"""Cross-cutting types shared across component families."""

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
        """Return the sum of all three phases."""
        return self.a + self.b + self.c

    def as_tuple(self) -> tuple[float, float, float]:
        """Return `(a, b, c)` as a plain tuple."""
        return (self.a, self.b, self.c)


@dataclass(frozen=True)
class TapPosition:
    """Regulator tap position per phase, as per-unit tap ratios.

    Each field is the tap ratio for the corresponding phase regulator.
    Phases set to `None` are left unchanged when applied.  At least
    one phase must be specified.

    Combine with [`at`][.at] and `|` to build a
    [`TapSchedule`][..TapSchedule]:

        TAP_STEP = 0.00625  # standard 5/8% tap step
        schedule = (
            TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP).at(t=0)
            | TapPosition(a=1.1).at(t=1500)
            | TapPosition(a=1.0625, c=1.0625).at(t=3300)
        )
    """

    a: float | None = None
    b: float | None = None
    c: float | None = None

    def __post_init__(self) -> None:
        if self.a is None and self.b is None and self.c is None:
            raise ValueError("TapPosition requires at least one phase (a, b, or c).")

    def at(self, t: float) -> TapSchedule:
        """Schedule this position at time *t* seconds."""
        return TapSchedule(((t, self),))


class TapSchedule:
    """Ordered sequence of scheduled tap positions.

    Build using [`TapPosition.at`][..TapPosition.at] and the `|`
    operator:

        TAP_STEP = 0.00625  # standard 5/8% tap step
        schedule = (
            TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP).at(t=0)
            | TapPosition(a=1.0 + 16 * TAP_STEP).at(t=25 * 60)
        )

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
            parts.append(f"TapPosition({', '.join(fields)}).at(t={t})")
        return " | ".join(parts)


class DatacenterCommand:
    """Base for commands targeting the datacenter backend.

    Subclass this for each concrete datacenter command kind.
    The coordinator routes commands to backends based on this type hierarchy.
    """


class GridCommand:
    """Base for commands targeting the grid backend.

    Subclass this for each concrete grid command kind.
    The coordinator routes commands to backends based on this type hierarchy.
    """


@dataclass(frozen=True)
class SetBatchSize(DatacenterCommand):
    """Set batch sizes for one or more models.

    Attributes:
        batch_size_by_model: Mapping of model label to target batch size.
        ramp_up_rate_by_model: Per-model requests/second ramp-up rate.
            Models not present get immediate changes (rate 0).
    """

    batch_size_by_model: dict[str, int]
    ramp_up_rate_by_model: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SetTaps(GridCommand):
    """Set regulator tap positions.

    Attributes:
        tap_position: Per-phase tap ratios. Phases set to `None` are
            unchanged.
    """

    tap_position: TapPosition


Command = DatacenterCommand | GridCommand


@dataclass(frozen=True)
class ControlAction:
    """Collection of control commands emitted by a controller.

    Use an empty `commands` list for a no-op action.
    """

    commands: list[DatacenterCommand | GridCommand] = field(default_factory=list)
