"""Cross-cutting types shared across component families."""

from __future__ import annotations

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
        ramp_up_rate: Requests/second ramp-up rate. 0 means immediate.
    """

    batch_size_by_model: dict[str, int]
    ramp_up_rate: float = 0.0


@dataclass(frozen=True)
class SetTaps(GridCommand):
    """Set regulator tap positions.

    Attributes:
        tap_changes: Mapping of regulator control name to tap ratio (pu).
    """

    tap_changes: dict[str, float]


Command = DatacenterCommand | GridCommand


@dataclass(frozen=True)
class ControlAction:
    """Collection of control commands emitted by a controller.

    Use an empty `commands` list for a no-op action.
    """

    commands: list[DatacenterCommand | GridCommand] = field(default_factory=list)
