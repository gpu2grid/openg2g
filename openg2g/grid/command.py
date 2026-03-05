"""Command types targeting grid backends."""

from __future__ import annotations

from dataclasses import dataclass

from openg2g.grid.config import TapPosition


class GridCommand:
    """Base for commands targeting the grid backend.

    Subclass this for each concrete grid command kind.
    The coordinator routes commands to backends based on this type hierarchy.
    """

    def __init__(self) -> None:
        if type(self) is GridCommand:
            raise TypeError("GridCommand cannot be instantiated directly; subclass it.")


@dataclass(frozen=True)
class SetTaps(GridCommand):
    """Set regulator tap positions.

    Attributes:
        tap_position: Per-phase tap ratios. Phases set to `None` are
            unchanged.
    """

    tap_position: TapPosition
