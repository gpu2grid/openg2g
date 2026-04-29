"""Command types targeting grid backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from openg2g.grid.config import TapPosition

if TYPE_CHECKING:
    from openg2g.grid.storage import EnergyStorage


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


@dataclass(frozen=True)
class SetStoragePower(GridCommand):
    """Set energy storage real/reactive power.

    Positive real power discharges storage into the grid; negative real power
    charges storage from the grid.

    Attributes:
        storage: Attached storage resource to command.
        power_kw: Real-power setpoint in kW.
        reactive_power_kvar: Reactive-power setpoint in kvar. Positive values
            inject reactive power.
    """

    storage: EnergyStorage
    power_kw: float
    reactive_power_kvar: float = 0.0
