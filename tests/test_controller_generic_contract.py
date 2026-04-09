from __future__ import annotations

import types
from fractions import Fraction
from typing import Any, cast

import pytest

from openg2g.clock import SimulationClock
from openg2g.common import ThreePhase
from openg2g.controller.base import Controller
from openg2g.datacenter.base import DatacenterBackend, DatacenterState, LLMBatchSizeControlledDatacenter
from openg2g.datacenter.command import DatacenterCommand
from openg2g.datacenter.offline import OfflineDatacenterState
from openg2g.events import EventEmitter
from openg2g.grid.base import BusVoltages, GridBackend, GridState, PhaseVoltages
from openg2g.grid.command import GridCommand
from openg2g.grid.opendss import OpenDSSGrid


class _DC(DatacenterBackend[DatacenterState]):
    def __init__(self) -> None:
        super().__init__(name="test")

    @property
    def dt_s(self) -> Fraction:
        return Fraction(1)

    def reset(self) -> None:
        pass

    def step(self, clock: SimulationClock, events: EventEmitter) -> DatacenterState:
        return DatacenterState(time_s=clock.time_s, power_w=ThreePhase(a=1.0, b=1.0, c=1.0))

    def apply_control(self, command: DatacenterCommand, events: EventEmitter) -> None:
        pass


class _Grid(GridBackend[GridState]):
    def __init__(self) -> None:
        super().__init__()

    @property
    def dt_s(self) -> Fraction:
        return Fraction(1)

    def reset(self) -> None:
        pass

    @property
    def v_index(self) -> list[tuple[str, int]]:
        return [("671", 1)]

    def step(
        self,
        clock: SimulationClock,
        power_samples_w: list[ThreePhase],
        events: EventEmitter,
    ) -> GridState:
        return GridState(
            time_s=clock.time_s,
            voltages=BusVoltages({"671": PhaseVoltages(1.0, 1.0, 1.0)}),
        )

    def apply_control(self, command: GridCommand, events: EventEmitter) -> None:
        pass

    def voltages_vector(self):
        import numpy as np

        return np.array([1.0], dtype=float)

    def dc_bus(self, dc):
        return "671"

    def estimate_sensitivity(self, perturbation_kw: float = 100.0, dc=None):
        import numpy as np

        return np.zeros((1, 3), dtype=float), np.array([1.0], dtype=float)


def test_controller_requires_explicit_generic_parameters():
    controller_any = cast(Any, Controller)
    with pytest.raises(TypeError, match="must explicitly specialize Controller generics"):
        type(
            "_BadMissingGeneric",
            (controller_any,),
            {
                "dt_s": property(lambda self: Fraction(1)),
                "step": lambda self, clock, grid, events: [],
            },
        )


def test_controller_rejects_reversed_generic_order():
    reversed_base = Controller.__class_getitem__((OpenDSSGrid, DatacenterBackend))
    with pytest.raises(TypeError, match="is not a subclass of DatacenterBackend"):
        types.new_class(
            "_BadReversed",
            (reversed_base,),
            exec_body=lambda ns: ns.update(
                {
                    "dt_s": property(lambda self: Fraction(1)),
                    "step": lambda self, clock, grid, events: [],
                }
            ),
        )


def test_controller_rejects_random_classes_in_generics():
    class _Random:
        pass

    random_base = Controller.__class_getitem__((_Random, OpenDSSGrid))
    with pytest.raises(TypeError, match="is not a subclass of DatacenterBackend"):
        types.new_class(
            "_BadRandom",
            (random_base,),
            exec_body=lambda ns: ns.update(
                {
                    "dt_s": property(lambda self: Fraction(1)),
                    "step": lambda self, clock, grid, events: [],
                }
            ),
        )


def test_controller_rejects_non_abc_subclass_for_grid_generic():
    class _NotGrid:
        pass

    bad_grid_base = Controller.__class_getitem__((_DC, _NotGrid))
    with pytest.raises(TypeError, match="is not a subclass of GridBackend"):
        types.new_class(
            "_BadGrid",
            (bad_grid_base,),
            exec_body=lambda ns: ns.update(
                {
                    "dt_s": property(lambda self: Fraction(1)),
                    "step": lambda self, clock, grid, events: [],
                }
            ),
        )


def test_controller_inherits_compatibility_from_typed_parent():
    class _BaseTyped(Controller[_DC, _Grid]):
        @property
        def dt_s(self) -> Fraction:
            return Fraction(1)

        @property
        def datacenters(self) -> list:
            return []

        def reset(self) -> None:
            pass

        def step(
            self,
            clock: SimulationClock,
            grid: _Grid,
            events: EventEmitter,
        ) -> list[DatacenterCommand | GridCommand]:
            return []

    class _Child(_BaseTyped):
        pass

    assert _Child.compatible_datacenter_types() == (_DC,)
    assert _Child.compatible_grid_types() == (_Grid,)


def test_controller_accepts_parameterized_backend_generics():
    """Controller[DatacenterBackend[X], GridBackend[Y]] should work via get_origin fallback."""

    class _Parameterized(Controller[LLMBatchSizeControlledDatacenter[OfflineDatacenterState], GridBackend[GridState]]):
        @property
        def dt_s(self) -> Fraction:
            return Fraction(1)

        @property
        def datacenters(self) -> list:
            return []

        def reset(self) -> None:
            pass

        def step(
            self,
            clock: SimulationClock,
            grid: GridBackend[GridState],
            events: EventEmitter,
        ) -> list[DatacenterCommand | GridCommand]:
            return []

    assert _Parameterized.compatible_datacenter_types() == (LLMBatchSizeControlledDatacenter,)
    assert _Parameterized.compatible_grid_types() == (GridBackend,)
