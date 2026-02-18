"""Abstract base class for controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Generic, TypeVar, Union, get_args, get_origin

from openg2g.clock import SimulationClock
from openg2g.datacenter.base import DatacenterBackend
from openg2g.events import EventEmitter
from openg2g.grid.base import GridBackend
from openg2g.types import ControlAction

DCBackendT = TypeVar("DCBackendT", bound=DatacenterBackend)
GridBackendT = TypeVar("GridBackendT", bound=GridBackend)


def _normalize_backend_type_arg(
    arg: object,
    *,
    required_base: type[object],
) -> tuple[type[object], ...]:
    if isinstance(arg, type):
        if issubclass(arg, required_base):
            return (arg,)
        raise TypeError(f"Controller generic type {arg!r} is not a subclass of {required_base.__name__}.")

    origin = get_origin(arg)

    # Handle parameterized generics like DatacenterBackend[OfflineDatacenterState]
    if isinstance(origin, type) and issubclass(origin, required_base):
        return (origin,)

    if origin is Union:
        out: list[type[object]] = []
        for item in get_args(arg):
            item_type = item if isinstance(item, type) else get_origin(item)
            if not isinstance(item_type, type) or not issubclass(item_type, required_base):
                raise TypeError(f"Controller generic type {item!r} is not a subclass of {required_base.__name__}.")
            out.append(item_type)
        return tuple(out)

    raise TypeError(
        f"Unsupported controller generic type argument: {arg!r}. Use a concrete class (or Union of concrete classes)."
    )


class Controller(Generic[DCBackendT, GridBackendT], ABC):
    """Interface for a control component in the G2G framework.

    Controllers receive datacenter and grid state and produce control actions.
    Multiple controllers compose in order within the coordinator.
    """

    _dc_types: tuple[type[DatacenterBackend], ...] = (DatacenterBackend,)
    _grid_types: tuple[type[GridBackend], ...] = (GridBackend,)

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        dc_types: tuple[type[DatacenterBackend], ...] | None = None
        grid_types: tuple[type[GridBackend], ...] | None = None
        for base in getattr(cls, "__orig_bases__", ()):
            if get_origin(base) is Controller:
                args = get_args(base)
                if len(args) != 2:
                    raise TypeError(
                        f"{cls.__name__} must specialize Controller with two generic args: "
                        "Controller[DatacenterType, GridType]."
                    )
                dc_raw, grid_raw = args
                dc_norm = _normalize_backend_type_arg(dc_raw, required_base=DatacenterBackend)
                grid_norm = _normalize_backend_type_arg(grid_raw, required_base=GridBackend)
                dc_types = tuple(t for t in dc_norm if issubclass(t, DatacenterBackend))
                grid_types = tuple(t for t in grid_norm if issubclass(t, GridBackend))
                break

        if dc_types is None or grid_types is None:
            inherited = [b for b in cls.__bases__ if issubclass(b, Controller)]
            inherited = [b for b in inherited if b is not Controller]
            if inherited:
                parent = inherited[0]
                cls._dc_types = parent.compatible_datacenter_types()
                cls._grid_types = parent.compatible_grid_types()
                return
            raise TypeError(
                f"{cls.__name__} must explicitly specialize Controller generics as "
                "Controller[DatacenterType, GridType]."
            )

        cls._dc_types = dc_types
        cls._grid_types = grid_types

    @classmethod
    def compatible_datacenter_types(cls) -> tuple[type[DatacenterBackend], ...]:
        return cls._dc_types

    @classmethod
    def compatible_grid_types(cls) -> tuple[type[GridBackend], ...]:
        return cls._grid_types

    @classmethod
    def compatibility_signature(cls) -> str:
        dc = " | ".join(t.__name__ for t in cls.compatible_datacenter_types())
        grid = " | ".join(t.__name__ for t in cls.compatible_grid_types())
        return f"Controller[{dc}, {grid}]"

    @property
    @abstractmethod
    def dt_s(self) -> Fraction:
        """Control interval as a Fraction (seconds)."""

    def start(self) -> None:
        """Acquire resources before simulation. No-op by default."""

    def stop(self) -> None:
        """Release resources after simulation. No-op by default."""

    @abstractmethod
    def step(
        self,
        clock: SimulationClock,
        datacenter: DCBackendT,
        grid: GridBackendT,
        events: EventEmitter,
    ) -> ControlAction:
        """Compute a control action. Must complete synchronously."""
