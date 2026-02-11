"""Shared state and action dataclasses for the G2G framework."""

from __future__ import annotations

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


@dataclass(frozen=True)
class OfflineDatacenterState(DatacenterState):
    """Extended state from the offline (trace-based) backend."""

    power_by_model_w: dict[str, float] = field(default_factory=dict)
    active_replicas_by_model: dict[str, int] = field(default_factory=dict)
    batch_size_by_model: dict[str, int] = field(default_factory=dict)
    avg_itl_by_model: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OnlineDatacenterState(DatacenterState):
    """Extended state from the online (live GPU) backend."""

    gpu_power_readings: dict[int, float] = field(default_factory=dict)
    batch_size_by_model: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class GridState:
    """State emitted by the grid simulator each timestep."""

    time_s: float
    voltages: BusVoltages


@dataclass(frozen=True)
class ControlAction:
    """Actions from a controller, applied to datacenter and/or grid."""

    batch_size_by_model: dict[str, int] | None = None
    tap_changes: dict[str, float] | None = None
