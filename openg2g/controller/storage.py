"""Storage controllers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from pydantic import BaseModel, ConfigDict

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.datacenter.base import DatacenterBackend
from openg2g.datacenter.command import DatacenterCommand
from openg2g.events import EventEmitter
from openg2g.grid.base import GridState, PhaseVoltages
from openg2g.grid.command import GridCommand, SetStoragePower
from openg2g.grid.opendss import OpenDSSGrid

VoltageWindowStatistic = Literal["minimum", "mean", "latest"]
StorageDroopMode = Literal["qv", "pv"]


@dataclass(frozen=True)
class _ControlledStorage:
    name: str
    bus: str
    output_limit: float
    droop_gain_per_pu: float


class StorageDroopConfig(BaseModel):
    """Configuration for local-voltage storage droop control."""

    model_config = ConfigDict(frozen=True)

    mode: StorageDroopMode = "qv"
    """Droop output mode: `qv` controls kvar, `pv` controls kW."""

    v_ref: float = 1.0
    """Reference local voltage in pu."""

    deadband_pu: float = 0.005
    """Symmetric deadband around `v_ref`, in pu."""

    full_output_voltage_error_pu: float = 0.05
    """Absolute voltage error from `v_ref` where storage reaches its output limit."""

    droop_gain_per_pu: float | None = None
    """Optional gain override in kW/pu for P-V mode or kvar/pu for Q-V mode."""

    max_abs_output: float | None = None
    """Optional output limit in kW for P-V mode or kvar for Q-V mode."""

    allow_negative_output: bool = True
    """Whether overvoltage may command charging in P-V mode or kvar absorption in Q-V mode."""

    voltage_statistic: VoltageWindowStatistic = "minimum"
    """How to reduce local voltage samples from the previous control window."""


class LocalVoltageStorageDroopController(Controller[DatacenterBackend, OpenDSSGrid]):
    """Proportional storage droop controller using only the storage's local voltage.

    The controller samples the grid history emitted since the previous control
    step, reads the voltage at each storage attachment bus, and emits
    [`SetStoragePower`][openg2g.grid.command.SetStoragePower] commands. In Q-V
    mode, positive output injects reactive power; in P-V mode, positive output
    discharges real power into the grid. Commands are held by each storage
    object until the next control step.
    """

    def __init__(
        self,
        *,
        grid: OpenDSSGrid,
        config: StorageDroopConfig,
        storage_name: str | None = None,
        storage_names: Sequence[str] | None = None,
        dt_s: Fraction = Fraction(1),
    ) -> None:
        self._validate_config(config)
        if dt_s <= 0:
            raise ValueError("dt_s must be positive.")

        self._grid = grid
        self._config = config
        self._dt_s = dt_s
        self._storages = self._resolve_controlled_storages(grid, config, storage_name, storage_names)
        self._history_cursor = 0
        self._bus_case_cache: dict[str, str] = {}

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    def reset(self) -> None:
        self._history_cursor = 0
        self._bus_case_cache.clear()

    def step(
        self,
        clock: SimulationClock,
        events: EventEmitter,
    ) -> list[DatacenterCommand | GridCommand]:
        history = self._grid.history()
        window = history[self._history_cursor :]
        self._history_cursor = len(history)

        if not window:
            return []

        commands: list[DatacenterCommand | GridCommand] = []
        for storage in self._storages:
            local_voltage_pu = self._window_local_voltage_pu(window, storage.bus)
            output = self._droop_output(local_voltage_pu, storage)
            if self._config.mode == "qv":
                power_kw = 0.0
                reactive_power_kvar = output
            else:
                power_kw = output
                reactive_power_kvar = 0.0

            events.emit(
                "controller.storage_droop.step",
                {
                    "time_s": clock.time_s,
                    "storage_name": storage.name,
                    "storage_bus": storage.bus,
                    "mode": self._config.mode,
                    "local_voltage_pu": local_voltage_pu,
                    "output": output,
                    "output_limit": storage.output_limit,
                    "power_kw": power_kw,
                    "reactive_power_kvar": reactive_power_kvar,
                    "v_ref": self._config.v_ref,
                    "deadband_pu": self._config.deadband_pu,
                    "window_size": len(window),
                    "voltage_statistic": self._config.voltage_statistic,
                },
            )
            commands.append(
                SetStoragePower(
                    storage_name=storage.name,
                    power_kw=power_kw,
                    reactive_power_kvar=reactive_power_kvar,
                )
            )
        return commands

    def _window_local_voltage_pu(self, window: list[GridState], storage_bus: str) -> float:
        samples = [self._state_local_voltage_pu(state, storage_bus) for state in window]
        stat = self._config.voltage_statistic
        if stat == "minimum":
            return min(samples)
        if stat == "mean":
            return sum(samples) / len(samples)
        if stat == "latest":
            return samples[-1]
        raise ValueError(f"Unsupported voltage statistic: {stat!r}")

    def _state_local_voltage_pu(self, state: GridState, storage_bus: str) -> float:
        bus = self._resolve_bus_in_state(state, storage_bus)
        phases = state.voltages[bus]
        values = _finite_phase_voltages(phases)
        if not values:
            raise ValueError(f"Storage bus {storage_bus!r} has no finite phase voltages.")
        return min(values)

    def _resolve_bus_in_state(self, state: GridState, storage_bus: str) -> str:
        if storage_bus in state.voltages:
            return storage_bus
        target = storage_bus.lower()
        cached = self._bus_case_cache.get(target)
        if cached is not None and cached in state.voltages:
            return cached
        for bus in state.voltages.buses():
            if bus.lower() == target:
                self._bus_case_cache[target] = bus
                return bus
        raise ValueError(f"Storage bus {storage_bus!r} not found in grid voltages.")

    def _droop_output(self, local_voltage_pu: float, storage: _ControlledStorage) -> float:
        error = self._config.v_ref - local_voltage_pu
        abs_error = abs(error)
        if abs_error <= self._config.deadband_pu:
            output = 0.0
        else:
            effective_error = math.copysign(abs_error - self._config.deadband_pu, error)
            output = storage.droop_gain_per_pu * effective_error

        if not self._config.allow_negative_output:
            output = max(output, 0.0)
        return max(-storage.output_limit, min(storage.output_limit, output))

    @classmethod
    def _resolve_controlled_storages(
        cls,
        grid: OpenDSSGrid,
        config: StorageDroopConfig,
        storage_name: str | None,
        storage_names: Sequence[str] | None,
    ) -> tuple[_ControlledStorage, ...]:
        if not grid.has_storage:
            raise ValueError("LocalVoltageStorageDroopController requires an attached storage resource.")
        if storage_name is not None and storage_names is not None:
            raise ValueError("Use either storage_name or storage_names, not both.")

        if storage_name is not None:
            requested = (storage_name,)
        elif storage_names is not None:
            if isinstance(storage_names, str):
                raise TypeError("storage_names must be a sequence of names; use storage_name for a single name.")
            requested = tuple(str(name) for name in storage_names)
        else:
            requested = grid.storage_names

        if not requested:
            raise ValueError("At least one storage resource must be selected.")

        resolved_names = tuple(cls._resolve_storage_name(grid, name) for name in requested)
        if len({name.lower() for name in resolved_names}) != len(resolved_names):
            raise ValueError("storage_names contains duplicates.")

        storages: list[_ControlledStorage] = []
        for name in resolved_names:
            output_limit = cls._storage_output_limit(grid, config, name)
            storages.append(
                _ControlledStorage(
                    name=name,
                    bus=grid.storage_bus(name),
                    output_limit=output_limit,
                    droop_gain_per_pu=cls._resolve_droop_gain(config, output_limit),
                )
            )
        return tuple(storages)

    @staticmethod
    def _resolve_storage_name(grid: OpenDSSGrid, storage_name: str) -> str:
        key = str(storage_name).lower()
        for name in grid.storage_names:
            if name.lower() == key:
                return name
        raise ValueError(f"Unknown storage {storage_name!r}. Known storage resources: {list(grid.storage_names)}")

    @staticmethod
    def _storage_output_limit(grid: OpenDSSGrid, config: StorageDroopConfig, storage_name: str) -> float:
        if config.mode == "qv":
            rating = float(grid.storage_rated_apparent_power_kva(storage_name))
        else:
            rating = float(grid.storage_rated_power_kw(storage_name))

        if not math.isfinite(rating) or rating <= 0.0:
            raise ValueError(f"Storage {storage_name!r} output rating must be positive.")

        if config.max_abs_output is not None:
            return min(rating, float(config.max_abs_output))
        return rating

    @staticmethod
    def _validate_config(config: StorageDroopConfig) -> None:
        if not math.isfinite(config.v_ref) or config.v_ref <= 0.0:
            raise ValueError("v_ref must be positive.")
        if not math.isfinite(config.deadband_pu) or config.deadband_pu < 0.0:
            raise ValueError("deadband_pu must be non-negative.")
        if not math.isfinite(config.full_output_voltage_error_pu) or config.full_output_voltage_error_pu <= 0.0:
            raise ValueError("full_output_voltage_error_pu must be positive.")
        if config.full_output_voltage_error_pu <= config.deadband_pu:
            raise ValueError("full_output_voltage_error_pu must be larger than deadband_pu.")
        if config.droop_gain_per_pu is not None and (
            not math.isfinite(config.droop_gain_per_pu) or config.droop_gain_per_pu <= 0.0
        ):
            raise ValueError("droop_gain_per_pu must be positive when provided.")
        if config.max_abs_output is not None and (
            not math.isfinite(config.max_abs_output) or config.max_abs_output <= 0.0
        ):
            raise ValueError("max_abs_output must be positive when provided.")

    @staticmethod
    def _resolve_droop_gain(config: StorageDroopConfig, output_limit: float) -> float:
        if config.droop_gain_per_pu is not None:
            return float(config.droop_gain_per_pu)
        effective_span_pu = config.full_output_voltage_error_pu - config.deadband_pu
        return output_limit / effective_span_pu


def _finite_phase_voltages(phases: PhaseVoltages) -> list[float]:
    values = [phases.a, phases.b, phases.c]
    return [float(v) for v in values if not math.isnan(float(v))]
