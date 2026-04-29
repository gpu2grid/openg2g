"""Storage controllers."""

from __future__ import annotations

import math
from collections.abc import Mapping
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
from openg2g.grid.storage import EnergyStorage

VoltageWindowStatistic = Literal["minimum", "mean", "latest"]
StorageDroopMode = Literal["qv", "pv"]


@dataclass(frozen=True)
class _ControlledStorage:
    storage: EnergyStorage
    bus: str
    output_limit: float
    droop_gain_per_pu: float


class StorageDroopConfig(BaseModel):
    """Configuration for local-voltage storage droop control.

    Attributes:
        mode: Droop output mode: `qv` controls kvar, `pv` controls kW.
        v_ref: Reference local voltage in pu.
        deadband_pu: Symmetric deadband around `v_ref`, in pu.
        full_output_voltage_error_pu: Absolute voltage error from `v_ref` where
            storage reaches its output limit.
        droop_gain_per_pu: Optional gain override in kW/pu for P-V mode or
            kvar/pu for Q-V mode.
        max_abs_output: Optional output limit in kW for P-V mode or kvar for
            Q-V mode.
        allow_negative_output: Whether overvoltage may command charging in P-V
            mode or kvar absorption in Q-V mode.
        voltage_statistic: How to reduce local voltage samples from the previous
            control window.
    """

    model_config = ConfigDict(frozen=True)

    mode: StorageDroopMode = "qv"
    v_ref: float = 1.0
    deadband_pu: float = 0.005
    full_output_voltage_error_pu: float = 0.05
    droop_gain_per_pu: float | None = None
    max_abs_output: float | None = None
    allow_negative_output: bool = True
    voltage_statistic: VoltageWindowStatistic = "minimum"


class LocalVoltageStorageDroopController(Controller[DatacenterBackend, OpenDSSGrid]):
    """Proportional storage droop controller using only the storage's local voltage.

    The controller samples the grid history emitted since the previous control
    step, reads the voltage at each storage attachment bus, and emits
    [`SetStoragePower`][openg2g.grid.command.SetStoragePower] commands. In Q-V
    mode, positive output injects reactive power; in P-V mode, positive output
    discharges real power into the grid. Commands are held by each storage
    object until the next control step.

    Args:
        grid: Grid backend.
        storages: Mapping from each [`EnergyStorage`][openg2g.grid.storage.EnergyStorage]
            to control onto its attachment bus. Each storage must be attached
            to *grid* at the bus given here.
        config: Droop configuration.
        dt_s: Control interval in seconds.
    """

    def __init__(
        self,
        *,
        grid: OpenDSSGrid,
        storages: Mapping[EnergyStorage, str],
        config: StorageDroopConfig,
        dt_s: Fraction = Fraction(1),
    ) -> None:
        self._validate_config(config)
        if dt_s <= 0:
            raise ValueError("dt_s must be positive.")
        if not storages:
            raise ValueError("storages must contain at least one storage resource.")

        self._grid = grid
        self._config = config
        self._dt_s = dt_s
        self._storages = tuple(
            _ControlledStorage(
                storage=storage,
                bus=bus,
                output_limit=self._storage_output_limit(config, storage),
                droop_gain_per_pu=self._resolve_droop_gain(config, self._storage_output_limit(config, storage)),
            )
            for storage, bus in storages.items()
        )
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
        for controlled in self._storages:
            local_voltage_pu = self._window_local_voltage_pu(window, controlled.bus)
            output = self._droop_output(local_voltage_pu, controlled)
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
                    "storage_name": controlled.storage.name,
                    "storage_bus": controlled.bus,
                    "mode": self._config.mode,
                    "local_voltage_pu": local_voltage_pu,
                    "output": output,
                    "output_limit": controlled.output_limit,
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
                    storage=controlled.storage,
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

    def _droop_output(self, local_voltage_pu: float, controlled: _ControlledStorage) -> float:
        error = self._config.v_ref - local_voltage_pu
        abs_error = abs(error)
        if abs_error <= self._config.deadband_pu:
            output = 0.0
        else:
            effective_error = math.copysign(abs_error - self._config.deadband_pu, error)
            output = controlled.droop_gain_per_pu * effective_error

        if not self._config.allow_negative_output:
            output = max(output, 0.0)
        return max(-controlled.output_limit, min(controlled.output_limit, output))

    @staticmethod
    def _storage_output_limit(config: StorageDroopConfig, storage: EnergyStorage) -> float:
        if config.mode == "qv":
            rating = float(storage.rated_apparent_power_kva)
        else:
            rating = float(storage.rated_power_kw)

        if not math.isfinite(rating) or rating <= 0.0:
            raise ValueError(f"Storage {storage.name!r} output rating must be positive.")

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
