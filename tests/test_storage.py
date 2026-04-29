from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import pytest

from openg2g.clock import SimulationClock
from openg2g.controller.storage import LocalVoltageStorageDroopController, StorageDroopConfig
from openg2g.grid.base import BusVoltages, GridState, PhaseVoltages
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.grid.storage import BatteryStorage, StorageState


def _battery() -> BatteryStorage:
    return BatteryStorage(name="battery", rated_power_kw=100.0, capacity_kwh=200.0, apparent_power_kva=120.0)


def _phase(v: float) -> PhaseVoltages:
    return PhaseVoltages(a=v, b=v, c=v)


def _grid_state(time_s: float, bus_voltages: dict[str, float]) -> GridState:
    return GridState(time_s=time_s, voltages=BusVoltages(_data={b: _phase(v) for b, v in bus_voltages.items()}))


@dataclass
class _StubEmitter:
    """No-op event emitter for tests; just records emitted topics."""

    topics: list[str] = field(default_factory=list)

    def emit(self, topic: str, data: dict | None = None) -> None:  # pragma: no cover - trivial
        self.topics.append(topic)


def test_battery_storage_reset_clears_setpoint_and_state() -> None:
    storage = _battery()
    storage.set_power_kw(50.0, reactive_power_kvar=20.0)
    storage.update_state(
        StorageState(
            time_s=1.0,
            stored_kwh=90.0,
            soc=0.45,
            power_kw=50.0,
            reactive_power_kvar=20.0,
            dss_state="DISCHARGING",
        )
    )

    storage.reset()

    assert storage.power_kw(0.0) == 0.0
    assert storage.reactive_power_kvar(0.0) == 0.0
    assert storage.state is None


def test_opendss_grid_reset_clears_attached_storage_setpoints(monkeypatch) -> None:
    from openg2g.grid import opendss as opendss_module

    monkeypatch.setattr(opendss_module, "dss", object())
    grid = OpenDSSGrid(dss_case_dir=".", dss_master_file="Master.dss")
    storage = _battery()
    grid.attach_storage(storage, bus="671")
    storage.set_power_kw(50.0, reactive_power_kvar=20.0)

    grid.reset()

    assert storage.power_kw(0.0) == 0.0
    assert storage.reactive_power_kvar(0.0) == 0.0


def _build_grid_with_storage(monkeypatch) -> tuple[OpenDSSGrid, BatteryStorage]:
    from openg2g.grid import opendss as opendss_module

    monkeypatch.setattr(opendss_module, "dss", object())
    grid = OpenDSSGrid(dss_case_dir=".", dss_master_file="Master.dss")
    storage = _battery()
    grid.attach_storage(storage, bus="bus1")
    return grid, storage


def test_droop_controller_qv_undervoltage_injects_kvar(monkeypatch) -> None:
    grid, storage = _build_grid_with_storage(monkeypatch)
    controller = LocalVoltageStorageDroopController(
        grid=grid,
        storages={storage: "bus1"},
        config=StorageDroopConfig(mode="qv", v_ref=1.0, deadband_pu=0.005, full_output_voltage_error_pu=0.05),
        dt_s=Fraction(1),
    )
    grid._history.append(_grid_state(time_s=1.0, bus_voltages={"bus1": 0.94}))

    cmds = controller.step(SimulationClock(tick_s=Fraction(1)), _StubEmitter())  # type: ignore[arg-type]

    assert len(cmds) == 1
    cmd = cmds[0]
    assert cmd.storage is storage
    assert cmd.power_kw == 0.0
    # Undervoltage by 0.06 pu, deadband 0.005, full at 0.05. Effective error
    # 0.055 pu but clipped to apparent rating 120 kvar.
    assert cmd.reactive_power_kvar == pytest.approx(120.0)


def test_droop_controller_pv_overvoltage_charges_storage(monkeypatch) -> None:
    grid, storage = _build_grid_with_storage(monkeypatch)
    controller = LocalVoltageStorageDroopController(
        grid=grid,
        storages={storage: "bus1"},
        config=StorageDroopConfig(mode="pv", v_ref=1.0, deadband_pu=0.005, full_output_voltage_error_pu=0.05),
        dt_s=Fraction(1),
    )
    grid._history.append(_grid_state(time_s=1.0, bus_voltages={"bus1": 1.02}))

    cmds = controller.step(SimulationClock(tick_s=Fraction(1)), _StubEmitter())  # type: ignore[arg-type]

    cmd = cmds[0]
    # Overvoltage by 0.02, deadband 0.005, effective error -0.015 pu.
    # Gain = rated_power_kw / (full - deadband) = 100 / 0.045 ~ 2222 kW/pu.
    expected_kw = 100.0 / (0.05 - 0.005) * (-0.015)
    assert cmd.reactive_power_kvar == 0.0
    assert cmd.power_kw == pytest.approx(expected_kw)


def test_droop_controller_deadband_emits_zero_output(monkeypatch) -> None:
    grid, storage = _build_grid_with_storage(monkeypatch)
    controller = LocalVoltageStorageDroopController(
        grid=grid,
        storages={storage: "bus1"},
        config=StorageDroopConfig(mode="qv", v_ref=1.0, deadband_pu=0.005, full_output_voltage_error_pu=0.05),
        dt_s=Fraction(1),
    )
    grid._history.append(_grid_state(time_s=1.0, bus_voltages={"bus1": 1.003}))

    cmds = controller.step(SimulationClock(tick_s=Fraction(1)), _StubEmitter())  # type: ignore[arg-type]

    assert cmds[0].reactive_power_kvar == 0.0
    assert cmds[0].power_kw == 0.0


def test_droop_controller_disallow_negative_output(monkeypatch) -> None:
    grid, storage = _build_grid_with_storage(monkeypatch)
    controller = LocalVoltageStorageDroopController(
        grid=grid,
        storages={storage: "bus1"},
        config=StorageDroopConfig(
            mode="qv",
            v_ref=1.0,
            deadband_pu=0.005,
            full_output_voltage_error_pu=0.05,
            allow_negative_output=False,
        ),
        dt_s=Fraction(1),
    )
    grid._history.append(_grid_state(time_s=1.0, bus_voltages={"bus1": 1.05}))

    cmds = controller.step(SimulationClock(tick_s=Fraction(1)), _StubEmitter())  # type: ignore[arg-type]

    # Overvoltage would normally absorb kvar; with allow_negative_output=False, clamp to 0.
    assert cmds[0].reactive_power_kvar == 0.0


def test_droop_controller_uses_minimum_phase_voltage(monkeypatch) -> None:
    """A bus with one severely sagged phase should drive output via that phase."""
    grid, storage = _build_grid_with_storage(monkeypatch)
    controller = LocalVoltageStorageDroopController(
        grid=grid,
        storages={storage: "bus1"},
        config=StorageDroopConfig(mode="qv", v_ref=1.0, deadband_pu=0.005, full_output_voltage_error_pu=0.05),
        dt_s=Fraction(1),
    )
    asym = GridState(
        time_s=1.0,
        voltages=BusVoltages(_data={"bus1": PhaseVoltages(a=1.0, b=1.0, c=0.92)}),
    )
    grid._history.append(asym)

    cmds = controller.step(SimulationClock(tick_s=Fraction(1)), _StubEmitter())  # type: ignore[arg-type]

    # Worst phase 0.92 -> error 0.08 pu, capped at apparent rating 120 kvar.
    assert cmds[0].reactive_power_kvar == pytest.approx(120.0)


def test_droop_controller_history_cursor_advances(monkeypatch) -> None:
    """Two successive step() calls should each consume only the new history."""
    grid, storage = _build_grid_with_storage(monkeypatch)
    controller = LocalVoltageStorageDroopController(
        grid=grid,
        storages={storage: "bus1"},
        config=StorageDroopConfig(mode="qv"),
        dt_s=Fraction(1),
    )
    clock = SimulationClock(tick_s=Fraction(1))
    grid._history.append(_grid_state(time_s=1.0, bus_voltages={"bus1": 0.94}))
    controller.step(clock, _StubEmitter())  # type: ignore[arg-type]
    cursor_after_first = controller._history_cursor

    cmds_no_new = controller.step(clock, _StubEmitter())  # type: ignore[arg-type]
    assert cmds_no_new == []
    assert controller._history_cursor == cursor_after_first

    grid._history.append(_grid_state(time_s=2.0, bus_voltages={"bus1": 0.96}))
    cmds_new = controller.step(clock, _StubEmitter())  # type: ignore[arg-type]
    assert len(cmds_new) == 1


def test_droop_controller_rejects_empty_storages(monkeypatch) -> None:
    grid, _storage = _build_grid_with_storage(monkeypatch)
    with pytest.raises(ValueError, match="at least one storage"):
        LocalVoltageStorageDroopController(
            grid=grid,
            storages={},
            config=StorageDroopConfig(),
            dt_s=Fraction(1),
        )
