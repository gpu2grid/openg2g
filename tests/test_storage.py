from __future__ import annotations

from openg2g.grid.opendss import OpenDSSGrid
from openg2g.grid.storage import BatteryStorage, StorageState


def _battery() -> BatteryStorage:
    return BatteryStorage(name="battery", rated_power_kw=100.0, capacity_kwh=200.0, apparent_power_kva=120.0)


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
