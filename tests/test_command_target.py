from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openg2g.datacenter.command import DatacenterCommand, SetBatchSize
from openg2g.grid.command import GridCommand, SetStoragePower, SetTaps
from openg2g.grid.config import TapPosition

_MOCK_DC = MagicMock()


def test_set_batch_size_is_datacenter_command() -> None:
    cmd = SetBatchSize(batch_size_by_model={"model_a": 64}, target=_MOCK_DC)
    assert isinstance(cmd, DatacenterCommand)
    assert cmd.batch_size_by_model == {"model_a": 64}
    assert cmd.ramp_up_rate_by_model == {}


def test_set_batch_size_with_ramp() -> None:
    cmd = SetBatchSize(batch_size_by_model={"model_a": 32}, ramp_up_rate_by_model={"model_a": 4.0}, target=_MOCK_DC)
    assert cmd.ramp_up_rate_by_model == {"model_a": 4.0}


def test_set_taps_is_grid_command() -> None:
    cmd = SetTaps(tap_position=TapPosition(a=1.05, b=1.0))
    assert isinstance(cmd, GridCommand)
    assert cmd.tap_position.a == 1.05
    assert cmd.tap_position.b == 1.0


def test_set_storage_power_is_grid_command() -> None:
    from openg2g.grid.storage import BatteryStorage

    storage = BatteryStorage(name="battery", rated_power_kw=10.0, capacity_kwh=20.0)
    cmd = SetStoragePower(storage=storage, power_kw=10.0, reactive_power_kvar=2.0)
    assert isinstance(cmd, GridCommand)
    assert cmd.storage is storage
    assert cmd.power_kw == 10.0
    assert cmd.reactive_power_kvar == 2.0


def test_command_types_are_disjoint() -> None:
    dc_cmd = SetBatchSize(batch_size_by_model={"a": 1}, target=_MOCK_DC)
    grid_cmd = SetTaps(tap_position=TapPosition(a=1.0))
    assert not isinstance(dc_cmd, GridCommand)
    assert not isinstance(grid_cmd, DatacenterCommand)


def test_base_command_classes_not_instantiable() -> None:
    with pytest.raises(TypeError, match="DatacenterCommand cannot be instantiated directly"):
        DatacenterCommand()
    with pytest.raises(TypeError, match="GridCommand cannot be instantiated directly"):
        GridCommand()
