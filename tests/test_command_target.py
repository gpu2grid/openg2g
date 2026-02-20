from __future__ import annotations

from openg2g.types import DatacenterCommand, GridCommand, SetBatchSize, SetTaps


def test_set_batch_size_is_datacenter_command() -> None:
    cmd = SetBatchSize(batch_size_by_model={"model_a": 64})
    assert isinstance(cmd, DatacenterCommand)
    assert cmd.batch_size_by_model == {"model_a": 64}
    assert cmd.ramp_up_rate == 0.0


def test_set_batch_size_with_ramp() -> None:
    cmd = SetBatchSize(batch_size_by_model={"model_a": 32}, ramp_up_rate=4.0)
    assert cmd.ramp_up_rate == 4.0


def test_set_taps_is_grid_command() -> None:
    cmd = SetTaps(tap_changes={"reg1": 1.05, "reg2": 1.0})
    assert isinstance(cmd, GridCommand)
    assert cmd.tap_changes == {"reg1": 1.05, "reg2": 1.0}


def test_command_types_are_disjoint() -> None:
    dc_cmd = SetBatchSize(batch_size_by_model={"a": 1})
    grid_cmd = SetTaps(tap_changes={"reg1": 1.0})
    assert not isinstance(dc_cmd, GridCommand)
    assert not isinstance(grid_cmd, DatacenterCommand)
