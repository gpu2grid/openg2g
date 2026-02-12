"""Tests for the Coordinator: multi-rate stepping order, dc_buffer accumulation."""

from __future__ import annotations

from unittest.mock import MagicMock

from openg2g.coordinator import Coordinator, gcd_float
from openg2g.types import (
    BusVoltages,
    ControlAction,
    DatacenterControlAction,
    DatacenterState,
    GridState,
    ThreePhase,
)


def testgcd_float():
    assert abs(gcd_float(0.1, 1.0) - 0.1) < 1e-9
    assert abs(gcd_float(0.5, 1.5) - 0.5) < 1e-9
    assert abs(gcd_float(60.0, 1.0) - 1.0) < 1e-9


def _make_mock_dc(dt_s: float = 0.1):
    dc = MagicMock()
    dc.dt_s = dt_s
    dc.step.return_value = DatacenterState(
        time_s=0.0,
        power_w=ThreePhase(a=100.0, b=100.0, c=100.0),
    )
    dc.apply_control = MagicMock()
    return dc


def _make_mock_grid(dt_s: float = 1.0):
    grid = MagicMock()
    grid.dt_s = dt_s
    grid.step.return_value = GridState(
        time_s=0.0,
        voltages=BusVoltages({"671": ThreePhase(a=1.0, b=1.0, c=1.0)}),
    )
    grid.apply_control = MagicMock()
    return grid


def _make_mock_ctrl(dt_s: float = 1.0):
    ctrl = MagicMock()
    ctrl.dt_s = dt_s
    ctrl.step.return_value = ControlAction()
    return ctrl


def test_coordinator_dc_fires_every_tick():
    """DC (dt=0.1) should fire 10x per grid step (dt=1.0)."""
    dc = _make_mock_dc(dt_s=0.1)
    grid = _make_mock_grid(dt_s=1.0)
    ctrl = _make_mock_ctrl(dt_s=1.0)

    coord = Coordinator(dc, grid, [ctrl], T_total_s=1.0, dc_bus="671")
    coord.run()

    # DC fires 10 times in 1 second (steps 0-9)
    assert dc.step.call_count == 10
    # Grid fires once at step 0
    assert grid.step.call_count == 1


def test_coordinator_dc_buffer_flush():
    """Grid receives full DC buffer at each grid step."""
    dc = _make_mock_dc(dt_s=0.1)
    grid = _make_mock_grid(dt_s=0.5)
    ctrl = _make_mock_ctrl(dt_s=0.5)

    coord = Coordinator(dc, grid, [ctrl], T_total_s=1.0, dc_bus="671")
    coord.run()

    # Grid fires at step 0 and step 5 (twice in 1 second with dt=0.5)
    assert grid.step.call_count == 2
    # First grid call (step 0): 1 DC sample accumulated
    # Second grid call (step 5): 5 DC samples (steps 1-5)
    sizes = [len(call[0][1]) for call in grid.step.call_args_list]
    assert sizes == [1, 5]


def test_coordinator_controller_order():
    """Controllers execute in order, after DC and grid."""
    call_order = []

    dc = _make_mock_dc(dt_s=1.0)
    dc.step.side_effect = lambda clock: (
        call_order.append("dc"),
        DatacenterState(time_s=0.0, power_w=ThreePhase(a=1.0, b=1.0, c=1.0)),
    )[-1]

    grid = _make_mock_grid(dt_s=1.0)
    grid.step.side_effect = lambda clock, buf, **kwargs: (
        call_order.append("grid"),
        GridState(
            time_s=0.0,
            voltages=BusVoltages({"671": ThreePhase(a=1.0, b=1.0, c=1.0)}),
        ),
    )[-1]

    ctrl1 = _make_mock_ctrl(dt_s=1.0)
    ctrl1.step.side_effect = lambda clock, dc_s, grid_s: (
        call_order.append("ctrl1"),
        ControlAction(),
    )[-1]

    ctrl2 = _make_mock_ctrl(dt_s=1.0)
    ctrl2.step.side_effect = lambda clock, dc_s, grid_s: (
        call_order.append("ctrl2"),
        ControlAction(),
    )[-1]

    coord = Coordinator(dc, grid, [ctrl1, ctrl2], T_total_s=1.0, dc_bus="671")
    coord.run()

    # First tick order should be: dc, grid, ctrl1, ctrl2
    assert call_order[0] == "dc"
    assert call_order[1] == "grid"
    assert call_order[2] == "ctrl1"
    assert call_order[3] == "ctrl2"


def test_coordinator_batch_action_applied():
    """Batch size changes from controllers are applied to datacenter."""
    dc = _make_mock_dc(dt_s=1.0)
    grid = _make_mock_grid(dt_s=1.0)
    ctrl = _make_mock_ctrl(dt_s=1.0)

    action_with_batch = DatacenterControlAction(batch_size_by_model={"model_a": 64})
    ctrl.step.return_value = action_with_batch

    coord = Coordinator(dc, grid, [ctrl], T_total_s=1.0, dc_bus="671")
    coord.run()

    dc.apply_control.assert_called_with(action_with_batch)
