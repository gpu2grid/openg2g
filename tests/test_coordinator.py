"""Tests for the Coordinator: multi-rate stepping order, dc_buffer accumulation."""

from __future__ import annotations

from unittest.mock import MagicMock

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.coordinator import Coordinator, gcd_float
from openg2g.datacenter.base import DatacenterBackend
from openg2g.events import EventEmitter
from openg2g.grid.base import GridBackend
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.types import (
    BusVoltages,
    Command,
    ControlAction,
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
    ctrl.step.return_value = ControlAction(commands=[])
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
    ctrl1.step.side_effect = lambda clock, datacenter, grid, events: (
        call_order.append("ctrl1"),
        ControlAction(commands=[]),
    )[-1]

    ctrl2 = _make_mock_ctrl(dt_s=1.0)
    ctrl2.step.side_effect = lambda clock, datacenter, grid, events: (
        call_order.append("ctrl2"),
        ControlAction(commands=[]),
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

    action_with_batch = ControlAction(
        commands=[
            Command(
                target="datacenter",
                kind="set_batch_size",
                payload={"batch_size_by_model": {"model_a": 64}},
            )
        ]
    )
    ctrl.step.return_value = action_with_batch

    coord = Coordinator(dc, grid, [ctrl], T_total_s=1.0, dc_bus="671")
    coord.run()

    dc.apply_control.assert_called_with(action_with_batch.commands[0])


def test_coordinator_exposes_clock_stamped_controller_events():
    dc = _make_mock_dc(dt_s=1.0)
    grid = _make_mock_grid(dt_s=1.0)
    ctrl = _make_mock_ctrl(dt_s=1.0)

    def _step(
        clock: SimulationClock,
        datacenter: DatacenterBackend,
        grid: GridBackend,
        events: EventEmitter,
    ) -> ControlAction:
        del clock, datacenter, grid
        events.emit("controller.test", {"value": 1})
        return ControlAction(commands=[])

    ctrl.step.side_effect = _step
    coord = Coordinator(dc, grid, [ctrl], T_total_s=1.0, dc_bus="671")
    log = coord.run()

    assert len(log.events) == 1
    ev = log.events[0]
    assert ev.source == "controller"
    assert ev.topic == "controller.test"
    assert ev.tick == 0
    assert abs(ev.t_s - 0.0) < 1e-12
    assert abs(ev.t_min - 0.0) < 1e-12
    assert abs(ev.t_hr - 0.0) < 1e-12


def test_batch_history_is_populated_from_datacenter_events():
    class _EventedDC(DatacenterBackend):
        def __init__(self) -> None:
            self._events: EventEmitter | None = None
            self._dt_s = 1.0
            self._state: DatacenterState | None = None
            self._history: list[DatacenterState] = []

        @property
        def dt_s(self) -> float:
            return self._dt_s

        @property
        def state(self) -> DatacenterState | None:
            return self._state

        def history(self, n: int | None = None) -> list[DatacenterState]:
            if n is None:
                return list(self._history)
            if n <= 0:
                return []
            return list(self._history[-int(n) :])

        def bind_event_emitter(self, emitter: EventEmitter) -> None:
            self._events = emitter

        def step(self, clock: SimulationClock) -> DatacenterState:
            state = DatacenterState(time_s=clock.time_s, power_w=ThreePhase(a=1.0, b=1.0, c=1.0))
            self._state = state
            self._history.append(state)
            return state

        def apply_control(self, command: Command) -> None:
            assert self._events is not None
            self._events.emit(
                "datacenter.batch_size.updated",
                {"batch_size_by_model": {"model_a": 64}},
            )

    dc = _EventedDC()
    grid = _make_mock_grid(dt_s=1.0)
    ctrl = _make_mock_ctrl(dt_s=1.0)
    ctrl.step.return_value = ControlAction(
        commands=[
            Command(
                target="datacenter",
                kind="set_batch_size",
                payload={"batch_size_by_model": {"model_a": 64}},
            )
        ]
    )

    coord = Coordinator(dc, grid, [ctrl], T_total_s=1.0, dc_bus="671")
    log = coord.run()
    assert log.batch_log_by_model["model_a"] == [64]


def test_controller_generic_types_auto_extracted():
    class _TypedController(Controller[DatacenterBackend, OpenDSSGrid]):
        @property
        def dt_s(self) -> float:
            return 1.0

        def step(
            self,
            clock: SimulationClock,
            datacenter: DatacenterBackend,
            grid: OpenDSSGrid,
            events: EventEmitter,
        ) -> ControlAction:
            del clock, datacenter, grid, events
            return ControlAction(commands=[])

    assert _TypedController.compatible_datacenter_types() == (DatacenterBackend,)
    assert _TypedController.compatible_grid_types() == (OpenDSSGrid,)


def test_controller_datacenter_mismatch_error_has_underlined_generic_snippet():
    class _ExpectedDC(DatacenterBackend):
        def __init__(self) -> None:
            self._state: DatacenterState | None = None

        @property
        def dt_s(self) -> float:
            return 1.0

        @property
        def state(self) -> DatacenterState | None:
            return self._state

        def history(self, n: int | None = None) -> list[DatacenterState]:
            del n
            return []

        def step(self, clock: SimulationClock) -> DatacenterState:
            state = DatacenterState(time_s=clock.time_s, power_w=ThreePhase(a=1.0, b=1.0, c=1.0))
            self._state = state
            return state

        def apply_control(self, command: Command) -> None:
            del command

    class _OtherDC(DatacenterBackend):
        def __init__(self) -> None:
            self._state: DatacenterState | None = None

        @property
        def dt_s(self) -> float:
            return 1.0

        @property
        def state(self) -> DatacenterState | None:
            return self._state

        def history(self, n: int | None = None) -> list[DatacenterState]:
            del n
            return []

        def step(self, clock: SimulationClock) -> DatacenterState:
            state = DatacenterState(time_s=clock.time_s, power_w=ThreePhase(a=1.0, b=1.0, c=1.0))
            self._state = state
            return state

        def apply_control(self, command: Command) -> None:
            del command

    class _NeedsExpectedDC(Controller[_ExpectedDC, OpenDSSGrid]):
        @property
        def dt_s(self) -> float:
            return 1.0

        def step(
            self,
            clock: SimulationClock,
            datacenter: _ExpectedDC,
            grid: OpenDSSGrid,
            events: EventEmitter,
        ) -> ControlAction:
            del clock, datacenter, grid, events
            return ControlAction(commands=[])

    dc = _OtherDC()
    grid = _make_mock_grid(dt_s=1.0)
    ctrl = _NeedsExpectedDC()
    coord = Coordinator(dc, grid, [ctrl], T_total_s=1.0, dc_bus="671")

    try:
        coord.run()
        raise AssertionError("Expected TypeError for controller/datacenter incompatibility.")
    except TypeError as exc:
        msg = str(exc)
        assert "Controller/datacenter type mismatch" in msg
        assert "_NeedsExpectedDC" in msg
        assert "class _NeedsExpectedDC(Controller[_ExpectedDC, OpenDSSGrid]):" in msg
        assert "^^^^" in msg
