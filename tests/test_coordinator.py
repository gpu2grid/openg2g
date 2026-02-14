"""Tests for the Coordinator: multi-rate stepping order, dc_buffer accumulation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

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


class _StubDC(DatacenterBackend):
    """Minimal datacenter for coordinator tests."""

    def __init__(self, dt_s: float = 0.1) -> None:
        self._dt_s = dt_s
        self._state: DatacenterState | None = None
        self._history: list[DatacenterState] = []
        self.step_count = 0
        self.apply_control_calls: list[Command] = []

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

    def step(self, clock: SimulationClock) -> DatacenterState:
        self.step_count += 1
        state = DatacenterState(
            time_s=clock.time_s,
            power_w=ThreePhase(a=100.0, b=100.0, c=100.0),
        )
        self._state = state
        self._history.append(state)
        return state

    def apply_control(self, command: Command) -> None:
        self.apply_control_calls.append(command)


class _StubGrid(GridBackend):
    """Minimal grid for coordinator tests."""

    def __init__(self, dt_s: float = 1.0) -> None:
        self._dt_s = dt_s
        self._state: GridState | None = None
        self._history: list[GridState] = []
        self.step_count = 0
        self.step_calls: list[tuple[SimulationClock, list[ThreePhase]]] = []

    @property
    def dt_s(self) -> float:
        return self._dt_s

    @property
    def state(self) -> GridState | None:
        return self._state

    def history(self, n: int | None = None) -> list[GridState]:
        if n is None:
            return list(self._history)
        if n <= 0:
            return []
        return list(self._history[-int(n) :])

    def step(
        self,
        clock: SimulationClock,
        load_trace_w: list[ThreePhase],
        **kwargs: object,
    ) -> GridState:
        self.step_count += 1
        self.step_calls.append((clock, list(load_trace_w)))
        state = GridState(
            time_s=clock.time_s,
            voltages=BusVoltages({"671": ThreePhase(a=1.0, b=1.0, c=1.0)}),
        )
        self._state = state
        self._history.append(state)
        return state

    @property
    def v_index(self) -> list[tuple[str, int]]:
        return [("671", 0), ("671", 1), ("671", 2)]

    def voltages_vector(self) -> np.ndarray:
        return np.array([1.0, 1.0, 1.0])

    def estimate_H(self, dp_kw: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((3, 3)), np.ones(3)

    def apply_control(self, command: Command) -> None:
        pass


class _StubController(Controller[DatacenterBackend, GridBackend]):
    """Test controller that delegates to a callback or returns a fixed action."""

    def __init__(
        self,
        dt_s: float = 1.0,
        action: ControlAction | None = None,
        on_step: Callable[
            [SimulationClock, DatacenterBackend, GridBackend, EventEmitter],
            ControlAction,
        ]
        | None = None,
    ):
        self._dt_s = dt_s
        self._action = action or ControlAction(commands=[])
        self._on_step = on_step
        self.call_count = 0

    @property
    def dt_s(self) -> float:
        return self._dt_s

    def step(
        self,
        clock: SimulationClock,
        datacenter: DatacenterBackend,
        grid: GridBackend,
        events: EventEmitter,
    ) -> ControlAction:
        self.call_count += 1
        if self._on_step is not None:
            return self._on_step(clock, datacenter, grid, events)
        return self._action


def test_coordinator_dc_fires_every_tick():
    """DC (dt=0.1) should fire 10x per grid step (dt=1.0)."""
    dc = _StubDC(dt_s=0.1)
    grid = _StubGrid(dt_s=1.0)
    ctrl = _StubController(dt_s=1.0)

    coord = Coordinator(dc, grid, [ctrl], T_total_s=1.0, dc_bus="671")
    coord.run()

    # DC fires 10 times in 1 second (steps 0-9)
    assert dc.step_count == 10
    # Grid fires once at step 0
    assert grid.step_count == 1


def test_coordinator_dc_buffer_flush():
    """Grid receives full DC buffer at each grid step."""
    dc = _StubDC(dt_s=0.1)
    grid = _StubGrid(dt_s=0.5)
    ctrl = _StubController(dt_s=0.5)

    coord = Coordinator(dc, grid, [ctrl], T_total_s=1.0, dc_bus="671")
    coord.run()

    # Grid fires at step 0 and step 5 (twice in 1 second with dt=0.5)
    assert grid.step_count == 2
    # First grid call (step 0): 1 DC sample accumulated
    # Second grid call (step 5): 5 DC samples (steps 1-5)
    sizes = [len(call[1]) for call in grid.step_calls]
    assert sizes == [1, 5]


def test_coordinator_controller_order():
    """Controllers execute in order, after DC and grid."""
    call_order: list[str] = []

    class _OrderDC(_StubDC):
        def step(self, clock: SimulationClock) -> DatacenterState:
            call_order.append("dc")
            return super().step(clock)

    class _OrderGrid(_StubGrid):
        def step(
            self,
            clock: SimulationClock,
            load_trace_w: list[ThreePhase],
            **kwargs: object,
        ) -> GridState:
            call_order.append("grid")
            return super().step(clock, load_trace_w, **kwargs)

    dc = _OrderDC(dt_s=1.0)
    grid = _OrderGrid(dt_s=1.0)

    ctrl1 = _StubController(
        dt_s=1.0,
        on_step=lambda clock, dc, g, ev: (
            call_order.append("ctrl1"),
            ControlAction(commands=[]),
        )[-1],
    )

    ctrl2 = _StubController(
        dt_s=1.0,
        on_step=lambda clock, dc, g, ev: (
            call_order.append("ctrl2"),
            ControlAction(commands=[]),
        )[-1],
    )

    coord = Coordinator(dc, grid, [ctrl1, ctrl2], T_total_s=1.0, dc_bus="671")
    coord.run()

    # First tick order should be: dc, grid, ctrl1, ctrl2
    assert call_order[0] == "dc"
    assert call_order[1] == "grid"
    assert call_order[2] == "ctrl1"
    assert call_order[3] == "ctrl2"


def test_coordinator_batch_action_applied():
    """Batch size changes from controllers are applied to datacenter."""
    dc = _StubDC(dt_s=1.0)
    grid = _StubGrid(dt_s=1.0)

    action_with_batch = ControlAction(
        commands=[
            Command(
                target="datacenter",
                kind="set_batch_size",
                payload={"batch_size_by_model": {"model_a": 64}},
            )
        ]
    )
    ctrl = _StubController(dt_s=1.0, action=action_with_batch)

    coord = Coordinator(dc, grid, [ctrl], T_total_s=1.0, dc_bus="671")
    coord.run()

    assert dc.apply_control_calls == [action_with_batch.commands[0]]


def test_coordinator_exposes_clock_stamped_controller_events():
    dc = _StubDC(dt_s=1.0)
    grid = _StubGrid(dt_s=1.0)

    def _on_step(
        clock: SimulationClock,
        datacenter: DatacenterBackend,
        grid: GridBackend,
        events: EventEmitter,
    ) -> ControlAction:

        events.emit("controller.test", {"value": 1})
        return ControlAction(commands=[])

    ctrl = _StubController(dt_s=1.0, on_step=_on_step)
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
    grid = _StubGrid(dt_s=1.0)
    ctrl = _StubController(
        dt_s=1.0,
        action=ControlAction(
            commands=[
                Command(
                    target="datacenter",
                    kind="set_batch_size",
                    payload={"batch_size_by_model": {"model_a": 64}},
                )
            ]
        ),
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

            return []

        def step(self, clock: SimulationClock) -> DatacenterState:
            state = DatacenterState(time_s=clock.time_s, power_w=ThreePhase(a=1.0, b=1.0, c=1.0))
            self._state = state
            return state

        def apply_control(self, command: Command) -> None:
            pass

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

            return []

        def step(self, clock: SimulationClock) -> DatacenterState:
            state = DatacenterState(time_s=clock.time_s, power_w=ThreePhase(a=1.0, b=1.0, c=1.0))
            self._state = state
            return state

        def apply_control(self, command: Command) -> None:
            pass

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

            return ControlAction(commands=[])

    dc = _OtherDC()
    grid = _StubGrid(dt_s=1.0)
    ctrl = _NeedsExpectedDC()
    coord = Coordinator(dc, grid, [ctrl], T_total_s=1.0, dc_bus="671")

    try:
        coord.run()
        raise AssertionError("Expected TypeError for controller/datacenter incompatibility.")
    except TypeError as exc:
        msg = str(exc)
        assert "_NeedsExpectedDC" in msg
        assert "_ExpectedDC" in msg
        assert "_OtherDC" in msg
