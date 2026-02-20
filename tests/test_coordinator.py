"""Tests for the Coordinator: multi-rate stepping order, dc_buffer accumulation."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from fractions import Fraction

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.coordinator import Coordinator, _gcd_fraction
from openg2g.datacenter.base import DatacenterBackend
from openg2g.events import EventEmitter
from openg2g.grid.base import GridBackend
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.types import (
    BusVoltages,
    ControlAction,
    DatacenterCommand,
    DatacenterState,
    GridCommand,
    GridState,
    SetBatchSize,
    ThreePhase,
)


def test_gcd_fraction():
    assert _gcd_fraction(Fraction(1, 10), Fraction(1)) == Fraction(1, 10)
    assert _gcd_fraction(Fraction(1, 2), Fraction(3, 2)) == Fraction(1, 2)
    assert _gcd_fraction(Fraction(60), Fraction(1)) == Fraction(1)


class _StubDC(DatacenterBackend[DatacenterState]):
    """Minimal datacenter for coordinator tests."""

    def __init__(self, dt_s: Fraction = Fraction(1, 10)) -> None:
        self._dt_s = dt_s
        self._state: DatacenterState | None = None
        self._history: list[DatacenterState] = []
        self.step_count = 0
        self.apply_control_calls: list[DatacenterCommand] = []

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    @property
    def state(self) -> DatacenterState:
        if self._state is None:
            raise RuntimeError("No state yet")
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

    def apply_control(self, command: DatacenterCommand) -> None:
        self.apply_control_calls.append(command)


class _StubGrid(GridBackend[GridState]):
    """Minimal grid for coordinator tests."""

    def __init__(self, dt_s: Fraction = Fraction(1)) -> None:
        self._dt_s = dt_s
        self._state: GridState | None = None
        self._history: list[GridState] = []
        self.step_count = 0
        self.step_calls: list[tuple[SimulationClock, list[ThreePhase]]] = []

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    @property
    def state(self) -> GridState:
        if self._state is None:
            raise RuntimeError("No state yet")
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
        power_samples_w: list[ThreePhase],
        **kwargs: object,
    ) -> GridState:
        self.step_count += 1
        self.step_calls.append((clock, list(power_samples_w)))
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

    def estimate_sensitivity(self, perturbation_kw: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((3, 3)), np.ones(3)

    def apply_control(self, command: GridCommand) -> None:
        pass


class _StubController(Controller[DatacenterBackend, GridBackend]):
    """Test controller that delegates to a callback or returns a fixed action."""

    def __init__(
        self,
        dt_s: Fraction = Fraction(1),
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
    def dt_s(self) -> Fraction:
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
    """DC (dt=1/10) should fire 10x per grid step (dt=1)."""
    dc = _StubDC(dt_s=Fraction(1, 10))
    grid = _StubGrid(dt_s=Fraction(1))
    ctrl = _StubController(dt_s=Fraction(1))

    coord = Coordinator(dc, grid, [ctrl], total_duration_s=1, dc_bus="671")
    coord.run()

    # DC fires 10 times in 1 second (steps 0-9)
    assert dc.step_count == 10
    # Grid fires once at step 0
    assert grid.step_count == 1


def test_coordinator_dc_buffer_flush():
    """Grid receives full DC buffer at each grid step."""
    dc = _StubDC(dt_s=Fraction(1, 10))
    grid = _StubGrid(dt_s=Fraction(1, 2))
    ctrl = _StubController(dt_s=Fraction(1, 2))

    coord = Coordinator(dc, grid, [ctrl], total_duration_s=1, dc_bus="671")
    coord.run()

    # Grid fires at step 0 and step 5 (twice in 1 second with dt=1/2)
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
            power_samples_w: list[ThreePhase],
            **kwargs: object,
        ) -> GridState:
            call_order.append("grid")
            return super().step(clock, power_samples_w, **kwargs)

    dc = _OrderDC(dt_s=Fraction(1))
    grid = _OrderGrid(dt_s=Fraction(1))

    ctrl1 = _StubController(
        dt_s=Fraction(1),
        on_step=lambda clock, dc, g, ev: (
            call_order.append("ctrl1"),
            ControlAction(commands=[]),
        )[-1],
    )

    ctrl2 = _StubController(
        dt_s=Fraction(1),
        on_step=lambda clock, dc, g, ev: (
            call_order.append("ctrl2"),
            ControlAction(commands=[]),
        )[-1],
    )

    coord = Coordinator(dc, grid, [ctrl1, ctrl2], total_duration_s=1, dc_bus="671")
    coord.run()

    # First tick order should be: dc, grid, ctrl1, ctrl2
    assert call_order[0] == "dc"
    assert call_order[1] == "grid"
    assert call_order[2] == "ctrl1"
    assert call_order[3] == "ctrl2"


def test_coordinator_batch_action_applied():
    """Batch size changes from controllers are applied to datacenter."""
    dc = _StubDC(dt_s=Fraction(1))
    grid = _StubGrid(dt_s=Fraction(1))

    action_with_batch = ControlAction(commands=[SetBatchSize(batch_size_by_model={"model_a": 64})])
    ctrl = _StubController(dt_s=Fraction(1), action=action_with_batch)

    coord = Coordinator(dc, grid, [ctrl], total_duration_s=1, dc_bus="671")
    coord.run()

    assert dc.apply_control_calls == [action_with_batch.commands[0]]


def test_coordinator_exposes_clock_stamped_controller_events():
    dc = _StubDC(dt_s=Fraction(1))
    grid = _StubGrid(dt_s=Fraction(1))

    def _on_step(
        clock: SimulationClock,
        datacenter: DatacenterBackend,
        grid: GridBackend,
        events: EventEmitter,
    ) -> ControlAction:

        events.emit("controller.test", {"value": 1})
        return ControlAction(commands=[])

    ctrl = _StubController(dt_s=Fraction(1), on_step=_on_step)
    coord = Coordinator(dc, grid, [ctrl], total_duration_s=1, dc_bus="671")
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
    class _EventedDC(DatacenterBackend[DatacenterState]):
        def __init__(self) -> None:
            self._events: EventEmitter | None = None
            self._dt_s = Fraction(1)
            self._state: DatacenterState | None = None
            self._history: list[DatacenterState] = []

        @property
        def dt_s(self) -> Fraction:
            return self._dt_s

        @property
        def state(self) -> DatacenterState:
            if self._state is None:
                raise RuntimeError("No state yet")
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

        def apply_control(self, command: DatacenterCommand) -> None:
            assert self._events is not None
            self._events.emit(
                "datacenter.batch_size.updated",
                {"batch_size_by_model": {"model_a": 64}},
            )

    dc = _EventedDC()
    grid = _StubGrid(dt_s=Fraction(1))
    ctrl = _StubController(
        dt_s=Fraction(1),
        action=ControlAction(commands=[SetBatchSize(batch_size_by_model={"model_a": 64})]),
    )

    coord = Coordinator(dc, grid, [ctrl], total_duration_s=1, dc_bus="671")
    log = coord.run()
    assert log.batch_log_by_model["model_a"] == [64]


def test_controller_generic_types_auto_extracted():
    class _TypedController(Controller[DatacenterBackend, OpenDSSGrid]):
        @property
        def dt_s(self) -> Fraction:
            return Fraction(1)

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
    class _ExpectedDC(DatacenterBackend[DatacenterState]):
        def __init__(self) -> None:
            self._state: DatacenterState | None = None

        @property
        def dt_s(self) -> Fraction:
            return Fraction(1)

        @property
        def state(self) -> DatacenterState:
            if self._state is None:
                raise RuntimeError("No state yet")
            return self._state

        def history(self, n: int | None = None) -> list[DatacenterState]:

            return []

        def step(self, clock: SimulationClock) -> DatacenterState:
            state = DatacenterState(time_s=clock.time_s, power_w=ThreePhase(a=1.0, b=1.0, c=1.0))
            self._state = state
            return state

        def apply_control(self, command: DatacenterCommand) -> None:
            pass

    class _OtherDC(DatacenterBackend[DatacenterState]):
        def __init__(self) -> None:
            self._state: DatacenterState | None = None

        @property
        def dt_s(self) -> Fraction:
            return Fraction(1)

        @property
        def state(self) -> DatacenterState:
            if self._state is None:
                raise RuntimeError("No state yet")
            return self._state

        def history(self, n: int | None = None) -> list[DatacenterState]:

            return []

        def step(self, clock: SimulationClock) -> DatacenterState:
            state = DatacenterState(time_s=clock.time_s, power_w=ThreePhase(a=1.0, b=1.0, c=1.0))
            self._state = state
            return state

        def apply_control(self, command: DatacenterCommand) -> None:
            pass

    class _NeedsExpectedDC(Controller[_ExpectedDC, OpenDSSGrid]):
        @property
        def dt_s(self) -> Fraction:
            return Fraction(1)

        def step(
            self,
            clock: SimulationClock,
            datacenter: _ExpectedDC,
            grid: OpenDSSGrid,
            events: EventEmitter,
        ) -> ControlAction:

            return ControlAction(commands=[])

    dc = _OtherDC()
    grid = _StubGrid(dt_s=Fraction(1))
    ctrl = _NeedsExpectedDC()
    coord = Coordinator(dc, grid, [ctrl], total_duration_s=1, dc_bus="671")

    try:
        coord.run()
        raise AssertionError("Expected TypeError for controller/datacenter incompatibility.")
    except TypeError as exc:
        msg = str(exc)
        assert "_NeedsExpectedDC" in msg
        assert "_ExpectedDC" in msg
        assert "_OtherDC" in msg


def test_coordinator_grid_uses_stale_power_when_dc_buffer_empty():
    """When dt_grid < dt_dc, grid gets the most recent DC power as fallback."""
    dc = _StubDC(dt_s=Fraction(1))
    grid = _StubGrid(dt_s=Fraction(1, 2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coord = Coordinator(dc, grid, [], total_duration_s=1, dc_bus="671")
    log = coord.run()

    # Grid fires at tick 0 (dc_buffer has 1 sample) and tick 5 (dc_buffer empty,
    # falls back to interval_start_power).
    assert grid.step_count == 2
    # Both grid steps should produce log entries
    assert len(log.grid_states) == 2


def test_coordinator_warns_dt_grid_lt_dt_dc():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Coordinator(
            _StubDC(dt_s=Fraction(1)),
            _StubGrid(dt_s=Fraction(1, 2)),
            [],
            total_duration_s=1,
        )
    msgs = [str(x.message) for x in w]
    assert any("dt_grid" in m and "dt_dc" in m for m in msgs)


def test_coordinator_warns_ctrl_dt_lt_grid_dt():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Coordinator(
            _StubDC(dt_s=Fraction(1)),
            _StubGrid(dt_s=Fraction(1)),
            [_StubController(dt_s=Fraction(1, 2))],
            total_duration_s=1,
        )
    msgs = [str(x.message) for x in w]
    assert any("stale voltages" in m for m in msgs)
