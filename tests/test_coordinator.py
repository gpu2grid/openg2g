"""Tests for the Coordinator: multi-rate stepping order, dc_buffer accumulation."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from fractions import Fraction

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.common import ThreePhase
from openg2g.controller.base import Controller
from openg2g.coordinator import Coordinator, _gcd_fraction
from openg2g.datacenter.base import DatacenterBackend, DatacenterState
from openg2g.datacenter.command import DatacenterCommand, SetBatchSize
from openg2g.events import EventEmitter
from openg2g.grid.base import BusVoltages, GridBackend, GridState, PhaseVoltages
from openg2g.grid.command import GridCommand
from openg2g.grid.opendss import OpenDSSGrid


def test_gcd_fraction():
    assert _gcd_fraction(Fraction(1, 10), Fraction(1)) == Fraction(1, 10)
    assert _gcd_fraction(Fraction(1, 2), Fraction(3, 2)) == Fraction(1, 2)
    assert _gcd_fraction(Fraction(60), Fraction(1)) == Fraction(1)


class _StubDC(DatacenterBackend[DatacenterState]):
    """Minimal datacenter for coordinator tests."""

    def __init__(self, dt_s: Fraction = Fraction(1, 10), name: str = "test") -> None:
        super().__init__(name=name)
        self._dt_s = dt_s
        self.step_count = 0
        self.apply_control_calls: list[DatacenterCommand] = []

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    def reset(self) -> None:
        self.step_count = 0
        self.apply_control_calls = []

    def step(self, clock: SimulationClock, events: EventEmitter) -> DatacenterState:
        self.step_count += 1
        return DatacenterState(
            time_s=clock.time_s,
            power_w=ThreePhase(a=100.0, b=100.0, c=100.0),
        )

    def apply_control(self, command: DatacenterCommand, events: EventEmitter) -> None:
        self.apply_control_calls.append(command)


class _StubGrid(GridBackend[GridState]):
    """Minimal grid for coordinator tests."""

    def __init__(self, dt_s: Fraction = Fraction(1)) -> None:
        super().__init__()
        self._dt_s = dt_s
        self.step_count = 0
        self.step_calls: list[tuple[SimulationClock, dict[str, list[ThreePhase]] | list[ThreePhase]]] = []

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    def reset(self) -> None:
        self.step_count = 0
        self.step_calls = []

    def step(
        self,
        clock: SimulationClock,
        power_samples_w: dict[str, list[ThreePhase]] | list[ThreePhase],
        events: EventEmitter,
    ) -> GridState:
        self.step_count += 1
        self.step_calls.append((clock, power_samples_w))
        return GridState(
            time_s=clock.time_s,
            voltages=BusVoltages({"671": PhaseVoltages(a=1.0, b=1.0, c=1.0)}),
        )

    @property
    def v_index(self) -> list[tuple[str, int]]:
        return [("671", 0), ("671", 1), ("671", 2)]

    def voltages_vector(self) -> np.ndarray:
        return np.array([1.0, 1.0, 1.0])

    def estimate_sensitivity(self, perturbation_kw: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((3, 3)), np.ones(3)

    def dc_bus(self, dc: DatacenterBackend) -> str:
        return "671"

    def apply_control(self, command: GridCommand, events: EventEmitter) -> None:
        pass


class _StubController(Controller[DatacenterBackend, GridBackend]):
    """Test controller that delegates to a callback or returns a fixed action."""

    def __init__(
        self,
        dt_s: Fraction = Fraction(1),
        commands: list[DatacenterCommand | GridCommand] | None = None,
        on_step: Callable[
            [SimulationClock, GridBackend, EventEmitter],
            list[DatacenterCommand | GridCommand],
        ]
        | None = None,
    ):
        self._dt_s = dt_s
        self._commands = commands if commands is not None else []
        self._on_step = on_step
        self.call_count = 0

    def reset(self) -> None:
        self.call_count = 0

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    def step(
        self,
        clock: SimulationClock,
        grid: GridBackend,
        events: EventEmitter,
    ) -> list[DatacenterCommand | GridCommand]:
        self.call_count += 1
        if self._on_step is not None:
            return self._on_step(clock, grid, events)
        return self._commands


def test_coordinator_dc_fires_every_tick():
    """DC (dt=1/10) should fire 10x per grid step (dt=1)."""
    dc = _StubDC(dt_s=Fraction(1, 10))
    grid = _StubGrid(dt_s=Fraction(1))
    ctrl = _StubController(dt_s=Fraction(1))

    coord = Coordinator(datacenters=[dc], grid=grid, controllers=[ctrl], total_duration_s=1)
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

    coord = Coordinator(datacenters=[dc], grid=grid, controllers=[ctrl], total_duration_s=1)
    coord.run()

    # Grid fires at step 0 and step 5 (twice in 1 second with dt=1/2)
    assert grid.step_count == 2
    # First grid call (step 0): 1 DC sample accumulated
    # Second grid call (step 5): 5 DC samples (steps 1-5)
    sizes = [len(next(iter(call[1].values()))) for call in grid.step_calls]
    assert sizes == [1, 5]


def test_coordinator_controller_order():
    """Controllers execute in order, after DC and grid."""
    call_order: list[str] = []

    class _OrderDC(_StubDC):
        def step(self, clock: SimulationClock, events: EventEmitter) -> DatacenterState:
            call_order.append("dc")
            return super().step(clock, events)

    class _OrderGrid(_StubGrid):
        def step(
            self,
            clock: SimulationClock,
            power_samples_w: dict[str, list[ThreePhase]] | list[ThreePhase],
            events: EventEmitter,
        ) -> GridState:
            call_order.append("grid")
            return super().step(clock, power_samples_w, events)

    dc = _OrderDC(dt_s=Fraction(1))
    grid = _OrderGrid(dt_s=Fraction(1))

    ctrl1 = _StubController(
        dt_s=Fraction(1),
        on_step=lambda clock, g, ev: (
            call_order.append("ctrl1"),
            [],
        )[-1],
    )

    ctrl2 = _StubController(
        dt_s=Fraction(1),
        on_step=lambda clock, g, ev: (
            call_order.append("ctrl2"),
            [],
        )[-1],
    )

    coord = Coordinator(datacenters=[dc], grid=grid, controllers=[ctrl1, ctrl2], total_duration_s=1)
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

    batch_cmd = SetBatchSize(batch_size_by_model={"model_a": 64}, target=dc)
    ctrl = _StubController(dt_s=Fraction(1), commands=[batch_cmd])

    coord = Coordinator(datacenters=[dc], grid=grid, controllers=[ctrl], total_duration_s=1)
    coord.run()

    assert dc.apply_control_calls == [batch_cmd]


def test_coordinator_exposes_clock_stamped_controller_events():
    dc = _StubDC(dt_s=Fraction(1))
    grid = _StubGrid(dt_s=Fraction(1))

    def _on_step(
        clock: SimulationClock,
        grid: GridBackend,
        events: EventEmitter,
    ) -> list[DatacenterCommand | GridCommand]:

        events.emit("controller.test", {"value": 1})
        return []

    ctrl = _StubController(dt_s=Fraction(1), on_step=_on_step)
    coord = Coordinator(datacenters=[dc], grid=grid, controllers=[ctrl], total_duration_s=1)
    log = coord.run()

    assert len(log.events) == 1
    ev = log.events[0]
    assert ev.source == "controller"
    assert ev.topic == "controller.test"
    assert ev.tick == 0
    assert abs(ev.t_s - 0.0) < 1e-12


def test_datacenter_events_are_recorded():
    class _EventedDC(DatacenterBackend[DatacenterState]):
        def __init__(self) -> None:
            super().__init__(name="evented")
            self._dt_s = Fraction(1)

        @property
        def dt_s(self) -> Fraction:
            return self._dt_s

        def reset(self) -> None:
            pass

        def step(self, clock: SimulationClock, events: EventEmitter) -> DatacenterState:
            return DatacenterState(time_s=clock.time_s, power_w=ThreePhase(a=1.0, b=1.0, c=1.0))

        def apply_control(self, command: DatacenterCommand, events: EventEmitter) -> None:
            events.emit(
                "datacenter.batch_size.updated",
                {"batch_size_by_model": {"model_a": 64}},
            )

    dc = _EventedDC()
    grid = _StubGrid(dt_s=Fraction(1))
    ctrl = _StubController(
        dt_s=Fraction(1),
        commands=[SetBatchSize(batch_size_by_model={"model_a": 64}, target=dc)],
    )

    coord = Coordinator(datacenters=[dc], grid=grid, controllers=[ctrl], total_duration_s=1)
    log = coord.run()
    batch_events = [e for e in log.events if e.topic == "datacenter.batch_size.updated"]
    assert len(batch_events) == 1
    assert batch_events[0].data["batch_size_by_model"]["model_a"] == 64


def test_controller_generic_types_auto_extracted():
    class _TypedController(Controller[DatacenterBackend, OpenDSSGrid]):
        @property
        def dt_s(self) -> Fraction:
            return Fraction(1)

        def reset(self) -> None:
            pass

        def step(
            self,
            clock: SimulationClock,
            grid: OpenDSSGrid,
            events: EventEmitter,
        ) -> list[DatacenterCommand | GridCommand]:

            return []

    assert _TypedController.compatible_datacenter_types() == (DatacenterBackend,)
    assert _TypedController.compatible_grid_types() == (OpenDSSGrid,)


def test_controller_datacenter_mismatch_error_has_underlined_generic_snippet():
    class _ExpectedDC(DatacenterBackend[DatacenterState]):
        def __init__(self) -> None:
            super().__init__(name="expected")

        @property
        def dt_s(self) -> Fraction:
            return Fraction(1)

        def reset(self) -> None:
            pass

        def step(self, clock: SimulationClock, events: EventEmitter) -> DatacenterState:
            return DatacenterState(time_s=clock.time_s, power_w=ThreePhase(a=1.0, b=1.0, c=1.0))

        def apply_control(self, command: DatacenterCommand, events: EventEmitter) -> None:
            pass

    class _OtherDC(DatacenterBackend[DatacenterState]):
        def __init__(self) -> None:
            super().__init__(name="other")

        @property
        def dt_s(self) -> Fraction:
            return Fraction(1)

        def reset(self) -> None:
            pass

        def step(self, clock: SimulationClock, events: EventEmitter) -> DatacenterState:
            return DatacenterState(time_s=clock.time_s, power_w=ThreePhase(a=1.0, b=1.0, c=1.0))

        def apply_control(self, command: DatacenterCommand, events: EventEmitter) -> None:
            pass

    class _NeedsExpectedDC(Controller[_ExpectedDC, OpenDSSGrid]):
        @property
        def dt_s(self) -> Fraction:
            return Fraction(1)

        def reset(self) -> None:
            pass

        def step(
            self,
            clock: SimulationClock,
            grid: OpenDSSGrid,
            events: EventEmitter,
        ) -> list[DatacenterCommand | GridCommand]:

            return []

    dc = _OtherDC()
    grid = _StubGrid(dt_s=Fraction(1))
    ctrl = _NeedsExpectedDC()
    coord = Coordinator(datacenters=[dc], grid=grid, controllers=[ctrl], total_duration_s=1)

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
        coord = Coordinator(datacenters=[dc], grid=grid, controllers=[], total_duration_s=1)
    log = coord.run()

    # Grid fires at tick 0 (dc_buffer has 1 sample) and tick 5 (dc_buffer empty,
    # grid reuses its internally cached previous power).
    assert grid.step_count == 2
    # Both grid steps should produce log entries
    assert len(log.grid_states) == 2


def test_coordinator_warns_dt_grid_lt_dt_dc():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Coordinator(
            datacenters=[_StubDC(dt_s=Fraction(1))],
            grid=_StubGrid(dt_s=Fraction(1, 2)),
            controllers=[],
            total_duration_s=1,
        )
    msgs = [str(x.message) for x in w]
    assert any("dt_grid" in m and "dt_dc" in m for m in msgs)


def test_coordinator_warns_ctrl_dt_lt_grid_dt():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Coordinator(
            datacenters=[_StubDC(dt_s=Fraction(1))],
            grid=_StubGrid(dt_s=Fraction(1)),
            controllers=[_StubController(dt_s=Fraction(1, 2))],
            total_duration_s=1,
        )
    msgs = [str(x.message) for x in w]
    assert any("stale voltages" in m for m in msgs)


def test_coordinator_run_twice_identical():
    """Calling run() twice on the same coordinator produces identical results."""
    dc = _StubDC(dt_s=Fraction(1, 10))
    grid = _StubGrid(dt_s=Fraction(1))
    ctrl = _StubController(dt_s=Fraction(1))
    coord = Coordinator(datacenters=[dc], grid=grid, controllers=[ctrl], total_duration_s=2)

    log1 = coord.run()
    log2 = coord.run()

    assert len(log1.dc_states) == len(log2.dc_states)
    assert len(log1.grid_states) == len(log2.grid_states)
    assert [s.time_s for s in log1.grid_states] == [s.time_s for s in log2.grid_states]
    assert dc.step_count == 20  # 20 per run, reset between
    assert grid.step_count == 2  # 2 per run, reset between
