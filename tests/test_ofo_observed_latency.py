from __future__ import annotations

from fractions import Fraction

import numpy as np
from mlenergy_data.modeling import LogisticModel

from openg2g.clock import SimulationClock
from openg2g.controller.ofo import OFOBatchController, PrimalCfg, VoltageDualCfg
from openg2g.datacenter.base import LLMBatchSizeControlledDatacenter
from openg2g.events import EventEmitter, SimEvent
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.models.spec import ModelSpec
from openg2g.types import BusVoltages, Command, DatacenterState, GridState, ThreePhase


class _GridStub(OpenDSSGrid):
    def __init__(self):
        self._v_index = [("671", 0), ("671", 1), ("671", 2)]
        self._state: GridState | None = None
        self._history: list[GridState] = []

    @property
    def dt_s(self) -> Fraction:
        return Fraction(1)

    @property
    def state(self) -> GridState | None:
        return self._state

    def history(self, n: int | None = None) -> list[GridState]:
        if n is None:
            return list(self._history)
        if n <= 0:
            return []
        return list(self._history[-int(n) :])

    @property
    def v_index(self) -> list[tuple[str, int]]:
        return self._v_index

    def voltages_vector(self) -> np.ndarray:
        return np.array([0.94, 0.96, 1.01], dtype=float)

    def estimate_H(self, dp_kw: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
        H = np.eye(3, dtype=float)
        v0 = np.array([1.0, 1.0, 1.0], dtype=float)
        return H, v0

    def step(
        self,
        clock: SimulationClock,
        load_trace_w: list[ThreePhase],
        *,
        interval_start_w: ThreePhase | None = None,
    ) -> GridState:

        self._state = GridState(
            time_s=clock.time_s,
            voltages=BusVoltages({"671": ThreePhase(a=0.94, b=0.96, c=1.01)}),
        )
        self._history.append(self._state)
        return self._state

    def set_state(self, state: GridState) -> None:
        self._state = state

    def apply_control(self, command: Command) -> None:
        pass


class _DCStub(LLMBatchSizeControlledDatacenter):
    def __init__(self):
        self._state: DatacenterState | None = None

    @property
    def dt_s(self) -> Fraction:
        return Fraction(1)

    @property
    def state(self) -> DatacenterState | None:
        return self._state

    def history(self, n: int | None = None) -> list[DatacenterState]:

        return [] if self._state is None else [self._state]

    def step(self, clock: SimulationClock) -> DatacenterState:

        if self._state is None:
            self._state = DatacenterState(
                time_s=0.0,
                power_w=ThreePhase(100.0, 100.0, 100.0),
                batch_size_by_model={"M1": 64},
                active_replicas_by_model={"M1": 5},
                observed_itl_s_by_model={"M1": 0.2},
            )
        return self._state

    def set_state(self, state: DatacenterState) -> None:
        self._state = state

    def apply_control(self, command: Command) -> None:
        pass


class _EventSink:
    def __init__(self) -> None:
        self.events: list[SimEvent] = []

    def emit(self, event: SimEvent) -> None:
        self.events.append(event)


def _make_fits() -> tuple[
    dict[str, LogisticModel], dict[str, LogisticModel], dict[str, LogisticModel]
]:
    """Create synthetic logistic fits for a single model M1."""
    power = {"M1": LogisticModel(L=100.0, x0=6.0, k=1.0, b0=50.0)}
    latency = {"M1": LogisticModel(L=0.05, x0=6.0, k=1.0, b0=0.02)}
    throughput = {"M1": LogisticModel(L=10.0, x0=6.0, k=1.0, b0=1.0)}
    return power, latency, throughput


def _build_controller() -> OFOBatchController:
    model = ModelSpec(model_label="M1", num_replicas=10, gpus_per_replica=1)
    power_fits, latency_fits, throughput_fits = _make_fits()
    return OFOBatchController(
        models=[model],
        power_fits=power_fits,
        latency_fits=latency_fits,
        throughput_fits=throughput_fits,
        Lth_by_model={"M1": 0.1},
        primal_cfg=PrimalCfg(eta_primal=0.05, w_latency=1.0, w_throughput=0.0, w_switch=0.0),
        voltage_dual_cfg=VoltageDualCfg(v_min=0.95, v_max=1.05, rho_v=0.5),
        batch_set=[8, 16, 32, 64, 128],
        batch_init=64,
        rho_l=1.0,
        dt_s=Fraction(1),
    )


def test_ofo_uses_observed_latency_for_dual_update():
    ctrl = _build_controller()
    grid = _GridStub()
    dc = _DCStub()

    dc_state = DatacenterState(
        time_s=0.0,
        power_w=ThreePhase(100.0, 100.0, 100.0),
        batch_size_by_model={"M1": 64},
        active_replicas_by_model={"M1": 5},
        observed_itl_s_by_model={"M1": 0.2},
    )
    grid_state = GridState(
        time_s=0.0,
        voltages=BusVoltages({"671": ThreePhase(a=0.94, b=0.96, c=1.01)}),
    )
    dc.set_state(dc_state)
    grid.set_state(grid_state)
    sink = _EventSink()
    events = EventEmitter(SimulationClock(Fraction(1)), sink, "controller")

    action = ctrl.step(SimulationClock(Fraction(1)), dc, grid, events)
    assert len(action.commands) == 1
    assert ctrl.mu_by_model["M1"] > 0.0


def test_ofo_requires_observed_latency_map():
    ctrl = _build_controller()
    grid = _GridStub()
    dc = _DCStub()

    dc_state = DatacenterState(
        time_s=0.0,
        power_w=ThreePhase(100.0, 100.0, 100.0),
        batch_size_by_model={"M1": 64},
        active_replicas_by_model={"M1": 5},
        observed_itl_s_by_model={},
    )
    dc.set_state(dc_state)
    sink = _EventSink()
    events = EventEmitter(SimulationClock(Fraction(1)), sink, "controller")

    try:
        ctrl.step(SimulationClock(Fraction(1)), dc, grid, events)
    except RuntimeError as exc:
        assert "observed_itl_s_by_model" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing observed latency")
