from __future__ import annotations

from fractions import Fraction

import numpy as np
from mlenergy_data.modeling import LogisticModel

from openg2g.clock import SimulationClock
from openg2g.common import ThreePhase
from openg2g.controller.ofo import LogisticModelStore, OFOBatchSizeController, OFOConfig
from openg2g.coordinator import SimulationLog
from openg2g.datacenter.base import LLMBatchSizeControlledDatacenter, LLMDatacenterState
from openg2g.datacenter.command import DatacenterCommand
from openg2g.datacenter.config import InferenceModelSpec
from openg2g.events import EventEmitter
from openg2g.grid.base import BusVoltages, GridBackend, GridState, PhaseVoltages
from openg2g.grid.command import GridCommand
from openg2g.grid.opendss import OpenDSSGrid


class _GridStub(OpenDSSGrid):
    def __init__(self):
        GridBackend.__init__(self)
        self._v_index_list = [("671", 0), ("671", 1), ("671", 2)]

    def reset(self) -> None:
        pass

    @property
    def dt_s(self) -> Fraction:
        return Fraction(1)

    @property
    def v_index(self) -> list[tuple[str, int]]:
        return self._v_index_list

    def voltages_vector(self) -> np.ndarray:
        return np.array([0.94, 0.96, 1.01], dtype=float)

    def estimate_sensitivity(self, perturbation_kw: float = 100.0, dc=None) -> tuple[np.ndarray, np.ndarray]:
        H = np.eye(3, dtype=float)
        v0 = np.array([1.0, 1.0, 1.0], dtype=float)
        return H, v0

    def step(
        self,
        clock: SimulationClock,
        power_samples_w: list[ThreePhase],
        events: EventEmitter,
    ) -> GridState:
        return GridState(
            time_s=clock.time_s,
            voltages=BusVoltages({"671": PhaseVoltages(a=0.94, b=0.96, c=1.01)}),
        )

    def set_state(self, state: GridState) -> None:
        self._state = state

    def apply_control(self, command: GridCommand, events: EventEmitter) -> None:
        pass


class _DCStub(LLMBatchSizeControlledDatacenter):
    def __init__(self):
        super().__init__(name="test")

    def reset(self) -> None:
        pass

    @property
    def dt_s(self) -> Fraction:
        return Fraction(1)

    def step(self, clock: SimulationClock, events: EventEmitter) -> LLMDatacenterState:
        if self._state is None:
            self._state = LLMDatacenterState(
                time_s=0.0,
                power_w=ThreePhase(100.0, 100.0, 100.0),
                batch_size_by_model={"M1": 64},
                active_replicas_by_model={"M1": 5},
                observed_itl_s_by_model={"M1": 0.2},
            )
        return self._state

    def set_state(self, state: LLMDatacenterState) -> None:
        self._state = state

    def apply_control(self, command: DatacenterCommand, events: EventEmitter) -> None:
        pass


def _make_model_store() -> LogisticModelStore:
    """Create a LogisticModelStore with synthetic fits for a single model M1."""
    power = {"M1": LogisticModel(L=100.0, x0=6.0, k=1.0, b0=50.0)}
    latency = {"M1": LogisticModel(L=0.05, x0=6.0, k=1.0, b0=0.02)}
    throughput = {"M1": LogisticModel(L=10.0, x0=6.0, k=1.0, b0=1.0)}
    return LogisticModelStore(power, latency, throughput)


def _build_controller(datacenter: LLMBatchSizeControlledDatacenter, grid=None) -> OFOBatchSizeController:
    model = InferenceModelSpec(
        model_id="test/Model",
        model_label="M1",
        gpus_per_replica=1,
        itl_deadline_s=0.1,
        feasible_batch_sizes=(8, 16, 32, 64, 128),
    )
    return OFOBatchSizeController(
        (model,),
        datacenter=datacenter,
        grid=grid,
        models=_make_model_store(),
        config=OFOConfig(
            primal_step_size=0.05,
            w_throughput=0.0,
            w_switch=0.0,
            v_min=0.95,
            v_max=1.05,
            voltage_dual_step_size=0.5,
            latency_dual_step_size=1.0,
        ),
        dt_s=Fraction(1),
        initial_batch_sizes={"M1": 64},
    )


def test_ofo_uses_observed_latency_for_dual_update():
    dc = _DCStub()
    grid = _GridStub()
    ctrl = _build_controller(dc, grid=grid)

    dc_state = LLMDatacenterState(
        time_s=0.0,
        power_w=ThreePhase(100.0, 100.0, 100.0),
        batch_size_by_model={"M1": 64},
        active_replicas_by_model={"M1": 5},
        observed_itl_s_by_model={"M1": 0.2},
    )
    grid_state = GridState(
        time_s=0.0,
        voltages=BusVoltages({"671": PhaseVoltages(a=0.94, b=0.96, c=1.01)}),
    )
    dc.set_state(dc_state)
    grid.set_state(grid_state)
    log = SimulationLog()
    events = EventEmitter(SimulationClock(Fraction(1)), log, "controller")

    action = ctrl.step(SimulationClock(Fraction(1)), events)
    assert len(action) == 1
    assert ctrl._latency_dual_by_model["M1"] > 0.0


def test_ofo_requires_observed_latency_map():
    dc = _DCStub()
    grid = _GridStub()
    ctrl = _build_controller(dc, grid=grid)

    dc_state = LLMDatacenterState(
        time_s=0.0,
        power_w=ThreePhase(100.0, 100.0, 100.0),
        batch_size_by_model={"M1": 64},
        active_replicas_by_model={"M1": 5},
        observed_itl_s_by_model={},
    )
    dc.set_state(dc_state)
    log = SimulationLog()
    events = EventEmitter(SimulationClock(Fraction(1)), log, "controller")

    try:
        ctrl.step(SimulationClock(Fraction(1)), events)
    except RuntimeError as exc:
        assert "observed_itl_s_by_model" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing observed latency")
