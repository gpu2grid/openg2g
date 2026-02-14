from __future__ import annotations

from pathlib import Path

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.controller.ofo import OFOBatchController, PrimalCfg, VoltageDualCfg
from openg2g.datacenter.base import DatacenterBackend
from openg2g.events import EventEmitter, SimEvent
from openg2g.grid.base import GridBackend
from openg2g.models.logistic import load_logistic_fits
from openg2g.models.spec import ModelSpec
from openg2g.types import BusVoltages, Command, DatacenterState, GridState, ThreePhase


class _GridStub(GridBackend):
    def __init__(self):
        self._v_index = [("671", 0), ("671", 1), ("671", 2)]
        self._state: GridState | None = None
        self._history: list[GridState] = []

    @property
    def dt_s(self) -> float:
        return 1.0

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
        _ = dp_kw
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
        del load_trace_w, interval_start_w
        self._state = GridState(
            time_s=clock.time_s,
            voltages=BusVoltages({"671": ThreePhase(a=0.94, b=0.96, c=1.01)}),
        )
        self._history.append(self._state)
        return self._state

    def set_state(self, state: GridState) -> None:
        self._state = state

    def apply_control(self, command: Command) -> None:
        del command


class _DCStub(DatacenterBackend):
    def __init__(self):
        self._state: DatacenterState | None = None

    @property
    def dt_s(self) -> float:
        return 1.0

    @property
    def state(self) -> DatacenterState | None:
        return self._state

    def history(self, n: int | None = None) -> list[DatacenterState]:
        del n
        return [] if self._state is None else [self._state]

    def step(self, clock: SimulationClock) -> DatacenterState:
        del clock
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
        del command


class _EventSink:
    def __init__(self) -> None:
        self.events: list[SimEvent] = []

    def emit(self, event: SimEvent) -> None:
        self.events.append(event)


def _write_fit_csv(path: Path) -> None:
    path.write_text(
        "metric,L,x0,k,b0\npower,100,6,1,50\nlatency,0.05,6,1,0.02\nthroughput,10,6,1,1\n"
    )


def _build_controller(tmp_path: Path) -> OFOBatchController:
    model = ModelSpec(model_label="M1", replicas=10, gpus_per_replica=1)
    fit_csv = tmp_path / "m1_fit.csv"
    _write_fit_csv(fit_csv)
    fits = load_logistic_fits({"M1": fit_csv})
    return OFOBatchController(
        models=[model],
        fits=fits,
        Lth_by_model={"M1": 0.1},
        primal_cfg=PrimalCfg(eta_primal=0.05, w_latency=1.0, w_throughput=0.0, w_switch=0.0),
        voltage_dual_cfg=VoltageDualCfg(v_min=0.95, v_max=1.05, rho_v=0.5),
        batch_set=[8, 16, 32, 64, 128],
        batch_init=64,
        rho_l=1.0,
        dt_s=1.0,
    )


def test_ofo_uses_observed_latency_for_dual_update(tmp_path: Path):
    ctrl = _build_controller(tmp_path)
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
    events = EventEmitter(SimulationClock(1.0), sink, "controller")

    action = ctrl.step(SimulationClock(1.0), dc, grid, events)
    assert len(action.commands) == 1
    assert ctrl.mu_by_model["M1"] > 0.0


def test_ofo_requires_observed_latency_map(tmp_path: Path):
    ctrl = _build_controller(tmp_path)
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
    events = EventEmitter(SimulationClock(1.0), sink, "controller")

    try:
        ctrl.step(SimulationClock(1.0), dc, grid, events)
    except RuntimeError as exc:
        assert "observed_itl_s_by_model" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing observed latency")
