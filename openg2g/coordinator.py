"""Central coordinator: multi-rate simulation loop."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.datacenter.base import DatacenterBackend
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.types import (
    ControlAction,
    DatacenterState,
    GridState,
    ThreePhase,
)


def gcd_float(a: float, b: float, tol: float = 1e-9) -> float:
    """GCD of two positive floats using Euclidean algorithm."""
    a, b = abs(a), abs(b)
    while b > tol:
        a, b = b, a % b
    return a


@dataclass
class SimulationLog:
    """Accumulated simulation data from a coordinator run."""

    dc_states: list[DatacenterState] = field(default_factory=list)
    grid_states: list[GridState] = field(default_factory=list)
    actions: list[ControlAction] = field(default_factory=list)

    # Per-timestep arrays for easy access
    time_s: list[float] = field(default_factory=list)
    Va: list[float] = field(default_factory=list)
    Vb: list[float] = field(default_factory=list)
    Vc: list[float] = field(default_factory=list)
    kW_A: list[float] = field(default_factory=list)
    kW_B: list[float] = field(default_factory=list)
    kW_C: list[float] = field(default_factory=list)

    # Per-model batch log
    batch_log_by_model: dict[str, list[int]] = field(default_factory=dict)
    # Voltage vector at each grid step
    v_vec_list: list[np.ndarray] = field(default_factory=list)
    # All-bus voltage snapshots
    Vabc_all_list: list[dict[str, ThreePhase]] = field(default_factory=list)
    # Eta norm
    eta_norm: list[float] = field(default_factory=list)

    def record_dc(self, state: DatacenterState) -> None:
        self.dc_states.append(state)

    def record_grid(self, state: GridState, *, dc_bus: str = "671") -> None:
        self.grid_states.append(state)
        self.time_s.append(state.time_s)

        v_dc = (
            state.voltages[dc_bus]
            if dc_bus in state.voltages
            else ThreePhase(a=float("nan"), b=float("nan"), c=float("nan"))
        )
        self.Va.append(v_dc.a)
        self.Vb.append(v_dc.b)
        self.Vc.append(v_dc.c)

    def record_action(self, action: ControlAction) -> None:
        self.actions.append(action)

    def record_power(self, power_w: ThreePhase) -> None:
        self.kW_A.append(power_w.a / 1e3)
        self.kW_B.append(power_w.b / 1e3)
        self.kW_C.append(power_w.c / 1e3)

    def record_batch(self, batch_by_model: dict[str, int]) -> None:
        for label, b in batch_by_model.items():
            if label not in self.batch_log_by_model:
                self.batch_log_by_model[label] = []
            self.batch_log_by_model[label].append(int(b))


class Coordinator:
    """Multi-rate simulation coordinator.

    Orchestrates datacenter, grid, and controller components at their
    respective rates.  The base tick is the GCD of all component periods.

    Args:
        datacenter: Datacenter backend (offline or online).
        grid: OpenDSS grid simulator.
        controllers: List of controllers, applied in order each tick.
        T_total_s: Total simulation duration (seconds).
        dc_bus: Bus name for DC voltage logging.
        live: If True, synchronize with wall-clock time.
    """

    def __init__(
        self,
        datacenter: DatacenterBackend,
        grid: OpenDSSGrid,
        controllers: list[Controller],
        T_total_s: float,
        dc_bus: str = "671",
        live: bool = False,
    ):
        self.datacenter = datacenter
        self.grid = grid
        self.controllers = list(controllers)
        self.T_total_s = float(T_total_s)
        self.dc_bus = str(dc_bus)

        # Compute tick as GCD of all component periods
        periods = [datacenter.dt_s, grid.dt_s] + [c.dt_s for c in controllers]
        tick = periods[0]
        for p in periods[1:]:
            tick = gcd_float(tick, p)

        self.clock = SimulationClock(tick_s=tick, live=live)

    def run(self) -> SimulationLog:
        """Run the full simulation and return the log."""
        dc_state: DatacenterState | None = None
        grid_state: GridState | None = None
        dc_buffer: list[ThreePhase] = []
        log = SimulationLog()

        n_ticks = int(round(self.T_total_s / self.clock.tick_s))

        for _ in range(n_ticks):
            # 1. Datacenter step (if due)
            if self.clock.is_due(self.datacenter.dt_s):
                dc_state = self.datacenter.step(self.clock)
                dc_buffer.append(dc_state.power_w)
                log.record_dc(dc_state)

            # 2. Grid step (if due) — pass full sub-trace since last grid step
            if self.clock.is_due(self.grid.dt_s) and dc_buffer:
                # In resample mode, use the DC's chunk power trace (which
                # includes the endpoint sample) for correct np.interp
                # resampling.  In other modes, use the accumulated dc_buffer.
                grid_buffer: list[ThreePhase] = list(dc_buffer)
                if self.grid.sub_step_mode == "resample":
                    _get_trace = getattr(self.datacenter, "chunk_power_trace", None)
                    if callable(_get_trace):
                        result = _get_trace()
                        if isinstance(result, list) and result:
                            grid_buffer = result
                grid_state = self.grid.step(self.clock, grid_buffer)
                # Record the last power sample for kW logging
                last_power = dc_buffer[-1]
                log.record_power(last_power)
                dc_buffer.clear()
                log.record_grid(grid_state, dc_bus=self.dc_bus)

            # 3. Controllers (if due) — in order, actions applied immediately
            for ctrl in self.controllers:
                if self.clock.is_due(ctrl.dt_s):
                    action = ctrl.step(self.clock, dc_state, grid_state)
                    if action.batch_size_by_model:
                        self.datacenter.apply_control(action)
                        log.record_batch(action.batch_size_by_model)
                    if action.tap_changes:
                        self.grid.apply_control(action)
                    log.record_action(action)

            self.clock.advance()

        return log
