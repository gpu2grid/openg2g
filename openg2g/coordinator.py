"""Central coordinator: multi-rate simulation loop."""

from __future__ import annotations

from dataclasses import dataclass, field

from openg2g.clock import SimulationClock
from openg2g.context import SimulationContext, validate_required_features
from openg2g.controller.base import Controller
from openg2g.datacenter.base import DatacenterBackend
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.types import (
    Command,
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
    """Accumulated simulation data from a coordinator run.

    All fields below are populated by the ``Coordinator.run()`` loop.

    State history (one entry per component step):
        dc_states: Every ``DatacenterState`` produced by the datacenter.
        grid_states: Every ``GridState`` produced by the grid.
        actions: Every ``ControlAction`` emitted by controllers.

    Per-grid-step time series (aligned with ``grid_states``):
        time_s: Simulation time at each grid step (seconds).
        Va, Vb, Vc: DC-bus voltage per phase at each grid step (pu).
        kW_A, kW_B, kW_C: DC load power per phase at each grid step (kW).

    Controller logs (populated only when controllers emit batch changes):
        batch_log_by_model: Per-model batch size history.
    """

    # -- State history --
    dc_states: list[DatacenterState] = field(default_factory=list)
    grid_states: list[GridState] = field(default_factory=list)
    actions: list[ControlAction] = field(default_factory=list)
    commands: list[Command] = field(default_factory=list)

    # -- Per-grid-step time series --
    time_s: list[float] = field(default_factory=list)
    Va: list[float] = field(default_factory=list)
    Vb: list[float] = field(default_factory=list)
    Vc: list[float] = field(default_factory=list)
    kW_A: list[float] = field(default_factory=list)
    kW_B: list[float] = field(default_factory=list)
    kW_C: list[float] = field(default_factory=list)

    # -- Controller logs --
    batch_log_by_model: dict[str, list[int]] = field(default_factory=dict)

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
        self.commands.extend(action.commands)

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
        self.context = self._build_context()

        # Compute tick as GCD of all component periods
        periods = [datacenter.dt_s, grid.dt_s] + [c.dt_s for c in controllers]
        tick = periods[0]
        for p in periods[1:]:
            tick = gcd_float(tick, p)

        self.clock = SimulationClock(tick_s=tick, live=live)

    def _build_context(self) -> SimulationContext:
        return SimulationContext(
            voltage=self.grid if hasattr(self.grid, "voltages_vector") else None,
            sensitivity=self.grid if hasattr(self.grid, "estimate_H") else None,
            raw={"grid": self.grid, "datacenter": self.datacenter},
        )

    def run(self) -> SimulationLog:
        """Run the full simulation and return the log."""
        for ctrl in self.controllers:
            validate_required_features(
                controller_name=ctrl.__class__.__name__,
                required=ctrl.required_features(),
                context=self.context,
            )

        dc_state: DatacenterState | None = None
        grid_state: GridState | None = None
        dc_buffer: list[ThreePhase] = []
        interval_start_power: ThreePhase | None = None
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
                grid_state = self.grid.step(
                    self.clock,
                    list(dc_buffer),
                    interval_start_w=interval_start_power,
                )
                # Record the last power sample for kW logging
                last_power = dc_buffer[-1]
                log.record_power(last_power)
                interval_start_power = last_power
                dc_buffer.clear()
                log.record_grid(grid_state, dc_bus=self.dc_bus)

            # 3. Controllers (if due) — in order, actions applied immediately
            for ctrl in self.controllers:
                if self.clock.is_due(ctrl.dt_s):
                    action = ctrl.step(self.clock, dc_state, grid_state, self.context)
                    for command in action.commands:
                        if command.target == "datacenter":
                            self.datacenter.apply_control(command)
                            if command.kind == "set_batch_size":
                                batch_map = command.payload.get("batch_size_by_model", {})
                                if isinstance(batch_map, dict):
                                    log.record_batch(batch_map)
                        elif command.target == "grid":
                            self.grid.apply_control(command)
                        elif command.target == "custom":
                            pass
                        else:
                            raise ValueError(f"Unsupported command target: {command.target!r}")
                    log.record_action(action)

            self.clock.advance()

        return log
