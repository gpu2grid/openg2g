"""Central coordinator: multi-rate simulation loop."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Generic

from openg2g.clock import SimulationClock
from openg2g.common import ThreePhase
from openg2g.controller.base import Controller
from openg2g.datacenter.base import DatacenterBackend, DCStateT
from openg2g.datacenter.command import DatacenterCommand
from openg2g.events import EventEmitter, SimEvent
from openg2g.grid.base import GridBackend, GridStateT, PhaseVoltages
from openg2g.grid.command import GridCommand

logger = logging.getLogger(__name__)


@dataclass
class SimulationLog(Generic[DCStateT, GridStateT]):
    """Accumulated simulation data from a coordinator run.

    Generic over the datacenter and grid state types. When constructed
    via [`Coordinator.run`][..Coordinator.run], the type parameters are
    inferred from the backends, giving typed access to backend-specific
    state fields.

    Attributes:
        dc_states: Every datacenter state produced by the datacenter.
        grid_states: Every grid state produced by the grid.
        commands: All commands emitted by controllers.
        time_s: Simulation time at each grid step (seconds).
        voltage_a_pu: DC-bus voltage phase A at each grid step (pu).
        voltage_b_pu: DC-bus voltage phase B at each grid step (pu).
        voltage_c_pu: DC-bus voltage phase C at each grid step (pu).
        events: Clock-stamped simulation events from all components.
    """

    dc_states: list[DCStateT] = field(default_factory=list)
    grid_states: list[GridStateT] = field(default_factory=list)
    commands: list[DatacenterCommand | GridCommand] = field(default_factory=list)

    time_s: list[float] = field(default_factory=list)
    voltage_a_pu: list[float] = field(default_factory=list)
    voltage_b_pu: list[float] = field(default_factory=list)
    voltage_c_pu: list[float] = field(default_factory=list)

    events: list[SimEvent] = field(default_factory=list)

    def record_datacenter(self, state: DCStateT) -> None:
        """Append a datacenter state snapshot."""
        self.dc_states.append(state)

    def record_grid(self, state: GridStateT, *, dc_bus: str) -> None:
        """Append a grid state snapshot and extract DC bus voltages."""
        self.grid_states.append(state)
        self.time_s.append(state.time_s)

        v_dc = (
            state.voltages[dc_bus]
            if dc_bus in state.voltages
            else PhaseVoltages(a=float("nan"), b=float("nan"), c=float("nan"))
        )
        self.voltage_a_pu.append(v_dc.a)
        self.voltage_b_pu.append(v_dc.b)
        self.voltage_c_pu.append(v_dc.c)

    def record_commands(self, commands: list[DatacenterCommand | GridCommand]) -> None:
        """Append control commands issued during a tick."""
        self.commands.extend(commands)

    def emit(self, event: SimEvent) -> None:
        """Event sink entrypoint for component-originated events."""
        self.events.append(event)


def _gcd_fraction(a: Fraction, b: Fraction) -> Fraction:
    """GCD of two positive Fractions using Euclidean algorithm."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


class Coordinator(Generic[DCStateT, GridStateT]):
    """Multi-rate simulation coordinator.

    Orchestrates datacenter, grid, and controller components at their
    respective rates.  The base tick is the GCD of all component
    periods.

    Generic over datacenter and grid state types. The type parameters
    are inferred from the backends and propagated to
    [`SimulationLog`][..SimulationLog].

    Args:
        datacenter: Datacenter backend (offline or online).
        grid: Grid simulator backend.
        controllers: List of controllers, applied in order each tick.
        total_duration_s: Total simulation duration (integer seconds).
        dc_bus: Bus name for DC voltage logging.
        live: If True, synchronize with wall-clock time.
    """

    def __init__(
        self,
        datacenter: DatacenterBackend[DCStateT],
        grid: GridBackend[GridStateT],
        controllers: Sequence[Controller[Any, Any]],
        total_duration_s: int,
        dc_bus: str,
        live: bool = False,
    ) -> None:
        self.datacenter = datacenter
        self.grid = grid
        self.controllers = list(controllers)
        self.total_duration_s = int(total_duration_s)
        self.dc_bus = str(dc_bus)

        # Compute tick as GCD of all component periods
        periods = [datacenter.dt_s, grid.dt_s] + [c.dt_s for c in controllers]
        tick = periods[0]
        for p in periods[1:]:
            tick = _gcd_fraction(tick, p)
        logger.info("Coordinator will run with tick %f s", float(tick))

        # Warn about potentially problematic dt configurations
        if grid.dt_s < datacenter.dt_s:
            warnings.warn(
                f"dt_grid ({grid.dt_s}) < dt_dc ({datacenter.dt_s}): "
                f"grid steps between DC steps will reuse the most recent DC power.",
                stacklevel=2,
            )
        for ctrl in controllers:
            if ctrl.dt_s < grid.dt_s:
                warnings.warn(
                    f"Controller {ctrl.__class__.__name__} dt_s ({ctrl.dt_s}) "
                    f"< dt_grid ({grid.dt_s}): controller may read stale voltages.",
                    stacklevel=2,
                )
        n_ticks_estimate = Fraction(self.total_duration_s) / tick
        if n_ticks_estimate > 10_000_000:
            warnings.warn(
                f"Simulation will run {int(n_ticks_estimate)} ticks. This may be slow. Consider coarser time steps.",
                stacklevel=2,
            )

        self.clock = SimulationClock(tick_s=tick, live=live)

    def reset(self) -> None:
        """Reset coordinator and all sub-components for a fresh run."""
        self.clock.reset()
        self.datacenter.do_reset()
        self.grid.do_reset()
        for ctrl in self.controllers:
            ctrl.reset()

    def start(self) -> None:
        """Acquire resources on all sub-components."""
        self.datacenter.start()
        self.grid.start()
        for ctrl in self.controllers:
            ctrl.start()

    def stop(self) -> None:
        """Release resources on all sub-components (LIFO order)."""
        for ctrl in reversed(self.controllers):
            ctrl.stop()
        self.grid.stop()
        self.datacenter.stop()

    def _validate_controller_compatibility(self) -> None:
        for ctrl in self.controllers:
            sig = ctrl.__class__.compatibility_signature()

            dc_types = ctrl.compatible_datacenter_types()
            try:
                dc_ok = isinstance(self.datacenter, dc_types)
            except TypeError:
                continue
            if not dc_ok:
                expected = " | ".join(t.__name__ for t in dc_types)
                got = type(self.datacenter).__name__
                raise TypeError(f"{ctrl.__class__.__name__} ({sig}) requires datacenter type {expected}, got {got}.")

            grid_types = ctrl.compatible_grid_types()
            try:
                grid_ok = isinstance(self.grid, grid_types)
            except TypeError:
                continue
            if not grid_ok:
                expected = " | ".join(t.__name__ for t in grid_types)
                got = type(self.grid).__name__
                raise TypeError(f"{ctrl.__class__.__name__} ({sig}) requires grid type {expected}, got {got}.")

    def run(self) -> SimulationLog[DCStateT, GridStateT]:
        """Run the full simulation and return the log."""
        log: SimulationLog[DCStateT, GridStateT] = SimulationLog()
        dc_events = EventEmitter(self.clock, log, "datacenter")
        grid_events = EventEmitter(self.clock, log, "grid")
        controller_events = EventEmitter(self.clock, log, "controller")

        self._validate_controller_compatibility()

        self.reset()
        self.start()

        dc_buffer: list[ThreePhase] = []

        ratio = Fraction(self.total_duration_s) / self.clock.tick_s
        if ratio.denominator != 1:
            raise ValueError(
                f"total_duration_s ({self.total_duration_s}) is not an exact multiple of tick_s ({self.clock.tick_s})"
            )
        n_ticks = int(ratio)

        logger.info(
            "Starting simulation: %d s, tick=%s s, %d ticks, dt_dc=%s s, dt_grid=%s s, %d controller(s)",
            self.total_duration_s,
            self.clock.tick_s,
            n_ticks,
            self.datacenter.dt_s,
            self.grid.dt_s,
            len(self.controllers),
        )

        try:
            for _ in range(n_ticks):
                # 1. Datacenter step (if due)
                if self.clock.is_due(self.datacenter.dt_s):
                    dc_state = self.datacenter.do_step(self.clock, dc_events)
                    dc_buffer.append(dc_state.power_w)
                    log.record_datacenter(dc_state)

                # 2. Grid step (if due). Pass full sub-trace since last grid step.
                if self.clock.is_due(self.grid.dt_s):
                    grid_state = self.grid.do_step(self.clock, list(dc_buffer), grid_events)
                    dc_buffer.clear()
                    log.record_grid(grid_state, dc_bus=self.dc_bus)

                # 3. Controllers (if due). In order, actions applied immediately.
                for ctrl in self.controllers:
                    if self.clock.is_due(ctrl.dt_s):
                        commands = ctrl.step(self.clock, self.datacenter, self.grid, controller_events)
                        for command in commands:
                            if isinstance(command, DatacenterCommand):
<<<<<<< HEAD
                                self.datacenter.apply_control(command, dc_events)
=======
                                # Route to target site if specified, else controller's default DC
                                target_site = getattr(command, "target_site_id", None)
                                target_dc = self._datacenters[target_site] if target_site else dc
                                target_dc.apply_control(command, dc_events)
>>>>>>> f03cf6c (Add multi-datacenter architecture: Coordinator accepts multiple DCs. Add functions to sweep ofo parameters, sweep DC locations, find DC hosting capacity, and optimize PV locations and capacities. Add IEEE 13, 34, 123 test feeders and example scripts. Include simulation outputs for IEEE 13, 34, 123 under multiple scenarios.)
                            elif isinstance(command, GridCommand):
                                self.grid.apply_control(command, grid_events)
                            else:
                                raise ValueError(f"Unsupported command type: {type(command).__name__}")
                        log.record_commands(commands)

                self.clock.advance()
        finally:
            self.stop()

        logger.info(
            "Simulation complete: %d grid steps, %d DC steps, %d commands",
            len(log.grid_states),
            len(log.dc_states),
            len(log.commands),
        )
        return log
