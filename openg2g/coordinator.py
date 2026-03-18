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
        dc_states: Every datacenter state produced by the datacenter (flat list, all sites).
        dc_states_by_site: Per-site datacenter states (multi-DC mode).
        grid_states: Every grid state produced by the grid.
        commands: All commands emitted by controllers.
        time_s: Simulation time at each grid step (seconds).
        voltage_a_pu: DC-bus voltage phase A at each grid step (pu).
        voltage_b_pu: DC-bus voltage phase B at each grid step (pu).
        voltage_c_pu: DC-bus voltage phase C at each grid step (pu).
        events: Clock-stamped simulation events from all components.
    """

    dc_states: list[DCStateT] = field(default_factory=list)
    dc_states_by_site: dict[str, list[DCStateT]] = field(default_factory=dict)
    grid_states: list[GridStateT] = field(default_factory=list)
    commands: list[DatacenterCommand | GridCommand] = field(default_factory=list)

    time_s: list[float] = field(default_factory=list)
    voltage_a_pu: list[float] = field(default_factory=list)
    voltage_b_pu: list[float] = field(default_factory=list)
    voltage_c_pu: list[float] = field(default_factory=list)

    events: list[SimEvent] = field(default_factory=list)

    def record_datacenter(self, state: DCStateT, *, site_id: str | None = None) -> None:
        """Append a datacenter state snapshot."""
        self.dc_states.append(state)
        if site_id is not None:
            self.dc_states_by_site.setdefault(site_id, []).append(state)

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

    Supports both single-DC and multi-DC modes:

    - **Single-DC**: Pass ``datacenter`` (a single backend).
    - **Multi-DC**: Pass ``datacenters`` (a dict mapping site IDs to backends).

    Args:
        datacenter: Single datacenter backend (legacy mode).
        datacenters: Dict of datacenter backends keyed by site ID (multi-DC).
        grid: Grid simulator backend.
        controllers: List of controllers, applied in order each tick.
        total_duration_s: Total simulation duration (integer seconds).
        dc_bus: Bus name for DC voltage logging.
        live: If True, synchronize with wall-clock time.
    """

    def __init__(
        self,
        datacenter: DatacenterBackend[DCStateT] | None = None,
        grid: GridBackend[GridStateT] | None = None,
        controllers: Sequence[Controller[Any, Any]] | None = None,
        total_duration_s: int = 0,
        dc_bus: str = "",
        live: bool = False,
        *,
        datacenters: dict[str, DatacenterBackend[DCStateT]] | None = None,
    ) -> None:
        # Build datacenters dict
        if datacenters is not None:
            self._datacenters = dict(datacenters)
        elif datacenter is not None:
            # Wrap single DC in a dict. Use "_single" as key; the grid
            # will receive a flat list (not a dict) in this mode.
            self._datacenters = {"_single": datacenter}
            self._single_dc_mode = True
        else:
            raise ValueError("Must provide either datacenter or datacenters.")

        if not hasattr(self, "_single_dc_mode"):
            self._single_dc_mode = False

        # Legacy single-DC property
        self.datacenter = next(iter(self._datacenters.values()))

        self.grid = grid
        self.controllers = list(controllers or [])
        self.total_duration_s = int(total_duration_s)
        self.dc_bus = str(dc_bus)

        # Compute tick as GCD of all component periods
        periods = [grid.dt_s] + [dc.dt_s for dc in self._datacenters.values()] + [c.dt_s for c in self.controllers]
        tick = periods[0]
        for p in periods[1:]:
            tick = _gcd_fraction(tick, p)
        logger.info("Coordinator will run with tick %f s", float(tick))

        # Warn about potentially problematic dt configurations
        for dc in self._datacenters.values():
            if grid.dt_s < dc.dt_s:
                warnings.warn(
                    f"dt_grid ({grid.dt_s}) < dt_dc ({dc.dt_s}): "
                    f"grid steps between DC steps will reuse the most recent DC power.",
                    stacklevel=2,
                )
        for ctrl in self.controllers:
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
        for dc in self._datacenters.values():
            dc.do_reset()
        self.grid.do_reset()
        for ctrl in self.controllers:
            ctrl.reset()

    def start(self) -> None:
        """Acquire resources on all sub-components."""
        for dc in self._datacenters.values():
            dc.start()
        self.grid.start()
        for ctrl in self.controllers:
            ctrl.start()

    def stop(self) -> None:
        """Release resources on all sub-components (LIFO order)."""
        for ctrl in reversed(self.controllers):
            ctrl.stop()
        self.grid.stop()
        for dc in self._datacenters.values():
            dc.stop()

    def _validate_controller_compatibility(self) -> None:
        for ctrl in self.controllers:
            sig = ctrl.__class__.compatibility_signature()

            dc_types = ctrl.compatible_datacenter_types()
            for dc in self._datacenters.values():
                try:
                    dc_ok = isinstance(dc, dc_types)
                except TypeError:
                    continue
                if not dc_ok:
                    expected = " | ".join(t.__name__ for t in dc_types)
                    got = type(dc).__name__
                    raise TypeError(
                        f"{ctrl.__class__.__name__} ({sig}) requires datacenter type {expected}, got {got}."
                    )

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

        # Per-site power buffers
        dc_buffers: dict[str, list[ThreePhase]] = {sid: [] for sid in self._datacenters}

        ratio = Fraction(self.total_duration_s) / self.clock.tick_s
        if ratio.denominator != 1:
            raise ValueError(
                f"total_duration_s ({self.total_duration_s}) is not an exact multiple of tick_s ({self.clock.tick_s})"
            )
        n_ticks = int(ratio)

        logger.info(
            "Starting simulation: %d s, tick=%s s, %d ticks, %d DC site(s), dt_grid=%s s, %d controller(s)",
            self.total_duration_s,
            self.clock.tick_s,
            n_ticks,
            len(self._datacenters),
            self.grid.dt_s,
            len(self.controllers),
        )

        try:
            for _ in range(n_ticks):
                # 1. Datacenter steps (if due)
                for site_id, dc in self._datacenters.items():
                    if self.clock.is_due(dc.dt_s):
                        dc_state = dc.do_step(self.clock, dc_events)
                        dc_buffers[site_id].append(dc_state.power_w)
                        log.record_datacenter(dc_state, site_id=site_id)

                # 2. Grid step (if due). Pass full sub-trace since last grid step.
                if self.clock.is_due(self.grid.dt_s):
                    if self._single_dc_mode:
                        # Single-DC: pass flat list for backward compatibility
                        power_arg = list(next(iter(dc_buffers.values())))
                    else:
                        # Multi-DC: pass dict keyed by site ID
                        power_arg = {sid: list(buf) for sid, buf in dc_buffers.items()}
                    grid_state = self.grid.do_step(self.clock, power_arg, grid_events)
                    for buf in dc_buffers.values():
                        buf.clear()
                    log.record_grid(grid_state, dc_bus=self.dc_bus)

                # 3. Controllers (if due). In order, actions applied immediately.
                for ctrl in self.controllers:
                    if self.clock.is_due(ctrl.dt_s):
                        # Route to the correct datacenter if the controller has a site_id
                        ctrl_site_id = getattr(ctrl, "_site_id", None)
                        ctrl_dc = (
                            self._datacenters.get(ctrl_site_id, self.datacenter) if ctrl_site_id else self.datacenter
                        )
                        commands = ctrl.step(self.clock, ctrl_dc, self.grid, controller_events)
                        for command in commands:
                            if isinstance(command, DatacenterCommand):
                                target_site = getattr(command, "target_site_id", None)
                                if target_site and target_site in self._datacenters:
                                    self._datacenters[target_site].apply_control(command, dc_events)
                                else:
                                    self.datacenter.apply_control(command, dc_events)
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
