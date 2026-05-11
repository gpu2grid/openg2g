"""Central coordinator: multi-rate simulation loop."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Generic

from openg2g.clock import SimulationClock
from openg2g.common import ThreePhase
from openg2g.controller.base import Controller
from openg2g.datacenter.base import DatacenterBackend, DCStateT
from openg2g.datacenter.command import DatacenterCommand
from openg2g.events import EventEmitter, SimEvent
from openg2g.grid.base import GridBackend, GridStateT
from openg2g.grid.command import GridCommand

logger = logging.getLogger(__name__)


@dataclass
class SimulationLog(Generic[DCStateT, GridStateT]):
    """Accumulated simulation data from a coordinator run.

    Attributes:
        dc_states: Every datacenter state (flat list, all sites).
        dc_states_by_site: Per-site datacenter states keyed by DC name.
        grid_states: Every grid state produced by the grid.
        commands: All commands emitted by controllers.
        time_s: Simulation time at each grid step (seconds).
        events: Clock-stamped simulation events from all components.
    """

    dc_states: list[DCStateT] = field(default_factory=list)
    dc_states_by_site: dict[str, list[DCStateT]] = field(default_factory=dict)
    grid_states: list[GridStateT] = field(default_factory=list)
    commands: list[DatacenterCommand | GridCommand] = field(default_factory=list)

    time_s: list[float] = field(default_factory=list)

    events: list[SimEvent] = field(default_factory=list)

    def record_datacenter(self, state: DCStateT, *, dc_name: str) -> None:
        """Append a datacenter state snapshot."""
        self.dc_states.append(state)
        self.dc_states_by_site.setdefault(dc_name, []).append(state)

    def record_grid(self, state: GridStateT) -> None:
        """Append a grid state snapshot."""
        self.grid_states.append(state)
        self.time_s.append(state.time_s)

    def record_commands(self, commands: list[DatacenterCommand | GridCommand]) -> None:
        """Append control commands issued during a tick."""
        self.commands.extend(commands)

    def emit(self, event: SimEvent) -> None:
        """Event sink entrypoint for component-originated events."""
        self.events.append(event)


@dataclass(frozen=True)
class TickOutput(Generic[DCStateT, GridStateT]):
    """Outputs produced by one base tick of [`Coordinator.step`][..Coordinator.step].

    Empty-or-`None` fields mean the corresponding component did not step on
    this tick. The base tick is the GCD of all component periods, so on any
    given tick only the components whose `dt_s` is currently due will appear.

    Attributes:
        t_s: Simulation time at the start of this tick (seconds).
        dc_states: Datacenter states produced this tick, keyed by DC name.
            Empty if no DC was due.
        grid_state: Grid state produced this tick, or `None` if the grid was
            not due.
        commands: Commands emitted by controllers and dispatched during this
            tick (in the order they fired).
        sim_events: `SimEvent`s emitted by all components during this tick,
            in chronological order.
    """

    t_s: float
    dc_states: dict[str, DCStateT]
    grid_state: GridStateT | None
    commands: list[DatacenterCommand | GridCommand]
    sim_events: list[SimEvent]


class _EventSink:
    """Per-run sink that EventEmitter writes into.

    Drained at the end of each [`Coordinator.step`][..Coordinator.step] so
    the resulting `TickOutput.sim_events` reflects only events emitted
    during that single tick.
    """

    def __init__(self) -> None:
        self._events: list[SimEvent] = []

    def emit(self, event: SimEvent) -> None:
        self._events.append(event)

    def drain(self) -> list[SimEvent]:
        out = self._events
        self._events = []
        return out


def _gcd_fraction(a: Fraction, b: Fraction) -> Fraction:
    """GCD of two positive Fractions using Euclidean algorithm."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


class Coordinator(Generic[DCStateT, GridStateT]):
    """Multi-rate simulation coordinator.

    Orchestrates datacenter, grid, and controller components at their
    respective rates. The base tick is the GCD of all component periods.

    Args:
        datacenters: List of datacenter backends.
        grid: Grid simulator backend.
        controllers: List of controllers, applied in order each tick.
        total_duration_s: Total simulation duration (integer seconds).
        live: If True, synchronize with wall-clock time.
    """

    def __init__(
        self,
        *,
        datacenters: Sequence[DatacenterBackend[DCStateT]],
        grid: GridBackend[GridStateT],
        controllers: Sequence[Controller[Any, Any]] | None = None,
        total_duration_s: int = 0,
        live: bool = False,
    ) -> None:
        self._datacenters = list(datacenters)
        if not self._datacenters:
            raise ValueError("At least one datacenter is required.")

        # Validate unique DC names
        names = [dc.name for dc in self._datacenters]
        if len(names) != len(set(names)):
            dupes = sorted(n for n in names if names.count(n) > 1)
            raise ValueError(f"Datacenter names must be unique. Duplicates: {dupes}")

        self.grid = grid
        self.controllers: list[Controller] = list(controllers or [])
        self.total_duration_s = int(total_duration_s)

        # Compute tick as GCD of all component periods
        periods = [grid.dt_s] + [dc.dt_s for dc in self._datacenters] + [c.dt_s for c in self.controllers]
        tick = periods[0]
        for p in periods[1:]:
            tick = _gcd_fraction(tick, p)
        logger.info("Coordinator will run with tick %f s", float(tick))

        # Warn about potentially problematic dt configurations
        for dc in self._datacenters:
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

        self._validate_controller_compatibility()

        # Per-run event scaffolding. Initialised by start(), torn down by stop().
        self._sink: _EventSink | None = None
        self._dc_events: EventEmitter | None = None
        self._grid_events: EventEmitter | None = None
        self._controller_events: EventEmitter | None = None
        self._dc_buffers: dict[DatacenterBackend[DCStateT], list[ThreePhase]] = {}

    def reset(self) -> None:
        """Reset coordinator and all sub-components for a fresh run."""
        self.clock.reset()
        for dc in self._datacenters:
            dc.do_reset()
        self.grid.do_reset()
        for ctrl in self.controllers:
            ctrl.reset()

    def start(self) -> None:
        """Acquire resources on all sub-components and prepare event routing.

        Must be called before [`step`][..Coordinator.step] or
        [`dispatch_commands`][..Coordinator.dispatch_commands].
        """
        for dc in self._datacenters:
            dc.start()
        self.grid.start()
        for ctrl in self.controllers:
            ctrl.start()
        self._sink = _EventSink()
        self._dc_events = EventEmitter(self.clock, self._sink, "datacenter")
        self._grid_events = EventEmitter(self.clock, self._sink, "grid")
        self._controller_events = EventEmitter(self.clock, self._sink, "controller")
        self._dc_buffers = {dc: [] for dc in self._datacenters}

    def stop(self) -> None:
        """Release resources on all sub-components (LIFO order)."""
        for ctrl in reversed(self.controllers):
            ctrl.stop()
        self.grid.stop()
        for dc in self._datacenters:
            dc.stop()
        self._sink = None
        self._dc_events = None
        self._grid_events = None
        self._controller_events = None
        self._dc_buffers = {}

    def _validate_controller_compatibility(self) -> None:
        for ctrl in self.controllers:
            sig = ctrl.__class__.compatibility_signature()

            dc_types = ctrl.compatible_datacenter_types()
            for dc in self._datacenters:
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

    def step(self) -> TickOutput[DCStateT, GridStateT]:
        """Advance the simulation by one base tick and return its outputs.

        The caller is responsible for the surrounding lifecycle: invoke
        [`reset`][..Coordinator.reset] and [`start`][..Coordinator.start]
        before the first call, and [`stop`][..Coordinator.stop] after the
        last. Iterating with `run_iter()` (`for tick in coord.run_iter()`) handles this
        automatically; [`run`][..Coordinator.run] is a thin wrapper that
        accumulates every `TickOutput` into a `SimulationLog`.

        Each tick:

        1. Step every datacenter whose `dt_s` is due.
        2. Step the grid if its `dt_s` is due, passing the per-DC power
           sub-trace accumulated since the previous grid step.
        3. Step every controller whose `dt_s` is due, in registration order,
           and dispatch its commands immediately via
           [`dispatch_commands`][..Coordinator.dispatch_commands].
        4. Drain `SimEvent`s emitted during the tick into the returned
           `TickOutput`, then advance the clock.
        """
        sink = self._sink
        dc_events = self._dc_events
        grid_events = self._grid_events
        controller_events = self._controller_events
        if sink is None or dc_events is None or grid_events is None or controller_events is None:
            raise RuntimeError("Coordinator must be started before step() can be called.")

        dc_states_this_tick: dict[str, DCStateT] = {}
        grid_state_this_tick: GridStateT | None = None
        commands_this_tick: list[DatacenterCommand | GridCommand] = []

        for dc in self._datacenters:
            if self.clock.is_due(dc.dt_s):
                dc_state = dc.do_step(self.clock, dc_events)
                self._dc_buffers[dc].append(dc_state.power_w)
                dc_states_this_tick[dc.name] = dc_state

        if self.clock.is_due(self.grid.dt_s):
            power_arg = {dc: list(buf) for dc, buf in self._dc_buffers.items()}
            grid_state_this_tick = self.grid.do_step(self.clock, power_arg, grid_events)
            for buf in self._dc_buffers.values():
                buf.clear()

        for ctrl in self.controllers:
            if self.clock.is_due(ctrl.dt_s):
                cmds = ctrl.step(self.clock, controller_events)
                self.dispatch_commands(cmds)
                commands_this_tick.extend(cmds)

        t_s = float(self.clock.time_s)
        sim_events = sink.drain()
        self.clock.advance()

        return TickOutput(
            t_s=t_s,
            dc_states=dc_states_this_tick,
            grid_state=grid_state_this_tick,
            commands=commands_this_tick,
            sim_events=sim_events,
        )

    def dispatch_commands(
        self,
        commands: Iterable[DatacenterCommand | GridCommand],
    ) -> None:
        """Route commands to the grid or to their target datacenter.

        Uses the same dispatch rules as the in-loop controller block: a
        `DatacenterCommand` is delivered to `command.target` (which must be
        set), and a `GridCommand` goes to the coordinator's grid. External
        drivers (e.g. an RL training environment) call this between
        [`step`][..Coordinator.step] invocations to inject control actions.
        """
        if self._dc_events is None or self._grid_events is None:
            raise RuntimeError("Coordinator must be started before dispatch_commands() can be called.")
        for command in commands:
            if isinstance(command, DatacenterCommand):
                command.target.apply_control(command, self._dc_events)
            elif isinstance(command, GridCommand):
                self.grid.apply_control(command, self._grid_events)
            else:
                raise ValueError(f"Unsupported command type: {type(command).__name__}")

    def run_iter(self) -> Iterator[TickOutput[DCStateT, GridStateT]]:
        """Yield one `TickOutput` per base tick until `total_duration_s`.

        Manages the [`reset`][..Coordinator.reset] /
        [`start`][..Coordinator.start] / [`stop`][..Coordinator.stop]
        lifecycle automatically, including on early termination (e.g. when
        the consumer breaks out of the loop). For fully external drivers
        that need per-tick control (e.g. an RL Gym env, an interactive
        viewer), use [`step`][..Coordinator.step] directly and manage the
        lifecycle by hand.
        """
        ratio = Fraction(self.total_duration_s) / self.clock.tick_s
        if ratio.denominator != 1:
            raise ValueError(
                f"total_duration_s ({self.total_duration_s}) is not an exact multiple of tick_s ({self.clock.tick_s})"
            )
        n_ticks = int(ratio)

        self.reset()
        self.start()
        try:
            for _ in range(n_ticks):
                yield self.step()
        finally:
            self.stop()

    def run(self) -> SimulationLog[DCStateT, GridStateT]:
        """Run the full simulation end-to-end and return the accumulated log."""
        log: SimulationLog[DCStateT, GridStateT] = SimulationLog()
        logger.info(
            "Starting simulation: %d s, tick=%s s, %d DC site(s), dt_grid=%s s, %d controller(s)",
            self.total_duration_s,
            self.clock.tick_s,
            len(self._datacenters),
            self.grid.dt_s,
            len(self.controllers),
        )

        for tick in self.run_iter():
            for dc_name, dc_state in tick.dc_states.items():
                log.record_datacenter(dc_state, dc_name=dc_name)
            if tick.grid_state is not None:
                log.record_grid(tick.grid_state)
            log.record_commands(tick.commands)
            log.events.extend(tick.sim_events)

        logger.info(
            "Simulation complete: %d grid steps, %d DC steps, %d commands",
            len(log.grid_states),
            len(log.dc_states),
            len(log.commands),
        )
        return log
