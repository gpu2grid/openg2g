# Architecture

This page describes how the components of OpenG2G fit together.

## Overview

The core abstractions provided by OpenG2G are the multi-rate simulation loop ([`Coordinator`][openg2g.coordinator.Coordinator]) and interfaces for datacenter ([`DatacenterBackend`][openg2g.datacenter.base.DatacenterBackend]), grid ([`GridBackend`][openg2g.grid.base.GridBackend]), and controller ([`Controller`][openg2g.controller.base.Controller]) components.

```
                    ┌─────────────────────────────┐
                    │        Coordinator          │
                    │   (main simulation loop)    │
                    │                             │
                    │   tick = GCD of all rates   │
                    │   e.g., tick = 0.1 s        │
                    └──┬──────────┬──────────┬────┘
                       │          │          │
        every 0.1 s    │          │          │   every 1.0 s
      ┌────────────────┘          │          └────────────────┐
      v                      every 0.5 s                      v
┌───────────────┐                 │                  ┌───────────────────┐
│  Datacenter   │                 v                  │  Controller(s)    │
│  Backend      │         ┌────────────────┐         │                   │
│               │         │  Grid Backend  │         │  Read DC & grid   │
│  Produces:    │<─power─>│                │         │  state, compute   │
│  power load   │         │  Power flow    │         │  commands         │
│  latency      │         │  solver        │<──cmds──│                   │
│  throughput   │         └────────────────┘         │ e.g. SetBatchSize │
│               │<────────────────cmds───────────────│      SetTaps      │
└───────────────┘                                    └───────────────────┘
```

Controllers return a list of [`GridCommand`][openg2g.grid.command.GridCommand] (e.g., [`SetTaps`][openg2g.grid.command.SetTaps]) and/or [`DatacenterCommand`][openg2g.datacenter.command.DatacenterCommand] (e.g., [`SetBatchSize`][openg2g.datacenter.command.SetBatchSize]) that the coordinator dispatches to the appropriate backend before the next tick.
Multiple controllers run in sequence each control step, so their actions compose naturally.

## Simulation Loop

The [`Coordinator`][openg2g.coordinator.Coordinator] drives the simulation.
It computes a base tick as the GCD of all component periods and advances a [`SimulationClock`][openg2g.clock.SimulationClock] each tick.

```
for each tick:
  1. if datacenter is due:    dc_state = datacenter.do_step(clock, events)
  2. if grid is due:          grid_state = grid.do_step(clock, power_samples, events)
  3. for each controller:
       if controller is due:  action = controller.step(clock, datacenter, grid, events)
                              apply action to datacenter and/or grid
```

The period of each component (`dt_s`) is specified as a `fractions.Fraction` object in seconds (e.g., `Fraction(1, 10)` for 0.1 s), which allows for exact representation of intervals and GCD calculation without floating-point issues.
The coordinator checks `clock.is_due(component.dt_s)` to determine if they should run.

### What Happens in One Tick

Zooming into a sequence of coordinator ticks (DC at 0.1 s, grid and controller at 1.0 s):

```
  ...

  simulation clock = 5.1 s      simulation clock = 5.2 s
  │                             │
  ├─ DC step -- YES             ├─ DC step -- YES
  │  └─ Return power sample     │  └─ Return power sample
  │     (+ workload metrics)    │     (+ workload metrics)
  │                             │
  ├─ Grid step? -- NO           ├─ Grid step? -- NO
  │  (grid runs at 1.0 s)       │
  │                             │
  ├─ Controller step? -- NO     ├─ Controller step? -- NO
  │  (ctrl runs at 1.0 s)       │
  │                             │
  │  Accumulate power samples   │  Accumulate power samples

  ...

  simulation clock = 6.0 s
  │
  ├─ DC step -- YES
  │  └─ Return power sample
  │
  ├─ Grid step? -- YES
  │  ├─ Receives accumulated power samples
  │  ├─ Runs power flow
  │  └─ Returns bus voltages
  │
  ├─ Controller step? -- YES
  │  ├─ Reads datacenter and grid state (e.g., power, latency, voltage)
  │  ├─ Computes control action (e.g., SetBatchSize, SetTaps)
  │  └─ Issues commands -> datacenter and/or grid
  │
  └─ Clear accumulated power samples
```

### Live Mode

When `live=True` is passed to the [`Coordinator`][openg2g.coordinator.Coordinator], it instantiates [`SimulationClock`][openg2g.clock.SimulationClock] in live mode.
In this mode, the simulation clock synchronizes with the wall-clock.
That is, when [`clock.advance()`][openg2g.clock.SimulationClock.advance] is called, the clock checks how much time has elapsed since the last tick and sleeps for the remaining time until the next tick is due at the wall clock.
This enables hardware-in-the-loop experiments where the [`OnlineDatacenter`][openg2g.datacenter.online.OnlineDatacenter] reads live GPU power and the controller reacts in real time, while sharing most of the code path with offline simulation.

## Component Interfaces

Each component type has an abstract base class in `openg2g`. For full typed code examples and guidelines for extending, see [Writing Custom Components](building-simulators.md#writing-custom-components).

### `DatacenterBackend`

[`openg2g.datacenter.base.DatacenterBackend`][openg2g.datacenter.base.DatacenterBackend]. Key methods:

- [`dt_s`][openg2g.datacenter.base.DatacenterBackend.dt_s]: The component's timestep
- [`step(clock, events)`][openg2g.datacenter.base.DatacenterBackend.step]: Produce one power sample (returns a [`DatacenterState`][openg2g.datacenter.base.DatacenterState] containing three-phase power in watts)
- [`apply_control(command, events)`][openg2g.datacenter.base.DatacenterBackend.apply_control]: Accept a command (e.g., [`SetBatchSize`][openg2g.datacenter.command.SetBatchSize])
- [`state`][openg2g.datacenter.base.DatacenterBackend.state] / [`history(n)`][openg2g.datacenter.base.DatacenterBackend.history]: Current and past states (managed automatically by the base class), readable by controllers

Built-in implementations: [`OfflineDatacenter`][openg2g.datacenter.offline.OfflineDatacenter], [`OnlineDatacenter`][openg2g.datacenter.online.OnlineDatacenter].

### `GridBackend`

[`openg2g.grid.base.GridBackend`][openg2g.grid.base.GridBackend]. Key methods:

- [`dt_s`][openg2g.grid.base.GridBackend.dt_s]: The grid solver's timestep
- [`step(clock, power_samples_w, events)`][openg2g.grid.base.GridBackend.step]: Run power flow on accumulated DC power samples, return per-bus per-phase voltages
- [`apply_control(command, events)`][openg2g.grid.base.GridBackend.apply_control]: Accept a command (e.g., [`SetTaps`][openg2g.grid.command.SetTaps])

Built-in implementation: [`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid].

### `Controller`

[`openg2g.controller.base.Controller`][openg2g.controller.base.Controller]. Key methods:

- [`dt_s`][openg2g.controller.base.Controller.dt_s]: The control interval
- [`step(clock, datacenter, grid, events)`][openg2g.controller.base.Controller.step]: Read [`datacenter.state`][openg2g.datacenter.base.DatacenterBackend.state] and [`grid.state`][openg2g.grid.base.GridBackend.state], and return a list of commands to be applied to the datacenter and/or grid this tick

Built-in implementations: [`OFOBatchSizeController`][openg2g.controller.ofo.OFOBatchSizeController], [`TapScheduleController`][openg2g.controller.tap_schedule.TapScheduleController], [`BatchSizeScheduleController`][openg2g.controller.batch_size_schedule.BatchSizeScheduleController], [`NoopController`][openg2g.controller.noop.NoopController].

## Component Lifecycle

All components implement the following lifecycle methods:

- **`__init__`** (or any class method that instantiates the object): Store configuration and do expensive one-time setup that is reusable across runs (e.g., build power templates, parse config). Does *not* acquire per-run resources. Backend subclasses must call `super().__init__()` to initialize the base class's state and history tracking.
- **`reset()`**: Clear component-specific simulation state (counters, RNG seeds, cached values). Configuration is not affected. For backends, history is cleared automatically by the coordinator's `do_reset()` wrapper.
- **`start()`**: Acquire per-run resources (compile DSS circuits, start threads). No-op by default.
- **`stop()`**: Release per-run resources. State is preserved for post-run inspection. No-op by default.

After the component is instantiated, the coordinator calls `do_reset()` (which clears history and calls `reset()`), then `start()`, then enters the simulation loop where it calls `do_step()` and `apply_control()` as needed, and finally calls `stop()` at the end of the run.

```
__init__() ──> do_reset() ──> start() ──> do_step() / apply_control() ──> stop()
                 ^                                                     │
                 └─────────────── (repeat from reset) ─────────────────┘
```

This is mainly to allow reuse component objects across multiple [`Coordinator.run()`][openg2g.coordinator.Coordinator.run] calls with different configurations without having to re-instantiate all of them all the time:

```python
grid = OpenDSSGrid(...)                    # stores config only
ctrl = OFOBatchSizeController(...)         # stores fits + tuning
for workload in workloads:
    dc = OfflineDatacenter(dc_config, workload, dt_s=dt)  # builds power templates
    coord = Coordinator(dc, grid, [ctrl], total_duration_s=3600, dc_bus="671")
    log = coord.run()                      # reset -> start -> loop -> stop
```


## State Types and Generics

Different backends produce different state.
Every datacenter returns `time_s` and `power_w`, but an LLM inference datacenter also reports `batch_size_by_model` and `observed_itl_s_by_model`, and the offline backend further adds `power_by_model_w`.
Similarly, not every controller works with every backend: an OFO controller needs LLM-specific state that a generic datacenter doesn't provide.

OpenG2G uses Python generics to encode these relationships.
This serves two purposes:

- Incompatible pairings (e.g., an OFO batch size controller with a non-LLM datacenter) are caught at construction time with type errors rather than runtime crashes or, worse, silent bugs.
- Type information propagates through the system so that downstream objects like [`SimulationLog`][openg2g.coordinator.SimulationLog] carry the specific state types rather than overly generic ones. For instance, `log.dc_states` will be correctly recognized as `list[OfflineDatacenterState]` instead of the generic `list[DatacenterState]`, giving you access to LLM-specific fields in autocompletion and type checking.

**State inheritance.** State types form a hierarchy where each level adds domain-specific fields:

```
DatacenterState                 (time_s, power_w)
└── LLMDatacenterState          (+ batch_size_by_model, active_replicas_by_model, ...)
    └── OfflineDatacenterState  (+ power_by_model_w)
    └── OnlineDatacenterState   (+ measured_power_w, augmentation_factor_by_model, ...)
```

**Backend generics.** Each backend declares which state type it produces:

```python
class DatacenterBackend(Generic[DCStateT], ABC): ...
class LLMBatchSizeControlledDatacenter(DatacenterBackend[DCStateT]): ...
class OfflineDatacenter(LLMBatchSizeControlledDatacenter[OfflineDatacenterState]): ...
```

**Controller generics.** Controllers declare which backends they require. The coordinator validates these constraints when you construct it:

```python
class Controller(Generic[DCBackendT, GridBackendT], ABC): ...

# Works with any datacenter and any grid
class TapScheduleController(Controller[DatacenterBackend, GridBackend]): ...

# Requires an LLM datacenter and OpenDSS specifically
class OFOBatchSizeController(Controller[LLMBatchSizeControlledDatacenter, OpenDSSGrid]): ...
```

## `SimulationLog`

[`Coordinator.run()`][openg2g.coordinator.Coordinator.run] returns a [`SimulationLog`][openg2g.coordinator.SimulationLog] that collects all state history.
Essentially, this object is a bill of materials for everything that happened during the simulation, and is the basis for all post-run analysis and plotting.
You can use `log.dc_states`, `log.grid_states`, and time-series arrays like `log.time_s`, `log.voltage_a_pu` for analysis. Per-phase power is available from `dc_states[i].power_w`.
See [Building Simulators: Analyzing Results](building-simulators.md#analyzing-results) for usage examples.
