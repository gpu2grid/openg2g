# Architecture

This page describes how the components of OpenG2G fit together. For the underlying optimization formulation, see the [GPU-to-Grid paper](https://arxiv.org/abs/2602.05116).

## Simulation Loop

The [`Coordinator`][openg2g.coordinator.Coordinator] drives the simulation. It computes a base tick as the GCD of all component periods and advances a [`SimulationClock`][openg2g.clock.SimulationClock] each tick. See the [overview diagram](../index.md#overview) for the high-level component layout.

The pseudocode for each tick:

```
for each tick:
    1. if datacenter is due:  dc_state = datacenter.step(clock)
    2. if grid is due:        grid_state = grid.step(clock, dc_buffer)
    3. for each controller:
       if controller is due:  action = controller.step(clock, datacenter, grid, events)
                              apply action to datacenter and/or grid
```

The clock uses integer tick counting to avoid floating-point drift. Components check `clock.is_due(period)` to determine if they should run.

### What Happens in One Tick

Zooming into a sequence of coordinator ticks (DC at 0.1 s, grid and controller at 1.0 s):

```
  t = 5.0 s                    t = 5.1 s
  │                             │
  ├─ DC step                    ├─ DC step
  │  └─ Return power sample     │  └─ Return power sample
  │     (3-phase kW + ITL)      │     ...
  │                             │
  ├─ Grid step? ── NO           ├─ Grid step? ── NO
  │  (grid runs at 1.0 s)       │
  │                             │
  ├─ Controller step? ── NO     ├─ Controller step? ── NO
  │  (ctrl runs at 1.0 s)       │
  │                             │
  │  Accumulate in dc_buffer    │  Accumulate in dc_buffer

  ...

  t = 6.0 s
  │
  ├─ DC step
  │  └─ Return power sample
  │
  ├─ Grid step? ── YES (due at 6.0 s)
  │  ├─ Receives 10 power samples from dc_buffer
  │  ├─ Resamples to 2 DSS points via interpolation
  │  ├─ Runs 2 OpenDSS power flow solves
  │  └─ Returns bus voltages
  │
  ├─ Controller step? ── YES (due at 6.0 s)
  │  ├─ Reads voltages from grid
  │  ├─ Reads ITL, replica counts from datacenter
  │  ├─ Updates voltage & latency dual variables
  │  ├─ Gradient descent on batch sizes (log2 space)
  │  └─ Issues SetBatchSize command → datacenter
  │
  └─ Clear dc_buffer, save last power for next interval
```

## Component Interfaces

Each component type has an abstract base class (ABC) in `openg2g`. For full typed code examples, see [Custom Components](custom-components.md).

### DatacenterBackend

Defined in [`openg2g.datacenter.base`][openg2g.datacenter.base.DatacenterBackend]. Key methods:

- `dt_s`: The component's timestep
- `step(clock)`: Produce one power sample (returns a [`DatacenterState`][openg2g.datacenter.base.DatacenterState] containing three-phase power in watts)
- `apply_control(command)`: Accept a command (e.g., [`SetBatchSize`][openg2g.types.SetBatchSize])
- `state` / `history(n)`: Current and past states, readable by controllers

The coordinator accumulates power samples in a buffer and flushes them to the grid at each grid step.

Two implementations ship with OpenG2G:

- **[`OfflineDatacenter`][openg2g.datacenter.offline.OfflineDatacenter]** replays pre-recorded GPU power traces with configurable noise, jitter, ramp profiles, and training overlays.
- **[`OnlineDatacenter`][openg2g.datacenter.online.OnlineDatacenter]** reads live GPU power via Zeus and dispatches batch size changes through a callback.

### GridBackend / OpenDSSGrid

Defined in [`openg2g.grid.base`][openg2g.grid.base.GridBackend] (implemented by [`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid]). Key methods:

- `dt_s`: The grid solver's timestep
- `step(clock, power_samples_w)`: Run power flow on accumulated DC power samples, return per-bus per-phase voltages
- `apply_control(command)`: Accept a command (e.g., [`SetTaps`][openg2g.types.SetTaps])
- `voltages_vector()`: Flat numpy array of all bus-phase voltages (used by the OFO controller)
- `estimate_sensitivity(perturbation_kw)`: Finite-difference estimate of the voltage sensitivity matrix dV/dP

When the grid runs at a coarser rate than the datacenter, it internally resamples the accumulated power buffer via interpolation.

### Controller

Defined in [`openg2g.controller.base`][openg2g.controller.base.Controller]. Key methods:

- `dt_s`: The control interval
- `step(clock, datacenter, grid, events)`: Read state, compute a [`ControlAction`][openg2g.types.ControlAction], return it

Controllers receive the full datacenter and grid backend objects, so they can read `datacenter.state`, `grid.state`, and their histories. They also receive an event emitter for logging. Multiple controllers compose in order within the coordinator.

Built-in command types:

- [`SetBatchSize`][openg2g.types.SetBatchSize]`(batch_size_by_model=...)`: Datacenter command
- [`SetTaps`][openg2g.types.SetTaps]`(tap_position=...)`: Grid command

Commands are typed dataclasses dispatched via `singledispatchmethod`. Backends raise `TypeError` for unsupported command types.

## Component Lifecycle

Each component follows a defined lifecycle managed by the coordinator:

```
__init__() ──> reset() ──> start() ──> step() / apply_control() ──> stop()
                 ^                                                     │
                 └─────────────── (repeat from reset) ─────────────────┘
```

What belongs in each method:

- **`__init__()`**: Store configuration and do expensive one-time setup that is reusable across runs (e.g., build power templates, parse config). Does NOT acquire per-run resources.
- **`reset()`**: Clear simulation state (history, counters, RNG seeds, cached values). Configuration is not affected.
- **`start()`**: Acquire per-run resources (compile DSS circuits, start threads). No-op by default.
- **`stop()`**: Release per-run resources. State is preserved for post-run inspection. No-op by default.

The coordinator sequences these for every `run()` call: `reset()` all -> `start()` all -> simulation loop -> `stop()` all. This means calling `run()` twice on the same coordinator produces identical results.

### Reuse pattern

Expensive objects (like [`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid], which compiles a DSS circuit) can be reused across configuration sweeps:

```python
grid = OpenDSSGrid(...)  # stores config only (cheap)
for config in sweep_configs:
    dc = OfflineDatacenter(**config)  # builds power templates (expensive, reusable)
    ctrl = OFOBatchController(...)
    coord = Coordinator(dc, grid, [ctrl], total_duration_s=3600, dc_bus="671")
    log = coord.run()  # reset -> start (compile DSS) -> loop -> stop
```

## The Datacenter Model

The [`OfflineDatacenter`][openg2g.datacenter.offline.OfflineDatacenter] replays real GPU power traces at controlled batch sizes (see Section IV-A of the [paper](https://arxiv.org/abs/2602.05116)):

```
  Per-model server fleet                Power assembly (3-phase)

  ┌─────────────────────┐              Phase A     Phase B     Phase C
  │ Llama-3.1-8B        │                │           │           │
  │  48 servers × 8 GPU │──┐             │  ┌─────┐  │  ┌─────┐  │  ┌─────┐
  │  batch = 256        │  │             ├──│srv 1│  ├──│srv 2│  ├──│srv 3│
  ├─────────────────────┤  │             │  └─────┘  │  └─────┘  │  └─────┘
  │ Llama-3.1-70B       │  │             │  ┌─────┐  │           │  ┌─────┐
  │  30 servers × 8 GPU │──┤  sum kW     ├──│srv 4│  │           ├──│srv 6│
  │  batch = 128        │  │──per phase─>│  └─────┘  │           │  └─────┘
  ├─────────────────────┤  │             │    ...    │    ...    │    ...
  │ Llama-3.1-405B      │  │             │           │           │
  │  16 servers × 8 GPU │──┤             │           │           │
  │  batch = 64         │  │             │  + training overlay   │
  ├─────────────────────┤  │             │  + noise + jitter     │
  │ (+ 2 MoE models)    │──┘             │                       │
  └─────────────────────┘                v           v           v
                                       P_A(t)     P_B(t)     P_C(t)
```

- Each server plays back a per-GPU power trace (from [ML.ENERGY Benchmark](https://ml.energy/data) data) scaled by GPU count
- Random restart offsets make servers desynchronized (realistic)
- An [`ActivationStrategy`][openg2g.datacenter.layout.ActivationStrategy] determines which servers are active at each timestep, supporting both ramp-up and ramp-down schedules. The default [`RampActivationStrategy`][openg2g.datacenter.layout.RampActivationStrategy] follows a [`ServerRampSchedule`][openg2g.datacenter.config.ServerRampSchedule] with random priority ordering. Custom strategies (e.g., phase-aware load balancing) can be implemented by subclassing [`ActivationStrategy`][openg2g.datacenter.layout.ActivationStrategy].
- Training workload overlays add transient high-power phases

## The OFO Controller

Online Feedback Optimization (primal-dual) regulates batch sizes to keep voltages safe. For the full mathematical formulation, see Section III of the [paper](https://arxiv.org/abs/2602.05116).

```
  ┌──────────────────────────────────────────────────────────────┐
  │                  OFO Controller (every 1 s)                  │
  │                                                              │
  │  INPUTS:                                                     │
  │    V(t)  ← grid voltages (all bus-phase pairs)               │
  │    P(t)  ← datacenter power                                  │
  │    ITL(t) ← observed inter-token latency per model           │
  │    H     ← voltage sensitivity dV/dP (re-estimated slowly)   │
  │                                                              │
  │  DUAL UPDATES (Eq. 5-7):                                    │
  │                                                              │
  │    Voltage:  λ⁺ ← [λ⁺ + ρ_v (V - V_max)]⁺                  │
  │              λ⁻ ← [λ⁻ + ρ_v (V_min - V)]⁺                  │
  │              η  = λ⁺ - λ⁻                                    │
  │                                                              │
  │    Latency:  μ_i ← [μ_i + ρ_l (ITL_i - L_thresh)]⁺         │
  │                                                              │
  │  PRIMAL UPDATE (Eq. 8):                                     │
  │                                                              │
  │    x_i = log₂(batch_i)                                      │
  │                                                              │
  │    ∇_i = w_L · dL/dx           (latency penalty)            │
  │         - w_T · dTh/dx         (throughput reward)           │
  │         + ηᵀ H eᵢ · dP/dx     (voltage dual × sensitivity)  │
  │         + μ_i · dL/dx          (latency dual)               │
  │         + w_S · (x - x_prev)   (switching cost)             │
  │                                                              │
  │    x_new = project(x - ρ_x · ∇)                             │
  │    batch_new = nearest_valid(2^x_new)                        │
  │                                                              │
  │  OUTPUT:                                                     │
  │    {model: batch_new} → sent as SetBatchSize to datacenter   │
  └──────────────────────────────────────────────────────────────┘

  Key: dP/dx, dL/dx, dTh/dx come from LogisticModel fits
       H comes from OpenDSS finite-difference perturbation
       Full gradient derivation: Appendix B (Eq. 18) of the paper
```

## Data Flow

For details on how benchmark data flows into the simulation, see the [Data Pipeline](data-pipeline.md) page. In brief:

1. At build time, GPU benchmark data is processed into CSV artifacts (power traces, logistic fits, latency fits).
2. At simulation time, the datacenter replays power traces and samples latency from mixture models.
3. Power samples accumulate in a buffer between grid steps.
4. The grid runs power flow and returns bus voltages.
5. Controllers read datacenter/grid state, then emit control actions.

## State Types

All state objects are frozen dataclasses defined in [`openg2g.types`][openg2g.types]:

| Type | Fields | Source |
|---|---|---|
| [`ThreePhase`][openg2g.types.ThreePhase] | `a`, `b`, `c` | Everywhere |
| [`DatacenterState`][openg2g.datacenter.base.DatacenterState] | `time_s`, `power_w`, `batch_size_by_model`, `active_replicas_by_model`, `observed_itl_s_by_model` | `DatacenterBackend.step()` |
| [`OfflineDatacenterState`][openg2g.datacenter.offline.OfflineDatacenterState] | + `power_by_model_w` | `OfflineDatacenter.step()` |
| [`OnlineDatacenterState`][openg2g.datacenter.online.OnlineDatacenterState] | + `measured_power_w`, `measured_power_w_by_model`, `augmented_power_w_by_model`, `augmentation_factor_by_model` | `OnlineDatacenter.step()` |
| [`GridState`][openg2g.grid.base.GridState] | `time_s`, `voltages: BusVoltages`, `tap_positions: TapPosition \| None` | `GridBackend.step()` |
| [`SetBatchSize`][openg2g.types.SetBatchSize] | `batch_size_by_model`, `ramp_up_rate_by_model` | Controller -> datacenter |
| [`SetTaps`][openg2g.types.SetTaps] | `tap_position` | Controller -> grid |
| [`ControlAction`][openg2g.types.ControlAction] | `commands: list` | `Controller.step()` |

## SimulationLog

[`Coordinator.run()`][openg2g.coordinator.Coordinator.run] returns a [`SimulationLog`][openg2g.coordinator.SimulationLog] that collects all state history. Use `log.dc_states`, `log.grid_states`, and time-series arrays like `log.time_s`, `log.kW_A`, `log.voltage_a_pu` for analysis. See [Composing Components: Analyzing Results](composing.md#analyzing-results) for usage examples.
