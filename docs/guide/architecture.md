# Architecture

This page describes how the components of OpenG2G fit together. For the underlying optimization formulation, see the [GPU-to-Grid paper](https://arxiv.org/abs/2602.05116).

## Simulation Loop

The `Coordinator` drives the simulation. It computes a base tick as the GCD of all component periods and advances a `SimulationClock` each tick. At each tick, it checks which components are due and dispatches accordingly:

```
                        ┌─────────────────────────────┐
                        │        Coordinator          │
                        │   (main simulation loop)    │
                        │                             │
                        │   tick = GCD of all rates   │
                        │   e.g. tick = 0.1 s         │
                        └──┬──────────┬──────────┬────┘
                           │          │          │
            every 0.1 s    │          │          │   every 1.0 s
          ┌────────────────┘          │          └────────────────┐
          v                           │                           v
  ┌───────────────┐          every 1.0 s              ┌───────────────────┐
  │  Datacenter   │                   │               │    Controller     │
  │  (Offline)    │                   v               │    (OFO)          │
  │               │          ┌────────────────┐       │                   │
  │ Power traces  │──power──>│  OpenDSS Grid  │──V──> │ Primal-dual       │
  │ Latency       │   (kW)   │  (IEEE 13-bus) │       │ batch optimizer   │
  │ Replicas      │          │                │       │                   │
  │               │<─batch───│  Power flow    │       │ Reads: V, P, ITL  │
  └───────────────┘  update  │  solver        │       │ Writes: batch cmd │
                             └────────────────┘       └───────────────────┘
```

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
  │  └─ Issues "set_batch_size" command → datacenter
  │
  └─ Clear dc_buffer, save last power for next interval
```

## Component Interfaces

### DatacenterBackend

Defined in `openg2g.datacenter.base`. Generic over the state type (`DCStateT`) emitted by `step()`:

```python
class DatacenterBackend(Generic[DCStateT], ABC):
    @property
    def dt_s(self) -> Fraction: ...
    @property
    def state(self) -> DCStateT | None: ...
    def history(self, n: int | None = None) -> Sequence[DCStateT]: ...
    def step(self, clock: SimulationClock) -> DCStateT: ...
    def apply_control(self, command: Command) -> None: ...
```

The `step()` method returns a state dataclass (a `DatacenterState` subclass) containing three-phase power. The coordinator accumulates these into a buffer that is flushed to the grid at each grid step. The state type propagates through the coordinator to the simulation log.

Two implementations ship with OpenG2G:

- **`OfflineDatacenter`** (`DatacenterBackend[OfflineDatacenterState]`) replays pre-recorded GPU power traces with configurable noise, jitter, ramp profiles, and training overlays.
- **`OnlineDatacenter`** (`DatacenterBackend[OnlineDatacenterState]`) reads live GPU power via Zeus and dispatches batch size changes through a callback.

### GridBackend / OpenDSSGrid

Defined in `openg2g.grid.base` (implemented by `openg2g.grid.opendss.OpenDSSGrid`). Generic over the state type (`GridStateT`) emitted by `step()`:

```python
class GridBackend(Generic[GridStateT], ABC):
    @property
    def dt_s(self) -> Fraction: ...
    @property
    def state(self) -> GridStateT | None: ...
    def history(self, n: int | None = None) -> Sequence[GridStateT]: ...
    @property
    def v_index(self) -> list[tuple[str, int]]: ...
    def step(
        self,
        clock: SimulationClock,
        power_samples_w: list[ThreePhase],
    ) -> GridStateT: ...
    def apply_control(self, command: Command) -> None: ...
    def voltages_vector(self) -> np.ndarray: ...
    def estimate_sensitivity(self, perturbation_kw: float = 100.0) -> tuple[np.ndarray, np.ndarray]: ...
```

The grid receives a list of power samples accumulated since the last grid step. When the grid runs at a coarser rate than the datacenter, the grid internally caches the last sample from the previous step to prepend for interpolation.

The grid returns a state dataclass containing per-bus, per-phase voltages and (optionally) current regulator tap positions.

### Controller

Defined in `openg2g.controller.base`. Generic over compatible datacenter and grid backend types:

```python
class Controller(Generic[DCType, GridType], ABC):
    @property
    def dt_s(self) -> Fraction: ...
    def step(self, clock, datacenter, grid, events) -> ControlAction: ...
```

Controllers receive full datacenter/grid backend objects and a clock-bound event emitter. They return a `ControlAction` containing command envelopes. The generic parameters declare which backend types the controller is compatible with; the coordinator checks compatibility at construction time.

Current built-in command kinds:
- `target=CommandTarget.DATACENTER` (`"datacenter"` also accepted), `kind="set_batch_size"` with `payload["batch_size_by_model"]`
- `target=CommandTarget.GRID` (`"grid"` also accepted), `kind="set_taps"` with `payload["tap_changes"]`

Multiple controllers compose in order within the coordinator.

## Component Lifecycle

Each component follows a defined lifecycle managed by the coordinator:

```
__init__() ──> reset() ──> start() ──> step() / apply_control() ──> stop()
                 ^                                                     │
                 └─────────────── (repeat from reset) ─────────────────┘
```

What belongs in each method:

- **`__init__()`**: Store configuration and do expensive one-time setup that is reusable across runs (e.g., build power templates, parse config). Does NOT acquire per-run resources.
- **`reset()`**: Clear simulation state -- history, counters, RNG seeds, cached values. Configuration is not affected. **Abstract** on all ABCs.
- **`start()`**: Acquire per-run resources -- compile DSS circuits, start threads. **Concrete no-op** by default.
- **`stop()`**: Release per-run resources. State is preserved for post-run inspection. Concrete no-op by default.

The coordinator sequences these for every `run()` call: `reset()` all -> `start()` all -> simulation loop -> `stop()` all. This means calling `run()` twice on the same coordinator produces identical results.

`reset()` and `start()` are separate because they have different forcing properties. `reset()` is abstract: every implementation must explicitly enumerate what state it clears. A forgotten field silently corrupts the second run, so the compiler catches missing implementations. `start()` is a concrete no-op: most components (5 of 7) have state to clear but no resources to acquire, so they only need to implement `reset()`.

### Reuse pattern

Expensive objects (like `OpenDSSGrid`, which compiles a DSS circuit) can be reused across configuration sweeps:

```python
grid = OpenDSSGrid(...)  # stores config only (cheap)
for config in sweep_configs:
    dc = OfflineDatacenter(**config)  # builds power templates (expensive, reusable)
    ctrl = OFOBatchController(...)
    coord = Coordinator(dc, grid, [ctrl], total_duration_s=3600)
    log = coord.run()  # reset -> start (compile DSS) -> loop -> stop
```

## The Datacenter Model

The `OfflineDatacenter` replays real GPU power traces at controlled batch sizes (see Section IV-A of the [paper](https://arxiv.org/abs/2602.05116)):

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
- A `ServerActivationPolicy` determines which servers are active at each timestep, supporting both ramp-up and ramp-down schedules. The default `ScheduleActivationPolicy` follows a `ServerRampSchedule` with random priority ordering. Custom policies (e.g., phase-aware load balancing) can be implemented by subclassing `ServerActivationPolicy`.
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
  │    {model: batch_new} → sent as command to datacenter        │
  └──────────────────────────────────────────────────────────────┘

  Key: dP/dx, dL/dx, dTh/dx come from LogisticModel fits
       H comes from OpenDSS finite-difference perturbation
       Full gradient derivation: Appendix B (Eq. 18) of the paper
```

## Data Flow

```
  ┌─────────── BUILD TIME (once, offline) ──────────────────────┐
  │                                                              │
  │   mlenergy-data                     data/offline/            │
  │                                     build_mlenergy_data.py  │
  │   LLMRuns  ──filter/group──>  For each (model, batch):     │
  │                                 LogisticModel.fit()  → CSV  │
  │   timelines()  ──extract──>     ITLMixtureModel.fit() → CSV │
  │                                 power trace → CSV           │
  └──────────────────────────────────────────────────────────────┘
                                         │  CSVs on disk
                                         v
  ┌─────────── RUN TIME (every simulation) ─────────────────────┐
  │                                                              │
  │   OfflineDatacenter reads:                                   │
  │     traces/*.csv ──> TraceByBatchCache (power templates)     │
  │     latency_fits.csv ──> ITLMixtureModel.sample_avg()        │
  │                                                              │
  │   OFO Controller reads:                                      │
  │     logistic_fits.csv ──> LogisticModel.eval() / .deriv()    │
  │                           called every control step          │
  └──────────────────────────────────────────────────────────────┘
```

1. At build time, GPU benchmark data is processed into CSV artifacts (power traces, logistic fits, latency fits).
2. At simulation time, the datacenter replays power traces and samples latency from mixture models.
3. Power samples accumulate in a buffer between grid steps.
4. The grid runs power flow and returns bus voltages.
5. Controllers read datacenter/grid state, then emit control actions.
6. Control commands are routed to datacenter/grid targets.

For details on the data pipeline, see the [Data Pipeline](data-pipeline.md) page.

## State Types

All state objects are frozen dataclasses defined in `openg2g.types`:

| Type | Fields | Source |
|---|---|---|
| `ThreePhase` | `a`, `b`, `c` | Everywhere |
| `DatacenterState` | `time_s`, `power_w`, `batch_size_by_model`, `active_replicas_by_model`, `observed_itl_s_by_model` | `DatacenterBackend.step()` |
| `OfflineDatacenterState` | + `power_by_model_w` | `OfflineDatacenter.step()` |
| `OnlineDatacenterState` | + `measured_power_w`, `measured_power_w_by_model`, `augmented_power_w_by_model`, `augmentation_factor_by_model` | `OnlineDatacenter.step()` |
| `GridState` | `time_s`, `voltages: BusVoltages`, `tap_positions: TapPosition \| None` | `GridBackend.step()` |
| `Command` | `target`, `kind`, `payload`, `metadata` | `Controller.step()` |
| `ControlAction` | `commands: list[Command]` | `Controller.step()` |

Bounded TypeVars `DCStateT` and `GridStateT` (defined in `openg2g.types`) parameterize the ABCs and propagate through the coordinator to the simulation log.

## SimulationLog

The `Coordinator.run()` method returns a `SimulationLog[DCStateT, GridStateT]` that accumulates:

- All datacenter states, grid states, and control actions (typed by the backend's state type)
- Time-series arrays for DC bus voltages and per-phase power
- Per-model batch size history
- Clock-stamped events from all components

The state type parameters are inferred from the backends passed to the coordinator. For example, `Coordinator(datacenter=OfflineDatacenter(...), grid=OpenDSSGrid(...), ...)` returns a `SimulationLog[OfflineDatacenterState, GridState]`, so `log.dc_states` is typed as `list[OfflineDatacenterState]` without any casts.
