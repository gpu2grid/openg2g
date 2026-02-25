# Building Simulators

This page shows how to build your own simulator for a particular scenario using OpenG2G's components.

## Minimal Example

A simulation requires three things: a datacenter, a grid, and at least one controller.

```python
from openg2g.coordinator import Coordinator
from openg2g.controller.noop import NoopController

coord = Coordinator(
    datacenter=dc,
    grid=grid,
    controllers=[NoopController(dt_s=1.0)],
    total_duration_s=3600,
    dc_bus="671",
)
log = coord.run()
```

The coordinator computes the base tick as the GCD of all component periods. In this example, if the datacenter runs at 0.1s and the grid at 1.0s, the tick is 0.1s.

## Component Lifecycle

Every `run()` call follows the sequence: `reset()` all -> `start()` all -> simulation loop -> `stop()` all. Calling `run()` twice produces identical results.

- **`reset()`** (abstract) clears simulation state (history, RNGs, counters). Configuration is not affected.
- **`start()`** (no-op by default) acquires per-run resources. [`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid] compiles its DSS circuit here; most offline components don't override this.
- **`stop()`** (no-op by default) releases per-run resources. Simulation state is preserved for inspection.

## Setting Up the Datacenter

### Offline (Trace Replay)

The [`OfflineDatacenter`][openg2g.datacenter.offline.OfflineDatacenter] replays CSV power traces. You first load traces from a manifest into a [`PowerTraceStore`][openg2g.datacenter.offline.PowerTraceStore], build periodic templates, then create the datacenter.

#### Direct construction

```python
from openg2g.datacenter.offline import OfflineDatacenter, PowerTraceStore
from openg2g.datacenter.layout import RampActivationStrategy
from openg2g.models.spec import LLMInferenceModelSpec
from openg2g.datacenter.config import ServerRamp, ServerRampSchedule

models = [
    LLMInferenceModelSpec(
        model_label="Llama-3.1-8B", num_replicas=720, gpus_per_replica=1, initial_batch_size=128,
    ),
    LLMInferenceModelSpec(
        model_label="Llama-3.1-70B", num_replicas=180, gpus_per_replica=4, initial_batch_size=128,
    ),
]

store = PowerTraceStore.load("data/generated/traces_summary.csv")
store.build_templates(duration_s=3600.0, timestep_s=0.1)

dc = OfflineDatacenter(
    trace_store=store,
    models=models,
    timestep_s=0.1,
    gpus_per_server=8,
    seed=0,
    amplitude_scale_range=(0.98, 1.02),
    noise_fraction=0.005,
    activation_strategy=RampActivationStrategy(
        ServerRampSchedule(entries=(ServerRamp(t_start=2500.0, t_end=3000.0, target=0.2),))
    ),
    base_kw_per_phase=500.0,
)
```

#### Using config objects

For more complex setups (training overlays, server ramp schedules), use the `from_config()` factory with [`DatacenterConfig`][openg2g.datacenter.config.DatacenterConfig] and [`WorkloadConfig`][openg2g.datacenter.config.WorkloadConfig]:

```python
from openg2g.datacenter.config import DatacenterConfig, WorkloadConfig
from openg2g.datacenter.offline import OfflineDatacenter, PowerTraceStore
from openg2g.models.spec import LLMInferenceModelSpec, LLMInferenceWorkload
from openg2g.datacenter.training_overlay import TrainingTrace
from openg2g.types import ServerRamp, TrainingRun

store = PowerTraceStore.load("data/generated/traces_summary.csv")
store.build_templates(duration_s=3600.0, timestep_s=0.1)

training_trace = TrainingTrace.load("data/generated/synthetic_training_trace.csv")

workload = WorkloadConfig(
    inference=LLMInferenceWorkload(models=(
        LLMInferenceModelSpec(
            model_label="Llama-3.1-8B", num_replicas=720, gpus_per_replica=1, initial_batch_size=128,
        ),
        LLMInferenceModelSpec(
            model_label="Llama-3.1-70B", num_replicas=180, gpus_per_replica=4, initial_batch_size=128,
        ),
    )),
    training=TrainingRun(t_start=1000.0, t_end=2000.0, n_gpus=2400, trace=training_trace),
    server_ramps=ServerRamp(t_start=2500.0, t_end=3000.0, target=0.2),
)

dc = OfflineDatacenter.from_config(
    datacenter=DatacenterConfig(gpus_per_server=8, base_kw_per_phase=500.0),
    workload=workload,
    trace_store=store,
    timestep_s=0.1,
    seed=0,
    amplitude_scale_range=(0.98, 1.02),
    noise_fraction=0.005,
)
```

### Online (Live GPU)

!!! Note
    The online (live) datacenter backend is currently in early development. The offline trace-replay backend is recommended for most users.

The [`OnlineDatacenter`][openg2g.datacenter.online.OnlineDatacenter] connects to real vLLM servers for load generation and ITL measurement, and to zeusd instances for live GPU power monitoring. Power readings from a small number of real GPUs are augmented to datacenter scale using temporal staggering.

```python
from zeus.monitor.power_streaming import PowerStreamingClient
from zeus.utils.zeusd import ZeusdConfig
from openg2g.datacenter.online import (
    OnlineDatacenter,
    OnlineModelDeployment,
    GPUEndpointMapping,
    PowerAugmentationConfig,
    LoadGenerationConfig,
)
from openg2g.models.spec import LLMInferenceModelSpec

deployments = [
    OnlineModelDeployment(
        spec=LLMInferenceModelSpec(
            model_label="Llama-3.1-8B", num_replicas=720,
            gpus_per_replica=1, initial_batch_size=128,
        ),
        vllm_base_url="http://node1:8000",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        gpu_endpoints=(
            GPUEndpointMapping(host="node1", gpu_indices=(0, 1, 2, 3)),
        ),
    ),
]

power_client = PowerStreamingClient(
    servers=[
        ZeusdConfig.tcp(ep.host, ep.port, gpu_indices=list(ep.gpu_indices), cpu_indices=[])
        for d in deployments for ep in d.gpu_endpoints
    ],
)

dc = OnlineDatacenter(
    deployments=deployments,
    power_client=power_client,
    requests_by_model={"Llama-3.1-8B": [...]},
    augmentation=PowerAugmentationConfig(base_kw_per_phase=500.0),
    load_gen=LoadGenerationConfig(max_output_tokens=512),
)
```

The coordinator calls `start()` to run health checks, wait for power readings, set initial batch sizes, and start load generation. `stop()` shuts everything down.

## Setting Up the Grid

Tap schedules are built using [`TapPosition`][openg2g.types.TapPosition] (per-unit tap ratios per phase) and the `|` operator:

```python
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.types import TapPosition

TAP_STEP = 0.00625  # standard 5/8% tap step

# Fixed taps (single position at t=0)
tap_schedule = TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP).at(t=0)

# Or scheduled changes at multiple times
tap_schedule = (
    TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP).at(t=0)
    | TapPosition(a=1.0 + 16 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 17 * TAP_STEP).at(t=25 * 60)
    | TapPosition(a=1.0 + 10 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 10 * TAP_STEP).at(t=55 * 60)
)

grid = OpenDSSGrid(
    dss_case_dir="examples/ieee13",
    dss_master_file="IEEE13Nodeckt.dss",
    dc_bus="671",
    dc_bus_kv=4.16,
    power_factor=0.95,
    dt_s=0.1,
    connection_type="wye",
    # dss_controls=False (default): OpenDSS is a passive power flow solver.
    # All voltage regulation is managed by our controllers.
)
```

The grid auto-detects its sub-step behavior based on how many DC power samples it receives per step. When `dt_grid == dt_dc` (e.g., both 0.1s), it receives one sample and runs one DSS solve. When `dt_grid > dt_dc` (e.g., grid at 1.0s, DC at 0.1s), it resamples the accumulated DC buffer to 2 DSS grid points via `np.interp`.

## Stacking Controllers

Controllers compose in order. Each controller gets full component handles plus an event emitter:

```python
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.controller.ofo import OFOBatchController

controllers = [
    TapScheduleController(schedule=[], dt_s=1.0),  # taps handled by grid schedule
    OFOBatchController(models=models, fits=fits, ...),
]

coord = Coordinator(
    datacenter=dc,
    grid=grid,
    controllers=controllers,
    total_duration_s=3600,
    dc_bus="671",
)
```

Initial tap positions are set via the `initial_tap_position` parameter on [`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid] (using the [`TapPosition`][openg2g.types.TapPosition] API). Scheduled changes are handled by [`TapScheduleController`][openg2g.controller.tap_schedule.TapScheduleController]. Actions from all controllers are applied before the next tick.

### Command Types

Commands are typed dataclasses routed to backends via `singledispatchmethod`:

- [`SetBatchSize`][openg2g.types.SetBatchSize]`(batch_size_by_model=...)`: Datacenter command
- [`SetTaps`][openg2g.types.SetTaps]`(tap_position=...)`: Grid command

Backends raise `TypeError` for unsupported command types.

Controller interface summary:

- `step(clock, datacenter, grid, events)` -> [`ControlAction`][openg2g.types.ControlAction]
- read current state through `datacenter.state` and `grid.state`
- read history through `datacenter.history(...)` and `grid.history(...)`
- emit events through `events.emit(topic, data)`

## Live Mode

For hardware-in-the-loop experiments, pass `live=True` to the coordinator:

```python
coord = Coordinator(
    datacenter=dc,
    grid=grid,
    controllers=controllers,
    total_duration_s=300,
    dc_bus="671",
    live=True,
)
```

In live mode, the clock synchronizes with wall time. If computation falls behind real time, a warning is issued.

## Analyzing Results

The [`SimulationLog`][openg2g.coordinator.SimulationLog] returned by `coord.run()` contains all state and action history:

```python
log = coord.run()

# Voltage statistics
from openg2g.metrics.voltage import compute_allbus_voltage_stats
stats = compute_allbus_voltage_stats(log.grid_states, v_min=0.95, v_max=1.05)
print(f"Violation time: {stats.violation_time_s:.1f} s")

# Plotting: the example scripts in examples/offline/ import shared plot
# functions from plotting.py (a sibling module in that directory).
# See examples/offline/plotting.py for reusable plot functions.
import numpy as np
time_s = np.array(log.time_s)
kW_A = np.array(log.kW_A)
kW_B = np.array(log.kW_B)
kW_C = np.array(log.kW_C)
```

## Configuration Sweeps

Since `run()` calls `reset()` on all components before each run, you can reuse expensive objects across a parameter sweep:

```python
from openg2g.coordinator import Coordinator
from openg2g.grid.opendss import OpenDSSGrid

grid = OpenDSSGrid(...)  # stores config only (cheap)

for batch_init in [64, 128, 256, 512]:
    dc = OfflineDatacenter(trace_store=store, models=models, ...)
    ctrl = OFOBatchController(models=models, ...)
    coord = Coordinator(dc, grid, [ctrl], total_duration_s=3600, dc_bus="671")
    log = coord.run()  # reset -> start (compile DSS) -> loop -> stop
    print(f"batch_init={batch_init}: violation={stats.integral_violation:.2f}")
```

Each `run()` resets simulation state, then `start()` compiles a fresh DSS circuit, so every iteration starts clean despite reusing the same [`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid] instance.

## Writing Custom Components

OpenG2G is designed for extensibility. You can implement your own datacenter backends and controllers by subclassing the provided abstract base classes. Before diving into the interface, let's look at how the built-in components are implemented as concrete examples.

### Case Study: The Offline Datacenter

The [`OfflineDatacenter`][openg2g.datacenter.offline.OfflineDatacenter] replays real GPU power traces at controlled batch sizes (see Section IV-A of the [G2G paper](https://arxiv.org/abs/2602.05116)):

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

### Case Study: The OpenDSS Grid

[`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid] is the built-in [`GridBackend`][openg2g.grid.base.GridBackend] implementation. Beyond the base interface, it provides:

- `voltages_vector()`: Flat numpy array of all bus-phase voltages (used by the OFO controller for gradient computation)
- `estimate_sensitivity(perturbation_kw)`: Finite-difference estimate of the voltage sensitivity matrix dV/dP

When the grid runs at a coarser rate than the datacenter, it internally resamples the accumulated power buffer via interpolation. For example, with DC at 0.1s and grid at 1.0s, 10 accumulated power samples are resampled to 2 DSS solve points via `np.interp`.

### Case Study: The OFO Controller

Online Feedback Optimization (primal-dual) regulates batch sizes to keep voltages safe. For the full mathematical formulation, see Section III of the [G2G paper](https://arxiv.org/abs/2602.05116).

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
  │  DUAL UPDATES (G2G paper Eqs. 5-7):                         │
  │                                                              │
  │    Voltage:  λ⁺ ← [λ⁺ + ρ_v (V - V_max)]⁺                  │
  │              λ⁻ ← [λ⁻ + ρ_v (V_min - V)]⁺                  │
  │              η  = λ⁺ - λ⁻                                    │
  │                                                              │
  │    Latency:  μ_i ← [μ_i + ρ_l (ITL_i - L_thresh)]⁺         │
  │                                                              │
  │  PRIMAL UPDATE (G2G paper Eq. 8):                            │
  │                                                              │
  │    x_i = log₂(batch_i)                                      │
  │                                                              │
  │    ∇_i = - w_T · dTh/dx         (throughput reward)           │
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
       Full gradient derivation: G2G paper, Appendix B (Eq. 18)
```

### Custom Controller

Controllers implement the [`Controller`][openg2g.controller.base.Controller] ABC from `openg2g.controller.base`. The [`Controller`][openg2g.controller.base.Controller] class is generic over its compatible datacenter and grid backend types:

```python
from __future__ import annotations

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.datacenter.base import DatacenterBackend
from openg2g.events import EventEmitter
from openg2g.grid.base import GridBackend
from openg2g.types import ControlAction, DatacenterState, GridState, SetBatchSize


class MyController(Controller[DatacenterBackend[DatacenterState], GridBackend[GridState]]):
    """A controller that reduces batch size when any voltage is below a threshold."""

    def __init__(self, v_threshold: float = 0.96, dt_s: float = 1.0):
        self._v_threshold = v_threshold
        self._dt_s = dt_s

    @property
    def dt_s(self) -> float:
        return self._dt_s

    def step(
        self,
        clock: SimulationClock,
        datacenter: DatacenterBackend[DatacenterState],
        grid: GridBackend[GridState],
        events: EventEmitter,
    ) -> ControlAction:
        if grid.state is None:
            return ControlAction(commands=[])

        # Check if any bus voltage is below threshold
        for bus in grid.state.voltages.buses():
            tp = grid.state.voltages[bus]
            for v in (tp.a, tp.b, tp.c):
                if v < self._v_threshold:
                    events.emit("controller.low_voltage", {"bus": bus, "time_s": clock.time_s})
                    return ControlAction(
                        commands=[SetBatchSize(batch_size_by_model={"MyModel": 64})]
                    )

        return ControlAction(
            commands=[SetBatchSize(batch_size_by_model={"MyModel": 128})]
        )
```

#### Controller Guidelines

- `step()` must return a [`ControlAction`][openg2g.types.ControlAction] on every call.
- Use [`ControlAction`][openg2g.types.ControlAction]`(commands=[])` for a no-op.
- Use [`SetBatchSize`][openg2g.types.SetBatchSize]`(batch_size_by_model=...)` for batch updates.
- Use [`SetTaps`][openg2g.types.SetTaps]`(tap_position=...)` for tap updates.
- Read current component state via `datacenter.state` and `grid.state`.
- Use `datacenter.history(...)` and `grid.history(...)` for non-Markovian logic.
- Keep `step()` fast. It runs synchronously in the simulation loop.
- Use `clock.time_s` for time-dependent logic.
- Use `events.emit(topic, data)` to log controller-side events.

#### Controller Generic Parameters

The two type parameters in `Controller[DC, Grid]` declare which backend types the controller is compatible with. The coordinator checks these at construction time. Common patterns:

- `Controller[DatacenterBackend[DatacenterState], GridBackend[GridState]]`: Works with any backend.
- `Controller[LLMBatchSizeControlledDatacenter[OfflineDatacenterState], OpenDSSGrid]`: Only works with the offline datacenter and OpenDSS grid.
- `Controller[LLMBatchSizeControlledDatacenter[DatacenterState], GridBackend[GridState]]`: Works with any LLM datacenter and any grid.

If your controller inherits from a typed parent, the generic parameters are inherited automatically and do not need to be re-specified.

### Custom Datacenter Backend

Datacenter backends implement the [`DatacenterBackend`][openg2g.datacenter.base.DatacenterBackend] ABC from `openg2g.datacenter.base`. The ABC is generic over the state type it emits. Parameterize it with the state dataclass your backend returns from `step()`:

```python
from __future__ import annotations

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.datacenter.base import DatacenterBackend
from openg2g.types import DatacenterCommand, DatacenterState, SetBatchSize, ThreePhase


class SyntheticDatacenter(DatacenterBackend[DatacenterState]):
    """A datacenter that generates sinusoidal power profiles."""

    def __init__(self, dt_s: float = 0.1, base_kw: float = 1000.0):
        self._dt = dt_s
        self._base_kw = base_kw
        self._batch: dict[str, int] = {}
        self._state: DatacenterState | None = None
        self._history: list[DatacenterState] = []

    @property
    def dt_s(self) -> float:
        return self._dt

    @property
    def state(self) -> DatacenterState | None:
        return self._state

    def history(self, n: int | None = None) -> list[DatacenterState]:
        if n is None:
            return list(self._history)
        if n <= 0:
            return []
        return list(self._history[-int(n) :])

    def step(self, clock: SimulationClock) -> DatacenterState:
        t = clock.time_s
        power_kw = self._base_kw * (1.0 + 0.3 * np.sin(2 * np.pi * t / 600))
        power_w = power_kw * 1e3 / 3  # split equally across phases
        st = DatacenterState(
            time_s=t,
            power_w=ThreePhase(a=power_w, b=power_w, c=power_w),
        )
        self._state = st
        self._history.append(st)
        return st

    def apply_control(self, command: DatacenterCommand) -> None:
        if isinstance(command, SetBatchSize):
            self._batch.update({str(k): int(v) for k, v in command.batch_size_by_model.items()})
```

If your backend needs richer state (e.g. per-model power breakdowns), define a [`DatacenterState`][openg2g.datacenter.base.DatacenterState] subclass and use it as the type parameter:

```python
@dataclass(frozen=True)
class MyState(DatacenterState):
    per_gpu_power_w: dict[int, float] = field(default_factory=dict)

class MyDatacenter(DatacenterBackend[MyState]):
    def step(self, clock: SimulationClock) -> MyState:
        ...
```

The state type propagates through the [`Coordinator`][openg2g.coordinator.Coordinator] to the [`SimulationLog`][openg2g.coordinator.SimulationLog], so `log.dc_states` will be correctly typed as `list[MyState]`.

#### Datacenter Guidelines

- `step()` is called at the rate specified by `dt_s`.
- Return a [`DatacenterState`][openg2g.datacenter.base.DatacenterState] (or subclass) with three-phase power in watts.
- `apply_control()` receives one command at a time.
- Implement `state` and `history(...)` to expose current/past states to controllers.
- For offline backends, consider using [`OfflineDatacenterState`][openg2g.datacenter.offline.OfflineDatacenterState] which includes per-model power and replica counts.

### Tips

- **Multiple controllers**: Controllers run in order. Put tap controllers before batch controllers if tap changes should be visible to the batch optimizer within the same tick.
- **State inspection**: The base [`DatacenterState`][openg2g.datacenter.base.DatacenterState] includes `batch_size_by_model` and `active_replicas_by_model`, so controllers can access per-model batch sizes and replica counts without knowing which backend is in use.
- **Testing**: Write unit tests for your controller by constructing mock [`DatacenterState`][openg2g.datacenter.base.DatacenterState] and [`GridState`][openg2g.grid.base.GridState] objects directly. They are simple frozen dataclasses.
