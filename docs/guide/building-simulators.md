# Building Simulators

This page walks through building your own simulator for your scenario using OpenG2G components.
For how the components fit together conceptually, see [Architecture](architecture.md).

## Essential Components

A simulator needs three things: a datacenter backend, a grid backend, and at least one controller.
You wire them together with the [`Coordinator`][openg2g.coordinator.Coordinator]:

```python
from openg2g.coordinator import Coordinator

coord = Coordinator(
    datacenter=dc,
    grid=grid,
    controllers=[controller1, controller2, ...],
    total_duration_s=3600,
    dc_bus="671",
)
log = coord.run()
```

The sections below show how to set up each component.

### Datacenter

The [`OfflineDatacenter`][openg2g.datacenter.offline.OfflineDatacenter] replays GPU power traces built from the [data pipeline](data-pipeline.md).
Load traces from the generated manifest, build templates for the simulation config, then create the datacenter with a [`DatacenterConfig`][openg2g.datacenter.config.DatacenterConfig] and [`OfflineWorkload`][openg2g.datacenter.offline.OfflineWorkload]:

```python
from fractions import Fraction
from pathlib import Path

from openg2g.datacenter.config import (
    DatacenterConfig, InferenceModelSpec, InferenceRamp,
    ModelDeployment, PowerAugmentationConfig, TrainingRun,
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.datacenter.workloads.training import TrainingTrace

spec_8b = InferenceModelSpec(
    model_label="Llama-3.1-8B", gpus_per_replica=1,
    itl_deadline_s=0.08, feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512),
)
spec_70b = InferenceModelSpec(
    model_label="Llama-3.1-70B", gpus_per_replica=4,
    itl_deadline_s=0.10, feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512),
)
models = (spec_8b, spec_70b)

data_dir = Path("data/offline")
inference_data = InferenceData.ensure(data_dir, models, dt_s=0.1)
training_trace = TrainingTrace.ensure(data_dir / "training_trace.csv")

# Replica counts are separate from model specs
replica_counts = {"Llama-3.1-8B": 720, "Llama-3.1-70B": 180}

workload = OfflineWorkload(
    inference_data=inference_data,
    replica_counts=replica_counts,
    inference_ramps=(
        InferenceRamp(target=360, model="Llama-3.1-8B").at(t_start=1500.0, t_end=2000.0)
        | InferenceRamp(target=864, model="Llama-3.1-8B").at(t_start=2500.0, t_end=3000.0)
    ),
    training=TrainingRun(n_gpus=2400, trace=training_trace).at(t_start=1000.0, t_end=2000.0),
)

dc_config = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=500.0)
dc = OfflineDatacenter(
    dc_config,
    workload,
    dt_s=Fraction(1, 10),
    seed=0,
    total_gpu_capacity=1440,
    power_augmentation=PowerAugmentationConfig(
        amplitude_scale_range=(0.98, 1.02),
        noise_fraction=0.005,
    ),
)
```

[`InferenceRamp`][openg2g.datacenter.config.InferenceRamp] controls the number of active replicas over time using absolute counts. Each ramp specifies its `target` replica count and the `model` it applies to. Ramps are scheduled with `at(t_start, t_end)` and chained with `|`. The datacenter validates that ramp targets do not exceed `total_gpu_capacity` before running.

!!! Warning "GPU capacity validation"
    The `total_gpu_capacity` parameter is the physical GPU count at the site and must be specified explicitly. At initialization, the datacenter checks that ramp schedules never exceed this capacity at any point in time.

### Grid

Construct an [`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid] with a DSS case directory and initial tap position. Tap *schedules* are built using [`TapPosition`][openg2g.grid.config.TapPosition] and the `|` operator, and passed to a [`TapScheduleController`][openg2g.controller.tap_schedule.TapScheduleController]:

```python
from fractions import Fraction

from openg2g.grid.opendss import OpenDSSGrid
from openg2g.grid.config import TapPosition

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
    dss_master_file="IEEE13Bus.dss",
    dc_bus="671",
    dc_bus_kv=4.16,
    power_factor=dc_config.power_factor,
    dt_s=Fraction(1, 10),
    connection_type="wye",
)
```

#### Grid Data Files

OpenG2G ships with IEEE test feeder models in `data/grid/`, one folder per system. Each folder contains a self-contained master `.dss` file and a shared line impedance file:

```
data/grid/
├── ieee13/
│   ├── IEEE13Bus.dss        # Circuit, transformers, regulators, loads, lines, bus coordinates
│   └── IEEELineCodes.dss    # Line impedance definitions (loaded via Redirect)
├── ieee34/
│   ├── IEEE34Bus.dss
│   └── IEEELineCodes.dss
└── ieee123/
    ├── IEEE123Bus.dss
    └── IEEELineCodes.dss
```

These are modified from the [OpenDSS IEEE Test Cases](https://github.com/tshort/OpenDSS/tree/master/Distrib/IEEETestCases) for studying datacenter loads on distribution grids. See the header comment in each `.dss` file for system-specific modifications.

**Regulator naming.** OpenDSS models voltage regulators as two separate objects — a `Transformer` (the physical hardware) and a `RegControl` (the control logic that adjusts the transformer's tap). Names follow the convention `reg{bank}{phase}` / `creg{bank}{phase}` (the `c` prefix stands for "control"):

| Component | Pattern | Example | Meaning |
|-----------|---------|---------|---------|
| Transformer | `reg{bank}{phase}` | `reg1a` | Bank 1, phase A |
| RegControl | `creg{bank}{phase}` | `creg1a` | Control for bank 1, phase A |

Each `RegControl` points to its transformer: `new regcontrol.creg1a transformer=reg1a winding=2 ...`. A typical bank has three per-phase regulators (e.g., `creg1a`, `creg1b`, `creg1c`).

**Adding a new test system.** When adding a new `.dss` file:

1. Name the master file `IEEE{N}Bus.dss`.
2. Use explicit phase suffixes on regulator transformer buses (e.g., `buses=[650.1 RG60.1]`). The simulator determines phase-to-regulator mapping from bus node data — if phase cannot be determined, users must address regulators by name in `TapPosition`.
3. Follow the `reg{bank}{phase}` / `creg{bank}{phase}` naming convention for regulators.
4. Use `SetBusXY bus=... X=... Y=...` for inline bus coordinates (no external CSV files).

### Controllers

Controllers compose in order. Pass them as a list to the coordinator:

```python
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.controller.ofo import OFOBatchSizeController

coord = Coordinator(
    datacenter=dc,
    grid=grid,
    controllers=[
        TapScheduleController(schedule=tap_schedule, dt_s=Fraction(1)),
        OFOBatchSizeController(...),
    ],
    total_duration_s=3600,
    dc_bus="671",
)
```

Actions from all controllers are applied before the next tick. Put tap controllers before batch controllers if tap changes should be visible to the batch optimizer within the same tick.

## Analyzing Results

[`Coordinator.run()`][openg2g.coordinator.Coordinator.run] returns a [`SimulationLog`][openg2g.coordinator.SimulationLog] containing all state and action history:

```python
log = coord.run()

# Voltage statistics
from openg2g.metrics.voltage import compute_allbus_voltage_stats
stats = compute_allbus_voltage_stats(log.grid_states, v_min=0.95, v_max=1.05)
print(f"Violation time: {stats.violation_time_s:.1f} s")

# Time-series data for plotting
import numpy as np
time_s = np.array(log.time_s)
dc_time_s = np.array([s.time_s for s in log.dc_states])
kW_A = np.array([s.power_w.a / 1e3 for s in log.dc_states])
kW_B = np.array([s.power_w.b / 1e3 for s in log.dc_states])
kW_C = np.array([s.power_w.c / 1e3 for s in log.dc_states])
```

See [`examples/offline/plot_all_figures.py`](https://github.com/gpu2grid/openg2g/blob/master/examples/offline/plot_all_figures.py) for reusable plot functions used by the example scripts.

## Writing Custom Components

OpenG2G is designed for extensibility.
Subclass the abstract base classes to implement your own datacenter backends and controllers, or subclass an existing implementation to create a custom variant.
For background on the type system and generics, see [Architecture: State Types and Generics](architecture.md#state-types-and-generics).

### Custom Datacenter

**Datacenter backend.**
Subclass [`DatacenterBackend`][openg2g.datacenter.base.DatacenterBackend] and parameterize it with the state type your backend returns from [`step()`][openg2g.datacenter.base.DatacenterBackend.step]. Your `__init__` must call `super().__init__()` to initialize the base class's state tracking. The coordinator calls [`do_step()`][openg2g.datacenter.base.DatacenterBackend.do_step] at the rate specified by [`dt_s`][openg2g.datacenter.base.DatacenterBackend.dt_s], which internally calls your `step()` and records the returned state. Each `step()` call should return a [`DatacenterState`][openg2g.datacenter.base.DatacenterState] (or subclass) with three-phase power in watts. Between steps, the coordinator may call [`apply_control(command, events)`][openg2g.datacenter.base.DatacenterBackend.apply_control] with individual [`DatacenterCommand`][openg2g.datacenter.command.DatacenterCommand] instances; changes take effect on the next `step()`. Current state is available through [`state`][openg2g.datacenter.base.DatacenterBackend.state] and past states through [`history(n)`][openg2g.datacenter.base.DatacenterBackend.history]; both are managed automatically by the base class.

[`reset()`][openg2g.datacenter.base.DatacenterBackend.reset] must clear all simulation state (counters, RNG state) while preserving configuration (`dt_s`, models, templates). History is cleared automatically by the coordinator (via [`do_reset()`][openg2g.datacenter.base.DatacenterBackend.do_reset]). Override [`start()`][openg2g.datacenter.base.DatacenterBackend.start] and [`stop()`][openg2g.datacenter.base.DatacenterBackend.stop] for backends that acquire resources (connections, solver circuits); the coordinator calls `start()` before the simulation loop and `stop()` in LIFO order afterward. See [Architecture: Component Lifecycle](architecture.md#component-lifecycle) for the full sequence.

```python
from __future__ import annotations

import functools
from fractions import Fraction

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.datacenter.base import DatacenterBackend
from openg2g.events import EventEmitter
from openg2g.common import ThreePhase
from openg2g.datacenter.base import DatacenterState
from openg2g.datacenter.command import DatacenterCommand, SetBatchSize


class SyntheticDatacenter(DatacenterBackend[DatacenterState]):
    """A datacenter that generates sinusoidal power profiles."""

    def __init__(self, dt_s: Fraction = Fraction(1, 10), base_kw: float = 1000.0):
        super().__init__()
        self._dt = dt_s
        self._base_kw = base_kw
        self._batch: dict[str, int] = {}

    @property
    def dt_s(self) -> Fraction:
        return self._dt

    def reset(self) -> None:
        self._batch.clear()

    def step(self, clock: SimulationClock, events: EventEmitter) -> DatacenterState:
        t = clock.time_s
        power_kw = self._base_kw * (1.0 + 0.3 * np.sin(2 * np.pi * t / 600))
        power_w = power_kw * 1e3 / 3  # split equally across phases
        return DatacenterState(
            time_s=t,
            power_w=ThreePhase(a=power_w, b=power_w, c=power_w),
        )

    @functools.singledispatchmethod
    def apply_control(self, command: DatacenterCommand, events: EventEmitter) -> None:
        raise TypeError(f"SyntheticDatacenter does not support {type(command).__name__}")

    @apply_control.register
    def _apply_set_batch_size(self, command: SetBatchSize, events: EventEmitter) -> None:
        self._batch.update({str(k): int(v) for k, v in command.batch_size_by_model.items()})
```

**Datacenter state.**
For richer state, define a [`DatacenterState`][openg2g.datacenter.base.DatacenterState] subclass:

```python
@dataclass(frozen=True)
class MyState(DatacenterState):
    per_gpu_power_w: dict[int, float] = field(default_factory=dict)

class MyDatacenter(DatacenterBackend[MyState]):
    def __init__(self):
        super().__init__()

    def step(self, clock: SimulationClock, events: EventEmitter) -> MyState:
        ...
```

The state type propagates through to [`SimulationLog`][openg2g.coordinator.SimulationLog], so `log.dc_states` will be correctly typed as `list[MyState]`.

**Datacenter commands.**
For custom commands, subclass [`DatacenterCommand`][openg2g.datacenter.command.DatacenterCommand] and register a handler with `@apply_control.register`:

```python
from dataclasses import dataclass
from openg2g.datacenter.command import DatacenterCommand

@dataclass(frozen=True)
class SetPowerCap(DatacenterCommand):
    cap_kw: float

class MyDatacenter(DatacenterBackend[DatacenterState]):
    @functools.singledispatchmethod
    def apply_control(self, command: DatacenterCommand, events: EventEmitter) -> None:
        raise TypeError(f"MyDatacenter does not support {type(command).__name__}")

    @apply_control.register
    def _apply_set_batch_size(self, command: SetBatchSize, events: EventEmitter) -> None:
        self._batch.update(command.batch_size_by_model)

    @apply_control.register
    def _apply_set_power_cap(self, command: SetPowerCap, events: EventEmitter) -> None:
        self._power_cap = command.cap_kw
```

The coordinator routes commands to backends based on their type hierarchy: [`DatacenterCommand`][openg2g.datacenter.command.DatacenterCommand] subclasses go to the datacenter, [`GridCommand`][openg2g.grid.command.GridCommand] subclasses go to the grid.

**Testing.** [`DatacenterState`][openg2g.datacenter.base.DatacenterState] and [`GridState`][openg2g.grid.base.GridState] are simple frozen dataclasses, so you can construct them directly in unit tests without running a full simulation.

### Custom Controller

**Controller.**
Subclass [`Controller`][openg2g.controller.base.Controller] and parameterize it with the datacenter and grid backend types it works with. The coordinator calls [`step(clock, datacenter, grid, events)`][openg2g.controller.base.Controller.step] at the rate specified by [`dt_s`][openg2g.controller.base.Controller.dt_s]; each call must return a list of [`DatacenterCommand`][openg2g.datacenter.command.DatacenterCommand] and/or [`GridCommand`][openg2g.grid.command.GridCommand] objects (return `[]` for a no-op). Read current state via [`datacenter.state`][openg2g.datacenter.base.DatacenterBackend.state] and [`grid.state`][openg2g.grid.base.GridBackend.state], and history via [`datacenter.history(...)`][openg2g.datacenter.base.DatacenterBackend.history] and [`grid.history(...)`][openg2g.grid.base.GridBackend.history]. Use [`events.emit(topic, data)`][openg2g.events.EventEmitter.emit] to log controller-side events. Keep `step()` fast since it runs synchronously in the simulation loop.

[`reset()`][openg2g.controller.base.Controller.reset] must clear all simulation state (e.g., dual variables, counters, cached matrices) while preserving configuration. Override [`start()`][openg2g.controller.base.Controller.start] and [`stop()`][openg2g.controller.base.Controller.stop] if the controller acquires external resources. See [Architecture: Component Lifecycle](architecture.md#component-lifecycle) for the full sequence.

```python
from __future__ import annotations

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.events import EventEmitter
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.datacenter.command import DatacenterCommand, SetBatchSize
from openg2g.grid.command import GridCommand


class MyController(Controller[MyDatacenter, OpenDSSGrid]):
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
        datacenter: MyDatacenter,
        grid: OpenDSSGrid,
        events: EventEmitter,
    ) -> list[DatacenterCommand | GridCommand]:
        if grid.state is None:
            return []

        for bus in grid.state.voltages.buses():
            tp = grid.state.voltages[bus]
            for v in (tp.a, tp.b, tp.c):
                if v < self._v_threshold:
                    events.emit("controller.low_voltage", {"bus": bus, "time_s": clock.time_s})
                    return [SetBatchSize(batch_size_by_model={"MyModel": 64})]

        return [SetBatchSize(batch_size_by_model={"MyModel": 128})]
```

**Backend-specific controllers.**
Controllers that need specific backend features (e.g., `voltages_vector()` from [`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid]) should bind their generic type parameters to concrete types instead of using the base classes. See the [OFO controller](#ofobatchsizecontroller) below for an example.

## Built-in Components

The following sections describe how the built-in components implement the interfaces above.

### `OfflineDatacenter`

[`OfflineDatacenter`][openg2g.datacenter.offline.OfflineDatacenter] implements [`DatacenterBackend`][openg2g.datacenter.base.DatacenterBackend] by replaying real GPU power traces at controlled batch sizes.

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
  │  batch = 64         │  │           + power from training overlays
  ├─────────────────────┤  │           + per-replica noise and jitter
  │ (+ 2 MoE models)    │──┘             │           │           │
  └─────────────────────┘                v           v           v
                                       P_A(t)     P_B(t)     P_C(t)
```

How it implements the interface:

- **`step(clock, events)`** indexes into pre-built per-GPU power templates using `(global_step + offset) % template_length` per server. Random restart offsets desynchronize servers for a realistic aggregate power profile. Returns an [`OfflineDatacenterState`][openg2g.datacenter.offline.OfflineDatacenterState] (extends [`LLMDatacenterState`][openg2g.datacenter.base.LLMDatacenterState]) with per-model batch sizes, replica counts, and observed ITL.
- **`apply_control(command, events)`** dispatches [`SetBatchSize`][openg2g.datacenter.command.SetBatchSize] and [`ShiftReplicas`][openg2g.datacenter.command.ShiftReplicas] commands. Batch size changes take effect immediately on the next `step()` call. `ShiftReplicas` adjusts the activation policy's base server count, moving replicas in/out of the datacenter subject to `total_gpu_capacity`.
- **`total_gpu_capacity`**: Maximum number of GPUs this datacenter can physically host. Auto-computed from initial model allocation if not specified. Incoming replica shifts are rejected or clamped when the datacenter is full. Exposed via `current_gpu_usage()` and `available_gpu_capacity()` methods.
- **`load_shift_headroom`**: Fraction of extra server capacity to pre-allocate (e.g., 0.3 = 30%) so incoming replicas have server slots available.
- **`reset()`** clears step counter, replica offsets, and RNG state. Rebuilds server layouts from the stored config so the next run starts fresh. History is cleared automatically by `do_reset()`.
- An [`ActivationPolicy`][openg2g.datacenter.layout.ActivationPolicy] determines which servers are active at each timestep. The default [`RampActivationPolicy`][openg2g.datacenter.layout.RampActivationPolicy] follows an [`InferenceRampSchedule`][openg2g.datacenter.config.InferenceRampSchedule] with random priority ordering. Ramps can target specific models via the `model` parameter and support targets above 1.0 (activating more replicas than the initial count).
- Training workload overlays add transient high-power phases.

### `OpenDSSGrid`

[`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid] implements [`GridBackend`][openg2g.grid.base.GridBackend] using the OpenDSS power flow solver on standard IEEE test feeders.

How it implements the interface:

- **`step(clock, power_samples_w, events)`** takes the most recent power sample from the accumulated buffer and runs a single OpenDSS power flow solve. If no samples are provided (grid runs faster than datacenter), the last known power is reused. Returns a [`GridState`][openg2g.grid.base.GridState] with per-bus, per-phase voltages and tap positions.
- **`apply_control(command, events)`** dispatches [`SetTaps`][openg2g.grid.command.SetTaps] commands to update regulator tap positions.
- **`reset()`** clears cached state and the `_started` flag. History is cleared automatically by `do_reset()`. The DSS circuit is recompiled on the next `start()`.
- **`start()`** compiles the DSS circuit from the case files, builds the bus-phase voltage index (`v_index`), and prepares snapshot indexing structures.
- **`voltages_vector()`** returns a flat numpy array of all bus-phase voltages in `v_index` order (used by the OFO controller for gradient computation).
- **`estimate_sensitivity(perturbation_kw)`** computes a finite-difference estimate of the voltage sensitivity matrix dV/dP.

### `OFOBatchSizeController`

[`OFOBatchSizeController`][openg2g.controller.ofo.OFOBatchSizeController] implements [`Controller`][openg2g.controller.base.Controller] using Online Feedback Optimization (primal-dual) to regulate batch sizes for voltage safety. For the full mathematical formulation, see the [G2G paper](https://arxiv.org/abs/2602.05116).

How it implements the interface:

- **`step(clock, datacenter, grid, events)`** reads `active_replicas_by_model` and `observed_itl_s_by_model` from `datacenter.state`, and `phase_share_by_model` from the datacenter itself. On the grid side, it calls `grid.v_index`, `grid.voltages_vector()`, and `grid.estimate_sensitivity()`. Returns a list of [`SetBatchSize`][openg2g.datacenter.command.SetBatchSize] commands.
- **`reset()`** clears dual variables (voltage and latency multipliers), primal state, step counters, and the cached sensitivity matrix.
- Binds its generic type parameters to [`LLMBatchSizeControlledDatacenter`][openg2g.datacenter.base.LLMBatchSizeControlledDatacenter] and [`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid], since it requires LLM-specific state fields and OpenDSS-specific methods (`voltages_vector`, `estimate_sensitivity`).
- In multi-DC setups, pass `site_id` to bind the controller to a specific datacenter site. The controller will call `estimate_sensitivity(site_id=...)` to compute gradients only for its site's loads.

### `RuleBasedBatchSizeController`

[`RuleBasedBatchSizeController`][openg2g.controller.rule_based.RuleBasedBatchSizeController] implements a proportional rule-based controller for batch-size regulation. Unlike OFO, it requires no sensitivity matrix, no logistic curve fits, and no dual variables — making it a simple baseline for comparison.

How it works:

- **`step(clock, datacenter, grid, events)`** reads all bus-phase voltages from `grid.state.voltages`, finds the worst violation, and adjusts batch sizes proportionally in log2-space. Undervoltage → reduce batch (less power draw); overvoltage → increase batch.
- Continuous internal state accumulates pressure across steps, enabling gradual batch transitions even with small violations.
- Optional **latency guard** prevents batch increases when ITL exceeds the model's deadline.
- Optional **deadband** (default 0.001 pu) filters tiny violations to prevent chattering.
- Configuration via [`RuleBasedConfig`][openg2g.controller.rule_based.RuleBasedConfig]: `step_size` (proportional gain), `v_min/v_max`, `deadband`, `latency_guard`.

### `LoadShiftController`

[`LoadShiftController`][openg2g.controller.load_shift.LoadShiftController] is a cross-site controller (`site_id=None`) that shifts LLM replicas between datacenters when batch-size control is exhausted and voltage violations persist. It must be placed **after** all per-site OFO controllers in the controller list so it sees the latest batch-size state.

**Rules:**

1. **Warm start only**: Only shifts models already running at both source and destination sites.
2. **Last resort**: Only acts when all models at the violated site have batch sizes at their feasible limit (OFO is saturated).
3. **Directional**: For undervoltage, shifts replicas OUT of the violated site to the site with the highest voltage. For overvoltage, shifts replicas IN from the site with the lowest voltage.
4. **Capacity-aware**: Checks `available_gpu_capacity()` on the destination before shifting. Shifts are rejected if the destination datacenter is full.
5. **Incremental**: Shifts `gpus_per_shift` GPUs worth of replicas per time step, repeating until the violation resolves.

**Wiring:**

```python
from openg2g.controller.load_shift import LoadShiftConfig, LoadShiftController

controllers = [
    TapScheduleController(...),
    OFOBatchSizeController(..., site_id="site_a"),
    OFOBatchSizeController(..., site_id="site_b"),
    # Load shift controller must be LAST
    LoadShiftController(
        config=LoadShiftConfig(enabled=True, gpus_per_shift=8),
        dt_s=Fraction(1),
        datacenters={"site_a": dc_a, "site_b": dc_b},
        site_bus_map={"site_a": "bus_1", "site_b": "bus_2"},
        models_by_site={"site_a": ["Model-A", "Model-B"], "site_b": ["Model-B", "Model-C"]},
        gpus_per_replica_by_model={"Model-A": 1, "Model-B": 4, "Model-C": 8},
        feasible_batch_sizes_by_model={"Model-A": [8, 16, 32], ...},
        v_min=0.95, v_max=1.05,
    ),
]
```

The `ShiftReplicas` command includes a `target_site_id` field, and the coordinator routes it to the correct datacenter automatically.

## Example Analysis Scripts

The `examples/offline/` directory includes ready-to-run analysis scripts. For detailed usage, examples across IEEE test systems, and interpretation guides, see the [Examples](../examples/) documentation:

| Script | Topic | Details |
|--------|-------|---------|
| `run_baseline.py`, `run_ofo.py` | Baseline and OFO simulations (all systems) | [GPU Flexibility](../examples/gpu-flexibility.md), [Voltage Strategies](../examples/voltage-regulation-strategies.md) |
| `analyze_different_controllers.py` | Controller comparison (baseline, rule-based, OFO) | [Voltage Strategies](../examples/voltage-regulation-strategies.md) |
| `sweep_ofo_parameters.py` | OFO parameter sensitivity sweep | [Parameter Sensitivity](../examples/controller-parameter-sensitivity.md) |
| `plot_topology.py` | System topology visualization | [Grid Topology](../examples/grid-topology-effects.md) |
| `sweep_hosting_capacities.py` | Per-bus hosting capacity analysis | [Hosting Capacity](../examples/hosting-capacity.md) |
| `sweep_dc_locations.py` | DC location sweep (1-D, 2-D, zone-constrained) | [DC Location Planning](../examples/dc-location-planning.md) |
| `analyze_LLM_load_shifting.py` | Cross-site LLM replica shifting comparison | [Multi-DC Coordination](../examples/multi-dc-coordination.md) |
| `optimize_pv_locations_and_capacities.py` | PV placement + capacity MILP | [PV + DC Siting](../examples/pv-dc-siting.md) |
| `optimize_pv_and_dc_locations.py` | Joint PV + DC location MILP | [PV + DC Siting](../examples/pv-dc-siting.md) |
