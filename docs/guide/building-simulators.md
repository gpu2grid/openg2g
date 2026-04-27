# Building Simulators

This page walks through building your own simulator for your scenario using OpenG2G components.
For how the components fit together conceptually, see [Architecture](architecture.md).

## Essential Components

A simulator needs three things: a datacenter backend, a grid backend, and at least one controller.
You wire them together with the [`Coordinator`][openg2g.coordinator.Coordinator]:

```python
from openg2g.coordinator import Coordinator

coord = Coordinator(
    datacenters=[dc],
    grid=grid,
    controllers=[controller1, controller2, ...],
    total_duration_s=3600,
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
    DatacenterConfig, InferenceModelSpec, ReplicaSchedule,
    ModelDeployment, PowerAugmentationConfig, TrainingRun,
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.datacenter.workloads.training import TrainingTrace

spec_8b = InferenceModelSpec(
    model_label="Llama-3.1-8B", model_id="meta-llama/Llama-3.1-8B-Instruct",
    gpu_model="H100", task="lm-arena-chat",
    gpus_per_replica=1, tensor_parallel=1, itl_deadline_s=0.08,
    batch_sizes=(8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768),
    feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512),
)
spec_70b = InferenceModelSpec(
    model_label="Llama-3.1-70B", model_id="meta-llama/Llama-3.1-70B-Instruct",
    gpu_model="H100", task="lm-arena-chat",
    gpus_per_replica=4, tensor_parallel=4, itl_deadline_s=0.10,
    batch_sizes=(8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024),
    feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512),
)
models = (spec_8b, spec_70b)

data_dir = Path("data/specs")
inference_data = InferenceData.ensure(data_dir, models, dt_s=0.1)
training_trace = TrainingTrace.ensure(Path("data/training_trace.csv"))

workload = OfflineWorkload(
    inference_data=inference_data,
    replica_schedules={
        "Llama-3.1-8B": ReplicaSchedule(initial=720).ramp_to(360, t_start=2500.0, t_end=3000.0),
        "Llama-3.1-70B": ReplicaSchedule(initial=180).ramp_to(90, t_start=2500.0, t_end=3000.0),
    },
    training=TrainingRun(n_gpus=2400, trace=training_trace).at(t_start=1000.0, t_end=2000.0),
)

dc_config = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=500.0)
dc = OfflineDatacenter(
    dc_config,
    workload,
    name="dc",
    dt_s=Fraction(1, 10),
    seed=0,
    total_gpu_capacity=1440,
    power_augmentation=PowerAugmentationConfig(
        amplitude_scale_range=(0.98, 1.02),
        noise_fraction=0.005,
    ),
)
```

[`ReplicaSchedule`][openg2g.datacenter.config.ReplicaSchedule] specifies each model's replica count over time. The initial count and optional ramps are defined in one object using method chaining: `ReplicaSchedule(initial=720).ramp_to(360, t_start=1500, t_end=2000)`. The datacenter validates that ramp targets do not exceed `total_gpu_capacity` before running.

!!! Warning "GPU capacity validation"
    The `total_gpu_capacity` parameter is the physical GPU count at the site and must be specified explicitly. At initialization, the datacenter checks that replica schedules never exceed this capacity at any ramp boundary.

### Grid

Construct an [`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid] with a DSS case directory, then attach dynamic components to specific buses. The DSS file defines the base network (lines, transformers, regulators, static loads). Attached components add dynamic power sources and sinks on top -- their power is updated each timestep during simulation.

The grid looks up bus voltage levels from the DSS model automatically, so you never need to specify `bus_kv`. See [`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid] for additional options (`source_pu`, `exclude_buses`, etc.).

```python
from fractions import Fraction

from openg2g.grid.opendss import OpenDSSGrid
from openg2g.grid.config import TapPosition
from openg2g.grid.generator import SyntheticPV
from openg2g.grid.load import SyntheticLoad
from openg2g.grid.storage import BatteryStorage

TAP_STEP = 0.00625  # standard 5/8% tap step

grid = OpenDSSGrid(
    dss_case_dir="data/grid/ieee13",
    dss_master_file="IEEE13Bus.dss",
    dt_s=Fraction(1, 10),
    initial_tap_position=TapPosition(
        regulators={"creg1a": 1.0 + 14 * TAP_STEP, "creg1b": 1.0 + 6 * TAP_STEP, "creg1c": 1.0 + 15 * TAP_STEP}
    ),
)

# Attach a datacenter load at bus 671
grid.attach_dc(dc, bus="671")

# Attach a PV generator at bus 675 (injected as negative load)
grid.attach_generator(SyntheticPV(peak_kw=10.0), bus="675")

# Attach a time-varying external load at bus 680
grid.attach_load(SyntheticLoad(peak_kw=500.0), bus="680")

# Attach a battery storage system at the datacenter bus
grid.attach_storage(
    BatteryStorage(
        name="bat_671",
        rated_power_kw=250.0,
        capacity_kwh=500.0,
        apparent_power_kva=300.0,
    ),
    bus="671",
)
```

All `attach_*` calls must happen before `start()` (which the [`Coordinator`][openg2g.coordinator.Coordinator] calls automatically). Static loads defined in the DSS file coexist with attached dynamic components -- the power flow sees both.

#### Generators and loads

[`Generator`][openg2g.grid.generator.Generator] and [`ExternalLoad`][openg2g.grid.load.ExternalLoad] are abstract base classes with a single method `power_kw(t)` that returns real power at simulation time `t`. Built-in implementations:

| Class | Description |
|-------|-------------|
| [`SyntheticPV`][openg2g.grid.generator.SyntheticPV] | Demonstration PV profile with cloud dips and trends |
| [`ConstantGenerator`][openg2g.grid.generator.ConstantGenerator] | Fixed power output |
| [`CSVProfileGenerator`][openg2g.grid.generator.CSVProfileGenerator] | Interpolated from a CSV time series |
| [`SyntheticLoad`][openg2g.grid.load.SyntheticLoad] | Demonstration load with diurnal bumps |
| [`ConstantLoad`][openg2g.grid.load.ConstantLoad] | Fixed power consumption |
| [`CSVProfileLoad`][openg2g.grid.load.CSVProfileLoad] | Interpolated from a CSV time series |

The CSV variants expect a two-column file (time in seconds, power in kW) with a header row; values between samples are linearly interpolated.

Subclass `Generator` or `ExternalLoad` to implement custom profiles (e.g., real weather-driven PV, measured load traces).

#### Energy storage

[`EnergyStorage`][openg2g.grid.storage.EnergyStorage] resources are bidirectional, stateful grid resources. Attach them with [`OpenDSSGrid.attach_storage`][openg2g.grid.opendss.OpenDSSGrid.attach_storage]:

```python
from openg2g.grid.storage import BatteryStorage

storage = BatteryStorage(
    name="bat_671",
    rated_power_kw=250.0,
    capacity_kwh=500.0,
    initial_soc=0.5,
    apparent_power_kva=300.0,
)
grid.attach_storage(storage, bus="671")
```

The OpenDSS backend creates a native `Storage` element during `start()`. The base DSS files are not modified; the element is added to the compiled circuit for the simulation run. Storage attachments must happen before `start()`, like the other `attach_*` APIs.

Positive real power discharges the battery into the grid, while negative real power charges it from the grid. Positive reactive power injects kvar; negative reactive power absorbs kvar. [`BatteryStorage`][openg2g.grid.storage.BatteryStorage] supports externally held setpoints via [`SetStoragePower`][openg2g.grid.command.SetStoragePower]:

```python
from openg2g.grid.command import SetStoragePower

# Discharge 50 kW.
SetStoragePower(storage_name="bat_671", power_kw=50.0)

# Charge 25 kW.
SetStoragePower(storage_name="bat_671", power_kw=-25.0)

# Pure reactive support: inject 100 kvar without a real-power command.
SetStoragePower(storage_name="bat_671", power_kw=0.0, reactive_power_kvar=100.0)
```

Controllers emit these commands and the coordinator routes them to the grid backend. The storage object holds the latest command between controller ticks, so a slower controller setpoint affects multiple faster grid/storage timesteps.

Use grid query helpers to discover and monitor storage resources:

```python
if grid.has_storage:
    for name in grid.storage_names:
        bus = grid.storage_bus(name)
        state = grid.storage_state(name)
        print(name, bus, state.soc, state.power_kw, state.reactive_power_kvar)
```

The state returned by [`storage_state`][openg2g.grid.opendss.OpenDSSGrid.storage_state] is read back from OpenDSS after the grid step. Treat realized `power_kw` and `reactive_power_kvar` as simulation results, even when they differ slightly from the commanded value.

For a built-in local voltage policy, use [`LocalVoltageStorageDroopController`][openg2g.controller.storage.LocalVoltageStorageDroopController]:

```python
from fractions import Fraction

from openg2g.controller.storage import LocalVoltageStorageDroopController, StorageDroopConfig

storage_controller = LocalVoltageStorageDroopController(
    grid=grid,
    config=StorageDroopConfig(mode="qv"),
    dt_s=Fraction(1),
)
```

The default Q-V mode emits reactive-power commands using each storage resource's local bus voltage from the previous control window. Set `StorageDroopConfig(mode="pv")` for real-power droop. In both modes, commands are local, fleet-capable, and zero-order held until the next controller step.

#### Tap schedules

Tap *schedules* are built using [`TapPosition`][openg2g.grid.config.TapPosition] and the `|` operator, and passed to a [`TapScheduleController`][openg2g.controller.tap_schedule.TapScheduleController]:

```python
tap_schedule = (
    TapPosition(regulators={"creg1a": 1.0 + 16 * TAP_STEP, "creg1b": 1.0 + 6 * TAP_STEP}).at(t=25 * 60)
    | TapPosition(regulators={"creg1a": 1.0 + 10 * TAP_STEP}).at(t=55 * 60)
)
```

For single-bank systems, the phase shorthand `TapPosition(a=..., b=..., c=...)` can be used instead of regulator names. Multi-bank systems must use explicit regulator names.

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

**Regulator naming.** OpenDSS models voltage regulators as two separate objects: a `Transformer` (the physical hardware) and a `RegControl` (the control logic that adjusts the transformer's tap). Names follow the convention `reg{bank}{phase}` / `creg{bank}{phase}` (the `c` prefix stands for "control"):

| Component | Pattern | Example | Meaning |
|-----------|---------|---------|---------|
| Transformer | `reg{bank}{phase}` | `reg1a` | Bank 1, phase A |
| RegControl | `creg{bank}{phase}` | `creg1a` | Control for bank 1, phase A |

Each `RegControl` points to its transformer: `new regcontrol.creg1a transformer=reg1a winding=2 ...`. A typical bank has three per-phase regulators (e.g., `creg1a`, `creg1b`, `creg1c`).

**Adding a new test system.** When adding a new `.dss` file:

1. Name the master file `IEEE{N}Bus.dss`.
2. Use explicit phase suffixes on regulator transformer buses (e.g., `buses=[650.1 RG60.1]`). The simulator determines phase-to-regulator mapping from bus node data; if phase cannot be determined, users must address regulators by name in `TapPosition`.
3. Follow the `reg{bank}{phase}` / `creg{bank}{phase}` naming convention for regulators.
4. Use `SetBusXY bus=... X=... Y=...` for inline bus coordinates (no external CSV files).

### Controllers

Controllers compose in order. Pass them as a list to the coordinator:

```python
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.controller.ofo import LogisticModelStore, OFOBatchSizeController, OFOConfig

logistic_models = LogisticModelStore.ensure(
    data_dir, models,
)

model_specs = tuple(m.spec for m in models)
coord = Coordinator(
    datacenters=[dc],
    grid=grid,
    controllers=[
        TapScheduleController(schedule=tap_schedule, dt_s=Fraction(1)),
        OFOBatchSizeController(
            model_specs,
            datacenter=dc,
            grid=grid,
            models=logistic_models,
            config=OFOConfig(v_min=0.95, v_max=1.05),
        ),
    ],
    total_duration_s=3600,
)
```

Actions from all controllers are applied before the next tick. Put tap controllers before batch controllers if tap changes should be visible to the batch optimizer within the same tick.

## Analyzing Results

[`Coordinator.run()`][openg2g.coordinator.Coordinator.run] returns a [`SimulationLog`][openg2g.coordinator.SimulationLog] containing all state and action history:

```python
log = coord.run()

# Voltage statistics (grid-side quality)
from openg2g.metrics.voltage import compute_allbus_voltage_stats
vstats = compute_allbus_voltage_stats(log.grid_states, v_min=0.95, v_max=1.05)
print(f"Violation time: {vstats.violation_time_s:.1f} s  (integral {vstats.integral_violation_pu_s:.4f} pu-s)")

# Performance statistics (datacenter-side quality of service)
from openg2g.metrics.performance import compute_performance_stats
pstats = compute_performance_stats(
    log.dc_states,
    itl_deadline_s_by_model={ms.model_label: ms.itl_deadline_s for ms in models},
)
print(f"Throughput: {pstats.mean_throughput_tps / 1e3:.1f} k tok/s  "
      f"(ITL over deadline on {pstats.itl_deadline_fraction * 100:.2f}% of samples)")

# Time-series data for plotting
import numpy as np
time_s = np.array(log.time_s)
dc_time_s = np.array([s.time_s for s in log.dc_states])
kW_A = np.array([s.power_w.a / 1e3 for s in log.dc_states])
kW_B = np.array([s.power_w.b / 1e3 for s in log.dc_states])
kW_C = np.array([s.power_w.c / 1e3 for s in log.dc_states])
```

Result CSVs written by the example scripts include a mode/case label plus the seven metric fields: `violation_time_s`, `integral_violation_pu_s`, `worst_vmin`, `worst_vmax`, `mean_throughput_tps`, `integrated_throughput_tokens`, `itl_deadline_fraction` (column ordering may differ between per-case and comparison CSVs). Voltage-only metrics give you grid operator quality; performance metrics give you DC operator quality. Examples like `analyze_different_controllers.py` show OFO wins on both axes simultaneously where rule-based keeps voltage in bounds but leaves throughput on the table.

See [`examples/offline/plotting.py`](https://github.com/gpu2grid/openg2g/blob/master/examples/offline/plotting.py) for reusable plot functions used by the example scripts.

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
Subclass [`Controller`][openg2g.controller.base.Controller] and parameterize it with the datacenter and grid backend types it works with. Bind the datacenter(s) and grid at construction time. The coordinator calls [`step(clock, events)`][openg2g.controller.base.Controller.step] at the rate specified by [`dt_s`][openg2g.controller.base.Controller.dt_s]; each call must return a list of [`DatacenterCommand`][openg2g.datacenter.command.DatacenterCommand] and/or [`GridCommand`][openg2g.grid.command.GridCommand] objects (return `[]` for a no-op). Read current state via the stored datacenter and grid references (e.g., `self._datacenter.state`, `self._grid.state`). Use [`events.emit(topic, data)`][openg2g.events.EventEmitter.emit] to log controller-side events. Keep `step()` fast since it runs synchronously in the simulation loop. All datacenter commands must set `target` to the datacenter they apply to.

[`reset()`][openg2g.controller.base.Controller.reset] must clear all simulation state (e.g., dual variables, counters, cached matrices) while preserving configuration. Override [`start()`][openg2g.controller.base.Controller.start] and [`stop()`][openg2g.controller.base.Controller.stop] if the controller acquires external resources. See [Architecture: Component Lifecycle](architecture.md#component-lifecycle) for the full sequence.

```python
from __future__ import annotations

from fractions import Fraction

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.datacenter.command import DatacenterCommand, SetBatchSize
from openg2g.events import EventEmitter
from openg2g.grid.command import GridCommand
from openg2g.grid.opendss import OpenDSSGrid


class MyController(Controller[MyDatacenter, OpenDSSGrid]):
    """A controller that reduces batch size when any voltage is below a threshold."""

    def __init__(self, *, datacenter: MyDatacenter, grid: OpenDSSGrid, v_threshold: float = 0.96, dt_s: Fraction = Fraction(1)):
        self._datacenter = datacenter
        self._grid = grid
        self._v_threshold = v_threshold
        self._dt_s = dt_s

    @property
    def dt_s(self) -> Fraction:
        return self._dt_s

    def reset(self) -> None:
        pass

    def step(
        self,
        clock: SimulationClock,
        events: EventEmitter,
    ) -> list[DatacenterCommand | GridCommand]:
        grid = self._grid
        if grid.state is None:
            return []

        for bus in grid.state.voltages.buses():
            tp = grid.state.voltages[bus]
            for v in (tp.a, tp.b, tp.c):
                if v < self._v_threshold:
                    events.emit("controller.low_voltage", {"bus": bus, "time_s": clock.time_s})
                    return [SetBatchSize(batch_size_by_model={"MyModel": 64}, target=self._datacenter)]

        return [SetBatchSize(batch_size_by_model={"MyModel": 128}, target=self._datacenter)]
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
- **`apply_control(command, events)`** dispatches [`SetBatchSize`][openg2g.datacenter.command.SetBatchSize] and [`ShiftReplicas`][openg2g.datacenter.command.ShiftReplicas] commands. Batch size changes take effect on the next `step()` call. `ShiftReplicas` adjusts a per-model replica offset; the controller is responsible for checking GPU capacity before issuing the command.
- **`total_gpu_capacity`**: Maximum number of GPUs this datacenter can physically host. The shared [`ServerPool`][openg2g.datacenter.layout.ServerPool] is sized from this. Exposed via `current_gpu_usage()` and `available_gpu_capacity()` for controllers to check before shifting replicas.
- **`reset()`** clears step counter, replica offsets, and RNG state. Rebuilds the server pool from the stored config so the next run starts fresh. History is cleared automatically by `do_reset()`.
- A shared [`ServerPool`][openg2g.datacenter.layout.ServerPool] holds `num_servers` virtual servers with model-independent properties (phase assignment, stagger offset, amplitude scale) and a per-model priority ordering. At each step, the datacenter computes each model's effective replica count (schedule + runtime offset) and the pool allocates servers via phase-balanced round-robin: each model gets `ceil(gpus_needed / gpus_per_server)` servers picked by cycling through phases to keep the allocation phase-balanced.
- Training workload overlays add transient high-power phases.

### `OpenDSSGrid`

[`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid] implements [`GridBackend`][openg2g.grid.base.GridBackend] using the OpenDSS power flow solver on standard IEEE test feeders.

How it implements the interface:

- **`step(clock, power_samples_w, events)`** takes the most recent power sample from the accumulated buffer and runs a single OpenDSS power flow solve. If no samples are provided (grid runs faster than datacenter), the last known power is reused. Returns a [`GridState`][openg2g.grid.base.GridState] with per-bus, per-phase voltages and tap positions.
- **`apply_control(command, events)`** dispatches [`SetTaps`][openg2g.grid.command.SetTaps] commands to update regulator tap positions and [`SetStoragePower`][openg2g.grid.command.SetStoragePower] commands to update storage setpoints.
- **`reset()`** clears cached state and the `_started` flag. History is cleared automatically by `do_reset()`. The DSS circuit is recompiled on the next `start()`.
- **`start()`** compiles the DSS circuit from the case files, builds the bus-phase voltage index (`v_index`), prepares snapshot indexing structures, and creates native OpenDSS `Storage` elements for any attached storage resources.
- **`voltages_vector()`** returns a flat numpy array of all bus-phase voltages in `v_index` order (used by the OFO controller for gradient computation).
- **`estimate_sensitivity(perturbation_kw)`** computes a finite-difference estimate of the voltage sensitivity matrix dV/dP.
- **Storage helpers**: `has_storage`, `storage_names`, `storage_state(name)`, `storage_bus(name)`, `storage_rated_power_kw(name)`, and `storage_rated_apparent_power_kva(name)` expose storage metadata and OpenDSS readback state without requiring controllers to inspect attachment internals.

### `BatteryStorage`

[`BatteryStorage`][openg2g.grid.storage.BatteryStorage] is a simple mutable setpoint storage resource backed by native OpenDSS `Storage` physics.

How it implements the interface:

- **`power_kw(t)` / `reactive_power_kvar(t)`** return the currently held setpoints. Positive real power discharges; negative real power charges. Positive reactive power injects kvar.
- **`set_power_kw(power_kw, reactive_power_kvar=0.0)`** updates the held setpoint used on subsequent grid steps. The method validates real-power and apparent-power limits before accepting the command.
- **`update_state(state)`** receives [`StorageState`][openg2g.grid.storage.StorageState] read back from OpenDSS after the grid step. The state includes stored energy, SOC, realized real/reactive power, and OpenDSS storage state.
- **`sized_for_datacenter(...)`** creates a battery sized relative to total datacenter real power. The default is 20% of datacenter power and 2 hours of storage duration.

### `OFOBatchSizeController`

[`OFOBatchSizeController`][openg2g.controller.ofo.OFOBatchSizeController] implements [`Controller`][openg2g.controller.base.Controller] using Online Feedback Optimization (primal-dual) to regulate batch sizes for voltage safety. For the full mathematical formulation, see the [G2G paper](https://arxiv.org/abs/2602.05116).

How it implements the interface:

- **`__init__(..., datacenter=, grid=)`** binds the controller to a specific datacenter and grid at construction time.
- **`step(clock, events)`** reads `active_replicas_by_model` and `observed_itl_s_by_model` from `self._datacenter.state`, and `phase_share_by_model` from the datacenter itself. On the grid side, it calls `self._grid.v_index`, `self._grid.voltages_vector()`, and `self._grid.estimate_sensitivity()`. Returns a list of [`SetBatchSize`][openg2g.datacenter.command.SetBatchSize] commands with `target` set explicitly.
- **`reset()`** clears dual variables (voltage and latency multipliers), primal state, step counters, and the cached sensitivity matrix.
- Binds its generic type parameters to [`LLMBatchSizeControlledDatacenter`][openg2g.datacenter.base.LLMBatchSizeControlledDatacenter] and [`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid], since it requires LLM-specific state fields and OpenDSS-specific methods (`voltages_vector`, `estimate_sensitivity`).
- In multi-DC setups, create one controller per DC. Each controller calls `estimate_sensitivity(dc=self._datacenter)` to compute gradients only for its DC's loads.

### `RuleBasedBatchSizeController`

[`RuleBasedBatchSizeController`][openg2g.controller.rule_based.RuleBasedBatchSizeController] implements a proportional rule-based controller for batch-size regulation. Unlike OFO, it requires no sensitivity matrix, no logistic curve fits, and no dual variables, making it a simple baseline for comparison.

How it works:

- **`step(clock, events)`** reads all bus-phase voltages from `self._grid.state.voltages`, finds the worst violation, and adjusts batch sizes proportionally in log2-space. Undervoltage reduces batch (less power draw); overvoltage increases batch.
- Continuous internal state accumulates pressure across steps, enabling gradual batch transitions even with small violations.
- Optional **latency guard** prevents batch increases when ITL exceeds the model's deadline.
- Optional **deadband** (default 0.001 pu) filters tiny violations to prevent chattering.
- Configuration via [`RuleBasedConfig`][openg2g.controller.rule_based.RuleBasedConfig]: `step_size` (proportional gain), `v_min/v_max`, `deadband`, `latency_guard`.

### `LoadShiftController`

[`LoadShiftController`][openg2g.controller.load_shift.LoadShiftController] is a cross-site controller that shifts LLM replicas between datacenters when batch-size control is exhausted and voltage violations persist. It holds references to all DCs and the grid at construction. It must be placed **after** all per-site OFO controllers in the controller list so it sees the latest batch-size state.

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
    OFOBatchSizeController(..., datacenter=dc_a, grid=grid),
    OFOBatchSizeController(..., datacenter=dc_b, grid=grid),
    # Load shift controller must be LAST
    LoadShiftController(
        config=LoadShiftConfig(enabled=True, gpus_per_shift=8),
        dt_s=Fraction(1),
        datacenters=[dc_a, dc_b],
        grid=grid,
        models_by_dc={dc_a: ["Model-A", "Model-B"], dc_b: ["Model-B", "Model-C"]},
        gpus_per_replica_by_model={"Model-A": 1, "Model-B": 4, "Model-C": 8},
        feasible_batch_sizes_by_model={"Model-A": [8, 16, 32], ...},
        v_min=0.95, v_max=1.05,
    ),
]
```

Every `DatacenterCommand` must set `target` to the datacenter it applies to. The `ShiftReplicas` command sets `target` explicitly on both the send and receive sides.

### `LocalVoltageStorageDroopController`

[`LocalVoltageStorageDroopController`][openg2g.controller.storage.LocalVoltageStorageDroopController] implements a local voltage droop policy for one or more attached storage resources.

How it works:

- **Local voltage only**: Each storage resource uses the voltage at its own attachment bus. The controller does not aggregate feeder-wide voltage.
- **Previous-window control**: The controller reads grid history emitted since its previous control tick and reduces the local voltage samples using `voltage_statistic` (`minimum`, `mean`, or `latest`).
- **Q-V mode**: `StorageDroopConfig(mode="qv")` emits reactive-power commands. Positive output injects kvar when local voltage is low; negative output absorbs kvar when local voltage is high.
- **P-V mode**: `StorageDroopConfig(mode="pv")` emits real-power commands. Positive output discharges when local voltage is low; negative output charges when local voltage is high.
- **Fleet targeting**: By default, the controller targets all attached storage resources. Pass `storage_name=` for a single resource or `storage_names=` for a subset.
- **Deadband and clipping**: `deadband_pu` avoids small commands near `v_ref`; `full_output_voltage_error_pu` sets the error where output reaches the storage rating. The output ramps continuously from the deadband edge and is clipped to the storage rating or `max_abs_output`.

## Example Analysis Scripts

The `examples/offline/` directory includes ready-to-run analysis scripts. For detailed usage, examples across IEEE test systems, and interpretation guides, see the [Examples](../examples/) documentation:

| Script | Topic | Details |
|--------|-------|---------|
| `run_ofo.py` | Baseline and OFO simulations (all systems) | [GPU Flexibility](../examples/gpu-flexibility.md), [Voltage Strategies](../examples/voltage-regulation-strategies.md) |
| `analyze_different_controllers.py` | Controller comparison (baseline, rule-based, OFO) | [Voltage Strategies](../examples/voltage-regulation-strategies.md) |
| `sweep_ofo_parameters.py` | OFO parameter sensitivity sweep | [Parameter Sensitivity](../examples/controller-parameter-sensitivity.md) |
| `plot_topology.py` | System topology visualization | [Grid Topology](../examples/grid-topology-effects.md) |
| `sweep_hosting_capacities.py` | Per-bus hosting capacity analysis | [Hosting Capacity](../examples/hosting-capacity.md) |
| `sweep_dc_locations.py` | DC location sweep (1-D, 2-D, zone-constrained) | [DC Location Planning](../examples/dc-location-planning.md) |
| `analyze_LLM_load_shifting.py` | Cross-site LLM replica shifting comparison | [Multi-DC Coordination](../examples/multi-dc-coordination.md) |
| `analyze_storage_coordination.py` | Storage-assisted OFO comparison | [Storage-Assisted Coordination](../examples/storage-coordination.md) |
| `optimize_pv_locations_and_capacities.py` | PV placement + capacity MILP (`--mode pv-only`) and joint PV + DC location MILP (`--mode pv-and-dc`) | [PV + DC Siting](../examples/pv-dc-siting.md) |
