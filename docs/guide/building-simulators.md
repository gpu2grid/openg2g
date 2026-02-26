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

from openg2g.datacenter.config import DatacenterConfig, PowerAugmentationConfig, ServerRamp, TrainingRun
from openg2g.datacenter.offline import OfflineDatacenter, OfflineInferenceData, OfflineWorkload, PowerTraceStore
from openg2g.datacenter.training_overlay import TrainingTrace
from openg2g.models.spec import LLMInferenceModelSpec, LLMInferenceWorkload

store = PowerTraceStore.load("data/generated/traces_summary.csv")
templates = store.build_templates(duration_s=3600.0, dt_s=Fraction(1, 10))

training_trace = TrainingTrace.load("data/generated/synthetic_training_trace.csv")

workload = OfflineWorkload(
    inference_data=OfflineInferenceData(
        LLMInferenceWorkload(models=(
            LLMInferenceModelSpec(
                model_label="Llama-3.1-8B", num_replicas=720, gpus_per_replica=1, initial_batch_size=128,
            ),
            LLMInferenceModelSpec(
                model_label="Llama-3.1-70B", num_replicas=180, gpus_per_replica=4, initial_batch_size=128,
            ),
        )),
        power_templates=templates,
    ),
    server_ramps=ServerRamp(t_start=2500.0, t_end=3000.0, target=0.2),
    training=TrainingRun(t_start=1000.0, t_end=2000.0, n_gpus=2400, trace=training_trace),
)

dc = OfflineDatacenter(
    DatacenterConfig(gpus_per_server=8, base_kw_per_phase=500.0),
    workload,
    dt_s=Fraction(1, 10),
    seed=0,
    power_augmentation=PowerAugmentationConfig(
        amplitude_scale_range=(0.98, 1.02),
        noise_fraction=0.005,
    ),
)
```

### Grid

Construct an [`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid] with a DSS case directory and tap schedule. Tap schedules are built using [`TapPosition`][openg2g.types.TapPosition] and the `|` operator:

```python
from fractions import Fraction

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
    dt_s=Fraction(1, 10),
    connection_type="wye",
)
```

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
kW_A = np.array(log.kW_A)
kW_B = np.array(log.kW_B)
kW_C = np.array(log.kW_C)
```

See [`examples/offline/plotting.py`](https://github.com/gpu2grid/openg2g/blob/master/data/offline/plotting.py) for reusable plot functions used by the example scripts.

## Writing Custom Components

OpenG2G is designed for extensibility.
Subclass the abstract base classes to implement your own datacenter backends and controllers, or subclass an existing implementation to create a custom variant.
For background on the type system and generics, see [Architecture: State Types and Generics](architecture.md#state-types-and-generics).

### Custom Datacenter

**Datacenter backend.**
Subclass [`DatacenterBackend`][openg2g.datacenter.base.DatacenterBackend] and parameterize it with the state type your backend returns from [`step()`][openg2g.datacenter.base.DatacenterBackend.step]. The coordinator calls `step()` at the rate specified by [`dt_s`][openg2g.datacenter.base.DatacenterBackend.dt_s]; each call should return a [`DatacenterState`][openg2g.datacenter.base.DatacenterState] (or subclass) with three-phase power in watts. Between steps, the coordinator may call [`apply_control(command)`][openg2g.datacenter.base.DatacenterBackend.apply_control] with individual [`DatacenterCommand`][openg2g.types.DatacenterCommand] instances; changes take effect on the next `step()`. Current state is available through [`state`][openg2g.datacenter.base.DatacenterBackend.state] and past states through [`history(n)`][openg2g.datacenter.base.DatacenterBackend.history].

[`reset()`][openg2g.datacenter.base.DatacenterBackend.reset] must clear all simulation state (history, counters, RNG state) while preserving configuration (`dt_s`, models, templates). Override [`start()`][openg2g.datacenter.base.DatacenterBackend.start] and [`stop()`][openg2g.datacenter.base.DatacenterBackend.stop] for backends that acquire resources (connections, solver circuits); the coordinator calls `start()` before the simulation loop and `stop()` in LIFO order afterward. See [Architecture: Component Lifecycle](architecture.md#component-lifecycle) for the full sequence.

```python
from __future__ import annotations

import functools

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

    @functools.singledispatchmethod
    def apply_control(self, command: DatacenterCommand) -> None:
        raise TypeError(f"SyntheticDatacenter does not support {type(command).__name__}")

    @apply_control.register
    def _apply_set_batch_size(self, command: SetBatchSize) -> None:
        self._batch.update({str(k): int(v) for k, v in command.batch_size_by_model.items()})
```

**Datacenter state.**
For richer state, define a [`DatacenterState`][openg2g.datacenter.base.DatacenterState] subclass:

```python
@dataclass(frozen=True)
class MyState(DatacenterState):
    per_gpu_power_w: dict[int, float] = field(default_factory=dict)

class MyDatacenter(DatacenterBackend[MyState]):
    def step(self, clock: SimulationClock) -> MyState:
        ...
```

The state type propagates through to [`SimulationLog`][openg2g.coordinator.SimulationLog], so `log.dc_states` will be correctly typed as `list[MyState]`.

**Datacenter commands.**
For custom commands, subclass [`DatacenterCommand`][openg2g.types.DatacenterCommand] and register a handler with `@apply_control.register`:

```python
from dataclasses import dataclass
from openg2g.types import DatacenterCommand

@dataclass(frozen=True)
class SetPowerCap(DatacenterCommand):
    cap_kw: float

class MyDatacenter(DatacenterBackend[DatacenterState]):
    @functools.singledispatchmethod
    def apply_control(self, command: DatacenterCommand) -> None:
        raise TypeError(f"MyDatacenter does not support {type(command).__name__}")

    @apply_control.register
    def _apply_set_batch_size(self, command: SetBatchSize) -> None:
        self._batch.update(command.batch_size_by_model)

    @apply_control.register
    def _apply_set_power_cap(self, command: SetPowerCap) -> None:
        self._power_cap = command.cap_kw
```

The coordinator routes commands to backends based on their type hierarchy: [`DatacenterCommand`][openg2g.types.DatacenterCommand] subclasses go to the datacenter, [`GridCommand`][openg2g.types.GridCommand] subclasses go to the grid.

**Testing.** [`DatacenterState`][openg2g.datacenter.base.DatacenterState] and [`GridState`][openg2g.grid.base.GridState] are simple frozen dataclasses, so you can construct them directly in unit tests without running a full simulation.

### Custom Controller

**Controller.**
Subclass [`Controller`][openg2g.controller.base.Controller] and parameterize it with the datacenter and grid backend types it works with. The coordinator calls [`step(clock, datacenter, grid, events)`][openg2g.controller.base.Controller.step] at the rate specified by [`dt_s`][openg2g.controller.base.Controller.dt_s]; each call must return a [`ControlAction`][openg2g.types.ControlAction] containing commands for the datacenter and/or grid (use `ControlAction(commands=[])` for a no-op). Read current state via [`datacenter.state`][openg2g.datacenter.base.DatacenterBackend.state] and [`grid.state`][openg2g.grid.base.GridBackend.state], and history via [`datacenter.history(...)`][openg2g.datacenter.base.DatacenterBackend.history] and [`grid.history(...)`][openg2g.grid.base.GridBackend.history]. Use [`events.emit(topic, data)`][openg2g.events.EventEmitter.emit] to log controller-side events. Keep `step()` fast since it runs synchronously in the simulation loop.

[`reset()`][openg2g.controller.base.Controller.reset] must clear all simulation state (e.g., dual variables, counters, cached matrices) while preserving configuration. Override [`start()`][openg2g.controller.base.Controller.start] and [`stop()`][openg2g.controller.base.Controller.stop] if the controller acquires external resources. See [Architecture: Component Lifecycle](architecture.md#component-lifecycle) for the full sequence.

```python
from __future__ import annotations

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.events import EventEmitter
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.types import ControlAction, SetBatchSize


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
    ) -> ControlAction:
        if grid.state is None:
            return ControlAction(commands=[])

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

- **`step(clock)`** indexes into pre-built per-GPU power templates using `(global_step + offset) % template_length` per server. Random restart offsets desynchronize servers for a realistic aggregate power profile. Returns an [`OfflineDatacenterState`][openg2g.datacenter.offline.OfflineDatacenterState] (extends [`LLMDatacenterState`][openg2g.datacenter.base.LLMDatacenterState]) with per-model batch sizes, replica counts, and observed ITL.
- **`apply_control(command)`** dispatches [`SetBatchSize`][openg2g.types.SetBatchSize] commands. Batch size changes take effect immediately on the next `step()` call.
- **`reset()`** clears history, step counter, and RNG state. Rebuilds server layouts from the stored config so the next run starts fresh.
- An [`ActivationStrategy`][openg2g.datacenter.layout.ActivationStrategy] determines which servers are active at each timestep. The default [`RampActivationStrategy`][openg2g.datacenter.layout.RampActivationStrategy] follows a [`ServerRampSchedule`][openg2g.datacenter.config.ServerRampSchedule] with random priority ordering.
- Training workload overlays add transient high-power phases.

### `OpenDSSGrid`

[`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid] implements [`GridBackend`][openg2g.grid.base.GridBackend] using the OpenDSS power flow solver on standard IEEE test feeders.

How it implements the interface:

- **`step(clock, power_samples_w)`** takes the most recent power sample from the accumulated buffer and runs a single OpenDSS power flow solve. If no samples are provided (grid runs faster than datacenter), the last known power is reused. Returns a [`GridState`][openg2g.grid.base.GridState] with per-bus, per-phase voltages and tap positions.
- **`apply_control(command)`** dispatches [`SetTaps`][openg2g.types.SetTaps] commands to update regulator tap positions.
- **`reset()`** clears history, cached state, and the `_started` flag. The DSS circuit is recompiled on the next `start()`.
- **`start()`** compiles the DSS circuit from the case files, builds the bus-phase voltage index (`v_index`), and prepares snapshot indexing structures.
- **`voltages_vector()`** returns a flat numpy array of all bus-phase voltages in `v_index` order (used by the OFO controller for gradient computation).
- **`estimate_sensitivity(perturbation_kw)`** computes a finite-difference estimate of the voltage sensitivity matrix dV/dP.

### `OFOBatchSizeController`

[`OFOBatchSizeController`][openg2g.controller.ofo.OFOBatchSizeController] implements [`Controller`][openg2g.controller.base.Controller] using Online Feedback Optimization (primal-dual) to regulate batch sizes for voltage safety. For the full mathematical formulation, see the [G2G paper](https://arxiv.org/abs/2602.05116).

How it implements the interface:

- **`step(clock, datacenter, grid, events)`** reads `active_replicas_by_model` and `observed_itl_s_by_model` from `datacenter.state`, and `phase_share_by_model` from the datacenter itself. On the grid side, it calls `grid.v_index`, `grid.voltages_vector()`, and `grid.estimate_sensitivity()`. Returns a [`ControlAction`][openg2g.types.ControlAction] containing [`SetBatchSize`][openg2g.types.SetBatchSize] commands.
- **`reset()`** clears dual variables (voltage and latency multipliers), primal state, step counters, and the cached sensitivity matrix.
- Binds its generic type parameters to [`LLMBatchSizeControlledDatacenter`][openg2g.datacenter.base.LLMBatchSizeControlledDatacenter] and [`OpenDSSGrid`][openg2g.grid.opendss.OpenDSSGrid], since it requires LLM-specific state fields and OpenDSS-specific methods (`voltages_vector`, `estimate_sensitivity`).
