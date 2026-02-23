# Composing Components

This page shows how to assemble a simulation from OpenG2G's building blocks.

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
