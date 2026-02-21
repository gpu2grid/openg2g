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
)
log = coord.run()
```

The coordinator computes the base tick as the GCD of all component periods. In this example, if the datacenter runs at 0.1s and the grid at 1.0s, the tick is 0.1s.

## Component Lifecycle

Every `run()` call follows the sequence: `reset()` all -> `start()` all -> simulation loop -> `stop()` all. Calling `run()` twice produces identical results.

- **`reset()`** (abstract) clears simulation state (history, RNGs, counters). Configuration is not affected.
- **`start()`** (no-op by default) acquires per-run resources. `OpenDSSGrid` compiles its DSS circuit here; most offline components don't override this.
- **`stop()`** (no-op by default) releases per-run resources. Simulation state is preserved for inspection.

## Setting Up the Datacenter

### Offline (Trace Replay)

The `OfflineDatacenter` replays CSV power traces. You first load traces into a cache, build periodic templates, then create the datacenter.

#### Direct construction

```python
from openg2g.datacenter.offline import (
    OfflineDatacenter,
    TraceByBatchCache,
    load_traces_by_batch_from_dir,
)
from openg2g.models.spec import LLMInferenceModelSpec

models = [
    LLMInferenceModelSpec(
        model_label="Llama-3.1-8B", num_replicas=720, gpus_per_replica=1, initial_batch_size=128,
    ),
    LLMInferenceModelSpec(
        model_label="Llama-3.1-70B", num_replicas=180, gpus_per_replica=4, initial_batch_size=128,
    ),
]

traces_by_batch = load_traces_by_batch_from_dir(
    base_dir="power_csvs_updated",
    batch_set=[128],
    required_measured_gpus={m.model_label: m.gpus_per_replica for m in models},
)

cache = TraceByBatchCache.from_traces(traces_by_batch, duration_s=3600.0, timestep_s=0.1)

dc = OfflineDatacenter(
    trace_cache=cache,
    models=models,
    timestep_s=0.1,
    gpus_per_server=8,
    seed=0,
    ramp_t_start=2500.0,
    ramp_t_end=3000.0,
    ramp_floor=0.2,
    base_kW_per_phase=500.0,
)
```

#### Using config objects

For more complex setups (training overlays, server ramp schedules), use the `from_config()` factory with `DatacenterConfig` and `WorkloadConfig`:

```python
from openg2g.datacenter.config import DatacenterConfig, WorkloadConfig
from openg2g.datacenter.offline import OfflineDatacenter, TraceByBatchCache
from openg2g.models.spec import LLMInferenceModelSpec, LLMInferenceWorkload
from openg2g.types import ServerRamp, TrainingRun

workload = WorkloadConfig(
    inference=LLMInferenceWorkload(models=(
        LLMInferenceModelSpec(
            model_label="Llama-3.1-8B", num_replicas=720, gpus_per_replica=1, initial_batch_size=128,
        ),
        LLMInferenceModelSpec(
            model_label="Llama-3.1-70B", num_replicas=180, gpus_per_replica=4, initial_batch_size=128,
        ),
    )),
    training=TrainingRun(t_start=1000.0, t_end=2000.0, n_gpus=2400),
    server_ramps=ServerRamp(t_start=2500.0, t_end=3000.0, floor=0.2),
)

dc = OfflineDatacenter.from_config(
    datacenter=DatacenterConfig(gpus_per_server=8, base_kW_per_phase=500.0),
    workload=workload,
    trace_cache=cache,
    timestep_s=0.1,
    seed=0,
)
```

### Online (Live GPU)

The `OnlineDatacenter` connects to real vLLM servers for load generation and ITL measurement, and to zeusd instances for live GPU power monitoring. Power readings from a small number of real GPUs are augmented to datacenter scale using temporal staggering.

```python
from zeus.monitor.power_streaming import PowerStreamingClient
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
    gpu_endpoints={ep.endpoint_key: list(ep.gpu_indices) for d in deployments for ep in d.gpu_endpoints}
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

Tap schedules are built using `TapPosition` (per-unit tap ratios per phase) and the `|` operator:

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
    controls_off=False,       # True for OFO (SolveNoControl)
    tap_schedule=tap_schedule,
    freeze_regcontrols=True,
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
)
```

Tap changes are typically configured via the `tap_schedule` parameter on `OpenDSSGrid` (using the `TapPosition` fluent API) rather than through `TapScheduleController`. Actions from all controllers are applied before the next tick.

### Command Kinds

Built-in controllers currently emit two command kinds:

- `set_batch_size` (target `datacenter`) with payload key `batch_size_by_model`
- `set_taps` (target `grid`) with payload key `tap_changes`

Backends validate command payloads and raise clear errors on unsupported kinds.

Controller interface summary:

- `step(clock, datacenter, grid, events) -> ControlAction`
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
    live=True,
)
```

In live mode, the clock synchronizes with wall time. If computation falls behind real time, a warning is issued.

## Analyzing Results

The `SimulationLog` returned by `coord.run()` contains all state and action history:

```python
log = coord.run()

# Voltage statistics
from openg2g.metrics.voltage import compute_allbus_voltage_stats
stats = compute_allbus_voltage_stats(log.grid_states, v_min=0.95, v_max=1.05)
print(f"Violation time: {stats.violation_time_s:.1f} s")

# Plotting (see examples/plotting.py for reusable plot functions)
from examples.plotting import plot_power_3ph, plot_allbus_voltages_per_phase
import numpy as np

plot_power_3ph(
    np.array(log.time_s),
    np.array(log.kW_A),
    np.array(log.kW_B),
    np.array(log.kW_C),
    save_path="power.png",
)
```

## Configuration Sweeps

Since `run()` calls `reset()` on all components before each run, you can reuse expensive objects across a parameter sweep:

```python
from openg2g.coordinator import Coordinator
from openg2g.grid.opendss import OpenDSSGrid

grid = OpenDSSGrid(...)  # stores config only (cheap)

for batch_init in [64, 128, 256, 512]:
    dc = OfflineDatacenter(trace_cache=cache, models=models, ...)
    ctrl = OFOBatchController(models=models, ...)
    coord = Coordinator(dc, grid, [ctrl], total_duration_s=3600)
    log = coord.run()  # reset -> start (compile DSS) -> loop -> stop
    print(f"batch_init={batch_init}: violation={stats.integral_violation:.2f}")
```

Each `run()` resets simulation state, then `start()` compiles a fresh DSS circuit, so every iteration starts clean despite reusing the same `OpenDSSGrid` instance.
