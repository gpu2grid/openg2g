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
    T_total_s=3600.0,
)
log = coord.run()
```

The coordinator computes the base tick as the GCD of all component periods. In this example, if the datacenter runs at 0.1s and the grid at 1.0s, the tick is 0.1s.

## Setting Up the Datacenter

### Offline (Trace Replay)

The `OfflineDatacenter` replays CSV power traces. You first load traces into a cache, build periodic templates, then create the datacenter:

```python
from openg2g.datacenter.offline import (
    OfflineDatacenter,
    TraceByBatchCache,
    load_traces_by_batch_from_dir,
)
from openg2g.models.spec import ModelSpec

models = [
    ModelSpec(model_label="Llama-3.1-8B", replicas=720, gpus_per_replica=1),
    ModelSpec(model_label="Llama-3.1-70B", replicas=180, gpus_per_replica=4),
]

traces_by_batch = load_traces_by_batch_from_dir(
    base_dir="power_csvs_updated",
    batch_set=[128],
    required_measured_gpus={m.model_label: m.gpus_per_replica for m in models},
)

cache = TraceByBatchCache(traces_by_batch)
cache.build_templates(T=3600.0, dt=0.1)

dc = OfflineDatacenter(
    trace_cache=cache,
    models=models,
    dt=0.1,
    batch_init=128,
    gpus_per_server=8,
    seed=0,
    chunk_steps=36000,
    ramp_t_start=2500.0,
    ramp_t_end=3000.0,
    ramp_floor=0.2,
    base_kW_per_phase=500.0,
)
```

### Online (Live GPU)

The `OnlineDatacenter` reads real-time GPU power via Zeus:

```python
from openg2g.datacenter.online import OnlineDatacenter

dc = OnlineDatacenter(
    gpu_indices=[0, 1, 2, 3],
    dt_s=0.1,
    batch_control_callback=my_batch_setter,
)
```

The `batch_control_callback` is called with a `{model_label: batch_size}` dict whenever the controller changes batch sizes.

## Setting Up the Grid

```python
from openg2g.grid.opendss import OpenDSSGrid

grid = OpenDSSGrid(
    case_dir="OpenDss_Test/13Bus",
    master="IEEE13Nodeckt.dss",
    dc_bus="671",
    dc_kv_ll=4.16,
    pf_dc=0.95,
    dt_s=0.1,
    dc_conn="wye",
    controls_off=False,       # True for OFO (SolveNoControl)
    sub_step_mode="all",      # "all", "resample", or "last"
    tap_schedule=[(0.0, {"reg1": 1.0875, "reg2": 1.0375, "reg3": 1.09375})],
    freeze_regcontrols=True,
)
```

### Sub-step Modes

The `sub_step_mode` controls how the grid processes the DC power buffer:

- **`"all"`** -- one DSS solve per DC sample. Use for baseline simulations where `dt_dss == dt_dc`.
- **`"resample"`** -- interpolates DC samples onto 2 DSS time points via `np.interp`. Use for OFO where `dt_dss > dt_dc`.
- **`"last"`** -- only the last DC sample is solved. Fastest but least accurate.

## Stacking Controllers

Controllers compose in order. Each controller sees the latest state and can emit actions:

```python
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.controller.ofo import OFOBatchController

controllers = [
    TapScheduleController(schedule=tap_schedule, dt_s=1.0),
    OFOBatchController(models=models, fits=fits, ...),
]

coord = Coordinator(
    datacenter=dc,
    grid=grid,
    controllers=controllers,
    T_total_s=3600.0,
)
```

Actions from all controllers are applied before the next tick.

## Live Mode

For hardware-in-the-loop experiments, pass `live=True` to the coordinator:

```python
coord = Coordinator(
    datacenter=dc,
    grid=grid,
    controllers=controllers,
    T_total_s=300.0,
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

# Plotting
from openg2g.plotting import plot_power_3ph, plot_allbus_voltages_per_phase
import numpy as np

plot_power_3ph(
    np.array(log.time_s),
    np.array(log.kW_A),
    np.array(log.kW_B),
    np.array(log.kW_C),
    save_path="power.png",
)
```
