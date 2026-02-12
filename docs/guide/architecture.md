# Architecture

This page describes how the components of OpenG2G fit together.

## Simulation Loop

The `Coordinator` drives the simulation. It computes a base tick as the GCD of all component periods and advances a `SimulationClock` each tick. At each tick, it checks which components are due and dispatches accordingly:

```
for each tick:
    1. if datacenter is due:  dc_state = datacenter.step(clock)
    2. if grid is due:        grid_state = grid.step(clock, dc_buffer)
    3. for each controller:
       if controller is due:  action = controller.step(clock, dc_state, grid_state)
                              apply action to datacenter and/or grid
```

The clock uses integer tick counting to avoid floating-point drift. Components check `clock.is_due(period)` to determine if they should run.

## Component Interfaces

### DatacenterBackend

Defined in `openg2g.datacenter.base`:

```python
class DatacenterBackend(ABC):
    @property
    def dt_s(self) -> float: ...
    def step(self, clock: SimulationClock) -> DatacenterState: ...
    def apply_control(self, action: ControlAction) -> None: ...
```

The `step()` method returns a `DatacenterState` containing three-phase power. The coordinator accumulates these into a buffer that is flushed to the grid at each grid step.

Two implementations ship with OpenG2G:

- **`OfflineDatacenter`** replays pre-recorded GPU power traces with configurable noise, jitter, ramp profiles, and training overlays.
- **`OnlineDatacenter`** reads live GPU power via Zeus and dispatches batch size changes through a callback.

### OpenDSSGrid

Defined in `openg2g.grid.opendss`:

```python
class OpenDSSGrid:
    @property
    def dt_s(self) -> float: ...
    def step(self, clock: SimulationClock, load_trace_w: list[ThreePhase]) -> GridState: ...
    def apply_control(self, action: ControlAction) -> None: ...
```

The grid receives a list of power samples accumulated since the last grid step. It supports three sub-step modes:

| Mode | Behavior | Use case |
|---|---|---|
| `"all"` | One DSS solve per sample | Baseline (exact) |
| `"resample"` | Interpolate to 2 DSS points | OFO (matches original) |
| `"last"` | Solve only the last sample | Fast approximation |

The grid returns a `GridState` containing per-bus, per-phase voltages.

### Controller

Defined in `openg2g.controller.base`:

```python
class Controller(ABC):
    @property
    def dt_s(self) -> float: ...
    def step(self, clock, dc_state, grid_state) -> ControlAction: ...
```

Controllers receive the latest datacenter and grid state and return a `ControlAction`. Actions can contain:

- `batch_size_by_model`: new batch sizes to apply to the datacenter
- `tap_changes`: regulator tap position changes to apply to the grid

Multiple controllers compose in order within the coordinator.

## Data Flow

```
Power traces (CSV)
       |
       v
OfflineDatacenter ──power_w──> DC buffer ──> OpenDSSGrid
       ^                                          |
       |                                     voltages
       |                                          |
       └──── batch_size ──── Controller <─────────┘
```

1. The datacenter generates three-phase power at its native rate.
2. Power samples accumulate in a buffer until the grid is due.
3. The grid runs power flow and returns bus voltages.
4. Controllers read voltages and datacenter state, then emit control actions.
5. Control actions (batch sizes, tap changes) are applied to the datacenter and grid.

## State Types

All state objects are frozen dataclasses defined in `openg2g.types`:

| Type | Fields | Source |
|---|---|---|
| `ThreePhase` | `a`, `b`, `c` | Everywhere |
| `DatacenterState` | `time_s`, `power_w`, `batch_size_by_model`, `active_replicas_by_model` | `DatacenterBackend.step()` |
| `OfflineDatacenterState` | + `power_by_model_w`, `avg_itl_by_model` | `OfflineDatacenter.step()` |
| `GridState` | `time_s`, `voltages: BusVoltages` | `OpenDSSGrid.step()` |
| `ControlAction` | _(base, no-op)_ | `Controller.step()` |
| `DatacenterControlAction` | `batch_size_by_model` | OFO controller |
| `GridControlAction` | `tap_changes` | Tap schedule controller |

## SimulationLog

The `Coordinator.run()` method returns a `SimulationLog` that accumulates:

- All datacenter states, grid states, and control actions
- Time-series arrays for DC bus voltages and per-phase power
- Per-model batch size history
