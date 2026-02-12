# Architecture

This page describes how the components of OpenG2G fit together.

## Simulation Loop

The `Coordinator` drives the simulation. It computes a base tick as the GCD of all component periods and advances a `SimulationClock` each tick. At each tick, it checks which components are due and dispatches accordingly:

```
for each tick:
    1. if datacenter is due:  dc_state = datacenter.step(clock)
    2. if grid is due:        grid_state = grid.step(clock, dc_buffer)
    3. for each controller:
       if controller is due:  action = controller.step(clock, dc_state, grid_state, context)
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
    def apply_control(self, command: Command) -> None: ...
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
    def apply_control(self, command: Command) -> None: ...
```

The grid receives a list of power samples accumulated since the last grid step.

The grid returns a `GridState` containing per-bus, per-phase voltages.

### Controller

Defined in `openg2g.controller.base`:

```python
class Controller(ABC):
    @property
    def dt_s(self) -> float: ...
    def step(self, clock, dc_state, grid_state, context) -> ControlAction: ...
```

Controllers receive the latest datacenter and grid state plus a context of feature interfaces. They return a `ControlAction` containing command envelopes.

Current built-in command kinds:
- `target="datacenter", kind="set_batch_size"` with `payload["batch_size_by_model"]`
- `target="grid", kind="set_taps"` with `payload["tap_changes"]`

Multiple controllers compose in order within the coordinator.

## Feature Interfaces (EE-facing)

A controller can declare which context features it needs:

| Feature name | Meaning | Provided by |
|---|---|---|
| `voltage` | Current voltage vector in fixed bus/phase order | `OpenDSSGrid` |
| `sensitivity` | Voltage sensitivity estimator (`estimate_H`) | `OpenDSSGrid` |

If a required feature is missing, the coordinator fails at startup with a plain-language error.

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
5. Control commands are routed to datacenter/grid targets.

## State Types

All state objects are frozen dataclasses defined in `openg2g.types`:

| Type | Fields | Source |
|---|---|---|
| `ThreePhase` | `a`, `b`, `c` | Everywhere |
| `DatacenterState` | `time_s`, `power_w`, `batch_size_by_model`, `active_replicas_by_model` | `DatacenterBackend.step()` |
| `OfflineDatacenterState` | + `power_by_model_w`, `avg_itl_by_model` | `OfflineDatacenter.step()` |
| `GridState` | `time_s`, `voltages: BusVoltages` | `OpenDSSGrid.step()` |
| `Command` | `target`, `kind`, `payload`, `metadata` | `Controller.step()` |
| `ControlAction` | `commands: list[Command]` | `Controller.step()` |

## SimulationLog

The `Coordinator.run()` method returns a `SimulationLog` that accumulates:

- All datacenter states, grid states, and control actions
- Time-series arrays for DC bus voltages and per-phase power
- Per-model batch size history
