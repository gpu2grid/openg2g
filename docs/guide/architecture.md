# Architecture

This page describes how the components of OpenG2G fit together.

## Simulation Loop

The `Coordinator` drives the simulation. It computes a base tick as the GCD of all component periods and advances a `SimulationClock` each tick. At each tick, it checks which components are due and dispatches accordingly:

```
for each tick:
    1. if datacenter is due:  dc_state = datacenter.step(clock)
    2. if grid is due:        grid_state = grid.step(clock, dc_buffer)
    3. for each controller:
       if controller is due:  action = controller.step(clock, datacenter, grid, events)
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
    @property
    def state(self) -> DatacenterState | None: ...
    def history(self, n: int | None = None) -> Sequence[DatacenterState]: ...
    def step(self, clock: SimulationClock) -> DatacenterState: ...
    def apply_control(self, command: Command) -> None: ...
```

The `step()` method returns a `DatacenterState` containing three-phase power. The coordinator accumulates these into a buffer that is flushed to the grid at each grid step.

Two implementations ship with OpenG2G:

- **`OfflineDatacenter`** replays pre-recorded GPU power traces with configurable noise, jitter, ramp profiles, and training overlays.
- **`OnlineDatacenter`** reads live GPU power via Zeus and dispatches batch size changes through a callback.

### GridBackend / OpenDSSGrid

Defined in `openg2g.grid.base` (implemented by `openg2g.grid.opendss.OpenDSSGrid`):

```python
class GridBackend(ABC):
    @property
    def dt_s(self) -> float: ...
    @property
    def state(self) -> GridState | None: ...
    def history(self, n: int | None = None) -> Sequence[GridState]: ...
    @property
    def v_index(self) -> list[tuple[str, int]]: ...
    def step(self, clock: SimulationClock, load_trace_w: list[ThreePhase]) -> GridState: ...
    def apply_control(self, command: Command) -> None: ...
    def voltages_vector(self) -> np.ndarray: ...
    def estimate_H(self, dp_kw: float = 100.0) -> tuple[np.ndarray, np.ndarray]: ...
```

The grid receives a list of power samples accumulated since the last grid step.

The grid returns a `GridState` containing per-bus, per-phase voltages.

### Controller

Defined in `openg2g.controller.base`:

```python
class Controller(ABC):
    @property
    def dt_s(self) -> float: ...
    def step(self, clock, datacenter, grid, events) -> ControlAction: ...
```

Controllers receive full datacenter/grid backend objects and a clock-bound event emitter. They return a `ControlAction` containing command envelopes.

Current built-in command kinds:
- `target=CommandTarget.DATACENTER` (`"datacenter"` also accepted), `kind="set_batch_size"` with `payload["batch_size_by_model"]`
- `target=CommandTarget.GRID` (`"grid"` also accepted), `kind="set_taps"` with `payload["tap_changes"]`

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
4. Controllers read datacenter/grid state directly from backend objects, then emit control actions.
5. Control commands are routed to datacenter/grid targets.

## State Types

All state objects are frozen dataclasses defined in `openg2g.types`:

| Type | Fields | Source |
|---|---|---|
| `ThreePhase` | `a`, `b`, `c` | Everywhere |
| `DatacenterState` | `time_s`, `power_w`, `batch_size_by_model`, `active_replicas_by_model` | `DatacenterBackend.step()` |
| `OfflineDatacenterState` | + `power_by_model_w`, `observed_itl_s_by_model`, `active_replicas_by_model` | `OfflineDatacenter.step()` |
| `GridState` | `time_s`, `voltages: BusVoltages` | `OpenDSSGrid.step()` |
| `Command` | `target`, `kind`, `payload`, `metadata` | `Controller.step()` |
| `ControlAction` | `commands: list[Command]` | `Controller.step()` |

## SimulationLog

The `Coordinator.run()` method returns a `SimulationLog` that accumulates:

- All datacenter states, grid states, and control actions
- Time-series arrays for DC bus voltages and per-phase power
- Per-model batch size history
