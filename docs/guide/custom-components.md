# Custom Components

OpenG2G is designed for extensibility. You can implement your own datacenter backends and controllers by subclassing the provided abstract base classes.

## Custom Controller

Controllers implement the `Controller` ABC from `openg2g.controller.base`. The `Controller` class is generic over its compatible datacenter and grid backend types:

```python
from __future__ import annotations

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.datacenter.base import DatacenterBackend
from openg2g.events import EventEmitter
from openg2g.grid.base import GridBackend
from openg2g.types import Command, ControlAction, DatacenterState, GridState


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
                        commands=[
                            Command(
                                target="datacenter",
                                kind="set_batch_size",
                                payload={"batch_size_by_model": {"MyModel": 64}},
                            )
                        ]
                    )

        return ControlAction(
            commands=[
                Command(
                    target="datacenter",
                    kind="set_batch_size",
                    payload={"batch_size_by_model": {"MyModel": 128}},
                )
            ]
        )
```

### Controller Guidelines

- `step()` must return a `ControlAction` on every call.
- Use `ControlAction(commands=[])` for a no-op.
- Emit `Command(target="datacenter", kind="set_batch_size", ...)` for batch updates.
- Emit `Command(target="grid", kind="set_taps", ...)` for tap updates.
- Read current component state via `datacenter.state` and `grid.state`.
- Use `datacenter.history(...)` and `grid.history(...)` for non-Markovian logic.
- Keep `step()` fast -- it runs synchronously in the simulation loop.
- Use `clock.time_s` for time-dependent logic.
- Use `events.emit(topic, data)` to log controller-side events.

### Controller Generic Parameters

The two type parameters in `Controller[DC, Grid]` declare which backend types the controller is compatible with. The coordinator checks these at construction time. Common patterns:

- `Controller[DatacenterBackend[DatacenterState], GridBackend[GridState]]` -- works with any backend.
- `Controller[LLMBatchSizeControlledDatacenter[OfflineDatacenterState], OpenDSSGrid]` -- only works with the offline datacenter and OpenDSS grid.
- `Controller[LLMBatchSizeControlledDatacenter[DatacenterState], GridBackend[GridState]]` -- works with any LLM datacenter and any grid.

If your controller inherits from a typed parent, the generic parameters are inherited automatically -- no need to re-specify them.

## Custom Datacenter Backend

Datacenter backends implement the `DatacenterBackend` ABC from `openg2g.datacenter.base`. The ABC is generic over the state type it emits -- parameterize it with the state dataclass your backend returns from `step()`:

```python
from __future__ import annotations

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.datacenter.base import DatacenterBackend
from openg2g.types import Command, DatacenterState, ThreePhase


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

    def apply_control(self, command: Command) -> None:
        if command.kind != "set_batch_size":
            return
        batch_map = command.payload.get("batch_size_by_model", {})
        if isinstance(batch_map, dict):
            self._batch.update({str(k): int(v) for k, v in batch_map.items()})
```

If your backend needs richer state (e.g. per-model power breakdowns), define a `DatacenterState` subclass and use it as the type parameter:

```python
@dataclass(frozen=True)
class MyState(DatacenterState):
    per_gpu_power_w: dict[int, float] = field(default_factory=dict)

class MyDatacenter(DatacenterBackend[MyState]):
    def step(self, clock: SimulationClock) -> MyState:
        ...
```

The state type propagates through the `Coordinator` to the `SimulationLog`, so `log.dc_states` will be correctly typed as `list[MyState]`.

### Datacenter Guidelines

- `step()` is called at the rate specified by `dt_s`.
- Return a `DatacenterState` (or subclass) with three-phase power in watts.
- `apply_control()` receives one command at a time.
- Implement `state` and `history(...)` to expose current/past states to controllers.
- For offline backends, consider using `OfflineDatacenterState` which includes per-model power and replica counts.

## Registering with the Coordinator

Custom components plug directly into the coordinator:

```python
from openg2g.coordinator import Coordinator

dc = SyntheticDatacenter(dt_s=0.1, base_kw=1500.0)
ctrl = MyController(v_threshold=0.96, dt_s=1.0)

coord = Coordinator(
    datacenter=dc,
    grid=grid,
    controllers=[ctrl],
    total_duration_s=3600,
    dc_bus="671",
)
log = coord.run()
```

The coordinator handles all the timing, buffering, and dispatch automatically. Your components just need to implement the interface.

## Tips

- **Multiple controllers**: Controllers run in order. Put tap controllers before batch controllers if tap changes should be visible to the batch optimizer within the same tick.
- **State inspection**: The base `DatacenterState` includes `batch_size_by_model` and `active_replicas_by_model`, so controllers can access per-model batch sizes and replica counts without knowing which backend is in use.
- **Testing**: Write unit tests for your controller by constructing mock `DatacenterState` and `GridState` objects directly -- they are simple frozen dataclasses.
