# Custom Components

OpenG2G is designed for extensibility. You can implement your own datacenter backends and controllers by subclassing the provided abstract base classes.

## Custom Controller

Controllers implement the `Controller` ABC from `openg2g.controller.base`:

```python
from __future__ import annotations

from openg2g.clock import SimulationClock
from openg2g.controller.base import Controller
from openg2g.context import SimulationContext
from openg2g.types import Command, ControlAction, DatacenterState, GridState


class MyController(Controller):
    """A controller that reduces batch size when voltage drops below a threshold."""

    def __init__(self, v_threshold: float = 0.96, dt_s: float = 1.0):
        self._v_threshold = v_threshold
        self._dt_s = dt_s

    @property
    def dt_s(self) -> float:
        return self._dt_s

    def step(
        self,
        clock: SimulationClock,
        dc_state: DatacenterState | None,
        grid_state: GridState | None,
        context: SimulationContext,
    ) -> ControlAction:
        if grid_state is None:
            return ControlAction(commands=[])

        # Check if any bus voltage is below threshold
        for bus in grid_state.voltages.buses():
            tp = grid_state.voltages[bus]
            for v in (tp.a, tp.b, tp.c):
                if v < self._v_threshold:
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
- `dc_state` and `grid_state` may be `None` if those components haven't produced state yet.
- Keep `step()` fast -- it runs synchronously in the simulation loop.
- Use `clock.time_s` for time-dependent logic.
- Declare required features by overriding `required_features()` if needed (for example OFO needs `{"voltage", "sensitivity"}`).

## Custom Datacenter Backend

Datacenter backends implement the `DatacenterBackend` ABC from `openg2g.datacenter.base`:

```python
from __future__ import annotations

import numpy as np

from openg2g.clock import SimulationClock
from openg2g.datacenter.base import DatacenterBackend
from openg2g.types import Command, DatacenterState, ThreePhase


class SyntheticDatacenter(DatacenterBackend):
    """A datacenter that generates sinusoidal power profiles."""

    def __init__(self, dt_s: float = 0.1, base_kw: float = 1000.0):
        self._dt = dt_s
        self._base_kw = base_kw
        self._batch: dict[str, int] = {}

    @property
    def dt_s(self) -> float:
        return self._dt

    def step(self, clock: SimulationClock) -> DatacenterState:
        t = clock.time_s
        power_kw = self._base_kw * (1.0 + 0.3 * np.sin(2 * np.pi * t / 600))
        power_w = power_kw * 1e3 / 3  # split equally across phases
        return DatacenterState(
            time_s=t,
            power_w=ThreePhase(a=power_w, b=power_w, c=power_w),
        )

    def apply_control(self, command: Command) -> None:
        if command.kind != "set_batch_size":
            return
        batch_map = command.payload.get("batch_size_by_model", {})
        if isinstance(batch_map, dict):
            self._batch.update({str(k): int(v) for k, v in batch_map.items()})
```

### Datacenter Guidelines

- `step()` is called at the rate specified by `dt_s`.
- Return a `DatacenterState` (or subclass) with three-phase power in watts.
- `apply_control()` receives one command at a time.
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
    T_total_s=3600.0,
)
log = coord.run()
```

The coordinator handles all the timing, buffering, and dispatch automatically. Your components just need to implement the interface.

## Tips

- **Multiple controllers**: Controllers run in order. Put tap controllers before batch controllers if tap changes should be visible to the batch optimizer within the same tick.
- **State inspection**: The base `DatacenterState` includes `batch_size_by_model` and `active_replicas_by_model`, so controllers can access per-model batch sizes and replica counts without knowing which backend is in use.
- **Testing**: Write unit tests for your controller by constructing mock `DatacenterState` and `GridState` objects directly -- they are simple frozen dataclasses.
