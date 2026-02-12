# OpenG2G Development Guide

## Project Overview

OpenG2G is a modular Python framework for simulating datacenter-grid interaction, focusing on LLM inference workloads. It models how GPU batch size control can regulate distribution-level voltages through Online Feedback Optimization (OFO).

## Architecture

The simulation loop is orchestrated by the `Coordinator`, which ticks a shared `SimulationClock` and dispatches to components at their respective rates:

```
Coordinator (tick = GCD of all component periods)
├── DatacenterBackend.step()     runs at dt_dc  (e.g., 0.1s)
├── OpenDSSGrid.step()           runs at dt_dss (e.g., 0.1s or 1.0s)
└── Controller.step()            runs at dt_ctrl (e.g., 1.0s or 60s)
```

### Key Components

| Component | Module | Purpose |
|---|---|---|
| `SimulationClock` | `openg2g.clock` | Integer-tick clock avoiding FP drift, multi-rate `is_due()` |
| `Coordinator` | `openg2g.coordinator` | Simulation loop, DC buffer, grid dispatch |
| `OfflineDatacenter` | `openg2g.datacenter.offline` | Trace-based DC power (from CSV power traces) |
| `OnlineDatacenter` | `openg2g.datacenter.online` | Live GPU power via Zeus |
| `OpenDSSGrid` | `openg2g.grid.opendss` | OpenDSS power flow solver |
| `OFOBatchController` | `openg2g.controller.ofo` | Primal-dual batch size optimizer |
| `TapScheduleController` | `openg2g.controller.tap_schedule` | Pre-scheduled regulator tap changes |
| `TapPosition` / `TapSchedule` | `openg2g.types` | Fluent tap schedule configuration |

### Extension Points

- **Datacenter backends** implement `DatacenterBackend` (ABC in `openg2g.datacenter.base`). The base `DatacenterState` carries `batch_size_by_model` and `active_replicas_by_model`, so controllers work with any backend without `isinstance` checks.
- **Controllers** implement `Controller` (ABC in `openg2g.controller.base`) and return command envelopes.
- Both are composed via the `Coordinator`.

### Control/Context Contract

- Controllers are called as `step(clock, dc_state, grid_state, context)`.
- `context` exposes feature interfaces (currently `voltage`, `sensitivity`) and `raw` handles for advanced experiments.
- A controller can declare required features via `required_features()`. Missing features fail fast at startup.
- Commands use envelope objects in `openg2g.types`:
  - `Command(target=..., kind=..., payload=..., metadata=...)`
  - `ControlAction(commands=[...])`
- Built-in command kinds:
  - `target="datacenter", kind="set_batch_size", payload={"batch_size_by_model": ...}`
  - `target="grid", kind="set_taps", payload={"tap_changes": ...}`

## Code Conventions

### Style

- Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections.
- `from __future__ import annotations` at the top of every module.
- Use built-in types for annotations: `list`, `dict`, `tuple`, `set`.
- Use `X | None` instead of `Optional[X]`. Use `X | Y` instead of `Union[X, Y]`.
- Do not import from `typing` unless absolutely necessary (e.g., `Callable`, `Sequence`).
- No decorative comment banners (e.g., `# --- section ---`). Use plain comments when needed.

### Naming

- Module-level constants: `UPPER_SNAKE_CASE`.
- Classes: `PascalCase`. Dataclasses preferred for plain data containers.
- Functions and methods: `snake_case`. Private: `_leading_underscore`.
- File names: `snake_case.py`.

### Imports

- Standard library, then third-party, then local. Separated by blank lines.
- Prefer explicit imports over star imports.

## Development

### Setup

```bash
uv sync          # installs project + dev dependencies
```

### Running

```bash
uv run python examples/run_baseline.py                  # baseline: no tap changes (default)
uv run python examples/run_baseline.py --mode tap-change # baseline: scheduled tap changes
uv run python examples/run_ofo.py                        # OFO closed-loop simulation
```

### Testing

```bash
uv run pytest tests/ -v
```

### Linting

```bash
bash scripts/lint.sh    # runs ruff format, ruff check, pyright
```

## Key Technical Details

### OfflineDatacenter Internals

- Generates `n+1` internal steps per chunk (endpoint-inclusive, matching the original `ceil(T/dt)+1` convention), serves only `n`. The extra sample preserves the RNG call sequence.
- Batch changes via `apply_control()` are deferred to the next chunk boundary to preserve RNG call sequence stability.
- Uses two RNGs: `_layout_rng` (`seed`) for server layout, `_rng` (`seed + 12345`) for noise/restart profiles.
- Per-model server layouts stored as `ServerLayout` dataclasses (frozen after first build).
- Factory: `TraceByBatchCache.from_traces(...)` builds templates in one step.

### Coordinator Buffer and `interval_start_w`

The coordinator accumulates DC power samples in `dc_buffer` between grid steps. After each grid step, it saves the last DC power sample as `interval_start_power`. On the next grid step, this is passed to `grid.step(interval_start_w=...)` so the grid has the full interval [start, end] for resampling.

### Tap Schedule Configuration

Tap schedules are built using the `TapPosition` and `TapSchedule` types:

```python
from openg2g.types import TapPosition

TAP_STEP = 0.00625  # standard 5/8% tap step
schedule = (
    TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP).at(t=0)
    | TapPosition(a=1.0 + 16 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 17 * TAP_STEP).at(t=25 * 60)
    | TapPosition(a=1.0 + 10 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 10 * TAP_STEP).at(t=55 * 60)
)
```

`TapPosition(a, b, c)` holds per-unit tap ratios per phase. The `|` operator combines schedule entries.

Two baseline modes are available in `examples/run_baseline.py`:

- `--mode no-tap`: Fixed tap positions throughout ("No control, no tap").
- `--mode tap-change`: Tap positions change at t=1500s and t=3300s ("Tap change only").

### OpenDSSGrid Sub-step Handling

The grid auto-detects the sub-step behavior from the number of DC power samples it receives:

- **Single sample** (`dt_grid == dt_dc`): one DSS solve per grid step.
- **Multiple samples** (`dt_grid > dt_dc`): resamples to 2 DSS grid points via `np.interp` and runs 2 solves.

When resampling, the coordinator passes `interval_start_w` (the last DC sample from the previous grid step) so the grid has the full interval [start, end] for interpolation.

### Voltage Metrics

- Integral violation uses sum-all-bus-phase formula: `sum_{bus,phase} [max(v_min - v, 0) + max(v - v_max, 0)]` per snapshot, integrated over time.
- `compute_allbus_voltage_stats` is vectorized: builds a `(T, N)` numpy matrix over all bus-phase pairs.
- Bus `rg60` is excluded by default (regulator bus with boosted voltage).

### OFO Controller

- Coupled latency RNG: accepts `latency_rng=dc.rng` to share the DC noise RNG.
- Reads `batch_size_by_model` and `active_replicas_by_model` from the base `DatacenterState` (no `isinstance` checks).
- Requires context features: `{"voltage", "sensitivity"}`.
- Zero-replica models get `l_hat = NaN`, which skips the latency dual update.
- Missing `active_replicas_by_model` entries now raise a clear runtime error (fail fast).
- `PrimalCfg` and `VoltageDualCfg` are documented with units and paper equation references. Gradient comments map each term to Eq. 18.
- Factory: `LogisticFitBank.from_csvs(...)` loads all fits in one step.

### SimulationLog

- Core fields: `dc_states`, `grid_states`, `actions` (always populated).
- Per-grid-step time series: `time_s`, `Va`/`Vb`/`Vc` (DC bus voltage), `kW_A`/`kW_B`/`kW_C` (power).
- Controller logs: `batch_log_by_model` (populated only when controllers emit batch changes).

## Data Dependencies

Simulation scripts require data files not checked into the repository:

- `power_csvs_updated/`: Per-model power trace CSVs, latency fit parameters, logistic fit parameters.
- `OpenDss_Test/13Bus/`: IEEE 13-bus OpenDSS case files.
- `reference_outputs/`: Frozen outputs from the original scripts for regression comparison.
