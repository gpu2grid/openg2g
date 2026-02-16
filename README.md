# OpenG2G

A modular Python framework for simulating datacenter-grid interaction, with a focus on LLM inference workloads.

OpenG2G provides the building blocks for studying how GPU-level controls (batch size, power capping) affect distribution-level voltages. It ships with an implementation of Online Feedback Optimization (OFO) for joint voltage regulation and latency management, alongside a trace-replay datacenter backend and an OpenDSS-based grid simulator.

## Key Features

- **Multi-rate simulation** -- datacenter, grid, and controller components run at independent rates, coordinated by a shared clock.
- **Pluggable architecture** -- swap datacenter backends (trace-based or live GPU) and controllers (OFO, tap scheduling, or your own) via simple abstract interfaces.
- **OpenDSS integration** -- power flow analysis on standard IEEE test feeders with tap scheduling (`TapPosition`/`TapSchedule` fluent API) and voltage monitoring.
- **Online Feedback Optimization** -- primal-dual batch size control balancing voltage regulation, inference latency, and throughput.
- **Live GPU support** -- `OnlineDatacenter` backend reads real-time GPU power via [Zeus](https://github.com/ml-energy/zeus) for hardware-in-the-loop experiments.

## Installation

Requires Python 3.10+.

```bash
pip install openg2g
```

For grid simulation with OpenDSS:

```bash
pip install "openg2g[opendss]"
```

### Development

```bash
git clone https://github.com/TODO/openg2g.git
cd openg2g
uv sync
```

## Quick Start

```python
from fractions import Fraction

from openg2g.coordinator import Coordinator
from openg2g.datacenter.offline import OfflineDatacenter, TraceByBatchCache
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.controller.noop import NoopController
from openg2g.models.spec import LLMInferenceModelSpec
from openg2g.types import TapPosition

# 1. Set up a trace-based datacenter
models = [
    LLMInferenceModelSpec("Llama-3.1-8B", num_replicas=720, gpus_per_replica=1, initial_batch_size=128),
    LLMInferenceModelSpec("Llama-3.1-70B", num_replicas=180, gpus_per_replica=4, initial_batch_size=128),
]
cache = TraceByBatchCache(traces_by_batch)
cache.build_templates(duration_s=3600, timestep_s=Fraction(1, 10))
dc = OfflineDatacenter(trace_cache=cache, models=models, timestep_s=Fraction(1, 10))

# 2. Set up the grid
TAP_STEP = 0.00625
grid = OpenDSSGrid(
    case_dir="examples/ieee13",
    master="IEEE13Nodeckt.dss",
    dc_bus="671",
    dc_bus_kv=4.16,
    power_factor=0.95,
    dt_s=Fraction(1, 10),
    tap_schedule=TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP).at(t=0),
)

# 3. Run the simulation
coord = Coordinator(
    datacenter=dc,
    grid=grid,
    controllers=[NoopController()],
    total_duration_s=3600,
)
log = coord.run()
```

See [`examples/`](examples/) for complete simulation scripts:

- `run_baseline.py --mode no-tap` -- fixed taps, no OFO ("No control, no tap")
- `run_baseline.py --mode tap-change` -- scheduled tap changes, no OFO ("Tap change only")
- `run_ofo.py` -- OFO closed-loop batch size control

## Running Example Simulations

Simulation data (power traces, latency fits, logistic fits) can be built from ML.ENERGY benchmark data using the [`mlenergy-data`](mlenergy-data/) library.

### 1. Build simulation data from benchmarks

The build script reads benchmark data and produces the trace CSVs, latency fit parameters, and logistic fit parameters that the simulation consumes. Model selection is controlled by a JSON config file ([`data/models.json`](data/models.json)).

Generated artifacts go into `data/generated/` (gitignored). Source files (`data/*.py`, `data/models.json`) are versioned.

```bash
uv run python data/build_mlenergy_data.py \
  --mlenergy-data-dir /path/to/compiled/data \
  --config data/models.json \
  --out-dir data/generated
```

### 2. Generate a synthetic training power trace

```bash
uv run python data/generate_training_trace.py \
  --out-csv data/generated/synthetic_training_trace.csv --seed 2
```

### 3. Run simulations

```bash
# Baseline: fixed taps
uv run python examples/run_baseline.py --mode no-tap \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv

# Baseline: scheduled tap changes
uv run python examples/run_baseline.py --mode tap-change \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv

# OFO closed-loop control
uv run python examples/run_ofo.py \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv
```

Without `--data-dir`, the simulation drivers default to the legacy `power_csvs_updated/` directory.

## Architecture

```
Coordinator
├── SimulationClock          tick = GCD of all component periods
├── DatacenterBackend        power generation at dt_dc
│   ├── OfflineDatacenter    trace replay from CSV power profiles
│   └── OnlineDatacenter     live GPU power via Zeus
├── OpenDSSGrid              power flow at dt_dss
└── Controller[]             control actions at dt_ctrl
    ├── OFOBatchController   primal-dual voltage/latency optimization
    ├── TapScheduleController pre-scheduled regulator tap changes
    └── NoopController       baseline (no control)
```

Each component exposes a `step()` method called by the coordinator at the appropriate rate. Controllers produce `ControlAction` objects that are applied to the datacenter and grid.

Current controller contract:

- `Controller.step(clock, datacenter, grid, events) -> ControlAction`
- `events` is always available and can be used to emit clock-stamped events.

## Documentation

Full documentation is available at the [documentation site](https://TODO.github.io/openg2g/), including:

- Installation and setup guide
- Running simulations
- Implementing custom components
- Architecture reference

## Citation

If you use OpenG2G in your research, please cite:

```bibtex
@inproceedings{openg2g2026,
  title     = {OpenG2G: A Framework for Datacenter-Grid Interaction with LLM Workloads},
  year      = {2026},
}
```
