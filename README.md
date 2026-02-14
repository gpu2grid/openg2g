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

For grid simulation (OpenDSS):

```bash
pip install "openg2g[grid]"
```

### Development

```bash
git clone https://github.com/your-org/openg2g.git
cd openg2g
uv sync  # installs all dev dependencies
```

## Quick Start

```python
from openg2g.coordinator import Coordinator
from openg2g.datacenter.offline import OfflineDatacenter, TraceByBatchCache
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.controller.noop import NoopController

# 1. Set up a trace-based datacenter
cache = TraceByBatchCache(traces_by_batch)
cache.build_templates(T=3600.0, dt=0.1)
dc = OfflineDatacenter(trace_cache=cache, models=models, ...)

# 2. Set up the grid
grid = OpenDSSGrid(case_dir="path/to/13Bus", master="IEEE13Nodeckt.dss", ...)

# 3. Run the simulation
coord = Coordinator(
    datacenter=dc,
    grid=grid,
    controllers=[NoopController()],
    T_total_s=3600.0,
)
log = coord.run()
```

See [`examples/`](examples/) for complete simulation scripts:

- `run_baseline.py --mode no-tap` -- fixed taps, no OFO ("No control, no tap")
- `run_baseline.py --mode tap-change` -- scheduled tap changes, no OFO ("Tap change only")
- `run_ofo.py` -- OFO closed-loop batch size control

## Data Pipeline

Simulation data (power traces, latency fits, logistic fits) can be built from raw ML.ENERGY benchmark data using the [`mlenergy-data`](mlenergy-data/) library.

### 1. Build simulation data from benchmarks

The build script reads raw benchmark results and produces the trace CSVs, latency fit parameters, and logistic fit parameters that the simulation consumes. Model selection is controlled by a JSON config file ([`data/openg2g_models.json`](data/openg2g_models.json)).

Generated artifacts go into `data/generated/` (gitignored). Source files (`data/*.py`, `data/openg2g_models.json`) are versioned.

```bash
uv run python data/build_mlenergy_data.py \
  --root /path/to/benchmark_root \
  --config data/openg2g_models.json \
  --llm-config-dir /path/to/benchmark_root/llm/configs \
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

Full documentation is available at the [documentation site](https://your-org.github.io/openg2g/), including:

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

## License

Apache 2.0
