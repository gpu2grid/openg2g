# OpenG2G

A modular Python framework for simulating datacenter-grid interaction, with a focus on LLM inference workloads.

OpenG2G provides the building blocks for studying how GPU-level controls (batch size, power capping) affect distribution-level voltages. It ships with an implementation of Online Feedback Optimization (OFO) for joint voltage regulation and latency management, alongside a trace-replay datacenter backend and an OpenDSS-based grid simulator.

## Key Features

- **Multi-rate simulation** -- datacenter, grid, and controller components run at independent rates, coordinated by a shared clock.
- **Pluggable architecture** -- swap datacenter backends (trace-based or live GPU) and controllers (OFO, tap scheduling, or your own) via simple abstract interfaces.
- **OpenDSS integration** -- power flow analysis on standard IEEE test feeders with tap scheduling and voltage monitoring.
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

See [`examples/`](examples/) for complete baseline and OFO simulation scripts.

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
