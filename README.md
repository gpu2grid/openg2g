<div align="center">
<h1>OpenG2G</h1>
</div>

<div align="center">
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-2ea44f.svg" alt="License: Apache-2.0"></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
<a href="https://arxiv.org/abs/2602.05116"><img src="https://img.shields.io/badge/arXiv-2602.05116-b31b1b.svg" alt="arXiv"></a>
</div>

</br>

A modular Python library for simulating datacenter-grid interaction, with a focus on LLM workloads.

OpenG2G provides the building blocks for studying how datacenter-level controls (e.g., LLM workload batch size) affect distribution-level voltages. It ships with an implementation of Online Feedback Optimization (OFO) for joint voltage regulation and latency management, alongside a trace-replay datacenter backend and an OpenDSS-based grid simulator.

## Key Features

- **Multi-rate simulation**: datacenter, grid, and controller components run at independent rates, coordinated by a shared clock.
- **Pluggable architecture**: swap datacenter backends (trace-based or live GPU) and controllers (OFO, tap scheduling, or your own) via simple abstract interfaces.
- **OpenDSS integration**: power flow analysis on standard IEEE test feeders with tap scheduling (`TapPosition`/`TapSchedule` API) and voltage monitoring.
- **Online Feedback Optimization**: primal-dual batch size control balancing voltage regulation, inference latency, and throughput.
- **Live GPU support**: `OnlineDatacenter` backend reads real-time GPU power via [Zeus](https://github.com/ml-energy/zeus) for hardware-in-the-loop experiments.

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
git clone https://github.com/gpu2grid/openg2g.git
cd openg2g
uv sync
```

## Quick Start

```python
from fractions import Fraction

from openg2g.coordinator import Coordinator
from openg2g.datacenter.offline import OfflineDatacenter, PowerTraceStore
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.controller.noop import NoopController
from openg2g.models.spec import LLMInferenceModelSpec
from openg2g.types import TapPosition

# 1. Set up a trace-based datacenter
models = [
    LLMInferenceModelSpec("Llama-3.1-8B", num_replicas=720, gpus_per_replica=1, initial_batch_size=128),
    LLMInferenceModelSpec("Llama-3.1-70B", num_replicas=180, gpus_per_replica=4, initial_batch_size=128),
]
store = PowerTraceStore.load("data/generated/traces_summary.csv")
store.build_templates(duration_s=3600, timestep_s=Fraction(1, 10))
dc = OfflineDatacenter(trace_store=store, models=models, timestep_s=Fraction(1, 10))

# 2. Set up the grid
TAP_STEP = 0.00625
grid = OpenDSSGrid(
    dss_case_dir="examples/ieee13",
    dss_master_file="IEEE13Nodeckt.dss",
    dc_bus="671",
    dc_bus_kv=4.16,
    power_factor=0.95,
    dt_s=Fraction(1, 10),
    initial_tap_position=TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP),
    connection_type="wye",
)

# 3. Run the simulation
coord = Coordinator(
    datacenter=dc,
    grid=grid,
    controllers=[NoopController()],
    total_duration_s=3600,
    dc_bus="671",
)
log = coord.run()
```

See [`examples/`](examples/) for complete simulation scripts:

- `run_baseline.py --mode no-tap` -- fixed taps, no OFO ("No control, no tap")
- `run_baseline.py --mode tap-change` -- scheduled tap changes, no OFO ("Tap change only")
- `run_ofo.py` -- OFO closed-loop batch size control

## Running Example Simulations

### 1. Build simulation data from benchmarks

The build script uses the [`mlenergy-data`](https://ml.energy/data) toolkit to download and process GPU benchmark data from the [ML.ENERGY Benchmark v3 dataset](https://huggingface.co/datasets/ml-energy/benchmark-v3) (gated -- [request access](https://huggingface.co/datasets/ml-energy/benchmark-v3) first). It produces the trace CSVs, latency fit parameters, and logistic fit parameters that the simulation consumes. Model selection is controlled by a JSON config file ([`data/offline/models.json`](data/offline/models.json)).

Generated artifacts go into `data/generated/` (gitignored). Source files (`data/offline/*.py`, `data/offline/models.json`) are versioned.

```bash
uv run python data/offline/build_mlenergy_data.py \
  --config data/offline/models.json \
  --out-dir data/generated
```

### 2. Generate a synthetic training power trace

```bash
uv run python data/offline/generate_training_trace.py \
  --out-csv data/generated/synthetic_training_trace.csv --seed 2
```

### 3. Run simulations

```bash
# Baseline: fixed taps
uv run python examples/offline/run_baseline.py --mode no-tap \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv

# Baseline: scheduled tap changes
uv run python examples/offline/run_baseline.py --mode tap-change \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv

# OFO closed-loop control
uv run python examples/offline/run_ofo.py \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv
```

`--data-dir` and `--training-trace` are required for all simulation drivers.

## Documentation

Full documentation is available at [https://gpu2grid.io/openg2g](https://gpu2grid.io/openg2g), including:

- Installation and setup guide
- Running simulations
- Implementing custom components
- Architecture reference

## Citation

If you use OpenG2G in your research, please cite:

```bibtex
@article{gpu2grid-arxiv26,
  title   = {{GPU-to-Grid}: Voltage Regulation via {GPU} Utilization Control},
  author  = {Zhirui Liang and Jae-Won Chung and Mosharaf Chowdhury and Jiasi Chen and Vladimir Dvorkin},
  year    = {2026},
  journal = {arXiv preprint arXiv:2602.05116},
}
```
