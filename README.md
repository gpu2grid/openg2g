<div align="center">
<h1>OpenG2G</h1>
</div>

<div align="center">
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-2ea44f.svg" alt="License: Apache-2.0"></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
<a href="https://arxiv.org/abs/2602.05116"><img src="https://img.shields.io/badge/arXiv-2602.05116-b31b1b.svg" alt="arXiv"></a>
<a href="https://gpu2grid.io/openg2g"><img src="https://img.shields.io/badge/Docs-gpu2grid.io-orange.svg" alt="Docs"></a>
</div>

<br>

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
uv sync  # or: pip install -e . --group dev
```

## Quick Start

For a full walkthrough including data setup, see the [Getting Started guide](https://gpu2grid.io/openg2g/getting-started/quickstart/). The snippet below illustrates the core API:

```python
from fractions import Fraction
from pathlib import Path

from openg2g.coordinator import Coordinator
from openg2g.datacenter.config import DatacenterConfig, InferenceModelSpec, ReplicaSchedule
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.controller.noop import NoopController
from openg2g.grid.config import TapPosition

# 1. Set up a trace-based datacenter
models = (
    InferenceModelSpec(
        model_label="Llama-3.1-8B", model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpus_per_replica=1, itl_deadline_s=0.08,
        feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512),
    ),
    InferenceModelSpec(
        model_label="Llama-3.1-70B", model_id="meta-llama/Llama-3.1-70B-Instruct",
        gpus_per_replica=4, itl_deadline_s=0.10,
        feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512),
    ),
)
data_dir = Path("data/offline")
inference_data = InferenceData.load(data_dir, models, duration_s=3600, dt_s=0.1)
dc_config = DatacenterConfig()
dc = OfflineDatacenter(
    dc_config,
    OfflineWorkload(
        inference_data=inference_data,
        replica_schedules={
            "Llama-3.1-8B": ReplicaSchedule(initial=720),
            "Llama-3.1-70B": ReplicaSchedule(initial=180),
        },
    ),
    name="dc",
    dt_s=Fraction(1, 10),
    total_gpu_capacity=1440,
)

# 2. Set up the grid and attach the datacenter
TAP_STEP = 0.00625
grid = OpenDSSGrid(
    dss_case_dir="data/grid/ieee13",
    dss_master_file="IEEE13Bus.dss",
    dt_s=Fraction(1, 10),
    initial_tap_position=TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP),
)
grid.attach_dc(dc, bus="671")

# 3. Run the simulation
coord = Coordinator(
    datacenters=[dc],
    grid=grid,
    controllers=[NoopController()],
    total_duration_s=3600,
)
log = coord.run()
```

See [`examples/`](examples/) for complete simulation scripts (offline trace-replay and online hardware-in-the-loop variants).

## Running Example Simulations

The first run downloads benchmark data from the [ML.ENERGY Benchmark v3 dataset](https://huggingface.co/datasets/ml-energy/benchmark-v3) (gated -- [request access](https://huggingface.co/datasets/ml-energy/benchmark-v3) first) and generates simulation artifacts. Subsequent runs load from cache.

```bash
export HF_TOKEN=hf_xxxxxxxxxxx  # needed for first run only

# Baseline: fixed taps
python examples/offline/run_ofo.py --system ieee13 --mode baseline-no-tap

# Baseline: scheduled tap changes
python examples/offline/run_ofo.py --system ieee13 --mode baseline-tap-change

# OFO closed-loop control
python examples/offline/run_ofo.py --system ieee13 --mode ofo-no-tap

# Run all four cases (both baselines + OFO with/without tap changes)
python examples/offline/run_ofo.py --system ieee13 --mode all
```

`--system` selects the IEEE test feeder (ieee13, ieee34, or ieee123). `--mode` selects one of `baseline-no-tap`, `baseline-tap-change`, `ofo-no-tap`, `ofo-tap-change`, or `all`. Data sources are in `data_sources.json`; all other experiment parameters are defined inline in each script. Generated data is cached in `data/offline/{hash}/`.

## Documentation

Full documentation is available at [https://gpu2grid.io/openg2g](https://gpu2grid.io/openg2g), including:

- Installation and setup guide
- Running simulations
- Implementing custom components
- Architecture reference

## Contact

Jae-Won Chung <jwnchung@umich.edu>

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
