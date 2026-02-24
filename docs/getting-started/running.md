# Running Simulation

## End-to-End Example

The commands below go from a fresh clone to running all three simulations.

!!! note "Dataset access required"
    The data build step uses the [`mlenergy-data`](https://ml.energy/data) toolkit to download and process GPU benchmark data from the [ML.ENERGY Benchmark v3 dataset](https://huggingface.co/datasets/ml-energy/benchmark-v3). This is a gated dataset -- you must request access on Hugging Face before running the build.

```bash
# Clone and install
git clone https://github.com/gpu2grid/openg2g.git
cd openg2g
uv sync
uv pip install -e mlenergy-data --config-setting editable_mode=compat

# Build simulation data from ML.ENERGY benchmark
# Requires access to the gated dataset: https://huggingface.co/datasets/ml-energy/benchmark-v3
uv run python data/offline/build_mlenergy_data.py \
  --config data/offline/models.json \
  --out-dir data/generated \
  --no-plot

# Generate synthetic training power trace
uv run python data/offline/generate_training_trace.py \
  --out-csv data/generated/synthetic_training_trace.csv --seed 2

# Run all three simulations
uv run python examples/offline/run_baseline.py --mode no-tap \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv

uv run python examples/offline/run_baseline.py --mode tap-change \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv

uv run python examples/offline/run_ofo.py \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv
```

Outputs (plots and logs) are saved to `outputs/baseline_no-tap/`, `outputs/baseline_tap-change/`, and `outputs/ofo/`.

## Example Scripts

OpenG2G ships with example simulations in the `examples/` directory:

- `offline/run_baseline.py`: Uncontrolled baseline (no OFO), with two modes:
    - `--mode no-tap` (default): Fixed tap positions ("No control, no tap")
    - `--mode tap-change`: Scheduled tap changes at t=1500s and t=3300s ("Tap change only")
- `offline/run_ofo.py`: OFO closed-loop control with batch size optimization

These correspond to the three evaluation cases in the [GPU-to-Grid paper](https://arxiv.org/abs/2602.05116).

## Data Requirements

!!! warning "Build simulation data before running"
    Simulation data is **not** included in the repository. You must build it from GPU benchmark data before running any examples. See the [Data Pipeline](../guide/data-pipeline.md) page for step-by-step instructions.

Both examples require:

- **Power trace CSVs**: Per-model GPU power traces at various batch sizes, latency fit parameters, and logistic fit parameters. Build from benchmark data with `data/offline/build_mlenergy_data.py`.
- **OpenDSS case files**: IEEE 13-bus test feeder files, included in the repo at `examples/ieee13/`.

## Baseline Simulation

The baseline runs the datacenter at a fixed batch size with no feedback control. Two modes correspond to two baselines in the paper:

```bash
uv run python examples/offline/run_baseline.py \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv

uv run python examples/offline/run_baseline.py --mode tap-change \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv
```

In `tap-change` mode, regulator taps on phases A and C change at t=1500s and t=3300s.

Key parameters (defined at the top of the script):

| Parameter | Value | Description |
|---|---|---|
| Datacenter timestep | 0.1 s | Power sample interval |
| Grid solver timestep | 0.1 s | OpenDSS solve interval |
| Initial batch size | 128 | Fixed batch size for all models |
| Total duration | 3600 s | Simulation horizon |

Outputs are saved to `outputs/baseline_no-tap/` or `outputs/baseline_tap-change/`:

- `dc_power_3ph.png`: Three-phase DC power over time
- `allbus_voltages_phase_{A,B,C}.png`: All-bus voltage trajectories
- `console_output.txt`: Full simulation log

## OFO Simulation

The OFO simulation uses the primal-dual batch size controller:

```bash
uv run python examples/offline/run_ofo.py \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv
```

The same base parameters from the [Baseline Simulation](#baseline-simulation) section apply here (datacenter timestep, grid solver timestep, initial batch size, total duration). The OFO controller adds these parameters:

| Parameter | Value | Description |
|---|---|---|
| Descent step size ($\rho_x$) | 0.1 | Primal gradient step size (G2G paper Eq. 8) |
| Throughput weight | 1e-3 | Weight on throughput maximization objective |
| Switching cost weight ($\gamma$) | 1.0 | Penalty for batch size changes (G2G paper Eq. 4a) |
| Voltage gradient scale | 1e6 | Scaling factor on voltage gradient term |
| Voltage dual step size ($\rho_v$) | 1.0 | Dual ascent rate for voltage constraints (G2G paper Eqs. 5-6) |
| Latency dual step size ($\rho_l$) | 1.0 | Dual ascent rate for latency constraints (G2G paper Eq. 7) |

Outputs are saved to `outputs/ofo/`:

- `dc_power_3ph.png`: Three-phase DC power over time
- `batch_schedule.png`: Batch size schedule per model
- `allbus_voltages_phase_{A,B,C}.png`: Voltage trajectories
- `console_output.txt`: Full simulation log with per-step OFO decisions

## Understanding the Output

### Voltage Statistics

Both simulations log voltage violation statistics at the end of the run. Baseline (tap-change) example:

```
21:00:09 run_baseline INFO === Voltage Statistics (all-bus) ===
21:00:09 run_baseline INFO   voltage_violation_time = 1050.8 s
21:00:09 run_baseline INFO   worst_vmin             = 0.935359
21:00:09 run_baseline INFO   worst_vmax             = 1.063965
21:00:09 run_baseline INFO   integral_violation     = 42.2543 pu·s
```

The OFO simulation logs each controller step with the current batch sizes, and the datacenter logs each batch size change as it is applied:

```
20:59:57 openg2g.controller.ofo INFO OFO step 51 (t=50.0 s): batch={'Llama-3.1-8B': 128, 'Llama-3.1-70B': 128, 'Llama-3.1-405B': 128, 'Qwen3-30B-A3B': 128, 'Qwen3-235B-A22B': 128}
20:59:57 openg2g.controller.ofo INFO OFO step 52 (t=51.0 s): batch={'Llama-3.1-8B': 256, 'Llama-3.1-70B': 128, 'Llama-3.1-405B': 128, 'Qwen3-30B-A3B': 128, 'Qwen3-235B-A22B': 128}
20:59:57 openg2g.datacenter.offline INFO Batch size Llama-3.1-8B: 128 -> 256
20:59:57 openg2g.controller.ofo INFO OFO step 53 (t=52.0 s): batch={'Llama-3.1-8B': 256, 'Llama-3.1-70B': 128, 'Llama-3.1-405B': 128, 'Qwen3-30B-A3B': 128, 'Qwen3-235B-A22B': 128}
...
```

At the end, the OFO simulation prints voltage statistics and a per-model batch schedule summary:

```
21:00:11 run_ofo INFO === Voltage Statistics (all-bus) ===
21:00:11 run_ofo INFO   voltage_violation_time = 220.1 s
21:00:11 run_ofo INFO   worst_vmin             = 0.944923
21:00:11 run_ofo INFO   worst_vmax             = 1.050551
21:00:11 run_ofo INFO   integral_violation     = 0.10456 pu·s
21:00:11 run_ofo INFO === Batch Schedule Summary ===
21:00:11 run_ofo INFO   Llama-3.1-405B: avg_batch=60.7, changes=6
21:00:11 run_ofo INFO   Llama-3.1-70B: avg_batch=110.9, changes=10
21:00:11 run_ofo INFO   Llama-3.1-8B: avg_batch=371.4, changes=30
21:00:11 run_ofo INFO   Qwen3-235B-A22B: avg_batch=80.9, changes=13
21:00:11 run_ofo INFO   Qwen3-30B-A3B: avg_batch=187.0, changes=21
```

- **violation_time**: total time any bus-phase voltage is outside [0.95, 1.05] pu
- **worst_vmin / worst_vmax**: extremes across all buses, phases, and time
- **integral_violation**: time-integrated sum of voltage violations across all bus-phase pairs (see Section IV-C of the [G2G paper](https://arxiv.org/abs/2602.05116))
