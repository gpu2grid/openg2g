# Running a Simulation

OpenG2G ships with example simulations in the `examples/` directory:

- `offline/run_baseline.py`: Uncontrolled baseline (no OFO, capacitor banks active), with two modes:
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

The baseline runs the datacenter at a fixed batch size with OpenDSS capacitor bank controls active. Two modes correspond to two baselines in the paper:

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
| `DT_DC` | 0.1s | Datacenter timestep |
| `DT_DSS` | 0.1s | Grid solver timestep |
| `BATCH_INIT` | 128 | Fixed batch size for all models |
| `T_TOTAL_S` | 3600s | Simulation duration |

Outputs are saved to `outputs/baseline_no-tap/` or `outputs/baseline_tap-change/`:

- `power_profiles.png`: Three-phase DC power over time
- `allbus_voltages_phase_{A,B,C}.png`: All-bus voltage trajectories
- `console_output.txt`: Full simulation log

## OFO Simulation

The OFO simulation uses the primal-dual batch size controller:

```bash
uv run python examples/offline/run_ofo.py \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv
```

This adds several OFO-specific parameters:

| Parameter | Value | Description |
|---|---|---|
| `ETA_PRIMAL` | 0.1 | Primal step size |
| `W_LATENCY` | 1.0 | Latency penalty weight |
| `K_V` | 1e6 | Voltage penalty gain |
| `RHO_V`, `RHO_L` | 1.0 | Dual step sizes (voltage, latency) |

Outputs are saved to `outputs/ofo/`:

- `dc_power_3ph.png`: Three-phase DC power
- `batch_schedule.png`: Batch size schedule per model
- `allbus_voltages_phase_{A,B,C}.png`: Voltage trajectories
- `console_output.txt`: Full simulation log with per-step OFO decisions

## Understanding the Output

### Voltage Statistics

Both simulations log voltage violation statistics at the end of the run:

```
run_baseline INFO === Voltage Statistics (all-bus) ===
run_baseline INFO   voltage_violation_time = 1007.0 s
run_baseline INFO   worst_vmin             = 0.935359
run_baseline INFO   worst_vmax             = 1.050525
run_baseline INFO   integral_violation     = 30.3320 pu·s
```

- **violation_time**: total time any bus-phase voltage is outside [0.95, 1.05] pu
- **worst_vmin / worst_vmax**: extremes across all buses, phases, and time
- **integral_violation**: time-integrated sum of voltage violations across all bus-phase pairs (see Section IV-C of the [paper](https://arxiv.org/abs/2602.05116))

### Batch Schedule (OFO only)

```
run_ofo INFO === Batch Schedule Summary ===
run_ofo INFO   Llama-3.1-8B: avg_batch=280.3, changes=22
run_ofo INFO   Llama-3.1-70B: avg_batch=70.1, changes=7
run_ofo INFO   Llama-3.1-405B: avg_batch=41.5, changes=4
run_ofo INFO   Qwen3-30B-A3B: avg_batch=328.0, changes=17
run_ofo INFO   Qwen3-235B-A22B: avg_batch=45.4, changes=7
```

Shows the average batch size and number of batch size changes per model over the simulation.
