# Quickstart

Let's get you from zero to up and running with an end-to-end simulation example.

## Power Trace Data

The data build step uses the [`mlenergy-data`](https://ml.energy/data) toolkit to download and process GPU benchmark data from the [ML.ENERGY Benchmark v3 dataset](https://huggingface.co/datasets/ml-energy/benchmark-v3).

See the [Data Pipeline](../guide/data-pipeline.md) page for more details on how the data is used and processed, but in brief:

1. Go to the [dataset page](https://huggingface.co/datasets/ml-energy/benchmark-v3) and request access. Approval should typically be immediate.
2. Create a [Hugging Face access token](https://huggingface.co/settings/tokens) and set it as an environment variable.
    ```bash
    export HF_TOKEN=hf_xxxxxxxxxxx
    ```

## Clone the Repository

This is to get the example scripts and data build utilities.

```bash
git clone https://github.com/gpu2grid/openg2g.git
cd openg2g
uv sync && source .venv/bin/activate  # or: pip install -e . --group dev
```

## Build Simulation Data

```bash
# Inference power trace from the ML.ENERGY Benchmark v3 dataset
python data/offline/build_mlenergy_data.py \
  --config data/offline/models.json \
  --out-dir data/generated

# Synthetic power trace
python data/offline/generate_training_trace.py \
  --out-csv data/generated/synthetic_training_trace.csv
```

## Run Simulations

These three runs correspond to the evaluation cases in the [GPU-to-Grid paper](https://arxiv.org/abs/2602.05116):

- **`run_baseline.py --mode no-tap`**: Fixed taps, no OFO control
- **`run_baseline.py --mode tap-change`**: Scheduled tap changes at 1500s and 3300s, no OFO
- **`run_ofo.py`**: OFO closed-loop batch size optimization

```bash
python examples/offline/run_baseline.py --mode no-tap \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv

python examples/offline/run_baseline.py --mode tap-change \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv

python examples/offline/run_ofo.py \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv
```

Outputs (plots and logs) are saved to `outputs/baseline_no-tap/`, `outputs/baseline_tap-change/`, and `outputs/ofo/`.

## Understanding the Output

Both simulations log voltage violation statistics at the end of the run. Baseline (tap-change) example:

```
21:00:09 run_baseline INFO === Voltage Statistics (all-bus) ===
21:00:09 run_baseline INFO   voltage_violation_time = 1050.8 s
21:00:09 run_baseline INFO   worst_vmin             = 0.935359
21:00:09 run_baseline INFO   worst_vmax             = 1.063965
21:00:09 run_baseline INFO   integral_violation     = 42.2543 pu·s
```

The OFO simulation additionally prints a per-model batch schedule summary:

```
21:00:11 run_ofo INFO === Batch Schedule Summary ===
21:00:11 run_ofo INFO   Llama-3.1-405B: avg_batch=60.7, changes=6
21:00:11 run_ofo INFO   Llama-3.1-70B: avg_batch=110.9, changes=10
21:00:11 run_ofo INFO   Llama-3.1-8B: avg_batch=371.4, changes=30
```

- **violation_time**: Total time any bus-phase voltage is outside [0.95, 1.05] pu
- **worst_vmin / worst_vmax**: Min and max observed voltage across all buses, phases, and time
- **integral_violation**: Time-integrated sum of voltage violations across all bus-phase pairs
