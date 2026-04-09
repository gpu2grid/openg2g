# Quickstart

Let's get you from zero to up and running with an end-to-end simulation example.

## Power Trace Data

The example scripts automatically download and process GPU benchmark data from the [ML.ENERGY Benchmark v3 dataset](https://huggingface.co/datasets/ml-energy/benchmark-v3) on the first run.

1. Go to the [dataset page](https://huggingface.co/datasets/ml-energy/benchmark-v3) and request access. Approval should typically be immediate.
2. Create a [Hugging Face access token](https://huggingface.co/settings/tokens) and set it as an environment variable.
    ```bash
    export HF_TOKEN=hf_xxxxxxxxxxx
    ```

## Clone the Repository

This is to get the example scripts and data files.

```bash
git clone https://github.com/gpu2grid/openg2g.git
cd openg2g
uv sync && source .venv/bin/activate  # or: pip install -e . --group dev
```

## Run Simulations

A single command builds all data (power traces, latency fits, training trace) and runs the simulation. Data is cached on disk so subsequent runs skip generation.

These three runs correspond to the evaluation cases in the [GPU-to-Grid paper](https://arxiv.org/abs/2602.05116):

- **`run_ofo.py --mode no-tap`**: Baseline (no tap changes) + OFO comparison
- **`run_ofo.py --mode tap-change`**: Baseline (with tap changes at 1500s and 3300s) + OFO comparison
- **`run_ofo.py --mode both`**: Baselines only (no-tap + tap-change, no OFO)

```bash
python examples/offline/run_ofo.py --system ieee13 --mode no-tap

python examples/offline/run_ofo.py --system ieee13 --mode tap-change

python examples/offline/run_ofo.py --system ieee13 --mode both
```

The first run will download benchmark data and generate simulation artifacts (this takes a few minutes). Subsequent runs load from the cache directory (`data/offline/{hash}/`).

Outputs (plots and logs) are saved to `outputs/baseline_no-tap/`, `outputs/baseline_tap-change/`, and `outputs/ofo/`.

## Understanding the Output

Both simulations log voltage violation statistics at the end of the run. Baseline (tap-change) example:

```
21:00:09 run_ofo INFO === Voltage Statistics (all-bus) ===
21:00:09 run_ofo INFO   voltage_violation_time = 1050.8 s
21:00:09 run_ofo INFO   worst_vmin             = 0.935359
21:00:09 run_ofo INFO   worst_vmax             = 1.063965
21:00:09 run_ofo INFO   integral_violation     = 42.2543 pu·s
```

The OFO simulation additionally prints a per-model batch schedule summary:

```
21:00:11 run_ofo INFO === Batch Schedule Summary ===
21:00:11 run_ofo INFO   Llama-3.1-405B: avg_batch=60.7, changes=6
21:00:11 run_ofo INFO   Llama-3.1-70B: avg_batch=110.9, changes=10
21:00:11 run_ofo INFO   Llama-3.1-8B: avg_batch=371.3, changes=28
21:00:11 run_ofo INFO   Qwen3-235B-A22B: avg_batch=80.9, changes=13
21:00:11 run_ofo INFO   Qwen3-30B-A3B: avg_batch=187.0, changes=21
```

- **violation_time**: Total time any bus-phase voltage is outside [0.95, 1.05] pu
- **worst_vmin / worst_vmax**: Min and max observed voltage across all buses, phases, and time
- **integral_violation**: Time-integrated sum of voltage violations across all bus-phase pairs
