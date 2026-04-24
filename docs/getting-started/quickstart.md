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

Run all evaluation cases from the [GPU-to-Grid paper](https://arxiv.org/abs/2602.05116) (baseline and OFO, with and without tap changes):

```bash
python examples/offline/run_ofo.py --system ieee13 --mode all
```

The first run will download benchmark data and generate simulation artifacts (this takes a few minutes). Subsequent runs reuse the per-spec cache under `data/specs/<spec-hash>/`.

Outputs (plots and CSVs) are saved to `outputs/ieee13/` with one subdirectory per case. See [Voltage Regulation Strategies](../examples/voltage-regulation-strategies.md) for individual `--mode` options.

## Understanding the Output

Each case logs voltage **and** performance statistics as it completes, plus a comparison table at the end:

```
INFO run_ofo   baseline_no-tap:    viol=993.8s  integral=31.2751  vmin=0.9353  vmax=1.0431  thpt=8080.0 k tok/s  itl_over_deadline=0.00%
INFO run_ofo   baseline_tap-change: viol=999.3s  integral=21.6594  vmin=0.9353  vmax=1.0568  thpt=8080.0 k tok/s  itl_over_deadline=0.00%
INFO run_ofo   ofo_no-tap:           viol=69.2s  integral=0.0989  vmin=0.9449  vmax=1.0444  thpt=8028.1 k tok/s  itl_over_deadline=1.11%
...
====================================================================================================
IEEE13 Comparison
====================================================================================================
Mode                      Viol(s)       Vmin       Vmax       Integral  Thpt(k tok/s)  ITL_miss%
----------------------------------------------------------------------------------------------------
baseline_no-tap             993.8     0.9353     1.0431        31.2751         8080.0      0.00%
baseline_tap-change         999.3     0.9353     1.0568        21.6594         8080.0      0.00%
ofo_no-tap                   69.2     0.9449     1.0444         0.0989         8028.1      1.11%
----------------------------------------------------------------------------------------------------
```

**Voltage metrics** (how well the grid stays within bounds):

- **Viol(s)**: Total time (seconds) any bus-phase voltage is outside `[0.95, 1.05]` pu.
- **Vmin / Vmax**: Worst observed voltage across all buses, phases, and time.
- **Integral**: Time-integrated sum of voltage violations across all bus-phase pairs (pu·s).

**Performance metrics** (how well the fleet serves tokens):

- **Thpt(k tok/s)**: Mean total throughput across all active models (thousands of tokens per second), computed each step as `batch / observed_itl * active_replicas` summed across models.
- **ITL_miss%**: Fraction of (timestep × model) samples where observed inter-token latency exceeded the model's deadline. The OFO controller trades a small ITL-miss rate for large throughput and voltage gains.
