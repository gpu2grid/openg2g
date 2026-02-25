# Data Pipeline

OpenG2G ships with built-in support for trace-replay simulations based on [real GPU benchmark data](https://ml.energy/data).
This page describes how raw benchmark measurements are compiled into artifacts that plug into simulation, and how those artifacts are consumed at runtime.

!!! Tip "LLM workloads"
    The data pipeline is focused on LLM workloads (inference from the ML.ENERGY Benchmark results, and training via synthetic generation), which is an important motivating workload for AI datacenter-grid interactions. We hope to improve and expand the data pipeline to support more workloads.

## Overview

The pipeline has two phases:

1. **Build time** (once, offline): [`data/offline/build_mlenergy_data.py`](https://github.com/gpu2grid/openg2g/blob/master/data/offline/build_mlenergy_data.py) in the repository processes real GPU LLM inference benchmark data from the [ML.ENERGY Benchmark](https://github.com/ml-energy/benchmark) into CSV artifacts: power traces, logistic curve fit parameters, and Inter-Token Latency (ITL) distribution parameters.
2. **Runtime** (every simulation): The [`OfflineDatacenter`][openg2g.datacenter.offline.OfflineDatacenter] replays power traces and samples latency from ITL fits. The OFO controller evaluates logistic fits for gradient computation.

!!! Note "Online Simulation"
    Online simulation with live GPUs do not use the power and ITL distributions, as they are supplied directly by running servers. However, the OFO controller can still use logistic fits for gradient estimation.

## Data Build Pipeline

Raw GPU benchmarks are processed into simulation-ready CSV artifacts by [`data/offline/build_mlenergy_data.py`](https://github.com/gpu2grid/openg2g/blob/master/data/offline/build_mlenergy_data.py).
Data loading, filtering, and model fitting are provided by the [`mlenergy-data`](https://ml.energy/data) toolkit.

A separate script, [`data/offline/generate_training_trace.py`](https://github.com/gpu2grid/openg2g/blob/master/data/offline/generate_training_trace.py), synthesizes a training power trace with configurable high/low plateaus, noise, brief dips, and a warm-up ramp. Generation is based on characteristics derived from real large model training measurements.

```
ML.ENERGY Benchmark Dataset                 mlenergy-data
    (Hugging Face hub)                         toolkit                             

  ┌─────────────────────┐
  │ results.json        │       LLMRuns.from_hf()
  │ (power, latency,    │────────────────────────────>┐
  │  throughput, ITL)   │    Load, filter, validate   │
  │  per model × batch  │                             │
  └─────────────────────┘                             │
                                                      v
                                  ┌───────────────────────────────────┐
         OpenG2G                  │ build_mlenergy_data.py            │
  Simulation model specs          │                                   │
  ┌─────────────────────┐         │ For each model × batch size:      │
  │ models.json         │         │  1. Extract power timelines       │
  │                     │────────>│  2. Resample to median-duration   │
  │ 5 models:           │         │  3. Fit LogisticModel (power,     │
  │  8B, 70B, 405B,     │         │     latency, throughput vs batch) │
  │  30B-A3B, 235B-A22B │         │  4. Fit ITLMixtureModel (latency  │
  │                     │         │     distribution per batch)       │
  └─────────────────────┘         └───────────┬───────────────────────┘
                                              │
                                              v
                        ┌────────────────────────────────┐
                        │ data/generated/                │
                        │                                │
   OpenG2G simulation   │  traces/*.csv                  │  <── GPU power timeseries
        inputs          │  traces_summary.csv            │  <── Trace manifest
                        │  logistic_fits.csv             │  <── Logistic fit params
                        │  latency_fits.csv              │  <── ITL distribution params
                        │  synthetic_training_trace.csv  │  <── Training power trace
                        └────────────────────────────────┘
```

### Model Config

The build script is driven by a config file ([`data/offline/models.json`](https://github.com/gpu2grid/openg2g/blob/master/data/offline/models.json)) that maps benchmark runs to simulation labels. Example entry:

```json
{
  "model_id": "Qwen/Qwen3-235B-A22B-Thinking-2507",  // Hugging Face model ID
  "label": "Qwen3-235B-A22B",                        // Nickname for simulation
  "task": "gpqa",
  "num_gpus": 8,
  "gpu": "H100",
  "batch_sizes": [8, 16, 32, 64, 96, 128, 192, 256, 384, 512]
}
```

### Logistic Curve Fitting

Workload parameters like batch size affect power, latency, and throughput in characteristic S-curve patterns. The build script fits four-parameter logistic curves:

$$p(x) = \frac{P_{\max}}{1 + \exp(-k_p(x - x_{0,p}))} + p_0, \quad x \triangleq \log_2(b)$$

where $P_{\max}$ is the saturation magnitude, $k_p$ controls transition sharpness, $x_{0,p}$ is the characteristic batch size threshold, and $p_0$ is an offset term. Latency and throughput use the same functional form with their own parameters.

OpenG2G uses [`LogisticModel`][mlenergy_data.modeling.LogisticModel] from [`mlenergy-data`](https://ml.energy/data) at both stages:

- **Build time**: [`LogisticModel.fit(x, y)`][mlenergy_data.modeling.logistic.LogisticModel.fit] fits the curve to benchmark data
- **Runtime**: [`LogisticModel.eval(batch)`][mlenergy_data.modeling.logistic.LogisticModel.eval] evaluates the curve, and [`LogisticModel.deriv_wrt_x(x)`][mlenergy_data.modeling.logistic.LogisticModel.deriv_wrt_x] computes gradients for the OFO controller

### ITL Mixture Model

Historical ITL measurements exhibit heavy-tailed behavior.
The build script captures this using a weighted mixture of two lognormal distributions per batch size.

OpenG2G uses [`ITLMixtureModel`][mlenergy_data.modeling.ITLMixtureModel] from [`mlenergy-data`](https://ml.energy/data) at both stages:

- **Build time**: [`ITLMixtureModel.fit(samples)`][mlenergy_data.modeling.latency.ITLMixtureModel.fit] fits the mixture to raw ITL samples
- **Runtime**: [`ITLMixtureModel.sample_avg(n_replicas, rng)`][mlenergy_data.modeling.latency.ITLMixtureModel.sample_avg] draws average latency across replicas

### Running the Build

The [`mlenergy-data`](https://ml.energy/data) toolkit automatically downloads benchmark data from the [ML.ENERGY Benchmark v3 dataset](https://huggingface.co/datasets/ml-energy/benchmark-v3) on first run.

To use the dataset:

1. [Request access on Hugging Face](https://huggingface.co/datasets/ml-energy/benchmark-v3)
1. [Create a Hugging Face access token](https://huggingface.co/settings/tokens)
1. Set the `HF_TOKEN` environment variable to your token before running the build.

```bash
# Set Hugging Face token after requesting access to the dataset
export HF_TOKEN=hf_xxxxxxxxxxx

python data/offline/build_mlenergy_data.py \
  --config data/offline/models.json \
  --out-dir data/generated

python data/offline/generate_training_trace.py \
  --out-csv data/generated/synthetic_training_trace.csv
```

## Runtime Integration

At simulation time, the generated artifacts are consumed by two components:

- **[`OfflineDatacenter`][openg2g.datacenter.offline.OfflineDatacenter]**: Loads power traces via [`PowerTraceStore.load(manifest)`][openg2g.datacenter.offline.PowerTraceStore.load] and builds periodic per-GPU templates. Latency fits ([`ITLMixtureModel`][mlenergy_data.modeling.ITLMixtureModel]) are sampled at each control interval.
- **[`OFOBatchController`][openg2g.controller.ofo.OFOBatchController]**: Loads logistic fits as [`LogisticModel`][mlenergy_data.modeling.LogisticModel] instances (one per metric per model). Calls `eval()` and `deriv_wrt_x()` at each control step to compute gradients.

### Passing Data to Simulations

```bash
python examples/offline/run_baseline.py --mode no-tap \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv

python examples/offline/run_ofo.py \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv
```

`--data-dir` and `--training-trace` are required for all simulation drivers.
