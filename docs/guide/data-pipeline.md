# Data Pipeline

OpenG2G ships with built-in support for trace-replay simulations based on [real GPU benchmark data](https://ml.energy/data).
This page describes how raw benchmark measurements are compiled into artifacts that plug into simulation, and how those artifacts are consumed at runtime.

!!! Tip "LLM workloads"
    The data pipeline is focused on LLM workloads (inference from the ML.ENERGY Benchmark results, and training via synthetic generation), which is an important motivating workload for AI datacenter-grid interactions. We hope to improve and expand the data pipeline to support more workloads.

## Overview

Data generation is integrated into the library classes that consume the data. Each class has `generate()`, `save()`, `load()`, and `ensure()` methods:

| Class | Generates | Consumed by |
|---|---|---|
| [`InferenceData`][openg2g.datacenter.workloads.inference.InferenceData] | Power traces + ITL distribution fits | [`OfflineDatacenter`][openg2g.datacenter.offline.OfflineDatacenter] |
| [`LogisticModelStore`][openg2g.controller.ofo.LogisticModelStore] | Logistic curve fits (power, latency, throughput) | [`OFOBatchSizeController`][openg2g.controller.ofo.OFOBatchSizeController] |
| [`TrainingTrace`][openg2g.datacenter.workloads.training.TrainingTrace] | Synthetic training power trace | [`OfflineDatacenter`][openg2g.datacenter.offline.OfflineDatacenter] |

Each class provides an `ensure()` classmethod that generates data if it doesn't exist and loads it:

```python
inference_data = InferenceData.ensure(data_dir, models, data_sources, dt_s=0.1)
training_trace = TrainingTrace.ensure(data_dir / "training_trace.csv", training_params)
logistic_models = LogisticModelStore.ensure(data_dir / "logistic_fits.csv", models, data_sources)
```

!!! Note "Online Simulation"
    Online simulation with live GPUs does not use the power and ITL distributions, as they are supplied directly by running servers. However, the OFO controller can still use logistic fits for gradient estimation.

## Config File

A single JSON config file drives both data generation and simulation. Example (`examples/offline/config.json`):

```json
{
  "models": [
    {
      "model_label": "Llama-3.1-8B",
      "model_id": "meta-llama/Llama-3.1-8B-Instruct",
      "gpus_per_replica": 1,
      "num_replicas": 720,
      "initial_batch_size": 128,
      "itl_deadline_s": 0.08,
      "feasible_batch_sizes": [8, 16, 32, 64, 128, 256, 512]
    }
  ],
  "data_sources": [
    {
      "model_label": "Llama-3.1-8B",
      "task": "lm-arena-chat",
      "gpu": "H100",
      "batch_sizes": [8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
    }
  ],
  "training_trace_params": {},
  "data_dir": null,
  "ieee_case_dir": "examples/ieee13",
  "mlenergy_data_dir": null
}
```

- `models[]` entries are parsed directly as [`InferenceModelSpec`][openg2g.datacenter.config.InferenceModelSpec].
- `data_sources[]` entries are parsed as [`MLEnergySource`][openg2g.datacenter.workloads.inference.MLEnergySource], linked to models by `model_label`.
- `training_trace_params` is parsed as [`TrainingTraceParams`][openg2g.datacenter.workloads.training.TrainingTraceParams]. Empty `{}` uses all defaults.
- `data_dir`: `null` means auto-generate a hash-based path (`data/offline/{hash}`). An explicit path skips hashing.
- `mlenergy_data_dir`: `null` loads benchmark data from the HuggingFace Hub.

## Lazy Generation and Caching

Each data class provides an `ensure()` classmethod that combines generate-if-missing and load into a single call:

```python
# First run: generates data to data_dir, then loads it.
# Subsequent runs: loads directly from cache.
inference_data = InferenceData.ensure(
    data_dir, models, data_sources,
    mlenergy_data_dir=config.mlenergy_data_dir,
    dt_s=0.1,
)
training_trace = TrainingTrace.ensure(
    data_dir / "training_trace.csv",
    config.training_trace_params,
)
logistic_models = LogisticModelStore.ensure(
    data_dir / "logistic_fits.csv",
    models, data_sources,
    mlenergy_data_dir=config.mlenergy_data_dir,
)
```

Under the hood, `ensure()` checks whether the output file or directory exists. If not, it calls `generate().save()` to create the artifacts. Then it calls `load()` to return the ready-to-use object.

### Default data path

When `data_dir` is `null` in the config, a hash-based path is computed from the data-relevant config keys (data sources and training trace parameters). Different configs automatically get different cache directories, so you can switch configs without manually clearing the cache.

## Inference Data Generation

[`InferenceData.generate()`][openg2g.datacenter.workloads.inference.InferenceData.generate] uses the [`mlenergy-data`](https://ml.energy/data) toolkit to download and process GPU benchmark data from the [ML.ENERGY Benchmark v3 dataset](https://huggingface.co/datasets/ml-energy/benchmark-v3).

For each model and batch size, it:

1. Extracts power timelines from benchmark runs
2. Resamples to a median-duration grid
3. Fits [`ITLMixtureModel`][mlenergy_data.modeling.ITLMixtureModel] distributions per batch size

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
         Config file              │ InferenceData.generate()          │
  ┌─────────────────────┐         │                                   │
  │ config.json         │         │ For each model x batch size:      │
  │                     │────────>│  1. Extract power timelines       │
  │ models[] +          │         │  2. Resample to median-duration   │
  │ data_sources[]      │         │  3. Fit ITLMixtureModel           │
  └─────────────────────┘         └───────────┬───────────────────────┘
                                              │
                                              v
                        ┌────────────────────────────────┐
                        │ data/offline/{hash}/         │
                        │                                │
                        │  traces/*.csv                  │  <── GPU power timeseries
                        │  traces_summary.csv            │  <── Trace manifest
                        │  latency_fits.csv              │  <── ITL distribution params
                        │  _manifest.json                │  <── Version stamp
                        └────────────────────────────────┘
```

## Logistic Curve Fitting

[`LogisticModelStore.generate()`][openg2g.controller.ofo.LogisticModelStore.generate] fits four-parameter logistic curves to power, latency, and throughput versus batch size:

$$p(x) = \frac{P_{\max}}{1 + \exp(-k_p(x - x_{0,p}))} + p_0, \quad x \triangleq \log_2(b)$$

where $P_{\max}$ is the saturation magnitude, $k_p$ controls transition sharpness, $x_{0,p}$ is the characteristic batch size threshold, and $p_0$ is an offset term. Latency and throughput use the same functional form with their own parameters.

OpenG2G uses [`LogisticModel`][mlenergy_data.modeling.LogisticModel] from [`mlenergy-data`](https://ml.energy/data) at both stages:

- **Generation**: [`LogisticModel.fit(x, y)`][mlenergy_data.modeling.logistic.LogisticModel.fit] fits the curve to benchmark data
- **Runtime**: [`LogisticModel.eval(batch)`][mlenergy_data.modeling.logistic.LogisticModel.eval] evaluates the curve, and [`LogisticModel.deriv_wrt_x(x)`][mlenergy_data.modeling.logistic.LogisticModel.deriv_wrt_x] computes gradients for the OFO controller

## ITL Mixture Model

Historical ITL measurements exhibit heavy-tailed behavior.
The generation step captures this using a weighted mixture of two lognormal distributions per batch size.

OpenG2G uses [`ITLMixtureModel`][mlenergy_data.modeling.ITLMixtureModel] from [`mlenergy-data`](https://ml.energy/data) at both stages:

- **Generation**: [`ITLMixtureModel.fit(samples)`][mlenergy_data.modeling.latency.ITLMixtureModel.fit] fits the mixture to raw ITL samples
- **Runtime**: [`ITLMixtureModel.sample_avg(n_replicas, rng)`][mlenergy_data.modeling.latency.ITLMixtureModel.sample_avg] draws average latency across replicas

## Training Trace Generation

[`TrainingTrace.generate()`][openg2g.datacenter.workloads.training.TrainingTrace.generate] synthesizes a training power trace with configurable high/low plateaus, noise, brief dips, and a warm-up ramp. Generation is based on characteristics derived from real large model training measurements.

Parameters are controlled via [`TrainingTraceParams`][openg2g.datacenter.workloads.training.TrainingTraceParams]. The empty dict `{}` in the config uses all defaults.

## Dataset Access

The [`mlenergy-data`](https://ml.energy/data) toolkit automatically downloads benchmark data from the [ML.ENERGY Benchmark v3 dataset](https://huggingface.co/datasets/ml-energy/benchmark-v3) on first run.

To use the dataset:

1. [Request access on Hugging Face](https://huggingface.co/datasets/ml-energy/benchmark-v3)
1. [Create a Hugging Face access token](https://huggingface.co/settings/tokens)
1. Set the `HF_TOKEN` environment variable to your token before running.

## Runtime Integration

At simulation time, the generated artifacts are consumed by two components:

- **[`OfflineDatacenter`][openg2g.datacenter.offline.OfflineDatacenter]**: Uses [`InferenceData`][openg2g.datacenter.workloads.inference.InferenceData] to replay periodic per-GPU power templates. Latency fits ([`ITLMixtureModel`][mlenergy_data.modeling.ITLMixtureModel]) are sampled at each control interval.
- **[`OFOBatchSizeController`][openg2g.controller.ofo.OFOBatchSizeController]**: Uses [`LogisticModelStore`][openg2g.controller.ofo.LogisticModelStore] for logistic curve evaluation. Calls `eval()` and `deriv_wrt_x()` at each control step to compute gradients.

### Running Simulations

```bash
python examples/offline/run_baseline.py --config examples/offline/config.json --mode no-tap

python examples/offline/run_ofo.py --config examples/offline/config.json
```

`--config` is the only required argument. The config file specifies all paths and data sources.
