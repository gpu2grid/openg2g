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
inference_data = InferenceData.ensure(data_dir, models, dt_s=0.1)
training_trace = TrainingTrace.ensure(Path("data/training_trace.csv"))
logistic_models = LogisticModelStore.ensure(data_dir, models)
```

!!! Note "Online Simulation"
    Online simulation with live GPUs does not use the power and ITL distributions, as they are supplied directly by running servers. However, the OFO controller can still use logistic fits for gradient estimation.

## Spec-Driven Configuration

Benchmark selection now lives directly on [`InferenceModelSpec`][openg2g.datacenter.config.InferenceModelSpec]. A spec includes the benchmark query fields (`model_id`, `gpu_model`, `task`, `gpus_per_replica`, `batch_sizes`) together with serving-time knobs (`itl_deadline_s`, `feasible_batch_sizes`, `fit_exclude_batch_sizes`).

```python
spec = InferenceModelSpec(
    model_label="Llama-3.1-8B",
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    gpu_model="H100",
    task="lm-arena-chat",
    gpus_per_replica=1,
    tensor_parallel=1,
    itl_deadline_s=0.08,
    batch_sizes=(8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768),
    feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512),
)
```

- `task` selects which benchmark prompt dataset to use (e.g. `lm-arena-chat`, `gpqa`).
- `gpu_model` and `gpus_per_replica` select the measured hardware configuration.
- `batch_sizes` controls which ML.ENERGY runs are requested.
- `fit_exclude_batch_sizes` lets you drop pathological measurements from logistic fitting without dropping them from trace extraction.
- First run downloads benchmark data from the HuggingFace Hub and caches it per spec under `data/specs/<spec-hash>/`.

Model specifications ([`InferenceModelSpec`][openg2g.datacenter.config.InferenceModelSpec]) are defined as Python constants inline in each example script. All other configuration (datacenter sizing, controller tuning, workload scenarios, grid setup) is also defined programmatically per-script. See [Building Simulators](building-simulators.md) for details.

## Lazy Generation and Caching

Each data class provides an `ensure()` classmethod that combines generate-if-missing and load into a single call:

```python
# First run: generates data to data_dir, then loads it.
# Subsequent runs: loads directly from cache.
inference_data = InferenceData.ensure(
    data_dir, models,
    dt_s=0.1,
)
training_trace = TrainingTrace.ensure(
    Path("data/training_trace.csv"),
)
logistic_models = LogisticModelStore.ensure(
    data_dir,
    models,
)
```

Under the hood, `ensure()` checks whether the output file or directory exists. If not, it calls `generate().save()` to create the artifacts. Then it calls `load()` to return the ready-to-use object.

### Default data path

Examples use a shared root such as `data/specs/`. Each `InferenceModelSpec` computes its own content-addressed cache key via `cache_hash()`, so changing one spec only regenerates that spec's artifacts.

## Inference Data Generation

[`InferenceData.generate()`][openg2g.datacenter.workloads.inference.InferenceData.generate] uses the [`mlenergy-data`](https://ml.energy/data) toolkit to download and process GPU benchmark data from the [ML.ENERGY Benchmark v3 dataset](https://huggingface.co/datasets/ml-energy/benchmark-v3).

For each model and batch size, it:

1. Extracts power timelines from benchmark runs
2. Resamples to a median-duration grid
3. Fits [`ITLMixtureModel`][mlenergy.data.modeling.ITLMixtureModel] distributions per batch size

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
         Model spec               │ InferenceData.generate()          │
  ┌─────────────────────┐         │                                   │
  │ InferenceModelSpec  │         │ For each model x batch size:      │
  │                     │────────>│  1. Extract power timelines       │
  │ batch_sizes, task,  │         │  2. Resample to median-duration   │
  │ gpu_model, etc.     │         │  3. Fit ITLMixtureModel           │
  │                     │         │                                   │
  └─────────────────────┘         └───────────┬───────────────────────┘
                                              │
                                              v
                        ┌────────────────────────────────┐
                        │ data/specs/<spec-hash>/        │
                        │                                │
                        │  trace.csv                     │  <── GPU power timeseries
                        │  itl_fit.json                  │  <── ITL distribution params
                        │  logistic_fit.json             │  <── OFO curve-fit params
                        │  _manifest.json                │  <── Version stamp + spec
                        └────────────────────────────────┘
```

## Logistic Curve Fitting

[`LogisticModelStore.generate()`][openg2g.controller.ofo.LogisticModelStore.generate] fits four-parameter logistic curves to power, latency, and throughput versus batch size:

$$p(x) = \frac{P_{\max}}{1 + \exp(-k_p(x - x_{0,p}))} + p_0, \quad x \triangleq \log_2(b)$$

where $P_{\max}$ is the saturation magnitude, $k_p$ controls transition sharpness, $x_{0,p}$ is the characteristic batch size threshold, and $p_0$ is an offset term. Latency and throughput use the same functional form with their own parameters.

OpenG2G uses [`LogisticModel`][mlenergy.data.modeling.LogisticModel] from [`mlenergy-data`](https://ml.energy/data) at both stages:

- **Generation**: [`LogisticModel.fit(x, y)`][mlenergy.data.modeling.logistic.LogisticModel.fit] fits the curve to benchmark data
- **Runtime**: [`LogisticModel.eval(batch)`][mlenergy.data.modeling.logistic.LogisticModel.eval] evaluates the curve, and [`LogisticModel.deriv_wrt_x(x)`][mlenergy.data.modeling.logistic.LogisticModel.deriv_wrt_x] computes gradients for the OFO controller

## ITL Mixture Model

Historical ITL measurements exhibit heavy-tailed behavior.
The generation step captures this using a weighted mixture of two lognormal distributions per batch size.

OpenG2G uses [`ITLMixtureModel`][mlenergy.data.modeling.ITLMixtureModel] from [`mlenergy-data`](https://ml.energy/data) at both stages:

- **Generation**: [`ITLMixtureModel.fit(samples)`][mlenergy.data.modeling.latency.ITLMixtureModel.fit] fits the mixture to raw ITL samples
- **Runtime**: [`ITLMixtureModel.sample_avg(n_replicas, rng)`][mlenergy.data.modeling.latency.ITLMixtureModel.sample_avg] draws average latency across replicas

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

- **[`OfflineDatacenter`][openg2g.datacenter.offline.OfflineDatacenter]**: Uses [`InferenceData`][openg2g.datacenter.workloads.inference.InferenceData] to replay periodic per-GPU power templates. Latency fits ([`ITLMixtureModel`][mlenergy.data.modeling.ITLMixtureModel]) are sampled at each control interval.
- **[`OFOBatchSizeController`][openg2g.controller.ofo.OFOBatchSizeController]**: Uses [`LogisticModelStore`][openg2g.controller.ofo.LogisticModelStore] for logistic curve evaluation. Calls `eval()` and `deriv_wrt_x()` at each control step to compute gradients.
