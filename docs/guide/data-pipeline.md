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
      "initial_num_replicas": 720,
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

- `models[]` entries are parsed directly as [`InferenceModelSpec`][openg2g.datacenter.config.InferenceModelSpec]. `initial_num_replicas` is the starting replica count; inference ramps with `target > 1.0` can scale beyond this.
- `data_sources[]` entries are parsed as [`MLEnergySource`][openg2g.datacenter.workloads.inference.MLEnergySource], linked to models by `model_label`.
- `training_trace_params` is parsed as [`TrainingTraceParams`][openg2g.datacenter.workloads.training.TrainingTraceParams]. Empty `{}` uses all defaults.
- `data_dir`: `null` means auto-generate a hash-based path (`data/offline/{hash}`). An explicit path skips hashing.
- `mlenergy_data_dir`: `null` loads benchmark data from the HuggingFace Hub.

<<<<<<< HEAD
=======
### Extended Config for Multi-DC and Scenario Simulations

The example scripts in `examples/offline/` use an extended config format (see `config_ieee34.json`) with additional fields for grid scenarios:

```json
{
  "models": [ ... ],
  "data_sources": [ ... ],

  "ieee_case_dir": "../../data/grid/ieee34",
  "dss_master_file": "ieee34Mod1_halfline.dss",
  "source_pu": 1.09,

  "dc_sites": {
    "upstream": {
      "bus": "850", "bus_kv": 24.9, "base_kw_per_phase": 120.0,
      "models": ["Llama-3.1-8B", "Llama-3.1-70B"],
      "connection_type": "wye", "seed": 0,
      "total_gpu_capacity": 520,
      "inference_ramps": [
        {"target": 0.5, "t_start": 1000, "t_end": 1500},
        {"target": 1.2, "t_start": 2500, "t_end": 3000, "model": "Llama-3.1-8B"}
      ]
    },
    "downstream": {
      "bus": "834", "bus_kv": 24.9, "base_kw_per_phase": 80.0,
      "models": ["Qwen3-30B-A3B"],
      "connection_type": "wye", "seed": 42,
      "total_gpu_capacity": 600
    }
  },

  "pv_systems": [
    {"bus": "848", "bus_kv": 24.9, "peak_kw": 130.0, "power_factor": 1.0}
  ],
  "time_varying_loads": [
    {"bus": "860", "bus_kv": 24.9, "peak_kw": 80.0, "power_factor": 0.96}
  ],

  "initial_taps": {
    "creg1a": "+12", "creg1b": "+8", "creg1c": "+10",
    "creg2a": "+0", "creg2b": "+0", "creg2c": "+0"
  },
  "regulator_zones": {
    "creg1": ["814r", "850", "816", "824", "828", "830", "854"],
    "creg2": ["852r", "832", "858", "834", "860", "836", "840", "862", "842", "844", "846", "848"]
  },

  "ofo": {
    "primal_step_size": 0.05, "w_throughput": 0.001, "w_switch": 1.0,
    "voltage_gradient_scale": 1e6, "voltage_dual_step_size": 20.0,
    "latency_dual_step_size": 1.0
  },
  "simulation": {
    "total_duration_s": 3600, "dt_dc": "1/10", "dt_grid": "1/10", "dt_ctrl": "1",
    "v_min": 0.95, "v_max": 1.05
  }
}
```

Key additional fields:

- `source_pu`: Substation source voltage in per-unit (default 1.0).
- `dc_sites`: Multi-datacenter sites, keyed by site ID. Each specifies the bus, voltage, base power, model assignment, connection type, and optional inference ramps. Maps to [`DCLoadSpec`][openg2g.grid.config.DCLoadSpec] for the grid and a separate [`OfflineDatacenter`][openg2g.datacenter.offline.OfflineDatacenter] per site.
    - `inference_ramps`: Optional list of inference server ramp events. Each entry has `target` (fraction of initial replicas; >1.0 for scale-up), `t_start`/`t_end` (ramp window in seconds), and optional `model` (apply to a specific model only). See [Building Simulators: Inference Ramps](building-simulators.md#inference-ramps).
    - `total_gpu_capacity`: Maximum number of GPUs the datacenter can physically host. If omitted, auto-computed from initial model allocation. Used to enforce capacity constraints during load shifting and for power estimation in PV expansion planning.
- `pv_systems`: Solar PV injections modeled as time-varying negative loads.
- `time_varying_loads`: Additional time-varying loads at arbitrary buses.
- `initial_taps`: Initial regulator tap positions, keyed by RegControl name. Supports integer step notation (`"+12"` = 1.0 + 12 × 0.00625) or direct per-unit values.
- `tap_schedule`: Optional dynamic tap changes during simulation. Each entry specifies a time `t` (seconds) and one or more regulator tap positions (e.g., `{"t": 1800, "creg4a": "+15"}`).
- `zones`: Optional per-zone bus lists for zone-constrained analysis (e.g., PV placement, DC location sweeps).
- `load_shift`: Optional cross-site load shifting configuration. When `enabled: true`, the [`LoadShiftController`][openg2g.controller.load_shift.LoadShiftController] shifts LLM replicas between datacenters to resolve voltage violations after batch-size control is exhausted. Fields: `gpus_per_shift` (GPUs moved per step), `headroom` (fraction of extra server capacity to pre-allocate).
- `regulator_zones`: Maps each regulator bank prefix to the list of downstream buses in its zone. Used for zone-aware tap optimization in the hosting capacity analysis.
- `training`: Optional training workload overlay. Fields: `dc_site` (site ID the training job runs on; `null` = first site), `n_gpus`, `target_peak_W_per_gpu`, `t_start`/`t_end` (seconds).
- `ofo`: OFO controller parameters.
- `simulation`: Simulation timing and voltage limits.

>>>>>>> f03cf6c (Add multi-datacenter architecture: Coordinator accepts multiple DCs. Add functions to sweep ofo parameters, sweep DC locations, find DC hosting capacity, and optimize PV locations and capacities. Add IEEE 13, 34, 123 test feeders and example scripts. Include simulation outputs for IEEE 13, 34, 123 under multiple scenarios.)
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
<<<<<<< HEAD
python examples/offline/run_baseline.py --config examples/offline/config.json --mode no-tap

python examples/offline/run_ofo.py --config examples/offline/config.json
```

`--config` is the only required argument. The config file specifies all paths and data sources.
=======
# Baseline and OFO simulations (works with any IEEE system)
python examples/offline/run_baseline.py --config examples/offline/config_ieee13.json --system ieee13
python examples/offline/run_ofo.py --config examples/offline/config_ieee13.json --system ieee13
python examples/offline/run_ofo.py --config examples/offline/config_ieee13.json --system ieee13 --mode all   # both no-tap and tap-change

python examples/offline/run_ofo.py --config examples/offline/config_ieee34.json --system ieee34
python examples/offline/run_ofo.py --config examples/offline/config_ieee123.json --system ieee123

# Analysis scripts
python examples/offline/sweep_hosting_capacities.py --config examples/offline/config_ieee34.json --system ieee34
python examples/offline/sweep_dc_locations.py --config examples/offline/config_ieee13.json --system ieee13   # 1-D (single DC)
python examples/offline/sweep_dc_locations.py --config examples/offline/config_ieee34.json --system ieee34   # 2-D (multi DC)
python examples/offline/sweep_dc_locations.py --config examples/offline/config_ieee123.json --system ieee123 --dt-screening 60 --top-k 4   # zone-constrained
python examples/offline/analyze_different_controllers.py --config examples/offline/config_ieee13.json --system ieee13
python examples/offline/optimize_pv_and_dc_locations.py --config examples/offline/config_ieee123.json --system ieee123 --n-pv 3
```

`--config` is the only required argument. The config file specifies all paths and data sources. The OFO and baseline scripts support `--mode no-tap` (default), `--mode tap-change`, or `--mode all` to control whether tap schedule changes are applied. For detailed usage guides and examples across all IEEE test systems, see the [Examples](../examples/) documentation.
>>>>>>> f03cf6c (Add multi-datacenter architecture: Coordinator accepts multiple DCs. Add functions to sweep ofo parameters, sweep DC locations, find DC hosting capacity, and optimize PV locations and capacities. Add IEEE 13, 34, 123 test feeders and example scripts. Include simulation outputs for IEEE 13, 34, 123 under multiple scenarios.)
