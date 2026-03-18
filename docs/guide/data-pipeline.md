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
