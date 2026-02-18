# Data Pipeline

OpenG2G simulations consume pre-processed GPU benchmark data. This page describes how raw benchmark measurements flow through the [mlenergy-data](https://ml.energy/data) toolkit into simulation-ready artifacts.

## Overview

Two Python packages work together:

```
  ┌──────────────────────────┐       ┌──────────────────────────────────┐
  │      mlenergy-data       │       │            OpenG2G               │
  │                          │       │                                  │
  │  Benchmark data toolkit  │──────>│  Grid-datacenter co-simulation   │
  │                          │       │                                  │
  │  "How do LLMs behave     │       │  "What happens to the            │
  │   at different batch     │       │   distribution feeder when       │
  │   sizes on real GPUs?"   │       │   you run these workloads?"      │
  └──────────────────────────┘       └──────────────────────────────────┘
       Data supply side                  Simulation & control side
```

- **mlenergy-data**: Loads, filters, and fits models to real GPU benchmark data (power, latency, throughput vs. batch size) from the [ML.ENERGY Benchmark](https://ml.energy/data).
- **OpenG2G**: Multi-rate time-domain simulation of an LLM inference datacenter connected to an IEEE 13-bus distribution feeder, with OFO batch-size control.

## The mlenergy-data Toolkit

Three capabilities used by OpenG2G:

### Typed data loading

```python
runs = LLMRuns.from_hf()                                    # load all runs
runs = runs.task("lm-arena-chat").gpu("H100").batch(min=8)  # fluent filtering
```

Each `LLMRun` is a typed record with 40 fields: power, latency, throughput, model metadata, GPU config, etc.

### Logistic curve fitting

Four-parameter logistic: `y = b0 + L * sigmoid(k * (x - x0))` where `x = log2(batch)`.

These curves model how power, latency, and throughput vary with batch size (Section II-C of the [paper](https://arxiv.org/abs/2602.05116), Eq. 1-3):

```
  Power (W)                              Latency (s)
    |          ___________                  |              ___________
    |         /                             |             /
    |        /                              |            /
    |    ___/                               |    _______/
    |___/                                   |___/
    └───────────────────── batch            └───────────────────── batch
       8   32  128  512                        8   32  128  512
```

- `LogisticModel.fit(x, y)` -- grid search + least squares
- `LogisticModel.eval(batch)` -- evaluate at any batch size
- `LogisticModel.deriv_wrt_x(x)` -- gradient for OFO controller (used in Eq. 18 of the paper)

### Inter-token latency (ITL) mixture model

Two-component lognormal mixture captures bimodal ITL distributions (steady decode vs. scheduling stall):

```
  Probability
    |
    |  **
    | ****
    | *****       *
    | ******     ***
    | *******   *****
    |********************
    └───────────────────────── ITL (ms)
     "steady"         "stall"
      (decode)     (scheduling)
```

- `ITLMixtureModel.fit(samples)` -- EM algorithm
- `ITLMixtureModel.sample_avg(n_replicas, rng)` -- draw average latency across replicas

## Build-Time Pipeline

Raw GPU benchmarks are processed into simulation-ready CSV artifacts by `data/offline/build_mlenergy_data.py`:

```
  ML.ENERGY Benchmark DB              mlenergy-data               OpenG2G simulation
  (HF Hub or local disk)                toolkit                      inputs

  ┌────────────────────┐
  │ results.json ×1000s│    LLMRuns.from_directory()
  │ (power, latency,   │────────────────────────────>┐
  │  throughput, ITL   │    Load, filter, validate   │
  │  per model × batch │                             │
  └────────────────────┘                             │
                                                     v
  ┌────────────────────┐         ┌───────────────────────────────────┐
  │ models.json        │         │ build_mlenergy_data.py            │
  │                    │────────>│                                   │
  │ 5 models:          │         │ For each model × batch size:      │
  │  8B, 70B, 405B,    │         │  1. Extract power timelines       │
  │  30B-A3B, 235B-A22B│         │  2. Resample to median-duration   │
  │                    │         │  3. Fit LogisticModel (power,     │
  └────────────────────┘         │     latency, throughput vs batch) │
                                 │  4. Fit ITLMixtureModel (latency  │
                                 │     distribution per batch)       │
                                 └──────────┬────────────────────────┘
                                            │
                                            v
                                 ┌──────────────────────────┐
                                 │ data/generated/          │
                                 │                          │
                                 │  traces/*.csv            │  <── per-GPU power time series
                                 │  logistic_fits.csv       │  <── 4-param curves (L, x0, k, b0)
                                 │  latency_fits.csv        │  <── 2-component lognormal mixture
                                 │  synthetic_training.csv  │  <── synthetic training overlay
                                 └──────────────────────────┘
```

### Running the build

```bash
python data/offline/build_mlenergy_data.py \
  --mlenergy-data-dir /path/to/compiled/data \
  --config data/offline/models.json \
  --out-dir data/generated

python data/offline/generate_training_trace.py \
  --out-csv data/generated/synthetic_training_trace.csv --seed 2
```

The config file (`data/offline/models.json`) maps benchmark model IDs to simulation labels.

## Runtime Integration

At simulation time, the generated CSV artifacts are consumed at two points:

```
  ┌─────────── RUN TIME (every simulation) ─────────────────────────────┐
  │                                                                     │
  │   OfflineDatacenter reads:                                          │
  │     traces/*.csv ──> TraceByBatchCache (periodic power templates)   │
  │     latency_fits.csv ──> ITLMixtureModel.sample_avg() per step      │
  │                                                                     │
  │   OFO Controller reads:                                             │
  │     logistic_fits.csv ──> LogisticModel.eval() / .deriv_wrt_x()     │
  │                           called every control step for gradients   │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
```

- **OfflineDatacenter**: Loads power trace CSVs into `TraceByBatchCache`, which tiles them into periodic templates. At each step, the datacenter indexes into these templates to produce per-server power. Latency fits are loaded as `ITLMixtureModel` instances and sampled at each control interval.

- **OFO Controller**: Loads logistic fits as `LogisticModel` instances (one per metric per model). At each control step, it calls `eval()` and `deriv_wrt_x()` to compute the gradient of the Lagrangian (Eq. 18 of the [paper](https://arxiv.org/abs/2602.05116)).

### Passing data to simulations

```bash
python examples/offline/run_baseline.py --mode no-tap \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv

python examples/offline/run_ofo.py \
  --data-dir data/generated \
  --training-trace data/generated/synthetic_training_trace.csv
```

Without `--data-dir`, simulation drivers default to `power_csvs_updated/` (legacy hand-curated data).
