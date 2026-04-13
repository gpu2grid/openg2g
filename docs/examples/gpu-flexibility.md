# GPU Flexibility for Voltage Regulation

## Research Question

How effective can GPU workload flexibility contribute to mitigating voltage violations caused by different sources: internal load changes (e.g., training task overlay, inference task fluctuations) and external load changes (e.g., time-varying load, renewable generation)?

## Overview

AI datacenter workload parameters (e.g., batch size for LLM inference) can be adjusted programmatically, which simultaneously affects GPU power consumption, inference latency, and token throughput. These power changes propagate through the distribution feeder, affecting bus voltages. This analysis explores whether adjusting batch sizes in response to grid conditions can effectively regulate voltages across different types of disturbances.

OpenG2G provides three IEEE test systems that represent different disturbance sources:

- **IEEE 13-bus**: DC-internal load changes: training task overlay (t=1000-2000s) and inference server ramp-down (t=2500-3000s) create large power swings within the datacenter itself.
- **IEEE 34-bus**: External load changes: time-varying loads and PV solar injection at various buses cause voltage fluctuations independent of the datacenter.
- **IEEE 123-bus**: Both internal and external: per-site inference ramps, PV systems, and time-varying loads all interact simultaneously across four datacenter zones.

## Scripts

| Script | Purpose |
|--------|---------|
| `run_ofo.py` | Baseline and OFO simulations (see [Voltage Regulation Strategies](voltage-regulation-strategies.md) for `--mode` options) |

## Usage

```bash
# All cases (baseline + OFO x no-tap + tap-change) for each system
python examples/offline/run_ofo.py --system ieee13 --mode all
python examples/offline/run_ofo.py --system ieee34 --mode all
python examples/offline/run_ofo.py --system ieee123 --mode all
```

- **IEEE 13-Bus**: DC-internal disturbances: training overlay (2400 GPUs, t=1000-2000s) and inference ramp-down to 50% (t=2500-3000s).
- **IEEE 34-Bus**: External disturbances: PV systems at buses 830/848 and time-varying loads at five buses. DC load is steady.
- **IEEE 123-Bus**: both internal and external, with per-site inference ramps (ramp-ups timed to stress each zone) across four DC zones, three PV systems, and time-varying loads.

## What to Look For

Each run writes voltage **and** performance columns to its result CSV (`violation_time_s, integral_violation_pu_s, worst_vmin, worst_vmax, mean_throughput_tps, integrated_throughput_tokens, itl_deadline_fraction`). Compare the four pillars between baseline and OFO:

- **Violation time / integral** (grid side): how often and how severely voltages exceed `[v_min, v_max]`.
- **Worst Vmin / Vmax** (grid side): peak voltage excursion.
- **Mean throughput** (DC side): tokens/s served by the fleet, derived from `batch / observed_itl * active_replicas` per model per step.
- **ITL-miss fraction** (DC side): share of samples where observed inter-token latency exceeded the model's deadline.

OFO is most effective against slow, predictable disturbances (training ramps, inference scaling) where the primal-dual optimization has time to converge -- and in those regimes it usually wins on *both* voltage and throughput at the cost of a small (typically < 5%) ITL-miss rate. Fast external disturbances (sudden PV cloud events) are harder to mitigate purely through batch-size control.

## Configuration

Experiment parameters are defined inline in each script's experiment functions. Key parameters:

- **Replica schedules**: `ReplicaSchedule(initial=...).ramp_to(target, t_start=, t_end=)` -- per-model server activation schedules (internal disturbance)
- **Training overlay**: `TrainingRun(n_gpus=..., trace=...).at(t_start, t_end)` for the training workload (internal disturbance)
- **Generators**: `SyntheticPV(peak_kw=...)` attached to grid buses for solar PV injection (external disturbance)
- **External loads**: `SyntheticLoad(peak_kw=...)` attached to grid buses (external disturbance)
- **OFO tuning**: `OFOConfig(...)`; see [Controller Parameter Sensitivity](controller-parameter-sensitivity.md)

Feeder constants (DSS paths, initial taps, excluded buses) are in `examples/offline/systems.py`. See [Building Simulators](../guide/building-simulators.md) for the full component API.
