# GPU Flexibility for Voltage Regulation

## Research Question

How effective can GPU workload flexibility contribute to mitigating voltage violations caused by different sources — internal load changes (e.g., training task overlay, inference task fluctuations) and external load changes (e.g., time-varying load, renewable generation)?

## Overview

AI datacenter workload parameters (e.g., batch size for LLM inference) can be adjusted programmatically, which simultaneously affects GPU power consumption, inference latency, and token throughput. These power changes propagate through the distribution feeder, affecting bus voltages. This analysis explores whether adjusting batch sizes in response to grid conditions can effectively regulate voltages across different types of disturbances.

OpenG2G provides three IEEE test systems that represent different disturbance sources:

- **IEEE 13-bus**: DC-internal load changes — training task overlay (t=1000–2000s) and inference server ramp-down (t=2500–3000s) create large power swings within the datacenter itself.
- **IEEE 34-bus**: External load changes — time-varying loads and PV solar injection at various buses cause voltage fluctuations independent of the datacenter.
- **IEEE 123-bus**: Both internal and external — per-site inference ramps, PV systems, and time-varying loads all interact simultaneously across four datacenter zones.

## Scripts

| Script | Purpose |
|--------|---------|
| `run_baseline.py` | Run without batch-size control (fixed batch sizes) |
| `run_ofo.py` | Run with OFO batch-size control |

## Usage

### IEEE 13-Bus: Internal Load Changes

The IEEE 13 config includes a training overlay (2400 GPUs, t=1000–2000s) and an inference ramp-down to 20% (t=2500–3000s), creating two distinct voltage stress periods within the datacenter.

```bash
# Baseline (no batch control, no tap changes)
python examples/offline/run_baseline.py --config examples/offline/config_ieee13.json --system ieee13

# OFO batch-size control
python examples/offline/run_ofo.py --config examples/offline/config_ieee13.json --system ieee13
```

### IEEE 34-Bus: External Load Changes

The IEEE 34 config has PV systems at buses 830 and 848 and time-varying loads at five buses. The datacenter load is steady (no ramps), so all voltage disturbances come from external sources.

```bash
python examples/offline/run_baseline.py --config examples/offline/config_ieee34.json --system ieee34
python examples/offline/run_ofo.py --config examples/offline/config_ieee34.json --system ieee34
```

### IEEE 123-Bus: Internal + External

The IEEE 123 config combines per-site inference ramps (four datacenters with different ramp schedules), three PV systems, and time-varying loads — representing the most realistic scenario.

```bash
python examples/offline/run_baseline.py --config examples/offline/config_ieee123.json --system ieee123
python examples/offline/run_ofo.py --config examples/offline/config_ieee123.json --system ieee123
```

## Key Results

Compare the voltage statistics between baseline and OFO runs:

- **Violation time** (seconds): How long voltages exceed [v_min, v_max] bounds
- **Integral violation** (pu-s): Severity-weighted violation (larger = worse)
- **Worst Vmin/Vmax**: Peak voltage excursion

OFO is most effective against slow, predictable disturbances (training ramps, inference scaling) where the primal-dual optimization has time to converge. Fast external disturbances (sudden PV cloud events) are harder to mitigate purely through batch-size control.

## Configuration

Key config fields for this analysis:

- `dc_sites[].inference_ramps`: Per-site server activation schedules (internal disturbance)
- `training`: Training workload overlay with `dc_site`, `n_gpus`, `t_start`/`t_end` (internal disturbance)
- `pv_systems`: Solar PV injections (external disturbance)
- `time_varying_loads`: Additional loads at arbitrary buses (external disturbance)
- `ofo`: OFO controller parameters (see [Controller Parameter Sensitivity](3-controller-parameter-sensitivity.md))

See [Data Pipeline](../guide/data-pipeline.md) for the full config format.
