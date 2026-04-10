# Controller Parameter Sensitivity

## Research Question

How do OFO tuning parameters (step sizes, dual weights, throughput/switching costs) affect voltage regulation performance, and what are the optimal operating points?

## Overview

The OFO controller has several tuning parameters that trade off between voltage regulation aggressiveness, throughput preservation, and batch-size stability. This analysis sweeps each parameter one-at-a-time (shared) or across sites independently (per-site) to characterize the sensitivity landscape.

The sweep mode auto-selects based on the number of DC sites:

- **1 DC site** (e.g., IEEE 13): shared sweep — varies each parameter while keeping others at baseline.
- **2+ DC sites** (e.g., IEEE 34): per-site sweep — sweeps all per-site parameter combinations independently, producing heatmap visualizations.

## Scripts

| Script | Purpose |
|--------|---------|
| `sweep_ofo_parameters.py` | Automated parameter sweep with comparison plots |

## Usage

### IEEE 13-Bus: Shared Sweep

```bash
python examples/offline/sweep_ofo_parameters.py \
    --system ieee13
```

### IEEE 34-Bus: Per-Site Sweep

```bash
# per-site sweep (independent parameters per site, auto-selected)
python examples/offline/sweep_ofo_parameters.py \
    --system ieee34

# Force shared sweep (shared parameters across sites)
python examples/offline/sweep_ofo_parameters.py \
    --system ieee34 --sweep-mode shared
```

### Overriding Time Resolution

For faster sweeps during exploration, use a coarser time step:

```bash
python examples/offline/sweep_ofo_parameters.py \
    --system ieee34 --dt 60
```

## Parameters Swept

| Parameter | `OFOConfig` field | Effect |
|-----------|------------------|--------|
| `voltage_dual_step_size` | `voltage_dual_step_size` | Aggressiveness of voltage constraint enforcement. **Most impactful** — values 5–10 can cut violation time 3x vs baseline (1.0). |
| `primal_step_size` | `primal_step_size` | Learning rate for batch-size updates. Sweet spot ~0.05; default 0.1 is slightly suboptimal. |
| `latency_dual_step_size` | `latency_dual_step_size` | Aggressiveness of latency constraint enforcement. Nearly no effect under standard load (latency dual not binding). |
| `w_throughput` | `w_throughput` | Weight on throughput preservation in objective. Values > 0.01 cause severe degradation; keep <= 1e-3. |
| `w_switch` | `w_switch` | Regularization for batch-size changes. Value of 5.0 gives mild improvement over baseline 1.0. |

## Key Results

Outputs are saved to `outputs/<system>/sweep_ofo_parameters/`:

- `results_<system>_sweep_ofo_parameters.csv` — One row per run with all metrics
- Per-parameter subdirectories with voltage and batch-size plots
- Heatmaps (per-site mode) showing cross-site parameter interactions

## Configuration

- **Baseline OFO parameters**: `OFOConfig(...)` — center of the sweep grid, defined inline in each script's experiment function
- **Simulation length**: `TOTAL_DURATION_S` constant in each script (default 3600s)

See [Building Simulators](../guide/building-simulators.md) for the full component API.
