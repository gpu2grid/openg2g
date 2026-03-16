# Controller Parameter Sensitivity

## Research Question

How do OFO tuning parameters (step sizes, dual weights, throughput/switching costs) affect voltage regulation performance, and what are the optimal operating points?

## Overview

The OFO controller has several tuning parameters that trade off between voltage regulation aggressiveness, throughput preservation, and batch-size stability. This analysis sweeps each parameter one-at-a-time (1-D) or across sites independently (2-D) to characterize the sensitivity landscape.

The sweep mode auto-selects based on the number of DC sites:

- **1 DC site** (e.g., IEEE 13): 1-D sweep — varies each parameter while keeping others at baseline.
- **2+ DC sites** (e.g., IEEE 34): 2-D sweep — sweeps all per-site parameter combinations independently, producing heatmap visualizations.

## Scripts

| Script | Purpose |
|--------|---------|
| `sweep_ofo_parameters.py` | Automated parameter sweep with comparison plots |

## Usage

### IEEE 13-Bus: 1-D Sweep

```bash
python examples/offline/sweep_ofo_parameters.py \
    --config examples/offline/config_ieee13.json --system ieee13
```

### IEEE 34-Bus: 2-D Per-Site Sweep

```bash
# 2-D sweep (independent parameters per site, auto-selected)
python examples/offline/sweep_ofo_parameters.py \
    --config examples/offline/config_ieee34.json --system ieee34

# Force 1-D sweep (shared parameters across sites)
python examples/offline/sweep_ofo_parameters.py \
    --config examples/offline/config_ieee34.json --system ieee34 --sweep-mode 1d
```

### Overriding Time Resolution

For faster sweeps during exploration, use a coarser time step:

```bash
python examples/offline/sweep_ofo_parameters.py \
    --config examples/offline/config_ieee34.json --system ieee34 --dt 60
```

## Parameters Swept

| Parameter | Config Field | Effect |
|-----------|-------------|--------|
| `voltage_dual_step_size` | `ofo.voltage_dual_step_size` | Aggressiveness of voltage constraint enforcement. **Most impactful** — values 5–10 can cut violation time 3× vs baseline (1.0). |
| `primal_step_size` | `ofo.primal_step_size` | Learning rate for batch-size updates. Sweet spot ~0.05; default 0.1 is slightly suboptimal. |
| `latency_dual_step_size` | `ofo.latency_dual_step_size` | Aggressiveness of latency constraint enforcement. Nearly no effect under standard load (latency dual not binding). |
| `w_throughput` | `ofo.w_throughput` | Weight on throughput preservation in objective. Values > 0.01 cause severe degradation; keep ≤ 1e-3. |
| `w_switch` | `ofo.w_switch` | Regularization for batch-size changes. Value of 5.0 gives mild improvement over baseline 1.0. |

## Key Results

Outputs are saved to `outputs/<system>/sweep_ofo_parameters/`:

- `results_<system>_sweep_ofo_parameters.csv` — One row per run with all metrics
- Per-parameter subdirectories with voltage and batch-size plots
- Heatmaps (2-D mode) showing cross-site parameter interactions

## Configuration

Key config fields:

- `ofo.*`: Baseline parameter values (center of the sweep grid)
- `simulation.total_duration_s`: Simulation length (affects sweep runtime)

See [Data Pipeline](../guide/data-pipeline.md) for the full config format.
