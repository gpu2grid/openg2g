# Voltage Regulation Strategies

## Research Question

How does datacenter-side batch-size control compare with grid-side regulator tap changes for voltage regulation? How do different batch-size control algorithms (e.g., OFO, rule-based) compare in maintaining grid stability metrics?

## Overview

Distribution feeders traditionally regulate voltage using regulator tap changers: mechanical devices that adjust transformer turns ratios. Datacenter batch-size control offers a complementary demand-side approach: adjusting GPU workload parameters to modulate power consumption in real time.

This analysis compares four control strategies:

1. **Baseline with tap changes**: Traditional grid-side control only (regulator tap schedule, no batch adjustment)
2. **Rule-based batch control**: Simple proportional controller that reduces batch on undervoltage and increases on overvoltage; no sensitivity matrix or model fits required
3. **OFO batch control**: Primal-dual optimization using voltage sensitivity matrices and logistic curve fits for gradient-based batch adjustment
4. **PPO batch control**: A reinforcement-learning policy (Proximal Policy Optimization) that maps a structured observation of grid + datacenter state to per-model batch-size actions. Requires a separate training run; see [Reinforcement Learning Controller (PPO)](rl-controller.md) for the end-to-end workflow.

## Scripts

| Script | Purpose |
|--------|---------|
| `run_ofo.py --mode baseline-no-tap` | Baseline without tap schedule |
| `run_ofo.py --mode baseline-tap-change` | Baseline with tap schedule |
| `run_ofo.py --mode ofo-no-tap` | OFO without tap schedule |
| `run_ofo.py --mode ofo-tap-change` | OFO with tap schedule |
| `analyze_different_controllers.py` | Side-by-side comparison of baseline, rule-based, and OFO |
| `evaluate_controllers.py` | Held-out scenario evaluation that also accepts trained PPO models via `--ppo-models` (see [Reinforcement Learning Controller (PPO)](rl-controller.md)) |

## Usage

### Controller Comparison (IEEE 13)

The comparison script runs all three controllers on the same system. Only the baseline uses tap schedule changes; OFO and rule-based run with fixed initial taps to isolate the effect of batch-size control.

```bash
python examples/offline/analyze_different_controllers.py \
    --system ieee13
```

Outputs (in `outputs/ieee13/controller_comparison/`):

- `voltage_comparison.png`: side-by-side voltage envelopes
- `batch_size_comparison.png`: per-model batch size traces over time
- `summary_bar_chart.png`: violation time, integral violation, worst Vmin
- `results_ieee13.csv`: metrics table (voltage + throughput + ITL-miss columns)

### Tap vs No-Tap Comparison (IEEE 13)

```bash
# Run all 4 cases: baseline ± tap, OFO ± tap
python examples/offline/run_ofo.py --system ieee13 --mode all
```

### Multi-DC System (IEEE 123)

```bash
python examples/offline/analyze_different_controllers.py \
    --system ieee123
```

### Tuning the Rule-Based Controller

The rule-based controller has two key parameters:

- `--rule-step-size`: Proportional gain (default 0.3). Higher = faster response but more oscillation.
- `--rule-deadband`: Minimum violation magnitude to act on (default 0.005 pu). Smaller = faster response but more chattering.

```bash
python examples/offline/analyze_different_controllers.py \
    --system ieee13 \
    --rule-step-size 1.0 --rule-deadband 0.001
```

## What to Look For

The script prints a summary table and writes `results_<system>.csv` with voltage + performance columns. Read both together:

- **Baseline** and **rule-based** typically land close on voltage (both at or near zero violations under moderate load), but rule-based never pushes batch sizes up -- it leaves throughput identical to baseline.
- **OFO** matches rule-based on voltage while serving several times more tokens per second, at the cost of a small ITL-over-deadline fraction (typically a few percent). This is the win OFO is designed for: it uses model-specific voltage sensitivity, acts proactively via dual variables, and operates in continuous log2-batch space with gradient information from logistic fits.
- Under harder stress (e.g., ieee34's two-DC setup), OFO additionally beats rule-based on voltage by orders of magnitude because rule-based can only apply a single uniform batch nudge while OFO drives each model independently.

## Configuration

Experiment parameters are defined inline in each script's setup functions:

- **Tap schedule**: `TapPosition(...).at(t=...) | TapPosition(...).at(t=...)`, which defines when and how regulator taps change (baseline only)
- **OFO tuning**: `OFOConfig(...)` for controller parameters
- **Initial taps**: `TapPosition(regulators={...})` giving starting tap positions (in `systems.py` feeder constants)

See [Building Simulators](../guide/building-simulators.md) for the full component API.
