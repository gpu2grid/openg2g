# Voltage Regulation Strategies

## Research Question

How does datacenter-side batch-size control compare with grid-side regulator tap changes for voltage regulation? How do different batch-size control algorithms (e.g., OFO, rule-based) compare in maintaining grid stability metrics?

## Overview

Distribution feeders traditionally regulate voltage using regulator tap changers — mechanical devices that adjust transformer turns ratios. Datacenter batch-size control offers a complementary demand-side approach: adjusting GPU workload parameters to modulate power consumption in real time.

This analysis compares three control strategies:

1. **Baseline with tap changes**: Traditional grid-side control only (regulator tap schedule, no batch adjustment)
2. **Rule-based batch control**: Simple proportional controller that reduces batch on undervoltage and increases on overvoltage — no sensitivity matrix or model fits required
3. **OFO batch control**: Primal-dual optimization using voltage sensitivity matrices and logistic curve fits for gradient-based batch adjustment

## Scripts

| Script | Purpose |
|--------|---------|
| `run_baseline.py` | Baseline with optional tap schedule (`--mode no-tap` or `--mode tap-change`) |
| `run_ofo.py` | OFO with optional tap schedule (`--mode no-tap` or `--mode tap-change`) |
| `analyze_different_controllers.py` | Side-by-side comparison of baseline, rule-based, and OFO |

## Usage

### Controller Comparison (IEEE 13)

The comparison script runs all three controllers on the same system. Only the baseline uses tap schedule changes; OFO and rule-based run with fixed initial taps to isolate the effect of batch-size control.

```bash
python examples/offline/analyze_different_controllers.py \
    --system ieee13
```

Outputs (in `outputs/ieee13/controller_comparison/`):

- `voltage_comparison.png` — Side-by-side voltage envelopes
- `batch_size_comparison.png` — Per-model batch size traces over time
- `summary_bar_chart.png` — Violation time, integral violation, worst Vmin
- `results_ieee13.csv` — Metrics table

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

- `--rule-step-size`: Proportional gain (default 10.0). Higher = faster response but more oscillation.
- `--rule-deadband`: Minimum violation magnitude to act on (default 0.001 pu). Smaller = faster response but more chattering.

```bash
python examples/offline/analyze_different_controllers.py \
    --system ieee13 \
    --rule-step-size 15.0 --rule-deadband 0.0005
```

## Key Results

Typical findings on IEEE 13-bus:

| Controller | Violation Time | Integral | Mechanism |
|-----------|---------------|----------|-----------|
| Baseline (with taps) | ~1050s | ~42 pu-s | Tap changes at scheduled times |
| Rule-based (no taps) | ~1250s | ~15 pu-s | Proportional batch reduction on violation |
| OFO (no taps) | ~120s | ~0.1 pu-s | Sensitivity-aware gradient descent |

OFO outperforms because it: (1) uses model-specific sensitivity to adjust the right models, (2) acts proactively via dual variables before violations occur, and (3) operates in continuous log2-space with gradient information from logistic fits.

## Configuration

- `tap_schedule`: Defines when and how regulator taps change (baseline only)
- `ofo`: OFO controller parameters
- `initial_taps`: Starting tap positions for all modes

See [Building Simulators](../guide/building-simulators.md) and `examples/offline/systems.py` for configuration details.
