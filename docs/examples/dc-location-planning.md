# Datacenter Location Planning

## Research Question

Which buses on a feeder are best suited for datacenter placement, how does location affect controllability, and how to find the best locations in a zonal system if multiple datacenters need to be built?

## Overview

Datacenter placement significantly affects voltage regulation. A datacenter at a bus with high voltage sensitivity will cause larger voltage swings but may also be more responsive to batch-size control. This analysis sweeps candidate bus locations to find optimal placements.

The sweep mode auto-selects based on the config:

- **1 DC site, no zones** (e.g., IEEE 13): 1-D sweep with per-bus tap optimization and 4-case comparison (baseline ± tap, OFO ± tap).
- **2+ DC sites, no zones** (e.g., IEEE 34): 2-D sweep over all unordered bus pairs with OFO control, producing heatmap visualizations.
- **N DC sites with zones** (e.g., IEEE 123): Zone-constrained 3-phase sweep for scalability.

### Zone-Constrained 3-Phase Sweep

For systems with many DC sites and large candidate bus sets, an exhaustive sweep is computationally infeasible. For example, IEEE 123 has 4 DC sites with 5, 15, 29, and 7 candidate buses per zone; a full Cartesian product would require 5 × 15 × 29 × 7 = 15,225 full-resolution simulations, each taking ~30 seconds, totaling ~127 hours.

The 3-phase approach reduces this to ~400 simulations (~1-2 hours):

- **Phase 1 (Screening)**: Sweep each zone independently while holding other zones at default buses. Uses a coarser time step (`--dt-screening`) for speed. Ranks candidates per zone and keeps the top-K.
- **Phase 2 (Combination)**: Run the Cartesian product of top-K candidates (K^N combinations). Uses a 60-second stress test with constant peak PV and full DC capacity to quickly rank combinations under worst-case conditions.
- **Phase 3 (Refinement, optional)**: Starting from the Phase 2 winner, iteratively re-sweep each zone at full resolution until no zone improves. Plots mark previous best (diamond) and new best (star) for each iteration.

## Scripts

| Script | Purpose |
|--------|---------|
| `sweep_dc_locations.py` | Automated DC location sweep (1-D, 2-D, or zone-constrained) |

## Usage

### IEEE 13-Bus: 1-D Sweep

```bash
python examples/offline/sweep_dc_locations.py \
    --system ieee13
```

### IEEE 34-Bus: 2-D Heatmap

```bash
python examples/offline/sweep_dc_locations.py \
    --system ieee34
```

### IEEE 123-Bus: Zone-Constrained 3-Phase Sweep

```bash
# Phase 1 + Phase 2 only (fast screening)
python examples/offline/sweep_dc_locations.py \
    --system ieee123 \
    --dt-screening 60 --top-k 4

# With Phase 3 refinement (full resolution re-sweep)
python examples/offline/sweep_dc_locations.py \
    --system ieee123 \
    --dt-screening 60 --top-k 4 --refine
```

## Outputs

Files follow a consistent naming convention:

| File | Description |
|------|-------------|
| `Phase_1_screening_results_{system}.png` | Per-zone screening bar charts |
| `Phase_2_combination_results_{system}.png` | Combination ranking bar charts |
| `Phase_3_refinement_results_{system}_iter{N}.png` | Per-iteration refinement (with previous/new best markers) |
| `Phase_3_refinement_results_{system}_iter{N}_{zone}.csv` | Detailed per-zone results per iteration |
| `sweep_dc_locations_final_results_{system}.csv` | Best bus combination found, with voltage metrics plus `mean_throughput_tps` and `itl_deadline_fraction` per candidate |

## Configuration

Experiment parameters are defined inline in each script's experiment functions:

- **DC sites**: Bus, `base_kw_per_phase`, model deployments, seed, constructed as `OfflineDatacenter` objects
- **Zones**: Per-zone candidate bus lists from `systems.py` feeder constants (required for zone-constrained mode)
- **Initial taps**: `TapPosition(regulators={...})` from `systems.py` feeder constants
- **Exclude buses**: From `systems.py` feeder constants
- **Generators and loads**: `SyntheticPV`/`SyntheticLoad` attached to grid (included in stress test)

See [Building Simulators](../guide/building-simulators.md) for the full component API.
