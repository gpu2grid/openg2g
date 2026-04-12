# Datacenter Sizing and Hosting Capacity

## Research Question

How large can an AI datacenter be on a given feeder before voltage violations become unmanageable? What is the maximum GPU count each bus can host?

## Overview

Hosting capacity analysis determines the maximum datacenter size (in GPUs or MW) that a bus can support while keeping voltage violations below a specified threshold. This is critical for grid interconnection planning: utilities need to know whether a proposed datacenter can be accommodated without infrastructure upgrades.

## Definition

The **hosting capacity** of a bus is defined as the maximum number of GPU replicas (running a given LLM model at a fixed batch size) that can be placed at that bus such that voltage violations do not exceed a specified fraction of the simulation duration.

For example, with a 10% violation fraction threshold and a 60-second simulation, the hosting capacity is the largest replica count where voltage violations occur for at most 6 seconds.

## Implementation

The hosting capacity is found via **binary search** for each candidate bus:

1. **For each bus** in the candidate set:
    - For each LLM model (to find which model yields the highest capacity):
        - Binary search over replica count `[0, max_replicas]`
        - For each candidate count, run a short simulation (60s staircase profile)
        - Measure voltage violation fraction
        - If violation < threshold → increase replicas; otherwise → decrease
    - The bus's hosting capacity is the model/count combination yielding the highest power (MW)

2. **Two passes** per bus:
    - **Pass 1 (Tap-only)**: Optimize regulator tap positions for the bus placement, no OFO control. This represents the hosting capacity with traditional grid-side control.
    - **Pass 2 (OFO + Tap)**: Run with OFO batch-size control and the optimized tap schedule. This shows how much additional capacity OFO enables.

3. **Staircase power profile**: Instead of running a full 3600s simulation, the script uses a compact staircase profile (60s total) where DC power ramps up and down, capturing the worst-case voltage response efficiently.

### 2-D Mode

In 2-D mode, the script sweeps all **unordered pairs** of candidate buses, placing identical datacenters at both locations simultaneously. The hosting capacity of a pair is the minimum across all models. This reveals which bus combinations can support co-located datacenters and produces heatmaps showing capacity across the feeder.

## Scripts

| Script | Purpose |
|--------|---------|
| `sweep_hosting_capacities.py` | Per-bus or bus-pair hosting capacity analysis |

## Usage

### IEEE 13-Bus: 1-D Per-Bus Hosting Capacity

```bash
python examples/offline/sweep_hosting_capacities.py \
    --system ieee13
```

### IEEE 13-Bus: 2-D Bus-Pair Heatmap

```bash
python examples/offline/sweep_hosting_capacities.py \
    --system ieee13 --mode 2d
```

### IEEE 34-Bus: With Zone-Aware Taps

```bash
python examples/offline/sweep_hosting_capacities.py \
    --system ieee34 \
    --violation-fractions 0.05,0.1
```

### Custom Parameters

```bash
# Higher power ceiling and finer search tolerance
python examples/offline/sweep_hosting_capacities.py \
    --system ieee13 \
    --max-power-mw 15 --tolerance 3
```

## Outputs

Saved to `outputs/<system>/hosting_capacity/`:

- `hosting_capacity_{system}_{fraction}.png`: bar chart of per-bus hosting capacity
- `hosting_capacity_{system}_{fraction}.csv`: detailed results (bus, capacity MW, best model, GPU count)
- Heatmaps (2-D mode) showing capacity for each bus pair

## What to Look For

Buses closer to the substation typically have higher hosting capacity due to lower voltage sensitivity. Downstream buses on long laterals may have an order of magnitude lower capacity.

## Configuration

Experiment parameters are defined inline in the script:

- **DC power levels**: `base_kw_per_phase` on `DatacenterConfig`
- **Initial taps**: `TapPosition(regulators={...})` from `systems.py` feeder constants (optimized per bus during analysis)
- **Regulator zones**: `regulator_zones` from `systems.py` feeder constants, maps regulators to downstream buses for zone-aware tap optimization
- **Exclude buses**: From `systems.py` feeder constants

See [Building Simulators](../guide/building-simulators.md) for the full component API.
