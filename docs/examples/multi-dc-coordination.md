# Multi-Datacenter Coordination

## Research Question

How do multiple datacenters at different grid locations interact, how should their controllers coordinate, and can shifting LLM replicas between sites further reduce voltage violations when batch-size control is exhausted?

## Overview

When multiple datacenters share a distribution feeder, their controllers must coordinate to avoid conflicting actions. OpenG2G supports multi-DC simulations where each datacenter has its own OFO controller bound to its site, and an optional cross-site load-shifting controller.

**OFO coordination**: Each per-site OFO controller independently optimizes batch sizes using voltage sensitivities computed for its own loads. The controllers share the same grid state but operate on different datacenter instances.

**Load shifting**: When a site's OFO controller has exhausted batch-size adjustments (all models at min or max feasible batch), the `LoadShiftController` can shift LLM replicas between sites. This is a last-resort mechanism that moves inference servers from high-violation sites to low-violation sites.

## Scripts

| Script | Purpose |
|--------|---------|
| `analyze_LLM_load_shifting.py` | Compare OFO with and without cross-site load shifting |
| `run_ofo.py` | Run multi-DC OFO simulation |

## Usage

### IEEE 123-Bus: OFO vs OFO + Load Shifting

The experiment is defined inline in `analyze_LLM_load_shifting.py` with multi-model DCs (at least 3 models per site for warm-start) and load shifting enabled.

```bash
python examples/offline/analyze_LLM_load_shifting.py --system ieee123
```

Outputs (in `outputs/ieee123/load_shift_comparison/`):

- `voltage_comparison.png` — Side-by-side voltage envelopes (OFO vs OFO+shift)
- `net_replica_shift.png` — Per-site net replica changes over time
- `power_comparison.png` — Per-site DC power traces
- `summary_bar_chart.png` — Violation time and integral comparison

### Load Shifting Rules

The `LoadShiftController` follows five rules:

1. **Warm start only**: Only shifts models already running at both source and destination.
2. **Last resort**: Only acts when all models at the violated site have batch sizes at their feasible limit.
3. **Directional**: Undervoltage → shift replicas OUT; overvoltage → shift replicas IN.
4. **Capacity-aware**: Checks `available_gpu_capacity()` on the destination before shifting.
5. **Incremental**: Shifts `gpus_per_shift` GPUs per time step, repeating until resolved.

## Configuration

```json
"load_shift": {
    "enabled": true,
    "gpus_per_shift": 8,
    "headroom": 0.3
}
```

Key config fields:

- `load_shift.enabled`: Enable/disable the load shifting controller
- `load_shift.gpus_per_shift`: GPUs moved per control step (default 8)
- `load_shift.headroom`: Fraction of extra server capacity to pre-allocate (default 0.3)
- `dc_sites[].total_gpu_capacity`: Maximum GPUs per site (enforced during shifting)
- `dc_sites[].models`: Each site must have at least 3 models for warm-start shifting

See [Building Simulators](../guide/building-simulators.md) and `examples/offline/systems.py` for configuration details.
