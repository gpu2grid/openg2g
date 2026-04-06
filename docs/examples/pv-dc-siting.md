# Joint PV + DC Siting and Sizing

## Research Question

How does co-located solar PV interact with datacenter loads? Where should PV systems be placed and sized to maximize economic return while minimizing voltage violations? Can PV and datacenter locations be co-optimized?

## Overview

This analysis uses sensitivity-based Mixed-Integer Linear Programming (MILP) to optimally place PV systems and/or datacenters on a distribution feeder. The MILP minimizes annualized cost (PV investment minus operational savings) plus voltage violation penalties, subject to linearized voltage constraints derived from OpenDSS power flow sensitivities.

Two scripts are available:

- **PV-only optimization** (`optimize_pv_locations_and_capacities.py`): Optimizes PV placement and capacity with fixed DC locations. Includes PV curtailment and regulator tap co-optimization.
- **PV + DC co-optimization** (`optimize_pv_and_dc_locations.py`): Jointly optimizes both PV and DC bus assignments. PV and DC capacities are fixed; only locations are decision variables.

Both scripts use [Gurobi](https://www.gurobi.com/) as the MIP solver (requires a license; academic licenses are free). The MILP formulation uses standard Gurobi API calls and could be adapted to other solvers (e.g., CPLEX, HiGHS, SCIP) by replacing the `gurobipy` interface.

## Scripts

| Script | Purpose |
|--------|---------|
| `optimize_pv_locations_and_capacities.py` | PV placement + capacity + tap co-optimization |
| `optimize_pv_and_dc_locations.py` | Joint PV + DC location co-optimization |

## Pipeline

Both scripts follow a 4-step pipeline:

1. **Sensitivity computation**: Finite-difference perturbation of OpenDSS power flow to compute voltage sensitivity matrices (H_pv, H_dc, H_tap, H_load).
2. **Scenario generation**: Five representative operating days (summer, summer_cloudy, spring, autumn_cloudy, winter) with 15-minute resolution, weighted to represent a full year.
3. **MILP formulation and solve**: Gurobi optimizes placement, capacity, curtailment, and tap positions.
4. **Validation**: Full nonlinear OpenDSS power flow to verify MILP solutions (catches linearization errors).

## Mathematical Formulation

### PV-Only Optimization

**Decision variables:**

| Variable | Type | Description |
|----------|------|-------------|
| $x_j$ | Binary | PV placement at candidate bus $j$ |
| $s_j$ | Continuous $\geq 0$ | Installed PV capacity (kW, 3-phase) at bus $j$ |
| $c_{j,t,s}$ | Continuous $\geq 0$ | Curtailed PV power at bus $j$, time $t$, scenario $s$ |
| $\Delta\tau_{r,s,h}$ | Integer | Tap change for regulator $r$, scenario $s$, 2-hour slot $h$ |
| $\sigma^+_{i,t,s}, \sigma^-_{i,t,s}$ | Continuous $\geq 0$ | Over/undervoltage slack at bus-phase $i$ |

**Objective (minimize):**

$$\min \; C_{\text{inv}} \sum_j s_j - 365 \sum_{s,t} w_s \cdot p_t \cdot \left(\sum_j s_j \cdot f_t^{pv} - c_{j,t,s}\right) \cdot \Delta h + C_{\text{viol}} \cdot 365 \sum_{i,t,s} w_s (\sigma^+_{i,t,s} + \sigma^-_{i,t,s}) + C_{\text{switch}} \cdot 365 \sum |\Delta\tau|$$

where $C_{\text{inv}}$ is annualized investment cost ($/kW/yr), $p_t$ is TOU electricity price, $f_t^{pv}$ is normalized PV output, $w_s$ is scenario weight, and $\Delta h$ is time step duration.

**Constraints:**

- PV count: $\sum_j x_j \leq N_{pv}$
- Big-M linking: $s_j \leq S_{\max} \cdot x_j$
- Total budget: $\sum_j s_j \leq S_{\text{total}}$ (optional)
- Curtailment: $c_{j,t,s} \leq s_j \cdot f_t^{pv}$
- Per-zone PV limit: $\sum_{j \in Z} x_j \leq K_{\text{zone}}$
- Voltage bounds:

$$v_{\min} - \sigma^-_{i,t,s} \leq v_0^i + \sum_j H^{pv}_{i,j} \frac{s_j f_t^{pv} - c_{j,t,s}}{3} + \sum_r H^{tap}_{i,r} \Delta\tau_{r,s,h(t)} + \delta^{load}_{i,t,s} \leq v_{\max} + \sigma^+_{i,t,s}$$

### PV + DC Co-Optimization

Extends the PV-only formulation with DC placement variables:

**Additional variables:**

| Variable | Type | Description |
|----------|------|-------------|
| $y_{k,b}$ | Binary | Assign DC site $k$ to candidate bus $b$ |

**Additional constraints:**

- DC assignment: $\sum_{b \in Z_k} y_{k,b} = 1$ for each site $k$ (exactly one bus per site)
- Mutual exclusion: $\sum_k y_{k,b} \leq 1$ for each bus $b$ (at most one DC per bus)

**Extended voltage constraint:**

$$v^i_{t,s} = v_0^i + \underbrace{\sum_j H^{pv}_{i,j} \frac{s_j f_t^{pv} - c_{j,t,s}}{3}}_{\text{PV effect}} + \underbrace{\sum_k D_{k,t,s} \sum_{b \in Z_k} H^{dc}_{i,b} \cdot y_{k,b}}_{\text{DC effect}} + \underbrace{\sum_r H^{tap}_{i,r} \Delta\tau_{r,s,h(t)}}_{\text{tap effect}} + \delta^{load}_{i,t,s}$$

where $D_{k,t,s}$ is the DC demand (kW/phase) for site $k$ at time $t$ in scenario $s$. The DC term is linear because $D_{k,t,s}$ is a known parameter and $y_{k,b}$ is binary.

**Efficiency**: The DC voltage expression $\sum_{b \in Z_k} H^{dc}_{i,b} \cdot y_{k,b}$ is precomputed once per (site, bus-phase) pair and reused across all time-scenario combinations.

No DC investment cost is included in the objective (capacity is fixed); DC placement quality is driven entirely by the voltage violation penalty.

### Sensitivity Computation

All sensitivity matrices are computed via finite-difference perturbation on a "bare" OpenDSS circuit (no PV, no DC loads):

- **$H^{pv}$**: Inject $+\Delta P$ kW/phase as negative load (generation) at each PV candidate bus. Measure $\Delta v / \Delta P$.
- **$H^{dc}$**: Inject $+\Delta P$ kW/phase as positive load at each DC candidate bus. Same as load sensitivity.
- **$H^{tap}$**: Perturb each regulator by +1 tap step. Measure $\Delta v$.
- **$H^{load}$**: Same method as $H^{dc}$, applied to time-varying load buses.

## Usage

### IEEE 34: PV-Only Optimization

```bash
python examples/offline/optimize_pv_locations_and_capacities.py \
    --system ieee34
```

### IEEE 123: PV + DC Co-Optimization

```bash
# Zone-constrained DC placement (default)
python examples/offline/optimize_pv_and_dc_locations.py \
    --system ieee123 \
    --n-pv 3 --s-max-kw 500

# Custom cost parameters
python examples/offline/optimize_pv_and_dc_locations.py \
    --system ieee123 \
    --n-pv 3 --s-max-kw 500 --c-inv 100 --c-viol 10000
```

### Sensitivity Analysis

Run the co-optimization with different cost parameters to understand solution robustness:

| Parameter | Baseline | Sweep Values | Effect |
|-----------|----------|-------------|--------|
| `c_viol` | 5000 | 500, 1000, 20000, 50000 | No effect (baseline already zero violations) |
| `c_inv` | 200 | 50, 100, 500, 1000 | At 500+, no PV placed (too expensive) |
| `c_switch` | 10 | 0, 50, 100 | At 0, different PV/DC locations (free tap switching enables alternatives) |
| `s_max_kw` | 500 | 250, 1000 | Smaller PV spreads across more buses |
| `n_pv` | 3 | 1, 5 | Fewer PVs → DCs shift to compensate |

## Key Results

Outputs are saved to `outputs/<system>/pv_dc_coopt/` (or `pv_expansion/`):

- `pv_placements_{system}.csv` — Selected PV buses and capacities
- `dc_assignments_{system}.csv` — DC site → bus assignments (co-optimization only)
- `optimized_topology_{system}.png` — Topology map with optimized locations
- `validation_{system}.csv` — Per-scenario nonlinear validation results
- `milp_summary_{system}.json` — Full optimization results

## Configuration

Key config fields:

- `dc_sites`: DC site templates with zone assignment and capacity
- `pv_systems`: PV system specs (used as reference; optimization may change locations)
- `zones`: Per-zone candidate buses for zone-constrained DC placement
- `initial_taps`: Starting regulator positions (co-optimized with PV/DC placement)
- `regulator_zones`: Maps regulators to downstream buses

See [Building Simulators](../guide/building-simulators.md) and `examples/offline/systems.py` for configuration details.
