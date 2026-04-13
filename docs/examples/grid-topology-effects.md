# Grid Topology and DER Effects

## Research Question

How does the choice of feeder (IEEE 13-bus, 34-bus, 123-bus, etc.), and the locations and capacities of loads and distributed energy resources (DERs), affect datacenter impact and controllability?

## Overview

Distribution feeder topology determines how datacenter power consumption propagates to bus voltages. Shorter, stiffer feeders (like IEEE 13) have lower voltage sensitivity, while longer feeders with downstream loads (like IEEE 34 and 123) exhibit larger voltage drops. The presence of DERs (solar PV, storage) and time-varying loads further complicates the voltage landscape.

OpenG2G includes three standard IEEE test feeders with pre-configured datacenter, PV, and load scenarios:

| System | Buses | Voltage | DC Sites | PV Sites | Regulators | Characteristics |
|--------|-------|---------|----------|----------|------------|-----------------|
| IEEE 13 | 16 | 4.16 kV | 1 | 1 | 1 bank (3-phase) | Short, stiff feeder; internal DC load changes dominate |
| IEEE 34 | 37 | 24.9 kV | 2 | 2 | 2 banks (per-phase) | Long rural feeder; PV and time-varying loads |
| IEEE 123 | 132 | 4.16 kV | 4 | 3 | 4 regulators | Large suburban feeder; 4 zones with mixed disturbances |

## Scripts

| Script | Purpose |
|--------|---------|
| `run_ofo.py` | Baseline and OFO simulations (see [Voltage Regulation Strategies](voltage-regulation-strategies.md) for `--mode` options) |
| `plot_topology.py` | Visualize system topology with DC, PV, and regulator locations |

## Usage

### Visualize System Topology

```bash
python examples/offline/plot_topology.py --system ieee13
python examples/offline/plot_topology.py --system ieee34
python examples/offline/plot_topology.py --system ieee123
```

The topology plot shows buses (colored by zone), DC locations (stars), PV systems (yellow triangles), and voltage regulators (red diamonds). It automatically parses connections and regulator definitions from the OpenDSS files.

### Compare Systems

Run all cases on each system and compare voltage statistics:

```bash
python examples/offline/run_ofo.py --system ieee13 --mode all
python examples/offline/run_ofo.py --system ieee34 --mode all
python examples/offline/run_ofo.py --system ieee123 --mode all
```

## Key Observations

- **Electrical distance matters**: Buses farther from the substation have higher voltage sensitivity to load changes, making them harder to regulate but also more responsive to batch-size control.
- **Regulator placement**: Systems with regulators between the substation and DC buses (like IEEE 34) offer an additional control degree of freedom via tap changes.
- **DER interaction**: PV injection can cause overvoltage at nearby buses while DC loads cause undervoltage; the net effect depends on relative locations and magnitudes.
- **Zone structure**: In multi-zone systems (IEEE 123), different zones may have independent voltage dynamics, requiring per-zone controller tuning.

## Configuration

The topology and DER configuration is defined inline in each experiment function (see `run_ofo.py`):

- **Feeder constants** (in `systems.py`): `dss_case_dir`, `dss_master_file`, `source_pu`, `initial_taps`, `exclude_buses`
- **Experiment parameters** (inline in each script): `dc_sites` (datacenter locations and power levels), `pv_systems` (solar PV locations and peak capacities), `time_varying_loads` (additional load locations and magnitudes)
