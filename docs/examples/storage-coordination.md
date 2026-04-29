# Storage-Assisted Datacenter-Grid Coordination

## Research Question

When datacenter OFO is already active, how does adding grid-connected storage change the datacenter-grid coordination problem? Can storage provide complementary voltage support and help datacenter-side OFO coordinate with grid objectives?

## Overview

This example adds energy storage to the standard IEEE 123-bus controller-comparison setup from `analyze_different_controllers.py`. By reusing the same base setup, the storage scenarios can be compared directly against the existing OFO case.

The scenarios compare standard OFO with storage attached at the datacenter sites:

1. **Baseline**: Standard IEEE 123-bus baseline from the controller-comparison example.
2. **OFO**: Standard IEEE 123-bus OFO case.
3. **OFO + idle storage**: Storage elements are attached but not actively controlled.
4. **OFO + Q-V storage droop**: Storage injects or absorbs reactive power using local voltage.
5. **OFO + P-V storage droop**: Storage charges or discharges real power using local voltage.

In this example, the storage systems are deliberately placed 1:1 with the standard datacenter buses (`8`, `23`, `60`, and `105`). This is a clear first placement rule rather than an optimized siting study: storage is placed next to the datacenter resources whose flexibility is already being studied.

## Script

| Script | Purpose |
|--------|---------|
| `analyze_storage_coordination.py` | Compare standard OFO with idle, Q-V, and P-V storage added at datacenter buses |

## Usage

Run all scenarios:

```bash
python examples/offline/analyze_storage_coordination.py
```

Run a smaller comparison:

```bash
python examples/offline/analyze_storage_coordination.py \
    --scenarios ofo ofo_qv_storage
```

Write outputs to a custom directory:

```bash
python examples/offline/analyze_storage_coordination.py \
    --output-dir outputs/storage_on_ofo
```

## Outputs

By default, outputs are saved to `examples/offline/outputs/ieee123/storage_on_ofo/`.

| Type | File | Contents |
|------|------|----------|
| CSV | `summary_metrics.csv` | Voltage and datacenter performance metrics for each scenario |
| CSV | `storage_timeseries.csv` | Per-storage realized `P`, realized `Q`, SOC, OpenDSS state, and local voltage over time |
| CSV | `storage_dispatch_summary.csv` | Per-storage dispatch extrema and SOC changes |
| CSV | `datacenter_power_timeseries.csv` | OFO-controlled datacenter fleet power over time |
| Figure | `voltage_envelope_comparison.png` | Feeder voltage envelope for each scenario |
| Figure | `local_storage_voltage_comparison.png` | Local voltage at each storage/datacenter bus |
| Figure | `storage_dispatch_soc.png` | Storage realized `P`, realized `Q`, and SOC. With the default scenarios this is a 3-by-3 plot: rows are `P`, `Q`, and SOC; columns are idle, P-V, and Q-V storage. Subset runs include only the selected storage scenarios. |
| Figure | `datacenter_fleet_power_comparison.png` | Total datacenter fleet power across scenarios |

## What to Look For

Read voltage and datacenter performance metrics together. Storage first changes the grid conditions; OFO then responds to the changed voltage environment. In some windows, improved voltage support means OFO can compress batch sizes less aggressively, which appears as higher datacenter fleet power. Storage is not directly changing datacenter demand.

The idle-storage case should closely match the standard OFO case. It is included to separate the effect of attaching native OpenDSS storage elements from the effect of actively controlling them.

Q-V storage is the main narrative case for voltage support because it provides reactive support without directly commanding real-power charge or discharge. P-V storage is useful as a complementary case because it uses real energy and shows how SOC evolves when storage charges or discharges.

The storage CSVs report OpenDSS readback values, not only commanded setpoints. Very small realized `P` or `Q` values can appear in the non-commanded channel as the solved circuit state changes; treat them as part of the simulation output rather than as a separate controller action.

## Configuration

Storage sites are defined inline in the script:

| Storage | Bus | Rated real power | Apparent power |
|---------|-----|------------------|----------------|
| `bat_z1` | `8` | 250 kW | 300 kVA |
| `bat_z2` | `23` | 250 kW | 300 kVA |
| `bat_z3` | `60` | 300 kW | 360 kVA |
| `bat_z4` | `105` | 350 kW | 420 kVA |

All four storage systems use a 2-hour duration and initial SOC of 0.5. The example leaves [`BatteryStorage.idle_loss_percent`][openg2g.grid.storage.BatteryStorage] at its default value of `0.0`, so idle and pure reactive operation do not intentionally drain stored energy.

The controlled storage scenarios use [`LocalVoltageStorageDroopController`][openg2g.controller.storage.LocalVoltageStorageDroopController] with [`StorageDroopConfig`][openg2g.controller.storage.StorageDroopConfig]:

| Scenario | Storage controller | Behavior |
|----------|--------------------|----------|
| `ofo_idle_storage` | None | Storage is attached, but no storage controller changes its setpoint. |
| `ofo_qv_storage` | `StorageDroopConfig(mode="qv")` | Uses local voltage to command reactive power. Positive output injects kvar when voltage is low. |
| `ofo_pv_storage` | `StorageDroopConfig(mode="pv")` | Uses local voltage to command real power. Positive output discharges storage when voltage is low. |

The droop controller targets all attached storage resources by default. Each storage system uses only its own bus voltage from the previous control window, with the minimum local phase voltage as the default voltage statistic. The example uses the default droop tuning: `v_ref = 1.0`, `deadband_pu = 0.005`, and `full_output_voltage_error_pu = 0.05`. Commands are emitted once per controller step and held by the storage resource until the next controller tick.

The controller does not optimize SOC. In Q-V mode, SOC should remain nearly unchanged when idle losses are zero; in P-V mode, SOC evolves through the native OpenDSS storage physics.

See [Building Simulators](../guide/building-simulators.md) for the storage attachment, command, and query APIs.
