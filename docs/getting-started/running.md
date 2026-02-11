# Running a Simulation

OpenG2G ships with two example simulations in the `examples/` directory:

- `run_baseline.py` -- uncontrolled baseline (no OFO, capacitor banks active)
- `run_ofo.py` -- OFO closed-loop control with batch size optimization

## Data Requirements

Both examples require data files that are not included in the repository:

- **Power trace CSVs** (`power_csvs_updated/`) -- per-model GPU power traces at various batch sizes, latency fit parameters, and logistic fit parameters.
- **OpenDSS case files** (`OpenDss_Test/13Bus/`) -- IEEE 13-bus test feeder files.

Place these directories in the project root before running.

## Baseline Simulation

The baseline runs the datacenter at a fixed batch size with OpenDSS capacitor bank controls active:

```bash
uv run python examples/run_baseline.py
```

Key parameters (defined at the top of the script):

| Parameter | Value | Description |
|---|---|---|
| `DT_DC` | 0.1s | Datacenter timestep |
| `DT_DSS` | 0.1s | Grid solver timestep |
| `BATCH_INIT` | 128 | Fixed batch size for all models |
| `T_TOTAL_S` | 3600s | Simulation duration |

Outputs are saved to `outputs/baseline/`:

- `power_profiles.png` -- three-phase DC power over time
- `voltage_trajectories_phase_{A,B,C}.png` -- all-bus voltage trajectories
- `console_output.txt` -- voltage violation statistics

## OFO Simulation

The OFO simulation uses the primal-dual batch size controller:

```bash
uv run python examples/run_ofo.py
```

This adds several OFO-specific parameters:

| Parameter | Value | Description |
|---|---|---|
| `ETA_PRIMAL` | 0.1 | Primal step size |
| `W_LATENCY` | 1.0 | Latency penalty weight |
| `K_V` | 1e6 | Voltage penalty gain |
| `RHO_V`, `RHO_L` | 1.0 | Dual step sizes (voltage, latency) |

Outputs are saved to `outputs/ofo/`:

- `dc_power_3ph.png` -- three-phase DC power
- `batch_schedule.png` -- batch size schedule per model
- `all_bus_voltages_phase_{A,B,C}.png` -- voltage trajectories

## Understanding the Output

### Voltage Statistics

Both simulations print voltage violation statistics:

```
=== Voltage Statistics (all-bus) ===
  voltage_violation_time = 1006.5 s
  worst_vmin             = 0.934839
  worst_vmax             = 1.050770
  integral_violation     = 31.2408 pu-s
```

- **violation_time**: total time any bus-phase voltage is outside [0.95, 1.05] pu
- **worst_vmin / worst_vmax**: extremes across all buses, phases, and time
- **integral_violation**: time-integrated sum of voltage violations across all bus-phase pairs

### Batch Schedule (OFO only)

```
=== Batch Schedule Summary ===
  Llama-3.1-8B: avg_batch=461.0, changes=11
```

Shows the average batch size and number of batch size changes per model over the simulation.
