"""Baseline (no OFO control) simulation using the openg2g library.

Reproduces the results of baseline_wo_control.py using the modular library
components: OfflineDatacenter + OpenDSSGrid + TapScheduleController + Coordinator.

Two modes correspond to two baselines in the paper:
  no-tap       "No control, no tap" — tap positions are fixed throughout.
  tap-change   "Tap change only" — regulator taps change at t=1500s and t=3300s.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import Coordinator
from openg2g.datacenter.offline import (
    OfflineDatacenter,
    TraceByBatchCache,
    load_traces_by_batch_from_dir,
)
from openg2g.datacenter.training_overlay import TrainingOverlayCache
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.voltage import compute_allbus_voltage_stats
from openg2g.models.spec import ModelSpec
from openg2g.plotting import plot_allbus_voltages_per_phase, plot_power_3ph
from openg2g.types import TapPosition

S = 0.00625  # standard 5/8% tap step

INITIAL_TAPS = TapPosition(a=1.0 + 14 * S, b=1.0 + 6 * S, c=1.0 + 15 * S)

TAP_SCHEDULES = {
    "no-tap": INITIAL_TAPS.at(t=0),
    "tap-change": (
        TapPosition(a=1.0 + 14 * S, b=1.0 + 6 * S, c=1.0 + 15 * S).at(t=0)
        | TapPosition(a=1.0 + 16 * S, b=1.0 + 6 * S, c=1.0 + 17 * S).at(t=25 * 60)
        | TapPosition(a=1.0 + 10 * S, b=1.0 + 6 * S, c=1.0 + 10 * S).at(t=55 * 60)
    ),
}


def main(mode: str = "no-tap") -> None:
    tap_schedule = TAP_SCHEDULES[mode]

    project_dir = Path(__file__).resolve().parent.parent
    trace_dir = project_dir / "power_csvs_updated"
    case_dir = project_dir / "OpenDss_Test" / "13Bus"
    training_csv = trace_dir / "synthetic_training_trace.csv"
    save_dir = project_dir / "outputs" / f"baseline_{mode}"
    save_dir.mkdir(parents=True, exist_ok=True)

    v_min = 0.95
    v_max = 1.05
    dc_bus = "671"
    gpus_per_server = 8
    dt_dc = 0.1
    t_total_s = 3600.0

    models = [
        ModelSpec(model_label="Llama-3.1-8B", replicas=720, gpus_per_replica=1),
        ModelSpec(model_label="Llama-3.1-70B", replicas=180, gpus_per_replica=4),
        ModelSpec(model_label="Llama-3.1-405B", replicas=90, gpus_per_replica=8),
        ModelSpec(model_label="Qwen3-30B-A3B", replicas=480, gpus_per_replica=2),
        ModelSpec(model_label="Qwen3-235B-A22B", replicas=210, gpus_per_replica=8),
    ]

    required_measured_gpus = {ms.model_label: ms.gpus_per_replica for ms in models}

    print("Loading power traces...")
    traces_by_batch = load_traces_by_batch_from_dir(
        base_dir=trace_dir,
        batch_set=[128],
        required_measured_gpus=required_measured_gpus,
        amp_jitter_default=(0.98, 1.02),
        noise_std_frac_default=0.005,
    )

    cache = TraceByBatchCache(traces_by_batch)
    cache.build_templates(T=t_total_s, dt=dt_dc)

    training_overlay = TrainingOverlayCache(training_csv, target_peak_W_per_gpu=400.0)

    print("Initializing OfflineDatacenter...")
    dc = OfflineDatacenter(
        trace_cache=cache,
        models=models,
        dt=dt_dc,
        batch_init=128,
        gpus_per_server=gpus_per_server,
        seed=0,
        chunk_steps=int(t_total_s / dt_dc),
        ramp_t_start=2500.0,
        ramp_t_end=3000.0,
        ramp_floor=0.2,
        base_kW_per_phase=500.0,
        training_overlay=training_overlay,
        training_t_add_start=1000.0,
        training_t_add_end=2000.0,
        training_n_train_gpus=300 * gpus_per_server,
    )

    print("Initializing OpenDSSGrid...")
    grid = OpenDSSGrid(
        case_dir=str(case_dir),
        master="IEEE13Nodeckt.dss",
        dc_bus=dc_bus,
        dc_kv_ll=4.16,
        pf_dc=0.95,
        dt_s=0.1,
        dc_conn="wye",
        controls_off=False,
        tap_schedule=tap_schedule,
        freeze_regcontrols=True,
    )

    ctrl = TapScheduleController(schedule=[], dt_s=1.0)

    print(f"Running simulation (mode={mode})...")
    coord = Coordinator(
        datacenter=dc,
        grid=grid,
        controllers=[ctrl],
        T_total_s=t_total_s,
        dc_bus=dc_bus,
    )
    log = coord.run()
    print(f"Simulation complete: {len(log.grid_states)} grid steps, {len(log.dc_states)} DC steps")

    stats = compute_allbus_voltage_stats(log.grid_states, v_min=v_min, v_max=v_max)
    print("\n=== Voltage Statistics (all-bus) ===")
    print(f"  voltage_violation_time = {stats.violation_time_s:.1f} s")
    print(f"  worst_vmin             = {stats.worst_vmin:.6f}")
    print(f"  worst_vmax             = {stats.worst_vmax:.6f}")
    print(f"  integral_violation     = {stats.integral_violation_pu_s:.4f} pu·s")

    time_s = np.array(log.time_s)
    kW_A = np.array(log.kW_A)
    kW_B = np.array(log.kW_B)
    kW_C = np.array(log.kW_C)

    plot_power_3ph(
        time_s,
        kW_A,
        kW_B,
        kW_C,
        save_path=save_dir / "power_profiles.png",
        title="DC Power by Phase",
    )
    plot_allbus_voltages_per_phase(
        log.grid_states,
        time_s,
        save_dir=save_dir,
        v_min=v_min,
        v_max=v_max,
        filename_template="voltage_trajectories_phase_{label}.png",
        title_template="Voltage Trajectories — Phase {label}",
    )

    with open(save_dir / "console_output.txt", "w") as f:
        f.write("=== Voltage Statistics (all-bus) ===\n")
        f.write(f"  voltage_violation_time = {stats.violation_time_s:.1f} s\n")
        f.write(f"  worst_vmin             = {stats.worst_vmin:.6f}\n")
        f.write(f"  worst_vmax             = {stats.worst_vmax:.6f}\n")
        f.write(f"  integral_violation     = {stats.integral_violation_pu_s:.4f} pu·s\n")

    print(f"\nOutputs saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline simulation (no OFO control)")
    parser.add_argument(
        "--mode",
        choices=["no-tap", "tap-change"],
        default="no-tap",
        help="Baseline variant: 'no-tap' (fixed taps) or 'tap-change' (scheduled tap changes)",
    )
    args = parser.parse_args()
    main(mode=args.mode)
