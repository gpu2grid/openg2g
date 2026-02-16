"""Baseline (no OFO control) simulation using the openg2g library.

Reproduces the results of baseline_wo_control.py using the modular library
components: OfflineDatacenter + OpenDSSGrid + TapScheduleController + Coordinator.

Two modes correspond to two baselines in the paper:
  no-tap       "No control, no tap": tap positions are fixed throughout.
  tap-change   "Tap change only": regulator taps change at t=1500s and t=3300s.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np
from plotting import (
    extract_per_model_timeseries,
    load_itl_fits_from_csv,
    plot_allbus_voltages_per_phase,
    plot_power_3ph,
    plot_power_and_itl_2panel,
)

from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import Coordinator
from openg2g.datacenter.config import DatacenterConfig, WorkloadConfig
from openg2g.datacenter.offline import (
    OfflineDatacenter,
    TraceByBatchCache,
    load_traces_by_batch_from_dir,
)
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.voltage import compute_allbus_voltage_stats
from openg2g.models.spec import LLMInferenceModelSpec, LLMInferenceWorkload
from openg2g.types import ServerRamp, TapPosition, TrainingRun

TAP_STEP = 0.00625  # standard 5/8% tap step

INITIAL_TAPS = TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP)

TAP_SCHEDULES = {
    "no-tap": INITIAL_TAPS.at(t=0),
    "tap-change": (
        TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP).at(t=0)
        | TapPosition(a=1.0 + 16 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 17 * TAP_STEP).at(
            t=25 * 60
        )
        | TapPosition(a=1.0 + 10 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 10 * TAP_STEP).at(
            t=55 * 60
        )
    ),
}

MODELS = (
    LLMInferenceModelSpec(
        "Llama-3.1-8B", num_replicas=720, gpus_per_replica=1, initial_batch_size=128
    ),
    LLMInferenceModelSpec(
        "Llama-3.1-70B", num_replicas=180, gpus_per_replica=4, initial_batch_size=128
    ),
    LLMInferenceModelSpec(
        "Llama-3.1-405B", num_replicas=90, gpus_per_replica=8, initial_batch_size=128
    ),
    LLMInferenceModelSpec(
        "Qwen3-30B-A3B", num_replicas=480, gpus_per_replica=2, initial_batch_size=128
    ),
    LLMInferenceModelSpec(
        "Qwen3-235B-A22B", num_replicas=210, gpus_per_replica=8, initial_batch_size=128
    ),
)

INFERENCE = LLMInferenceWorkload(models=MODELS)


def main(args: argparse.Namespace) -> None:
    mode = args.mode
    tap_schedule = TAP_SCHEDULES[mode]

    project_dir = Path(__file__).resolve().parent.parent
    case_dir = Path(__file__).resolve().parent / "ieee13"
    save_dir = project_dir / "outputs" / f"baseline_{mode}"
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir:
        data_dir = Path(args.data_dir)
        trace_dir = data_dir / "traces"
    else:
        data_dir = None
        trace_dir = project_dir / "power_csvs_updated"

    if args.training_trace:
        training_csv = Path(args.training_trace)
    elif args.data_dir:
        raise SystemExit(
            "Error: --training-trace is required when using --data-dir "
            "(training trace is not part of the build output)"
        )
    else:
        training_csv = project_dir / "power_csvs_updated" / "synthetic_training_trace.csv"

    v_min = 0.95
    v_max = 1.05
    dc_bus = "671"
    gpus_per_server = 8
    dt_dc = Fraction(1, 10)
    t_total_s = 3600

    print("Loading power traces...")
    traces_by_batch = load_traces_by_batch_from_dir(
        base_dir=trace_dir,
        batch_set=[128],
        required_measured_gpus=INFERENCE.required_measured_gpus,
        amp_jitter_default=(0.98, 1.02),
        noise_std_frac_default=0.005,
    )

    cache = TraceByBatchCache(traces_by_batch)
    cache.build_templates(T=t_total_s, dt=dt_dc)

    print("Loading latency fits...")
    if data_dir is not None:
        itl_fits = load_itl_fits_from_csv(data_dir / "latency_fits.csv")
    else:
        itl_fits = load_itl_fits_from_csv(trace_dir / "ALL_MODELS_latency_fit_parameters_ALL.csv")

    dc_config = DatacenterConfig(gpus_per_server=gpus_per_server, base_kW_per_phase=500.0)
    workload = WorkloadConfig(
        inference=INFERENCE,
        training=TrainingRun(
            t_start=1000.0,
            t_end=2000.0,
            n_gpus=300 * gpus_per_server,
            trace_csv=training_csv,
            target_peak_W_per_gpu=400.0,
        ),
        server_ramps=ServerRamp(t_start=2500.0, t_end=3000.0, target=0.2),
    )

    print("Initializing OfflineDatacenter...")
    dc = OfflineDatacenter.from_config(
        dc_config,
        workload,
        trace_cache=cache,
        dt=dt_dc,
        seed=0,
        chunk_steps=int(t_total_s / dt_dc),
        latency_fits=itl_fits,
        latency_exact_threshold=30,
        latency_seed=0,
    )

    print("Initializing OpenDSSGrid...")
    grid = OpenDSSGrid(
        case_dir=str(case_dir),
        master="IEEE13Nodeckt.dss",
        dc_bus=dc_bus,
        dc_kv_ll=4.16,
        pf_dc=0.95,
        dt_s=Fraction(1, 10),
        dc_conn="wye",
        controls_off=False,
        tap_schedule=tap_schedule,
        freeze_regcontrols=True,
    )

    ctrl = TapScheduleController(schedule=[], dt_s=Fraction(1))

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

    per_model = extract_per_model_timeseries(log.dc_states)

    # Fig. 5: 2-panel power + ITL
    plot_power_and_itl_2panel(
        time_s,
        kW_A,
        kW_B,
        kW_C,
        avg_itl_by_model=per_model.itl_s,
        itl_time_s=per_model.time_s,
        save_path=save_dir / "power_latency_subfigs.png",
    )

    # Fig. 6: All-bus voltages with bus colormap
    plot_allbus_voltages_per_phase(
        log.grid_states,
        time_s,
        save_dir=save_dir,
        v_min=v_min,
        v_max=v_max,
        title_template="Voltage trajectories without GPU flexibility (Phase {label})",
    )

    # Standalone power plot (backward-compatible)
    plot_power_3ph(
        time_s,
        kW_A,
        kW_B,
        kW_C,
        save_path=save_dir / "power_profiles.png",
        title="DC Power by Phase",
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
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Toolkit-generated data directory (contains traces/). "
        "Defaults to power_csvs_updated/ for legacy mode.",
    )
    parser.add_argument(
        "--training-trace",
        default=None,
        help="Path to synthetic training trace CSV. Required when using "
        "--data-dir; defaults to power_csvs_updated/synthetic_training_trace.csv.",
    )
    args = parser.parse_args()
    main(args)
