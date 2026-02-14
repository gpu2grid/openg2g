"""OFO closed-loop simulation using the openg2g library.

Reproduces the results of final_ofo_test.py using the modular library
components: OfflineDatacenter + OpenDSSGrid + [TapScheduleController,
OFOBatchController] + Coordinator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openg2g.controller.ofo import (
    OFOBatchController,
    PrimalCfg,
    VoltageDualCfg,
)
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
from openg2g.models.latency import load_latency_fits
from openg2g.models.logistic import load_logistic_fits, load_logistic_fits_merged
from openg2g.models.spec import ModelSpec
from openg2g.plotting import (
    plot_allbus_voltages_per_phase,
    plot_batch_schedule,
    plot_power_3ph,
)
from openg2g.types import TapPosition


def main(args: argparse.Namespace) -> None:
    project_dir = Path(__file__).resolve().parent.parent
    case_dir = project_dir / "OpenDss_Test" / "13Bus"
    save_dir = project_dir / "outputs" / "ofo"
    save_dir.mkdir(parents=True, exist_ok=True)

    v_min = 0.95
    v_max = 1.05
    dc_bus = "671"
    gpus_per_server = 8
    batch_init = 128
    batch_set = [8, 16, 32, 64, 128, 256, 512]
    dt_dc = 0.1
    dt_ctrl = 1.0
    t_total_s = 3600.0

    models = [
        ModelSpec(model_label="Llama-3.1-8B", replicas=720, gpus_per_replica=1),
        ModelSpec(model_label="Llama-3.1-70B", replicas=180, gpus_per_replica=4),
        ModelSpec(model_label="Llama-3.1-405B", replicas=90, gpus_per_replica=8),
        ModelSpec(model_label="Qwen3-30B-A3B", replicas=480, gpus_per_replica=2),
        ModelSpec(model_label="Qwen3-235B-A22B", replicas=210, gpus_per_replica=8),
    ]

    required_measured_gpus = {ms.model_label: ms.gpus_per_replica for ms in models}

    TAP_STEP = 0.00625  # standard 5/8% tap step
    tap_schedule = TapPosition(
        a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP
    ).at(t=0)

    data_dir = Path(args.data_dir) if args.data_dir else None
    if data_dir is not None:
        trace_dir = data_dir / "traces"
    else:
        trace_dir = project_dir / "power_csvs_updated"

    if args.training_trace:
        training_csv = Path(args.training_trace)
    elif data_dir is not None:
        raise SystemExit(
            "Error: --training-trace is required when using --data-dir "
            "(training trace is not part of the build output)"
        )
    else:
        training_csv = project_dir / "power_csvs_updated" / "synthetic_training_trace.csv"

    print("Loading power traces...")
    traces_by_batch = load_traces_by_batch_from_dir(
        base_dir=trace_dir,
        batch_set=batch_set,
        required_measured_gpus=required_measured_gpus,
        amp_jitter_default=(0.98, 1.02),
        noise_std_frac_default=0.005,
    )

    cache = TraceByBatchCache(traces_by_batch)
    cache.build_templates(T=600.0, dt=dt_dc)

    training_overlay = TrainingOverlayCache(training_csv, target_peak_W_per_gpu=400.0)

    print("Loading latency fits...")
    if data_dir is not None:
        latency_fits = load_latency_fits(data_dir / "latency_fits.csv")
    else:
        latency_fits = load_latency_fits(trace_dir / "ALL_MODELS_latency_fit_parameters_ALL.csv")

    print("Initializing OfflineDatacenter...")
    dc = OfflineDatacenter(
        trace_cache=cache,
        models=models,
        dt=dt_dc,
        batch_init=batch_init,
        gpus_per_server=gpus_per_server,
        seed=0,
        chunk_steps=int(dt_ctrl / dt_dc),
        ramp_t_start=2500.0,
        ramp_t_end=3000.0,
        ramp_floor=0.2,
        base_kW_per_phase=500.0,
        training_overlay=training_overlay,
        training_t_add_start=1000.0,
        training_t_add_end=2000.0,
        training_n_train_gpus=300 * gpus_per_server,
        latency_fits=latency_fits,
        latency_exact_threshold=30,
        latency_seed=0,
    )

    print("Loading logistic fits...")
    if data_dir is not None:
        fits = load_logistic_fits_merged(data_dir / "logistic_fits.csv")
    else:
        fits = load_logistic_fits(
            {
                label: trace_dir / f"{label}_logistic_fit_parameters_combined.csv"
                for label in required_measured_gpus
            }
        )

    print("Initializing OpenDSSGrid...")
    grid = OpenDSSGrid(
        case_dir=str(case_dir),
        master="IEEE13Nodeckt.dss",
        dc_bus=dc_bus,
        dc_kv_ll=4.16,
        pf_dc=0.95,
        dt_s=dt_dc,
        dc_conn="wye",
        controls_off=True,
        tap_schedule=tap_schedule,
        freeze_regcontrols=True,
    )

    tap_ctrl = TapScheduleController(schedule=[], dt_s=dt_ctrl)

    ofo_ctrl = OFOBatchController(
        models=models,
        fits=fits,
        Lth_by_model={
            "Llama-3.1-405B": 0.12,
            "Llama-3.1-70B": 0.10,
            "Llama-3.1-8B": 0.08,
            "Qwen3-235B-A22B": 0.14,
            "Qwen3-30B-A3B": 0.06,
        },
        primal_cfg=PrimalCfg(
            eta_primal=0.1,
            w_latency=1.0,
            w_throughput=1e-3,
            w_switch=1.0,
            k_v=1e6,
        ),
        voltage_dual_cfg=VoltageDualCfg(
            v_min=v_min,
            v_max=v_max,
            rho_v=1.0,
        ),
        batch_set=batch_set,
        batch_init=batch_init,
        rho_l=1.0,
        dt_s=dt_ctrl,
        estimate_H_every=3600,
        estimate_H_dp_kw=100.0,
    )

    print("Running simulation...")
    coord = Coordinator(
        datacenter=dc,
        grid=grid,
        controllers=[tap_ctrl, ofo_ctrl],
        T_total_s=t_total_s,
        dc_bus=dc_bus,
    )
    log = coord.run()
    print(f"Simulation complete: {len(log.grid_states)} grid steps, {len(log.dc_states)} DC steps")

    stats = compute_allbus_voltage_stats(log.grid_states, v_min=v_min, v_max=v_max)
    print("\n=== Voltage Statistics (all-bus) ===")
    print(f"  voltage_violation_time = {stats.violation_time_s / 60:.2f} min")
    print(f"  worst_vmin             = {stats.worst_vmin:.6f}")
    print(f"  worst_vmax             = {stats.worst_vmax:.6f}")
    print(f"  integral_violation     = {stats.integral_violation_pu_s:.5f} pu·s")

    print("\n=== Batch Schedule Summary ===")
    for label, batches in log.batch_log_by_model.items():
        if batches:
            avg = np.mean(batches)
            changes = sum(1 for i in range(1, len(batches)) if batches[i] != batches[i - 1])
            print(f"  {label}: avg_batch={avg:.1f}, changes={changes}")

    time_s = np.array(log.time_s)
    kW_A = np.array(log.kW_A)
    kW_B = np.array(log.kW_B)
    kW_C = np.array(log.kW_C)

    plot_power_3ph(
        time_s,
        kW_A,
        kW_B,
        kW_C,
        save_path=save_dir / "dc_power_3ph.png",
        title="DC Power by Phase (OFO)",
    )
    if log.batch_log_by_model:
        plot_batch_schedule(
            log.batch_log_by_model,
            dt_ctrl,
            save_path=save_dir / "batch_schedule.png",
            title="Batch Size Schedule (OFO)",
        )
    plot_allbus_voltages_per_phase(
        log.grid_states,
        time_s,
        save_dir=save_dir,
        v_min=v_min,
        v_max=v_max,
        title_template="All-Bus Voltages — Phase {label} (OFO)",
    )

    with open(save_dir / "console_output.txt", "w") as f:
        f.write("=== Voltage Statistics (all-bus) ===\n")
        f.write(f"  voltage_violation_time = {stats.violation_time_s / 60:.2f} min\n")
        f.write(f"  worst_vmin             = {stats.worst_vmin:.6f}\n")
        f.write(f"  worst_vmax             = {stats.worst_vmax:.6f}\n")
        f.write(f"  integral_violation     = {stats.integral_violation_pu_s:.5f} pu·s\n")
        f.write("\n=== Batch Schedule Summary ===\n")
        for label, batches in log.batch_log_by_model.items():
            if batches:
                avg = np.mean(batches)
                changes = sum(1 for i in range(1, len(batches)) if batches[i] != batches[i - 1])
                f.write(f"  {label}: avg_batch={avg:.1f}, changes={changes}\n")

    print(f"\nOutputs saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OFO closed-loop simulation")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Toolkit-generated data directory (contains traces/, latency_fits.csv, "
        "logistic_fits.csv). Defaults to power_csvs_updated/ for legacy mode.",
    )
    parser.add_argument(
        "--training-trace",
        default=None,
        help="Path to synthetic training trace CSV. Required when using "
        "--data-dir; defaults to power_csvs_updated/synthetic_training_trace.csv.",
    )
    args = parser.parse_args()
    main(args)
