"""Baseline (no OFO control) simulation using the openg2g library.

Reproduces the results of baseline_wo_control.py using the modular library
components: OfflineDatacenter + OpenDSSGrid + TapScheduleController + Coordinator.

Two modes correspond to two baselines in the paper:
  no-tap       "No control, no tap": tap positions are fixed throughout.
  tap-change   "Tap change only": regulator taps change at t=1500s and t=3300s.
"""

from __future__ import annotations

import argparse
import logging
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

logger = logging.getLogger("run_baseline")

TAP_STEP = 0.00625
TAP_SCHEDULES = {
    "no-tap": TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP).at(t=0),
    "tap-change": (
        TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP).at(t=0)
        | TapPosition(a=1.0 + 16 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 17 * TAP_STEP).at(t=25 * 60)
        | TapPosition(a=1.0 + 10 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 10 * TAP_STEP).at(t=55 * 60)
    ),
}

MODELS = (
    LLMInferenceModelSpec("Llama-3.1-8B", num_replicas=720, gpus_per_replica=1, initial_batch_size=128),
    LLMInferenceModelSpec("Llama-3.1-70B", num_replicas=180, gpus_per_replica=4, initial_batch_size=128),
    LLMInferenceModelSpec("Llama-3.1-405B", num_replicas=90, gpus_per_replica=8, initial_batch_size=128),
    LLMInferenceModelSpec("Qwen3-30B-A3B", num_replicas=480, gpus_per_replica=2, initial_batch_size=128),
    LLMInferenceModelSpec("Qwen3-235B-A22B", num_replicas=210, gpus_per_replica=8, initial_batch_size=128),
)

INFERENCE = LLMInferenceWorkload(models=MODELS)


def main(args: argparse.Namespace) -> None:
    mode = args.mode
    tap_schedule = TAP_SCHEDULES[mode]

    project_dir = Path(__file__).resolve().parent.parent
    case_dir = Path(__file__).resolve().parent / "ieee13"
    save_dir = project_dir / "outputs" / f"baseline_{mode}"
    save_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(save_dir / "console_output.txt", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    if args.data_dir:
        data_dir = Path(args.data_dir)
        trace_dir = data_dir / "traces"
    else:
        data_dir = None
        trace_dir = project_dir / "power_csvs_updated"

    if args.training_trace:
        training_csv = Path(args.training_trace)
    elif args.data_dir:
        raise ValueError(
            "Error: --training-trace is required when using --data-dir (training trace is not part of the build output)"
        )
    else:
        training_csv = project_dir / "power_csvs_updated" / "synthetic_training_trace.csv"

    v_min = 0.95
    v_max = 1.05
    dc_bus = "671"
    gpus_per_server = 8
    dt_dc = Fraction(1, 10)
    t_total_s = 3600

    logger.info("Loading power traces...")
    traces_by_batch = load_traces_by_batch_from_dir(
        base_dir=trace_dir,
        batch_set=[128],
        required_measured_gpus=INFERENCE.required_measured_gpus,
        amp_jitter_default=(0.98, 1.02),
        noise_std_frac_default=0.005,
    )

    cache = TraceByBatchCache(traces_by_batch)
    cache.build_templates(duration_s=t_total_s, timestep_s=dt_dc)

    logger.info("Loading latency fits...")
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

    logger.info("Initializing OfflineDatacenter...")
    dc = OfflineDatacenter.from_config(
        dc_config,
        workload,
        trace_cache=cache,
        timestep_s=dt_dc,
        seed=0,
        chunk_steps=int(t_total_s / dt_dc),
        itl_distributions=itl_fits,
        latency_exact_threshold=30,
        latency_seed=0,
    )

    logger.info("Initializing OpenDSSGrid...")
    grid = OpenDSSGrid(
        case_dir=str(case_dir),
        master="IEEE13Nodeckt.dss",
        dc_bus=dc_bus,
        dc_bus_kv=4.16,
        power_factor=0.95,
        dt_s=Fraction(1, 10),
        connection_type="wye",
        controls_off=False,
        tap_schedule=tap_schedule,
        freeze_regcontrols=True,
    )

    ctrl = TapScheduleController(schedule=[], dt_s=Fraction(1))

    logger.info("Running simulation (mode=%s)...", mode)
    coord = Coordinator(
        datacenter=dc,
        grid=grid,
        controllers=[ctrl],
        total_duration_s=t_total_s,
        dc_bus=dc_bus,
    )
    log = coord.run()

    stats = compute_allbus_voltage_stats(log.grid_states, v_min=v_min, v_max=v_max)
    logger.info("=== Voltage Statistics (all-bus) ===")
    logger.info("  voltage_violation_time = %.1f s", stats.violation_time_s)
    logger.info("  worst_vmin             = %.6f", stats.worst_vmin)
    logger.info("  worst_vmax             = %.6f", stats.worst_vmax)
    logger.info("  integral_violation     = %.4f pu·s", stats.integral_violation_pu_s)

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

    logger.info("Outputs saved to: %s", save_dir)


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
        help="Toolkit-generated data directory (contains traces/). Defaults to power_csvs_updated/ for legacy mode.",
    )
    parser.add_argument(
        "--training-trace",
        default=None,
        help="Path to synthetic training trace CSV. Required when using "
        "--data-dir; defaults to power_csvs_updated/synthetic_training_trace.csv.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )

    main(args)
