"""Baseline simulation using the openg2g library.

Reproduces the results of baseline_wo_control.py using the modular library
components: OfflineDatacenter + OpenDSSGrid + TapScheduleController + Coordinator.

Two modes correspond to two baselines in the G2G paper:
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
from openg2g.datacenter.config import DatacenterConfig, ServerRamp, TrainingRun, WorkloadConfig
from openg2g.datacenter.offline import (
    OfflineDatacenter,
    PowerTraceStore,
)
from openg2g.datacenter.training_overlay import TrainingTrace
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.voltage import compute_allbus_voltage_stats
from openg2g.models.spec import LLMInferenceModelSpec, LLMInferenceWorkload
from openg2g.types import TapPosition, TapSchedule

logger = logging.getLogger("run_baseline")

TAP_STEP = 0.00625
INITIAL_TAPS = TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP)
TAP_CHANGE_SCHEDULE = TapPosition(a=1.0 + 16 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 17 * TAP_STEP).at(
    t=25 * 60
) | TapPosition(a=1.0 + 10 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 10 * TAP_STEP).at(t=55 * 60)

MODELS = (
    LLMInferenceModelSpec(
        "Llama-3.1-8B", num_replicas=720, gpus_per_replica=1, initial_batch_size=128, itl_deadline_s=0.08
    ),
    LLMInferenceModelSpec(
        "Llama-3.1-70B", num_replicas=180, gpus_per_replica=4, initial_batch_size=128, itl_deadline_s=0.10
    ),
    LLMInferenceModelSpec(
        "Llama-3.1-405B", num_replicas=90, gpus_per_replica=8, initial_batch_size=128, itl_deadline_s=0.12
    ),
    LLMInferenceModelSpec(
        "Qwen3-30B-A3B", num_replicas=480, gpus_per_replica=2, initial_batch_size=128, itl_deadline_s=0.06
    ),
    LLMInferenceModelSpec(
        "Qwen3-235B-A22B", num_replicas=210, gpus_per_replica=8, initial_batch_size=128, itl_deadline_s=0.14
    ),
)

INFERENCE = LLMInferenceWorkload(models=MODELS)


def main(args: argparse.Namespace) -> None:
    mode = args.mode

    project_dir = Path(__file__).resolve().parent.parent.parent
    case_dir = Path(__file__).resolve().parent.parent / "ieee13"
    save_dir = project_dir / "outputs" / f"baseline_{mode}"
    save_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(save_dir / "console_output.txt", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    data_dir = Path(args.data_dir)
    training_trace = TrainingTrace.load(Path(args.training_trace))

    v_min = 0.95
    v_max = 1.05
    dc_bus = "671"
    gpus_per_server = 8
    dt_dc = Fraction(1, 10)
    t_total_s = 3600

    logger.info("Loading power traces...")
    store = PowerTraceStore.load(data_dir / "traces_summary.csv")
    store.build_templates(duration_s=600.0, timestep_s=dt_dc)

    logger.info("Loading latency fits...")
    itl_fits = load_itl_fits_from_csv(data_dir / "latency_fits.csv")

    dc_config = DatacenterConfig(gpus_per_server=gpus_per_server, base_kw_per_phase=500.0)
    workload = WorkloadConfig(
        inference=INFERENCE,
        training=TrainingRun(
            t_start=1000.0,
            t_end=2000.0,
            n_gpus=300 * gpus_per_server,
            trace=training_trace,
            target_peak_W_per_gpu=400.0,
        ),
        server_ramps=ServerRamp(t_start=2500.0, t_end=3000.0, target=0.2),
    )

    logger.info("Initializing OfflineDatacenter...")
    dc = OfflineDatacenter.from_config(
        dc_config,
        workload,
        trace_store=store,
        timestep_s=dt_dc,
        seed=0,
        amplitude_scale_range=(0.98, 1.02),
        noise_fraction=0.005,
        itl_distributions=itl_fits,
        latency_exact_threshold=30,
        latency_seed=0,
    )

    logger.info("Initializing OpenDSSGrid...")
    grid = OpenDSSGrid(
        dss_case_dir=str(case_dir),
        dss_master_file="IEEE13Nodeckt.dss",
        dc_bus=dc_bus,
        dc_bus_kv=4.16,
        power_factor=0.95,
        dt_s=Fraction(1, 10),
        connection_type="wye",
        initial_tap_position=INITIAL_TAPS,
    )

    tap_ctrl_schedule = TAP_CHANGE_SCHEDULE if mode == "tap-change" else TapSchedule(())
    ctrl = TapScheduleController(schedule=tap_ctrl_schedule, dt_s=Fraction(1))

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

    plot_power_3ph(
        time_s,
        kW_A,
        kW_B,
        kW_C,
        save_path=save_dir / "dc_power_3ph.png",
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
        required=True,
        help="Toolkit-generated data directory (contains traces/, traces_summary.csv, latency_fits.csv).",
    )
    parser.add_argument(
        "--training-trace",
        required=True,
        help="Path to synthetic training trace CSV.",
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
