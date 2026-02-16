"""OFO closed-loop simulation using the openg2g library.

Reproduces the results of final_ofo_test.py using the modular library
components: OfflineDatacenter + OpenDSSGrid + [TapScheduleController,
OFOBatchController] + Coordinator.
"""

from __future__ import annotations

import argparse
import logging
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from mlenergy_data.modeling import LogisticModel
from plotting import (
    extract_per_model_timeseries,
    load_itl_fits_from_csv,
    plot_allbus_voltages_per_phase,
    plot_batch_schedule,
    plot_latency_samples,
    plot_model_timeseries_4panel,
    plot_per_model_power,
    plot_power_3ph,
    plot_voltage_dc_bus,
)

from openg2g.controller.ofo import (
    OFOBatchController,
    PrimalConfig,
    VoltageDualConfig,
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

logger = logging.getLogger(__name__)

BATCH_SET = (8, 16, 32, 64, 128, 256, 512)

MODELS = (
    LLMInferenceModelSpec(
        "Llama-3.1-8B",
        num_replicas=720,
        gpus_per_replica=1,
        feasible_batch_sizes=BATCH_SET,
        initial_batch_size=128,
        itl_deadline_s=0.08,
    ),
    LLMInferenceModelSpec(
        "Llama-3.1-70B",
        num_replicas=180,
        gpus_per_replica=4,
        feasible_batch_sizes=BATCH_SET,
        initial_batch_size=128,
        itl_deadline_s=0.10,
    ),
    LLMInferenceModelSpec(
        "Llama-3.1-405B",
        num_replicas=90,
        gpus_per_replica=8,
        feasible_batch_sizes=BATCH_SET,
        initial_batch_size=128,
        itl_deadline_s=0.12,
    ),
    LLMInferenceModelSpec(
        "Qwen3-30B-A3B",
        num_replicas=480,
        gpus_per_replica=2,
        feasible_batch_sizes=BATCH_SET,
        initial_batch_size=128,
        itl_deadline_s=0.06,
    ),
    LLMInferenceModelSpec(
        "Qwen3-235B-A22B",
        num_replicas=210,
        gpus_per_replica=8,
        feasible_batch_sizes=BATCH_SET,
        initial_batch_size=128,
        itl_deadline_s=0.14,
    ),
)

INFERENCE = LLMInferenceWorkload(models=MODELS)


def _load_logistic_fits_from_csvs(
    csv_by_model: dict[str, Path],
) -> tuple[dict[str, LogisticModel], dict[str, LogisticModel], dict[str, LogisticModel]]:
    """Load per-model logistic fit CSVs (columns: metric, L, x0, k, b0).

    Legacy format: each model has its own CSV with rows for power/latency/throughput.
    Subject to removal when legacy power_csvs_updated/ data is retired.
    """
    power: dict[str, LogisticModel] = {}
    latency: dict[str, LogisticModel] = {}
    throughput: dict[str, LogisticModel] = {}
    targets = {"power": power, "latency": latency, "throughput": throughput}
    for label, csv_path in csv_by_model.items():
        df = pd.read_csv(csv_path)
        for row in df.to_dict(orient="records"):
            metric = str(row["metric"]).strip().lower()
            if metric in targets:
                targets[metric][label] = LogisticModel.from_dict(row)
    return power, latency, throughput


def _load_logistic_fits_merged(
    csv_path: Path,
) -> tuple[dict[str, LogisticModel], dict[str, LogisticModel], dict[str, LogisticModel]]:
    """Load power, latency, throughput fits from a merged CSV."""
    df = pd.read_csv(csv_path)
    power: dict[str, LogisticModel] = {}
    latency: dict[str, LogisticModel] = {}
    throughput: dict[str, LogisticModel] = {}
    targets = {"power": power, "latency": latency, "throughput": throughput}
    for row in df.to_dict(orient="records"):
        metric = str(row["metric"]).strip().lower()
        if metric in targets:
            targets[metric][str(row["model_label"])] = LogisticModel.from_dict(row)
    return power, latency, throughput


def main(args: argparse.Namespace) -> None:
    project_dir = Path(__file__).resolve().parent.parent
    case_dir = Path(__file__).resolve().parent / "ieee13"
    save_dir = project_dir / "outputs" / "ofo"
    save_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(save_dir / "console_output.txt", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    v_min = 0.95
    v_max = 1.05
    dc_bus = "671"
    gpus_per_server = 8
    dt_dc = Fraction(1, 10)
    dt_ctrl = Fraction(1)
    t_total_s = 3600

    TAP_STEP = 0.00625  # standard 5/8% tap step
    tap_schedule = TapPosition(a=1.0 + 14 * TAP_STEP, b=1.0 + 6 * TAP_STEP, c=1.0 + 15 * TAP_STEP).at(t=0)

    data_dir = Path(args.data_dir) if args.data_dir else None
    if data_dir is not None:
        trace_dir = data_dir / "traces"
    else:
        trace_dir = project_dir / "power_csvs_updated"

    if args.training_trace:
        training_csv = Path(args.training_trace)
    elif data_dir is not None:
        raise SystemExit(
            "Error: --training-trace is required when using --data-dir (training trace is not part of the build output)"
        )
    else:
        training_csv = project_dir / "power_csvs_updated" / "synthetic_training_trace.csv"

    logger.info("Loading power traces...")
    traces_by_batch = load_traces_by_batch_from_dir(
        base_dir=trace_dir,
        batch_set=list(BATCH_SET),
        required_measured_gpus=INFERENCE.required_measured_gpus,
        amp_jitter_default=(0.98, 1.02),
        noise_std_frac_default=0.005,
    )

    cache = TraceByBatchCache(traces_by_batch)
    cache.build_templates(duration_s=600.0, timestep_s=dt_dc)

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
        chunk_steps=int(dt_ctrl / dt_dc),
        itl_distributions=itl_fits,
        latency_exact_threshold=30,
        latency_seed=0,
    )

    logger.info("Loading logistic fits...")
    if data_dir is not None:
        power_fits, latency_fits, throughput_fits = _load_logistic_fits_merged(data_dir / "logistic_fits.csv")
    else:
        power_fits, latency_fits, throughput_fits = _load_logistic_fits_from_csvs(
            {
                label: trace_dir / f"{label}_logistic_fit_parameters_combined.csv"
                for label in INFERENCE.required_measured_gpus
            }
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
        controls_off=True,
        tap_schedule=tap_schedule,
        freeze_regcontrols=True,
    )

    tap_ctrl = TapScheduleController(schedule=[], dt_s=dt_ctrl)

    ofo_ctrl = OFOBatchController.from_workload(
        workload=INFERENCE,
        power_fits=power_fits,
        latency_fits=latency_fits,
        throughput_fits=throughput_fits,
        primal_config=PrimalConfig(
            descent_step_size=0.1,
            w_latency=1.0,
            w_throughput=1e-3,
            w_switch=1.0,
            voltage_gradient_scale=1e6,
        ),
        voltage_dual_config=VoltageDualConfig(
            v_min=v_min,
            v_max=v_max,
            ascent_step_size=1.0,
        ),
        latency_dual_step_size=1.0,
        dt_s=dt_ctrl,
        sensitivity_update_interval=3600,
        sensitivity_perturbation_kw=100.0,
    )

    logger.info("Running simulation...")
    coord = Coordinator(
        datacenter=dc,
        grid=grid,
        controllers=[tap_ctrl, ofo_ctrl],
        total_duration_s=t_total_s,
        dc_bus=dc_bus,
    )
    log = coord.run()

    stats = compute_allbus_voltage_stats(log.grid_states, v_min=v_min, v_max=v_max)
    logger.info("=== Voltage Statistics (all-bus) ===")
    logger.info("  voltage_violation_time = %.2f min", stats.violation_time_s / 60)
    logger.info("  worst_vmin             = %.6f", stats.worst_vmin)
    logger.info("  worst_vmax             = %.6f", stats.worst_vmax)
    logger.info("  integral_violation     = %.5f pu·s", stats.integral_violation_pu_s)

    logger.info("=== Batch Schedule Summary ===")
    for label, batches in log.batch_log_by_model.items():
        if batches:
            avg = np.mean(batches)
            changes = sum(1 for i in range(1, len(batches)) if batches[i] != batches[i - 1])
            logger.info("  %s: avg_batch=%.1f, changes=%d", label, avg, changes)

    time_s = np.array(log.time_s)
    kW_A = np.array(log.kW_A)
    kW_B = np.array(log.kW_B)
    kW_C = np.array(log.kW_C)

    per_model = extract_per_model_timeseries(log.dc_states)

    # Fig. 8: 4-panel model timeseries
    plot_model_timeseries_4panel(
        per_model.time_s,
        log.batch_log_by_model,
        per_model,
        model_labels=INFERENCE.model_labels,
        dt_ctrl_s=float(dt_ctrl),
        save_path=save_dir / "model_timeseries_4panel.png",
    )

    # Fig. 7: All-bus voltages with bus colormap
    plot_allbus_voltages_per_phase(
        log.grid_states,
        time_s,
        save_dir=save_dir,
        v_min=v_min,
        v_max=v_max,
        title_template="Voltage trajectories with GPU flexibility (Phase {label})",
    )

    # Standalone plots
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
            float(dt_ctrl),
            save_path=save_dir / "batch_schedule.png",
            title="Batch Size Schedule (OFO)",
        )

    plot_voltage_dc_bus(
        time_s,
        np.array(log.voltage_a_pu),
        np.array(log.voltage_b_pu),
        np.array(log.voltage_c_pu),
        v_min=v_min,
        v_max=v_max,
        save_path=save_dir / "voltage_dc_bus.png",
    )

    plot_latency_samples(
        per_model,
        itl_deadlines=INFERENCE.itl_deadline_by_model,
        save_path=save_dir / "latency_samples.png",
    )

    if per_model.power_w:
        plot_per_model_power(
            per_model,
            save_path=save_dir / "per_model_power.png",
        )

    logger.info("Outputs saved to: %s", save_dir)


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
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    main(args)
