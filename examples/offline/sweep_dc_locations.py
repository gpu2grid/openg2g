"""Unified DC location sweep: 1-D, 2-D, or zone-constrained.

Automatically selects sweep mode based on the system configuration:
  - 1 DC site, no zones  -> 1-D sweep: sweep single DC across candidate buses,
                            with tap optimization and 4-case comparison
  - 2+ DC sites, no zones -> 2-D sweep: sweep all unordered pairs of candidate
                            buses, OFO control for each site, heatmap output
  - N DC sites with zones -> zone-constrained sweep (3 phases):
      Phase 1 (Screening):  Sweep each zone independently while holding other
                            zones at default buses.  Keep top-K per zone.
                            Uses coarse dt (default 60 s) over the full
                            simulation duration (e.g. 3600 s) for fast ranking.
      Phase 2 (Combination): Cartesian product of top-K per zone.  Uses native
                            config resolution (e.g. dt=0.1 s) but only 60 s
                            duration as a stress test with full DC capacity.
      Phase 3 (Refinement):  Optional (--refine).  Iteratively re-sweep each
                            zone from the Phase 2 winner.  Uses native config
                            resolution over the full simulation duration.

Usage:
    python sweep_dc_locations.py --system ieee13
    python sweep_dc_locations.py --system ieee34
    python sweep_dc_locations.py --system ieee123
    python sweep_dc_locations.py --system ieee123 --top-k 6 --refine
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import logging
import math
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.use("Agg")

from utils import discover_candidate_buses, extract_all_voltages, find_violations

from openg2g.controller.ofo import (
    LogisticModelStore,
    OFOBatchSizeController,
    OFOConfig,
)
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import Coordinator
from openg2g.datacenter.config import (
    DatacenterConfig,
    InferenceModelSpec,
    ModelDeployment,
    PowerAugmentationConfig,
    ReplicaSchedule,
    TrainingRun,
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData, MLEnergySource
from openg2g.datacenter.workloads.training import TrainingTrace
from openg2g.grid.config import TapPosition, TapSchedule
from openg2g.grid.generator import ConstantGenerator, SyntheticPV
from openg2g.grid.load import SyntheticLoad
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.performance import compute_performance_stats
from openg2g.metrics.voltage import compute_allbus_voltage_stats

from plotting import plot_allbus_voltages_per_phase, plot_model_timeseries_4panel
from systems import SYSTEMS, TAP_STEP, tap

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DT_DC = Fraction(1, 10)
DT_GRID = Fraction(1, 10)
DT_CTRL = Fraction(1)
V_MIN, V_MAX = 0.95, 1.05
TOTAL_DURATION_S = 3600
POWER_AUG = PowerAugmentationConfig(amplitude_scale_range=(0.98, 1.02), noise_fraction=0.005)

LLAMA_8B = InferenceModelSpec(
    model_label="Llama-3.1-8B",
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    gpus_per_replica=1,
    itl_deadline_s=0.08,
    feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
)
LLAMA_70B = InferenceModelSpec(
    model_label="Llama-3.1-70B",
    model_id="meta-llama/Llama-3.1-70B-Instruct",
    gpus_per_replica=4,
    itl_deadline_s=0.10,
    feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
)
LLAMA_405B = InferenceModelSpec(
    model_label="Llama-3.1-405B",
    model_id="meta-llama/Llama-3.1-405B-Instruct-FP8",
    gpus_per_replica=8,
    itl_deadline_s=0.12,
    feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
)
QWEN_30B = InferenceModelSpec(
    model_label="Qwen3-30B-A3B",
    model_id="Qwen/Qwen3-30B-A3B-Thinking-2507",
    gpus_per_replica=2,
    itl_deadline_s=0.06,
    feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
)
QWEN_235B = InferenceModelSpec(
    model_label="Qwen3-235B-A22B",
    model_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
    gpus_per_replica=8,
    itl_deadline_s=0.14,
    feasible_batch_sizes=[8, 16, 32, 64, 128, 256, 512],
)
ALL_MODEL_SPECS = (LLAMA_8B, LLAMA_70B, LLAMA_405B, QWEN_30B, QWEN_235B)
MODEL_SPECS = {s.model_label: s for s in ALL_MODEL_SPECS}


def deploy(label, num_replicas, initial_batch_size=128):
    """Shorthand: `deploy("Llama-3.1-8B", 720, 128)` -> `(ModelDeployment, ReplicaSchedule)`."""
    return (
        ModelDeployment(spec=MODEL_SPECS[label], initial_batch_size=initial_batch_size),
        ReplicaSchedule(initial=num_replicas),
    )


def load_data_sources(config_path=None):
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "data_sources.json"
    with open(config_path) as f:
        cfg = json.load(f)
    sources_raw = cfg["data_sources"]
    data_sources = {s["model_label"]: MLEnergySource(**s) for s in sources_raw}
    blob = json.dumps(
        sorted(sources_raw, key=lambda s: s["model_label"]),
        sort_keys=True,
    ).encode()
    data_dir = _PROJECT_ROOT / "data" / "offline" / hashlib.sha256(blob).hexdigest()[:16]
    return data_sources, data_dir


@dataclass
class _DCSiteConfig:
    bus: str
    base_kw_per_phase: float
    total_gpu_capacity: int
    models: tuple[tuple[ModelDeployment, ReplicaSchedule], ...] = ()
    seed: int = 0
    connection_type: str = "wye"
    replica_schedules: dict[str, ReplicaSchedule] | None = None


logger = logging.getLogger("sweep_dc_locations")


# 1-D SWEEP (single DC site)


MAX_TAP_ITERATIONS = 20


def optimize_taps_for_bus(
    *,
    sys: dict,
    dc_bus: str,
    base_kw_per_phase: float,
    connection_type: str,
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    training: TrainingRun | None,
    replica_schedules: dict[str, ReplicaSchedule] | None,
    initial_tap_position: TapPosition,
    total_gpu_capacity: int = 0,
    pv_systems: list | None = None,
    time_varying_loads: list | None = None,
    power_augmentation: PowerAugmentationConfig = POWER_AUG,
    t_total_s: int = TOTAL_DURATION_S,
    t_analysis_start: int = 0,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
) -> TapPosition:
    regs = dict(initial_tap_position.regulators)
    reg_names = list(regs.keys())

    dt_coarse = Fraction(60)

    for iteration in range(MAX_TAP_ITERATIONS):
        tap_pos = TapPosition(regulators=dict(regs))
        logger.debug(
            "  Tap opt iter %d: %s",
            iteration,
            ", ".join(f"{k}=%.5f (%+d)" % (regs[k], round((regs[k] - 1.0) / TAP_STEP)) for k in reg_names),
        )

        dc_config = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=base_kw_per_phase)

        workload_kwargs: dict = {"inference_data": inference_data}
        if replica_schedules is not None:
            workload_kwargs["replica_schedules"] = replica_schedules
        if training is not None:
            workload_kwargs["training"] = training
        workload = OfflineWorkload(**workload_kwargs)

        dc = OfflineDatacenter(
            dc_config,
            workload,
            name="dc",
            dt_s=dt_coarse,
            seed=0,
            power_augmentation=power_augmentation,
            total_gpu_capacity=total_gpu_capacity,
        )

        grid = OpenDSSGrid(
            dss_case_dir=sys["dss_case_dir"],
            dss_master_file=sys["dss_master_file"],
            source_pu=sys["source_pu"],
            dt_s=dt_coarse,
            initial_tap_position=tap_pos,
            exclude_buses=sys.get("exclude_buses", []),
        )
        grid.attach_dc(dc, bus=dc_bus, connection_type=connection_type, power_factor=dc_config.power_factor)
        for bus, gen in pv_systems or []:
            grid.attach_generator(gen, bus=bus)
        for bus, ld in time_varying_loads or []:
            grid.attach_load(ld, bus=bus)

        ctrl = TapScheduleController(schedule=TapSchedule(()), dt_s=dt_coarse)
        coord = Coordinator(
            datacenters=[dc],
            grid=grid,
            controllers=[ctrl],
            total_duration_s=t_total_s,
        )
        log = coord.run()

        analysis_states = [gs for gs in log.grid_states if gs.time_s >= t_analysis_start]
        if len(analysis_states) < 2:
            analysis_states = log.grid_states

        voltages = extract_all_voltages(analysis_states)
        violations = find_violations(voltages, v_min=v_min, v_max=v_max)

        if not violations:
            logger.debug("    No violations. Taps optimized.")
            return tap_pos

        phase_adjust = {"a": 0, "b": 0, "c": 0}
        for _bus, phase, vtype, _val, _mag in violations:
            if vtype == "under":
                phase_adjust[phase] = max(phase_adjust[phase], 1)
            elif vtype == "over":
                phase_adjust[phase] = min(phase_adjust[phase], -1)

        has_conflict = False
        for phase in ("a", "b", "c"):
            under = any(v[1] == phase and v[2] == "under" for v in violations)
            over = any(v[1] == phase and v[2] == "over" for v in violations)
            if under and over:
                has_conflict = True
        if has_conflict:
            logger.debug("    Conflicting violations - cannot resolve. Stopping.")
            return tap_pos

        prev_regs = dict(regs)
        for rname in reg_names:
            rname_lower = rname.lower()
            if rname_lower.endswith("a"):
                regs[rname] += phase_adjust["a"] * TAP_STEP
            elif rname_lower.endswith("b"):
                regs[rname] += phase_adjust["b"] * TAP_STEP
            elif rname_lower.endswith("c"):
                regs[rname] += phase_adjust["c"] * TAP_STEP

        for rname in reg_names:
            regs[rname] = max(0.9, min(1.1, regs[rname]))

        if regs == prev_regs:
            logger.debug("    Taps clamped at limits. Stopping.")
            return tap_pos

    return TapPosition(regulators=dict(regs))


def optimize_taps_multiscenario(
    *,
    sys: dict,
    dc_bus: str,
    base_kw_per_phase: float,
    connection_type: str,
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    training: TrainingRun | None,
    replica_schedules: dict[str, ReplicaSchedule] | None,
    initial_tap_position: TapPosition,
    tap_schedule_entries: list[tuple[float, TapPosition]] | None = None,
    training_t_start: float = 0,
    training_t_end: float = 0,
    ramp_t_end: float = 0,
    total_gpu_capacity: int = 0,
    pv_systems: list | None = None,
    time_varying_loads: list | None = None,
    power_augmentation: PowerAugmentationConfig = POWER_AUG,
    total_duration_s: int = TOTAL_DURATION_S,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
) -> dict[str, TapPosition]:
    """Optimize taps for multiple scenarios (inference, training, low_load).

    Parameters
    ----------
    training_t_start, training_t_end : float
        Timing window for the training run (used for scenario splitting).
    ramp_t_end : float
        End time of the inference ramp (used for low_load scenario).
    """

    # Build scenario base taps from tap_schedule_entries
    bases: dict[str, TapPosition] = {"inference": initial_tap_position}
    sorted_entries = sorted(tap_schedule_entries or [], key=lambda e: e[0])

    if len(sorted_entries) >= 1 and training is not None:
        bases["training"] = sorted_entries[0][1]

    if len(sorted_entries) >= 2 and replica_schedules is not None:
        bases["low_load"] = sorted_entries[-1][1]

    COARSE_TICK = 60

    def _round_up(t: int) -> int:
        return ((t + COARSE_TICK - 1) // COARSE_TICK) * COARSE_TICK

    scenarios: list[tuple[str, int, int, TapPosition]] = []

    t_ts = int(training_t_start)
    t_te = int(training_t_end)

    if t_ts > 0:
        scenarios.append(("inference", _round_up(min(t_ts, 180)), 0, bases["inference"]))
    else:
        scenarios.append(("inference", _round_up(180), 0, bases["inference"]))

    if training is not None and "training" in bases:
        scenarios.append(("training", _round_up(t_te + 120), t_ts, bases["training"]))

    if replica_schedules is not None and "low_load" in bases:
        t_re = int(ramp_t_end) if ramp_t_end > 0 else total_duration_s
        scenarios.append(("low_load", _round_up(total_duration_s), t_re, bases["low_load"]))

    results = {}
    for name, t_total, t_start, init_taps in scenarios:
        taps_str = ", ".join(f"{k}=%+d" % round((v - 1.0) / TAP_STEP) for k, v in init_taps.regulators.items())
        logger.info(
            "    Scenario '%s': T=%d, analysis>=%d, base taps: %s",
            name,
            t_total,
            t_start,
            taps_str,
        )
        optimal = optimize_taps_for_bus(
            sys=sys,
            dc_bus=dc_bus,
            base_kw_per_phase=base_kw_per_phase,
            connection_type=connection_type,
            inference_data=inference_data,
            training_trace=training_trace,
            training=training,
            replica_schedules=replica_schedules,
            initial_tap_position=init_taps,
            total_gpu_capacity=total_gpu_capacity,
            pv_systems=pv_systems,
            time_varying_loads=time_varying_loads,
            power_augmentation=power_augmentation,
            t_total_s=t_total,
            t_analysis_start=t_start,
            v_min=v_min,
            v_max=v_max,
        )
        results[name] = optimal
        opt_str = ", ".join(f"{k}=%+d" % round((v - 1.0) / TAP_STEP) for k, v in optimal.regulators.items())
        logger.info("    -> %s taps: %s", name, opt_str)
    return results


def run_case_1d(
    *,
    sys: dict,
    dc_bus: str,
    base_kw_per_phase: float,
    case_name: str,
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
    all_models: tuple[InferenceModelSpec, ...],
    initial_taps: TapPosition,
    tap_schedule: TapSchedule | None,
    use_ofo: bool,
    ofo_config: OFOConfig | None = None,
    training: TrainingRun | None = None,
    replica_schedules: dict[str, ReplicaSchedule] | None = None,
    total_gpu_capacity: int = 0,
    pv_systems: list | None = None,
    time_varying_loads: list | None = None,
    total_duration_s: int = TOTAL_DURATION_S,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    power_augmentation: PowerAugmentationConfig = POWER_AUG,
    connection_type: str = "wye",
    dt_dc: Fraction = DT_DC,
    dt_grid: Fraction = DT_GRID,
    dt_ctrl: Fraction = DT_CTRL,
    save_dir: Path | None = None,
) -> dict:
    """Run a single 1-D simulation case and return metrics dict."""
    dc_config = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=base_kw_per_phase)

    workload_kwargs: dict = {"inference_data": inference_data}
    if replica_schedules is not None:
        workload_kwargs["replica_schedules"] = replica_schedules
    if training is not None:
        workload_kwargs["training"] = training
    workload = OfflineWorkload(**workload_kwargs)

    dc = OfflineDatacenter(
        dc_config,
        workload,
        name="dc",
        dt_s=dt_dc,
        seed=0,
        power_augmentation=power_augmentation,
        total_gpu_capacity=total_gpu_capacity,
    )

    grid = OpenDSSGrid(
        dss_case_dir=sys["dss_case_dir"],
        dss_master_file=sys["dss_master_file"],
        source_pu=sys["source_pu"],
        dt_s=dt_grid,
        initial_tap_position=initial_taps,
        exclude_buses=sys.get("exclude_buses", []),
    )
    grid.attach_dc(dc, bus=dc_bus, connection_type=connection_type, power_factor=dc_config.power_factor)
    for bus, gen in pv_systems or []:
        grid.attach_generator(gen, bus=bus)
    for bus, ld in time_varying_loads or []:
        grid.attach_load(ld, bus=bus)

    controllers: list = []
    schedule = tap_schedule if tap_schedule else TapSchedule(())
    tap_ctrl = TapScheduleController(schedule=schedule, dt_s=dt_ctrl)
    controllers.append(tap_ctrl)

    if use_ofo and ofo_config is not None:
        ofo_ctrl = OFOBatchSizeController(
            all_models,
            datacenter=dc,
            models=logistic_models,
            config=ofo_config,
            dt_s=dt_ctrl,
            grid=grid,
        )
        controllers.append(ofo_ctrl)

    coord = Coordinator(
        datacenters=[dc],
        grid=grid,
        controllers=controllers,
        total_duration_s=total_duration_s,
    )
    log = coord.run()

    stats = compute_allbus_voltage_stats(
        log.grid_states, v_min=v_min, v_max=v_max, exclude_buses=tuple(sys.get("exclude_buses", ()))
    )
    pstats = compute_performance_stats(
        log.dc_states, itl_deadline_s_by_model={ms.model_label: ms.itl_deadline_s for ms in all_models}
    )

    dc_kW = np.array([(s.power_w.a + s.power_w.b + s.power_w.c) / 3e3 for s in log.dc_states])
    avg_power = float(dc_kW.mean()) if dc_kW.size > 0 else 0.0

    result = {
        "violation_time_s": stats.violation_time_s,
        "worst_vmin": stats.worst_vmin,
        "worst_vmax": stats.worst_vmax,
        "integral_violation_pu_s": stats.integral_violation_pu_s,
        "avg_power_kw_per_phase": avg_power,
        "mean_throughput_tps": pstats.mean_throughput_tps,
        "itl_deadline_fraction": pstats.itl_deadline_fraction,
    }

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        time_s = np.array(log.time_s)
        plot_allbus_voltages_per_phase(
            log.grid_states,
            time_s,
            save_dir=save_dir,
            v_min=v_min,
            v_max=v_max,
            title_template=f"DC@{dc_bus} {case_name} -- Voltage (Phase {{label}})",
        )
        if use_ofo:
            plot_model_timeseries_4panel(
                log.dc_states,
                model_labels=[m.model_label for m in all_models],
                regime_shading=False,
                save_path=save_dir / "OFO_results.png",
            )

    return result


def _plot_bus_comparison(all_rows: list[dict], save_path: Path) -> None:
    buses = [r["dc_bus"] for r in all_rows]
    cases = [
        ("baseline_no_tap", "Baseline (no tap)"),
        ("baseline_tap_change", "Baseline (tap change)"),
        ("ofo_no_tap", "OFO (no tap)"),
        ("ofo_tap_change", "OFO (tap change)"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(buses))
    width = 0.18

    for i, (prefix, label) in enumerate(cases):
        viol = [r.get(f"{prefix}_violation_time_s", float("nan")) for r in all_rows]
        integ = [r.get(f"{prefix}_integral_violation_pu_s", float("nan")) for r in all_rows]
        offset = (i - 1.5) * width
        ax1.bar(x + offset, viol, width, label=label)
        ax2.bar(x + offset, integ, width, label=label)

    ax1.set_xlabel("DC Bus")
    ax1.set_ylabel("Violation Time (s)")
    ax1.set_title("Voltage Violation Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(buses)
    ax1.legend(fontsize=8)

    ax2.set_xlabel("DC Bus")
    ax2.set_ylabel("Integral Violation (pu·s)")
    ax2.set_title("Integral Voltage Violation")
    ax2.set_xticks(x)
    ax2.set_xticklabels(buses)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison plot saved to: %s", save_path)


def main_1d(
    *,
    sys: dict,
    system: str,
    dc_site: _DCSiteConfig,
    all_models: tuple[InferenceModelSpec, ...],
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
    ofo_config: OFOConfig,
    training: TrainingRun | None = None,
    replica_schedules: dict[str, ReplicaSchedule] | None = None,
    tap_schedule_entries: list[tuple[float, TapPosition]] | None = None,
    training_t_start: float = 0,
    training_t_end: float = 0,
    ramp_t_end: float = 0,
    pv_systems: list | None = None,
    time_varying_loads: list | None = None,
    power_augmentation: PowerAugmentationConfig = POWER_AUG,
    total_duration_s: int = TOTAL_DURATION_S,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    buses: list[str] | None = None,
    dt_override: str | None = None,
    output_dir: Path | None = None,
) -> None:
    """1-D sweep: single DC site swept across candidate buses."""
    if replica_schedules is None:
        replica_schedules = {md.spec.model_label: s for md, s in dc_site.models}
    site_gpu_capacity = dc_site.total_gpu_capacity

    dt_dc = DT_DC
    dt_grid = DT_GRID
    dt_ctrl = DT_CTRL

    if dt_override is not None:
        frac = Fraction(dt_override) if "/" in dt_override else Fraction(int(dt_override))
        dt_dc = dt_grid = dt_ctrl = frac
        logger.info("Time resolution override: dt = %s s for all components", dt_override)

    save_dir = (
        output_dir if output_dir else (Path(__file__).resolve().parent / "outputs" / system / "sweep_dc_locations_1d")
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    # Discover candidate buses
    if buses:
        candidate_buses = buses
        logger.info("Using user-specified buses: %s", candidate_buses)
    else:
        logger.info("Discovering candidate 3-phase buses at %.2f kV...", sys["bus_kv"])
        candidate_buses = discover_candidate_buses(
            sys["dss_case_dir"],
            sys["dss_master_file"],
            sys["bus_kv"],
            exclude=set(sys["exclude_buses"]),
        )
        logger.info("Found %d candidate buses: %s", len(candidate_buses), candidate_buses)

    if not candidate_buses:
        logger.error("No candidate buses found!")
        return

    initial_taps = sys["initial_taps"]

    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 1: Tap optimization for %d buses", len(candidate_buses))
    logger.info("=" * 70)

    optimal_taps: dict[str, dict[str, TapPosition]] = {}

    for dc_bus in candidate_buses:
        logger.info("")
        logger.info("  Bus %s: optimizing taps...", dc_bus)
        try:
            taps = optimize_taps_multiscenario(
                sys=sys,
                dc_bus=dc_bus,
                base_kw_per_phase=dc_site.base_kw_per_phase,
                connection_type=dc_site.connection_type,
                inference_data=inference_data,
                training_trace=training_trace,
                training=training,
                replica_schedules=replica_schedules,
                initial_tap_position=initial_taps,
                tap_schedule_entries=tap_schedule_entries,
                training_t_start=training_t_start,
                training_t_end=training_t_end,
                ramp_t_end=ramp_t_end,
                total_gpu_capacity=site_gpu_capacity,
                pv_systems=pv_systems,
                time_varying_loads=time_varying_loads,
                power_augmentation=power_augmentation,
                total_duration_s=total_duration_s,
                v_min=v_min,
                v_max=v_max,
            )
            optimal_taps[dc_bus] = taps
        except Exception as e:
            logger.error("  Bus %s: tap optimization FAILED: %s", dc_bus, e)
            optimal_taps[dc_bus] = {"inference": initial_taps}

    # Save tap optimization results
    tap_csv_path = save_dir / "optimal_taps.csv"
    with open(tap_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        all_reg_names: list[str] = []
        for scenarios in optimal_taps.values():
            for tp in scenarios.values():
                for rn in tp.regulators:
                    if rn not in all_reg_names:
                        all_reg_names.append(rn)
        header = ["dc_bus", "scenario"]
        for rn in all_reg_names:
            header.extend([f"tap_{rn}", f"step_{rn}"])
        writer.writerow(header)
        for dc_bus, scenarios in optimal_taps.items():
            for scenario, tp in scenarios.items():
                row = [dc_bus, scenario]
                for rn in all_reg_names:
                    val = tp.regulators.get(rn, 1.0)
                    row.extend([f"{val:.5f}", round((val - 1.0) / TAP_STEP)])
                writer.writerow(row)
    logger.info("Tap optimization results saved to: %s", tap_csv_path)

    # Phase 2: Run 4 comparison cases per bus
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 2: Running 4 comparison cases for %d buses", len(candidate_buses))
    logger.info("=" * 70)

    all_rows = []
    total_cases = len(candidate_buses) * 4
    case_idx = 0

    # Build tap schedule from optimized per-bus taps
    for dc_bus in candidate_buses:
        bus_taps = optimal_taps.get(dc_bus, {})
        inference_taps = bus_taps.get("inference", initial_taps)

        bus_tap_schedule_entries = []
        if "training" in bus_taps and training is not None:
            bus_tap_schedule_entries.append((training_t_start, bus_taps["training"]))
        if "low_load" in bus_taps and replica_schedules is not None:
            t_low = ramp_t_end + 120 if ramp_t_end > 0 else total_duration_s
            bus_tap_schedule_entries.append((t_low, bus_taps["low_load"]))

        bus_tap_schedule = (
            TapSchedule(()) if not bus_tap_schedule_entries else TapSchedule(tuple(bus_tap_schedule_entries))
        )

        cases = [
            ("baseline_no_tap", False, False),
            ("baseline_tap_change", False, True),
            ("ofo_no_tap", True, False),
            ("ofo_tap_change", True, True),
        ]

        bus_results: dict[str, str | float] = {"dc_bus": dc_bus}

        for case_name, use_ofo, use_tap_change in cases:
            case_idx += 1
            logger.info("")
            logger.info("[%d/%d] Bus %s -- %s", case_idx, total_cases, dc_bus, case_name)

            case_save_dir = save_dir / f"bus_{dc_bus}" / case_name

            try:
                result = run_case_1d(
                    sys=sys,
                    dc_bus=dc_bus,
                    base_kw_per_phase=dc_site.base_kw_per_phase,
                    case_name=case_name,
                    inference_data=inference_data,
                    training_trace=training_trace,
                    logistic_models=logistic_models,
                    all_models=all_models,
                    initial_taps=inference_taps,
                    tap_schedule=bus_tap_schedule if use_tap_change else None,
                    use_ofo=use_ofo,
                    ofo_config=ofo_config,
                    training=training,
                    replica_schedules=replica_schedules,
                    total_gpu_capacity=site_gpu_capacity,
                    pv_systems=pv_systems,
                    time_varying_loads=time_varying_loads,
                    total_duration_s=total_duration_s,
                    v_min=v_min,
                    v_max=v_max,
                    power_augmentation=power_augmentation,
                    connection_type=dc_site.connection_type,
                    dt_dc=dt_dc,
                    dt_grid=dt_grid,
                    dt_ctrl=dt_ctrl,
                    save_dir=case_save_dir,
                )
                logger.info(
                    "  -> viol=%.1fs  integral=%.4f pu·s  vmin=%.4f  vmax=%.4f",
                    result["violation_time_s"],
                    result["integral_violation_pu_s"],
                    result["worst_vmin"],
                    result["worst_vmax"],
                )
                for k, v in result.items():
                    bus_results[f"{case_name}_{k}"] = v
            except Exception as e:
                logger.error("  Bus %s -- %s FAILED: %s", dc_bus, case_name, e)
                for k in [
                    "violation_time_s",
                    "worst_vmin",
                    "worst_vmax",
                    "integral_violation_pu_s",
                    "avg_power_kw_per_phase",
                ]:
                    bus_results[f"{case_name}_{k}"] = float("nan")

        all_rows.append(bus_results)

    # Save results
    if all_rows:
        csv_path = save_dir / f"results_{system}_sweep_dc_locations.csv"
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        logger.info("")
        logger.info("Results saved to: %s", csv_path)
        _plot_bus_comparison(all_rows, save_dir / f"comparison_{system}_sweep_dc_locations.png")

    # Summary table
    logger.info("")
    logger.info("=" * 160)
    logger.info("FULL COMPARISON: No-Tap vs Tap-Change vs OFO vs OFO+Tap-Change")
    logger.info("=" * 160)
    logger.info(
        "%-8s | %12s %12s | %12s %12s | %12s %12s | %12s %12s",
        "Bus",
        "NoTap V(s)",
        "NoTap I",
        "TapChg V(s)",
        "TapChg I",
        "OFO V(s)",
        "OFO I",
        "OFO+TC V(s)",
        "OFO+TC I",
    )
    logger.info("-" * 160)
    for row in all_rows:
        bus = row["dc_bus"]
        logger.info(
            "%-8s | %12.1f %12.4f | %12.1f %12.4f | %12.1f %12.4f | %12.1f %12.4f",
            bus,
            row.get("baseline_no_tap_violation_time_s", float("nan")),
            row.get("baseline_no_tap_integral_violation_pu_s", float("nan")),
            row.get("baseline_tap_change_violation_time_s", float("nan")),
            row.get("baseline_tap_change_integral_violation_pu_s", float("nan")),
            row.get("ofo_no_tap_violation_time_s", float("nan")),
            row.get("ofo_no_tap_integral_violation_pu_s", float("nan")),
            row.get("ofo_tap_change_violation_time_s", float("nan")),
            row.get("ofo_tap_change_integral_violation_pu_s", float("nan")),
        )

    logger.info("")
    logger.info("All outputs saved to: %s", save_dir)


# 2-D SWEEP (multiple DC sites)


def _plot_heatmaps(df: pd.DataFrame, buses: list[str], save_dir: Path) -> None:
    n = len(buses)
    bus_idx = {b: i for i, b in enumerate(buses)}

    for metric, label, fname in [
        ("violation_time_s", "Violation Time (s)", "heatmap_violation_time.png"),
        ("integral_violation_pu_s", "Integral Violation (pu·s)", "heatmap_integral_violation.png"),
    ]:
        mat = np.full((n, n), np.nan)

        for _, row in df.iterrows():
            i = bus_idx.get(row["bus_A"])
            j = bus_idx.get(row["bus_B"])
            if i is None or j is None:
                continue
            val = row[metric]
            mat[i, j] = val
            mat[j, i] = val

        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
        im = ax.imshow(mat, cmap="YlOrRd", aspect="equal", origin="lower")

        ax.set_xticks(range(n))
        ax.set_xticklabels(buses, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(buses, fontsize=9)

        ax.set_xlabel("DC2 Bus", fontsize=12)
        ax.set_ylabel("DC1 Bus", fontsize=12)
        ax.set_title(f"2-D DC Location Sweep: {label}", fontsize=14)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(label, fontsize=11)

        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                if np.isfinite(val):
                    text_color = "white" if val > (np.nanmax(mat) + np.nanmin(mat)) / 2 else "black"
                    fmt = f"{val:.0f}" if metric == "violation_time_s" else f"{val:.2f}"
                    ax.text(j, i, fmt, ha="center", va="center", fontsize=7, color=text_color)

        fig.tight_layout()
        fig.savefig(save_dir / fname, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved heatmap: %s", fname)


def main_2d(
    *,
    sys: dict,
    system: str,
    dc_sites: dict[str, _DCSiteConfig],
    all_models: tuple[InferenceModelSpec, ...],
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
    ofo_config: OFOConfig,
    training: TrainingRun | None = None,
    pv_systems: list | None = None,
    time_varying_loads: list | None = None,
    power_augmentation: PowerAugmentationConfig = POWER_AUG,
    total_duration_s: int = TOTAL_DURATION_S,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    dt_override: str | None = None,
    output_dir: Path | None = None,
) -> None:
    """2-D sweep: multiple DC sites, sweep all unordered pairs of candidate buses."""
    dt_dc = DT_DC
    dt_grid = DT_GRID
    dt_ctrl = DT_CTRL

    if dt_override is not None:
        frac = Fraction(dt_override) if "/" in dt_override else Fraction(int(dt_override))
        dt_dc = dt_grid = dt_ctrl = frac
        logger.info("Time resolution override: dt = %s s for all components", dt_override)

    save_dir = (
        output_dir if output_dir else (Path(__file__).resolve().parent / "outputs" / system / "sweep_dc_locations_2d")
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    # Discover candidate buses
    site_ids = list(dc_sites.keys())
    assert len(site_ids) >= 2, "Need at least 2 dc_sites for 2-D sweep"
    site_A_id = site_ids[0]
    site_B_id = site_ids[1]
    site_A = dc_sites[site_A_id]
    site_B = dc_sites[site_B_id]

    target_kv = sys["bus_kv"]

    logger.info("Discovering candidate 3-phase buses at %.1f kV...", target_kv)
    candidate_buses = discover_candidate_buses(
        sys["dss_case_dir"],
        sys["dss_master_file"],
        target_kv,
        exclude=set(sys["exclude_buses"]),
    )
    logger.info("Found %d candidate buses: %s", len(candidate_buses), candidate_buses)

    if len(candidate_buses) < 2:
        logger.error("Need at least 2 candidate buses for 2-D sweep!")
        return

    specs_A = tuple(md.spec for md, _ in site_A.models)
    specs_B = tuple(md.spec for md, _ in site_B.models)
    inference_A = inference_data.filter_models(specs_A)
    inference_B = inference_data.filter_models(specs_B)

    initial_taps = sys["initial_taps"]
    exclude_buses = tuple(sys["exclude_buses"])

    # Generate all unordered pairs
    pairs = list(itertools.combinations(candidate_buses, 2))
    total = len(pairs)
    logger.info("Total pairs: %d (from %d buses)", total, len(candidate_buses))

    # Run sweep
    rows: list[dict] = []

    for idx, (bus_A, bus_B) in enumerate(pairs, start=1):
        logger.info("[%d/%d] DC1(%s)@%s + DC2(%s)@%s", idx, total, site_A_id, bus_A, site_B_id, bus_B)

        try:
            dc_config_A = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=site_A.base_kw_per_phase)
            dc_config_B = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=site_B.base_kw_per_phase)

            rs_A = site_A.replica_schedules or {md.spec.model_label: s for md, s in site_A.models}
            rs_B = site_B.replica_schedules or {md.spec.model_label: s for md, s in site_B.models}

            wl_kwargs_A: dict = {"inference_data": inference_A, "replica_schedules": rs_A}
            if training is not None:
                wl_kwargs_A["training"] = training

            wl_kwargs_B: dict = {"inference_data": inference_B, "replica_schedules": rs_B}
            if training is not None:
                wl_kwargs_B["training"] = training

            dc_A = OfflineDatacenter(
                dc_config_A,
                OfflineWorkload(**wl_kwargs_A),
                name=site_A_id,
                dt_s=dt_dc,
                seed=site_A.seed,
                power_augmentation=power_augmentation,
                total_gpu_capacity=site_A.total_gpu_capacity,
            )
            dc_B = OfflineDatacenter(
                dc_config_B,
                OfflineWorkload(**wl_kwargs_B),
                name=site_B_id,
                dt_s=dt_dc,
                seed=site_B.seed,
                power_augmentation=power_augmentation,
                total_gpu_capacity=site_B.total_gpu_capacity,
            )

            grid = OpenDSSGrid(
                dss_case_dir=sys["dss_case_dir"],
                dss_master_file=sys["dss_master_file"],
                source_pu=sys["source_pu"],
                dt_s=dt_grid,
                initial_tap_position=initial_taps,
                exclude_buses=exclude_buses,
            )
            grid.attach_dc(
                dc_A, bus=bus_A, connection_type=site_A.connection_type, power_factor=dc_config_A.power_factor
            )
            grid.attach_dc(
                dc_B, bus=bus_B, connection_type=site_B.connection_type, power_factor=dc_config_B.power_factor
            )
            for bus, gen in pv_systems or []:
                grid.attach_generator(gen, bus=bus)
            for bus, ld in time_varying_loads or []:
                grid.attach_load(ld, bus=bus)

            ofo_A = OFOBatchSizeController(
                specs_A,
                datacenter=dc_A,
                models=logistic_models,
                config=ofo_config,
                dt_s=dt_ctrl,
                grid=grid,
            )
            ofo_B = OFOBatchSizeController(
                specs_B,
                datacenter=dc_B,
                models=logistic_models,
                config=ofo_config,
                dt_s=dt_ctrl,
                grid=grid,
            )

            coord = Coordinator(
                datacenters=[dc_A, dc_B],
                grid=grid,
                controllers=[ofo_A, ofo_B],
                total_duration_s=total_duration_s,
            )

            t0 = time.monotonic()
            log = coord.run()
            wall_s = time.monotonic() - t0

            vstats = compute_allbus_voltage_stats(
                log.grid_states,
                v_min=v_min,
                v_max=v_max,
                exclude_buses=exclude_buses,
            )
            pstats = compute_performance_stats(
                log.dc_states,
                itl_deadline_s_by_model={ms.model_label: ms.itl_deadline_s for ms in all_models},
            )

            row = {
                "bus_A": bus_A,
                "bus_B": bus_B,
                "violation_time_s": vstats.violation_time_s,
                "integral_violation_pu_s": vstats.integral_violation_pu_s,
                "worst_vmin": vstats.worst_vmin,
                "worst_vmax": vstats.worst_vmax,
                "mean_throughput_tps": pstats.mean_throughput_tps,
                "itl_deadline_fraction": pstats.itl_deadline_fraction,
                "wall_time_s": wall_s,
            }
            rows.append(row)
            logger.info(
                "  -> viol=%.1fs  integral=%.4f pu·s  vmin=%.4f  thpt=%.1f k tok/s  itl_miss=%.2f%%  wall=%.1fs",
                vstats.violation_time_s,
                vstats.integral_violation_pu_s,
                vstats.worst_vmin,
                pstats.mean_throughput_tps / 1e3,
                pstats.itl_deadline_fraction * 100.0,
                wall_s,
            )

        except Exception:
            logger.exception("Pair (%s, %s) failed; skipping.", bus_A, bus_B)

    # Save CSV
    if not rows:
        logger.warning("No successful runs; nothing to save.")
        return

    df = pd.DataFrame(rows)
    csv_path = save_dir / f"results_{system}_sweep_dc_locations_2d.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved %d rows to %s", len(df), csv_path)

    _plot_heatmaps(df, candidate_buses, save_dir)
    logger.info("All outputs in: %s", save_dir)


# ZONE-CONSTRAINED SWEEP (N DC sites, each in its own zone)


def _run_multi_dc_case(
    *,
    bus_map: dict[str, str],
    site_ids: list[str],
    dc_sites: dict[str, _DCSiteConfig],
    site_inference_data: dict[str, InferenceData],
    sys: dict,
    training_trace: TrainingTrace,
    training: TrainingRun | None,
    logistic_models: LogisticModelStore,
    ofo_config: OFOConfig,
    target_kv: float,
    initial_taps: TapPosition | None,
    exclude_buses: tuple[str, ...],
    pv_systems: list | None = None,
    time_varying_loads: list | None = None,
    power_augmentation: PowerAugmentationConfig = POWER_AUG,
    dt_dc: Fraction = DT_DC,
    dt_grid: Fraction = DT_GRID,
    dt_ctrl: Fraction = DT_CTRL,
    total_duration_s: int | None = None,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    stress_test: bool = False,
) -> dict:
    """Run a single multi-DC simulation and return metrics dict.

    When stress_test=True, PV runs at constant peak power and time-varying
    loads are disabled.  Each datacenter uses an immediate inference ramp to
    fill its total_gpu_capacity, simulating the extreme case.
    """
    duration = total_duration_s if total_duration_s is not None else TOTAL_DURATION_S

    # Build datacenters
    datacenters = {}
    for sid in site_ids:
        site = dc_sites[sid]
        dc_config = DatacenterConfig(
            gpus_per_server=8,
            base_kw_per_phase=site.base_kw_per_phase,
        )
        rs = site.replica_schedules or {md.spec.model_label: s for md, s in site.models}
        wl_kwargs: dict = {"inference_data": site_inference_data[sid]}

        if stress_test:
            # Immediate per-model ramps to fill total GPU capacity
            current_gpus = sum(s.initial * md.spec.gpus_per_replica for md, s in site.models)
            total_cap = site.total_gpu_capacity or current_gpus
            scale = total_cap / current_gpus if current_gpus > 0 else 1.0
            if scale > 1.0:
                stress_rs = {}
                for md, s in site.models:
                    abs_target = round(scale * s.initial)
                    stress_rs[md.spec.model_label] = ReplicaSchedule(initial=s.initial).ramp_to(
                        abs_target, t_start=0, t_end=1
                    )
                wl_kwargs["replica_schedules"] = stress_rs
            else:
                wl_kwargs["replica_schedules"] = rs
        else:
            wl_kwargs["replica_schedules"] = rs
            if training is not None:
                wl_kwargs["training"] = training

        dc = OfflineDatacenter(
            dc_config,
            OfflineWorkload(**wl_kwargs),
            name=sid,
            dt_s=dt_dc,
            seed=site.seed,
            power_augmentation=power_augmentation,
            total_gpu_capacity=site.total_gpu_capacity,
        )
        datacenters[sid] = dc

    # Build grid
    grid = OpenDSSGrid(
        dss_case_dir=sys["dss_case_dir"],
        dss_master_file=sys["dss_master_file"],
        source_pu=sys["source_pu"],
        dt_s=dt_grid,
        initial_tap_position=initial_taps,
        exclude_buses=exclude_buses,
    )
    for sid in site_ids:
        grid.attach_dc(
            datacenters[sid],
            bus=bus_map[sid],
            connection_type=dc_sites[sid].connection_type,
            power_factor=DatacenterConfig().power_factor,
        )
    if stress_test:
        # Constant PV at peak power, no time-varying loads
        for bus, gen in pv_systems or []:
            grid.attach_generator(ConstantGenerator(gen._peak_kw), bus=bus)
    else:
        for bus, gen in pv_systems or []:
            grid.attach_generator(gen, bus=bus)
        for bus, ld in time_varying_loads or []:
            grid.attach_load(ld, bus=bus)

    # Build OFO controllers
    dc_list = list(datacenters.values())
    controllers = []
    for sid in site_ids:
        site_specs = tuple(md.spec for md, _ in dc_sites[sid].models)
        ofo_ctrl = OFOBatchSizeController(
            site_specs,
            datacenter=datacenters[sid],
            models=logistic_models,
            config=ofo_config,
            dt_s=dt_ctrl,
            grid=grid,
        )
        controllers.append(ofo_ctrl)

    coord = Coordinator(
        datacenters=dc_list,
        grid=grid,
        controllers=controllers,
        total_duration_s=duration,
    )

    t0 = time.monotonic()
    log = coord.run()
    wall_s = time.monotonic() - t0

    vstats = compute_allbus_voltage_stats(
        log.grid_states,
        v_min=v_min,
        v_max=v_max,
        exclude_buses=exclude_buses,
    )
    all_site_specs = tuple(ms for sid in site_ids for md, _ in dc_sites[sid].models for ms in (md.spec,))
    pstats = compute_performance_stats(
        log.dc_states,
        itl_deadline_s_by_model={ms.model_label: ms.itl_deadline_s for ms in all_site_specs},
    )

    row: dict = {}
    for sid in site_ids:
        row[f"bus_{sid}"] = bus_map[sid]
    row.update(
        {
            "violation_time_s": vstats.violation_time_s,
            "integral_violation_pu_s": vstats.integral_violation_pu_s,
            "worst_vmin": vstats.worst_vmin,
            "worst_vmax": vstats.worst_vmax,
            "mean_throughput_tps": pstats.mean_throughput_tps,
            "itl_deadline_fraction": pstats.itl_deadline_fraction,
            "wall_time_s": wall_s,
        }
    )
    return row


def _log_results_table(
    df: pd.DataFrame,
    site_ids: list[str],
    title: str,
) -> None:
    """Print a formatted results table to the logger."""
    logger.info("")
    logger.info("=" * 120)
    logger.info(title)
    logger.info("=" * 120)
    bus_cols = [f"bus_{sid}" for sid in site_ids]
    header = "  ".join(f"{col:>10s}" for col in bus_cols)
    header += f"  {'Viol(s)':>12s} {'Integral':>12s} {'Vmin':>10s} {'Vmax':>10s} {'Wall(s)':>10s}"
    logger.info(header)
    logger.info("-" * 120)
    for _, row in df.iterrows():
        line = "  ".join(f"{row[col]:>10s}" for col in bus_cols)
        line += "  {:12.1f} {:12.4f} {:10.4f} {:10.4f} {:10.1f}".format(
            row["violation_time_s"],
            row["integral_violation_pu_s"],
            row["worst_vmin"],
            row["worst_vmax"],
            row["wall_time_s"],
        )
        logger.info(line)


def _plot_zoned_summary(
    df: pd.DataFrame,
    site_ids: list[str],
    save_dir: Path,
    system: str,
    suffix: str = "",
) -> None:
    """Plot bar chart of zone-constrained sweep results."""
    n_show = min(20, len(df))
    df_top = df.head(n_show)

    bus_cols = [f"bus_{sid}" for sid in site_ids]
    labels = ["/".join(str(row[c]) for c in bus_cols) for _, row in df_top.iterrows()]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, n_show * 0.7), 10))

    x = np.arange(n_show)
    ax1.bar(x, df_top["violation_time_s"].values, color="steelblue")
    ax1.set_ylabel("Violation Time (s)")
    title_tag = f" ({suffix})" if suffix else ""
    ax1.set_title(f"{system.upper()} DC Location Sweep{title_tag} -- Violation Time (top {n_show})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    ax2.bar(x, df_top["integral_violation_pu_s"].values, color="coral")
    ax2.set_ylabel("Integral Violation (pu·s)")
    ax2.set_title(f"{system.upper()} DC Location Sweep{title_tag} -- Integral Violation (top {n_show})")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    fname = (
        f"Phase_2_combination_results_{system}.png"
        if suffix == "phase2"
        else f"comparison_{system}_zoned{('_' + suffix) if suffix else ''}.png"
    )
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison plot: %s", fname)


def _plot_screening_bars(
    screening_results: dict[str, pd.DataFrame],
    save_dir: Path,
    system: str,
) -> None:
    """Plot per-zone screening results as grouped bar charts."""
    n_zones = len(screening_results)
    fig, axes = plt.subplots(1, n_zones, figsize=(5 * n_zones, 5), squeeze=False)

    for ax, (sid, df) in zip(axes[0], screening_results.items(), strict=False):
        buses = df[f"bus_{sid}"].values
        viol = df["violation_time_s"].values
        x = np.arange(len(buses))
        ax.bar(x, viol, color="steelblue")
        ax.set_xlabel("Bus")
        ax.set_ylabel("Violation Time (s)")
        ax.set_title(f"Zone '{sid}' Screening")
        ax.set_xticks(x)
        ax.set_xticklabels(buses, rotation=45, ha="right", fontsize=9)

    fig.tight_layout()
    fname = f"Phase_1_screening_results_{system}.png"
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved screening plot: %s", fname)


def _plot_refinement_iteration(
    iteration_results: dict[str, list[dict]],
    site_ids: list[str],
    save_dir: Path,
    system: str,
    iteration: int,
    prev_best: dict[str, str] | None = None,
    new_best: dict[str, str] | None = None,
) -> None:
    """Plot Phase 3 refinement results: one subplot per zone, bar charts of
    violation time and integral violation for each candidate bus.

    Marks the previous best bus (from the prior iteration or Phase 2) with a
    diamond marker and the new best bus (from this iteration) with a star.
    """
    n_zones = len(site_ids)
    fig, axes = plt.subplots(2, n_zones, figsize=(5 * n_zones, 8), squeeze=False)

    for col, sid in enumerate(site_ids):
        zone_rows = iteration_results.get(sid, [])
        if not zone_rows:
            axes[0][col].set_title(f"Zone '{sid}' (no data)")
            axes[1][col].set_title(f"Zone '{sid}' (no data)")
            continue

        df = pd.DataFrame(zone_rows).sort_values(
            ["violation_time_s", "integral_violation_pu_s"],
        )
        buses = df[f"bus_{sid}"].values
        x = np.arange(len(buses))

        prev_bus = prev_best.get(sid) if prev_best else None
        new_bus = new_best.get(sid) if new_best else None

        for ax_row, metric, ylabel, color, title_suffix in [
            (0, "violation_time_s", "Violation Time (s)", "steelblue", "Violation Time"),
            (1, "integral_violation_pu_s", "Integral Violation (pu·s)", "coral", "Integral Violation"),
        ]:
            ax = axes[ax_row][col]
            vals = df[metric].values
            ax.bar(x, vals, color=color)

            # Mark previous best with diamond, new best with star
            for idx, bus in enumerate(buses):
                bus_str = str(bus)
                if prev_bus and bus_str == prev_bus and bus_str != (new_bus or ""):
                    ax.plot(idx, vals[idx], "Dk", markersize=10, zorder=5)
                if new_bus and bus_str == new_bus:
                    ax.plot(
                        idx,
                        vals[idx],
                        "*",
                        color="gold",
                        markersize=16,
                        markeredgecolor="black",
                        markeredgewidth=0.8,
                        zorder=5,
                    )

            ax.set_ylabel(ylabel)
            ax.set_title(f"Zone '{sid}' -- {title_suffix}")
            ax.set_xticks(x)
            ax.set_xticklabels(buses, rotation=45, ha="right", fontsize=9)

    # Add legend for markers
    legend_handles = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="black", markersize=8, label="Previous best"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gold",
            markersize=14,
            markeredgecolor="black",
            label="New best",
        ),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=0.9)

    fig.suptitle(
        f"{system.upper()} Phase 3 Refinement -- Iteration {iteration}",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fname = f"Phase_3_refinement_results_{system}_iter{iteration}.png"
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved refinement plot: %s", fname)


def main_zoned(
    *,
    sys: dict,
    system: str,
    dc_sites: dict[str, _DCSiteConfig],
    all_models: tuple[InferenceModelSpec, ...],
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    training: TrainingRun | None,
    logistic_models: LogisticModelStore,
    ofo_config: OFOConfig,
    zones: dict[str, list[str]],
    pv_systems: list | None = None,
    time_varying_loads: list | None = None,
    power_augmentation: PowerAugmentationConfig = POWER_AUG,
    total_duration_s: int = TOTAL_DURATION_S,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    dt_override: str | None = None,
    dt_screening: str | None = None,
    output_dir: Path | None = None,
    top_k: int = 4,
    refine: bool = False,
) -> None:
    """Zone-constrained sweep with screening + combination + optional refinement.

    Three-phase design with different resolution/duration per phase:

    Phase 1 (Screening): Sweep each zone independently while holding other zones
        at their default config buses.  Uses coarse resolution (`dt_screening`,
        default 60 s) over the full simulation duration (typically 3600 s) for
        fast ranking.  Keep top-K per zone.
    Phase 2 (Combination): Cartesian product of top-K per zone.  Uses native
        config resolution (typically dt=0.1 s) but only 60 s duration as a
        stress test with full DC capacity and constant PV.
    Phase 3 (Refinement, optional): Starting from the Phase 2 winner, re-sweep
        each zone one at a time (holding others at best).  Uses native config
        resolution over the full simulation duration for accurate evaluation.
    """
    # Full-resolution dt (for Phase 2 & 3)
    dt_dc = DT_DC
    dt_grid = DT_GRID
    dt_ctrl = DT_CTRL
    logger.info("Phase 2/3 resolution: dt_dc=%s, dt_grid=%s, dt_ctrl=%s", dt_dc, dt_grid, dt_ctrl)

    # Screening dt (Phase 1): coarse resolution for fast ranking
    if dt_screening is not None:
        frac = Fraction(dt_screening) if "/" in dt_screening else Fraction(int(dt_screening))
        dt_dc_screen = dt_grid_screen = dt_ctrl_screen = frac
    elif dt_override is not None:
        frac = Fraction(dt_override) if "/" in dt_override else Fraction(int(dt_override))
        dt_dc_screen = dt_grid_screen = dt_ctrl_screen = frac
    else:
        # Default: 60 s for screening (coarse but fast)
        dt_dc_screen = dt_grid_screen = dt_ctrl_screen = Fraction(60)
    logger.info("Phase 1 screening resolution: dt = %s s", dt_dc_screen)

    save_dir = (
        output_dir
        if output_dir
        else (Path(__file__).resolve().parent / "outputs" / system / "sweep_dc_locations_zoned")
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    site_ids = list(dc_sites.keys())

    # Discover all 3-phase candidate buses
    target_kv = sys["bus_kv"]

    logger.info("Discovering candidate 3-phase buses at %.1f kV...", target_kv)
    all_candidate_buses = set(
        discover_candidate_buses(
            sys["dss_case_dir"],
            sys["dss_master_file"],
            target_kv,
            exclude=set(sys["exclude_buses"]),
        )
    )
    logger.info("Found %d candidate 3-phase buses total", len(all_candidate_buses))

    # Build per-zone candidate lists (intersect zone bus lists with 3-phase candidates)
    zone_candidates: dict[str, list[str]] = {}
    for site_id in site_ids:
        zone_buses = zones.get(site_id, [])
        if not zone_buses:
            logger.warning("Site '%s' has no zone defined; using all candidate buses", site_id)
            zone_candidates[site_id] = sorted(all_candidate_buses)
        else:
            zone_set = {b.lower() for b in zone_buses}
            candidates = sorted(b for b in all_candidate_buses if b.lower() in zone_set)
            zone_candidates[site_id] = candidates
        logger.info(
            "  Zone '%s': %d candidate buses: %s",
            site_id,
            len(zone_candidates[site_id]),
            zone_candidates[site_id],
        )

    # Check for empty zones
    for site_id, cands in zone_candidates.items():
        if not cands:
            logger.error("Zone '%s' has no 3-phase candidate buses! Aborting.", site_id)
            return

    # Prepare per-site inference data
    site_inference_data: dict[str, InferenceData] = {}
    for sid in site_ids:
        site_specs = tuple(md.spec for md, _ in dc_sites[sid].models)
        site_inference_data[sid] = inference_data.filter_models(site_specs)

    initial_taps = sys["initial_taps"]
    exclude_buses = tuple(sys["exclude_buses"])

    # Shared kwargs for _run_multi_dc_case (without dt, added per-phase)
    _base_kwargs = dict(
        site_ids=site_ids,
        dc_sites=dc_sites,
        site_inference_data=site_inference_data,
        sys=sys,
        training_trace=training_trace,
        training=training,
        logistic_models=logistic_models,
        ofo_config=ofo_config,
        target_kv=target_kv,
        initial_taps=initial_taps,
        exclude_buses=exclude_buses,
        pv_systems=pv_systems,
        time_varying_loads=time_varying_loads,
        power_augmentation=power_augmentation,
        v_min=v_min,
        v_max=v_max,
    )
    screening_kwargs = dict(
        **_base_kwargs,
        dt_dc=dt_dc_screen,
        dt_grid=dt_grid_screen,
        dt_ctrl=dt_ctrl_screen,
    )
    full_res_kwargs = dict(
        **_base_kwargs,
        dt_dc=dt_dc,
        dt_grid=dt_grid,
        dt_ctrl=dt_ctrl,
    )

    # Default bus map from config
    default_bus_map = {sid: dc_sites[sid].bus for sid in site_ids}

    # Phase 1: Screening
    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 1: SCREENING -- sweep each zone independently")
    logger.info("=" * 80)
    logger.info("Default bus map: %s", default_bus_map)

    screening_results: dict[str, pd.DataFrame] = {}
    top_k_per_zone: dict[str, list[str]] = {}

    total_screening = sum(len(zone_candidates[sid]) for sid in site_ids)
    screen_idx = 0

    for sweep_sid in site_ids:
        zone_rows: list[dict] = []
        cands = zone_candidates[sweep_sid]
        logger.info("")
        logger.info("── Screening zone '%s': %d candidates ──", sweep_sid, len(cands))

        for bus in cands:
            screen_idx += 1
            # Use default bus map but override the zone being swept
            bus_map = dict(default_bus_map)
            bus_map[sweep_sid] = bus
            combo_str = ", ".join(f"{sid}@{b}" for sid, b in bus_map.items())
            logger.info(
                "[%d/%d] Screening %s: %s",
                screen_idx,
                total_screening,
                sweep_sid,
                combo_str,
            )

            try:
                row = _run_multi_dc_case(bus_map=bus_map, **screening_kwargs)
                zone_rows.append(row)
                logger.info(
                    "  -> viol=%.1fs  integral=%.4f  vmin=%.4f  wall=%.1fs",
                    row["violation_time_s"],
                    row["integral_violation_pu_s"],
                    row["worst_vmin"],
                    row["wall_time_s"],
                )
            except Exception:
                logger.exception("  Screening %s@%s failed; skipping.", sweep_sid, bus)

        if not zone_rows:
            logger.error("Zone '%s' screening produced no results!", sweep_sid)
            return

        df_zone = pd.DataFrame(zone_rows).sort_values(
            ["violation_time_s", "integral_violation_pu_s"],
        )
        screening_results[sweep_sid] = df_zone

        # Select top-K (capped at available)
        k = min(top_k, len(df_zone))
        top_buses = df_zone[f"bus_{sweep_sid}"].head(k).tolist()
        top_k_per_zone[sweep_sid] = top_buses
        logger.info(
            "  Zone '%s' top-%d: %s",
            sweep_sid,
            k,
            top_buses,
        )

        # Save per-zone screening CSV
        csv_path = save_dir / f"Phase_1_screening_{system}_{sweep_sid}.csv"
        df_zone.to_csv(csv_path, index=False)

    # Save screening plots
    _plot_screening_bars(screening_results, save_dir, system)

    total_phase1 = screen_idx
    logger.info("")
    logger.info("Phase 1 complete: %d simulations", total_phase1)
    for sid, buses_list in top_k_per_zone.items():
        logger.info("  %s top-%d: %s", sid, len(buses_list), buses_list)

    # Phase 2: Combination (1-min stress test: constant PV + full DC)
    # Duration must yield at least 2 grid steps; ensure >= 2 x dt_grid.
    min_duration = int(math.ceil(float(dt_grid) * 2))
    phase2_duration_s = max(60, min_duration)
    logger.info("")
    logger.info("=" * 80)
    logger.info(
        "PHASE 2: COMBINATION -- Cartesian product of top-%d per zone "
        "(%ds stress test: constant PV + full DC capacity, no time-varying loads)",
        top_k,
        phase2_duration_s,
    )
    logger.info("=" * 80)

    zone_bus_lists = [top_k_per_zone[sid] for sid in site_ids]
    combinations = list(itertools.product(*zone_bus_lists))
    total = len(combinations)
    logger.info(
        "Combinations: %d (%s)",
        total,
        " x ".join(str(len(z)) for z in zone_bus_lists),
    )

    phase2_rows: list[dict] = []

    for idx, combo in enumerate(combinations, start=1):
        bus_map = {sid: bus for sid, bus in zip(site_ids, combo, strict=False)}
        combo_str = ", ".join(f"{sid}@{bus}" for sid, bus in bus_map.items())
        logger.info("[%d/%d] %s", idx, total, combo_str)

        try:
            row = _run_multi_dc_case(
                bus_map=bus_map,
                total_duration_s=phase2_duration_s,
                stress_test=True,
                **full_res_kwargs,
            )
            phase2_rows.append(row)
            logger.info(
                "  -> viol=%.1fs  integral=%.4f  vmin=%.4f  wall=%.1fs",
                row["violation_time_s"],
                row["integral_violation_pu_s"],
                row["worst_vmin"],
                row["wall_time_s"],
            )
        except Exception:
            logger.exception("Combination %s failed; skipping.", combo_str)

    if not phase2_rows:
        logger.warning("Phase 2 produced no results!")
        return

    df_phase2 = pd.DataFrame(phase2_rows).sort_values(
        ["violation_time_s", "integral_violation_pu_s"],
    )
    csv_path = save_dir / f"Phase_2_combination_results_{system}.csv"
    df_phase2.to_csv(csv_path, index=False)
    logger.info("Saved Phase 2 results (%d rows) to %s", len(df_phase2), csv_path)

    _log_results_table(df_phase2, site_ids, "PHASE 2 RESULTS (sorted by violation time)")
    _plot_zoned_summary(df_phase2, site_ids, save_dir, system, suffix="phase2")

    # Best from Phase 2
    best_row = df_phase2.iloc[0]
    best_bus_map = {sid: str(best_row[f"bus_{sid}"]) for sid in site_ids}
    logger.info("")
    logger.info(
        "Phase 2 best: %s  (viol=%.1fs, integral=%.4f)",
        ", ".join(f"{sid}@{b}" for sid, b in best_bus_map.items()),
        best_row["violation_time_s"],
        best_row["integral_violation_pu_s"],
    )

    total_sims = total_phase1 + len(phase2_rows)

    # Save final summary with the best result from Phase 2
    final_csv_path = save_dir / f"sweep_dc_locations_final_results_{system}.csv"
    df_phase2.head(1).to_csv(final_csv_path, index=False)
    logger.info("Saved best result to: %s", final_csv_path)

    logger.info("")
    logger.info("=" * 80)
    logger.info("BEST RESULT (Phase 2)")
    logger.info("=" * 80)
    logger.info(
        "  %s  ->  viol=%.1fs  integral=%.4f  vmin=%.4f  vmax=%.4f",
        ", ".join(f"{sid}@{b}" for sid, b in best_bus_map.items()),
        best_row["violation_time_s"],
        best_row["integral_violation_pu_s"],
        best_row["worst_vmin"],
        best_row["worst_vmax"],
    )

    # Phase 3: Refinement (optional)
    if not refine:
        logger.info("")
        logger.info("Phase 3 (refinement) skipped. Use --refine to enable.")
        logger.info("Total simulations: %d (Phase 1: %d, Phase 2: %d)", total_sims, total_phase1, len(phase2_rows))
        logger.info("All outputs in: %s", save_dir)
        return

    logger.info("")
    logger.info("=" * 80)
    logger.info("PHASE 3: REFINEMENT -- re-sweep each zone from Phase 2 best")
    logger.info("=" * 80)

    current_best = dict(best_bus_map)
    improved = True
    iteration = 0

    # Collect per-iteration, per-zone results for CSV and plotting
    all_refinement_rows: dict[int, dict[str, list[dict]]] = {}

    while improved:
        iteration += 1
        improved = False
        prev_best = dict(current_best)  # snapshot before this iteration
        logger.info("")
        logger.info("── Refinement iteration %d ──", iteration)
        logger.info("Current best: %s", current_best)

        all_refinement_rows[iteration] = {}

        for sweep_sid in site_ids:
            cands = zone_candidates[sweep_sid]
            logger.info(
                "  Re-sweeping zone '%s': %d candidates",
                sweep_sid,
                len(cands),
            )

            best_for_zone = current_best[sweep_sid]
            best_viol = float("inf")
            best_integral = float("inf")
            zone_rows: list[dict] = []

            for bus in cands:
                bus_map = dict(current_best)
                bus_map[sweep_sid] = bus
                combo_str = ", ".join(f"{sid}@{b}" for sid, b in bus_map.items())

                try:
                    row = _run_multi_dc_case(bus_map=bus_map, **full_res_kwargs)
                    total_sims += 1
                    viol = row["violation_time_s"]
                    integ = row["integral_violation_pu_s"]
                    logger.info(
                        "    %s@%s -> viol=%.1fs  integral=%.4f",
                        sweep_sid,
                        bus,
                        viol,
                        integ,
                    )
                    zone_rows.append(row)
                    if (viol, integ) < (best_viol, best_integral):
                        best_viol = viol
                        best_integral = integ
                        best_for_zone = bus
                except Exception:
                    logger.exception("    %s@%s failed; skipping.", sweep_sid, bus)

            all_refinement_rows[iteration][sweep_sid] = zone_rows

            if best_for_zone != current_best[sweep_sid]:
                logger.info(
                    "  Zone '%s' improved: %s -> %s (viol=%.1fs)",
                    sweep_sid,
                    current_best[sweep_sid],
                    best_for_zone,
                    best_viol,
                )
                current_best[sweep_sid] = best_for_zone
                improved = True
            else:
                logger.info(
                    "  Zone '%s' unchanged: %s",
                    sweep_sid,
                    current_best[sweep_sid],
                )

        # Save per-iteration CSV and plot
        for sweep_sid in site_ids:
            zone_rows = all_refinement_rows[iteration].get(sweep_sid, [])
            if zone_rows:
                df_iter_zone = pd.DataFrame(zone_rows).sort_values(
                    ["violation_time_s", "integral_violation_pu_s"],
                )
                csv_path = save_dir / (f"Phase_3_refinement_results_{system}_iter{iteration}_{sweep_sid}.csv")
                df_iter_zone.to_csv(csv_path, index=False)

        # Plot this iteration: one subplot per zone
        _plot_refinement_iteration(
            all_refinement_rows[iteration],
            site_ids,
            save_dir,
            system,
            iteration,
            prev_best=prev_best,
            new_best=dict(current_best),
        )

    # Final result
    logger.info("")
    logger.info("── Refinement converged after %d iteration(s) ──", iteration)
    final_str = ", ".join(f"{sid}@{b}" for sid, b in current_best.items())

    # Run one final sim at full config resolution to get exact metrics
    final_row = _run_multi_dc_case(bus_map=current_best, **full_res_kwargs)
    total_sims += 1

    df_final = pd.DataFrame([final_row])
    # Also overwrite the final summary with the refined result
    df_final.to_csv(
        save_dir / f"sweep_dc_locations_final_results_{system}.csv",
        index=False,
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL RESULT (after refinement)")
    logger.info("=" * 80)
    logger.info(
        "  %s  ->  viol=%.1fs  integral=%.4f  vmin=%.4f  vmax=%.4f",
        final_str,
        final_row["violation_time_s"],
        final_row["integral_violation_pu_s"],
        final_row["worst_vmin"],
        final_row["worst_vmax"],
    )
    logger.info("")
    logger.info(
        "Total simulations: %d (Phase 1: %d, Phase 2: %d, Phase 3: %d)",
        total_sims,
        total_phase1,
        len(phase2_rows),
        total_sims - total_phase1 - len(phase2_rows),
    )
    logger.info("All outputs in: %s", save_dir)


# Per-system experiment definitions


def setup_ieee13(sys_const, training_trace):
    """IEEE 13-bus: single DC at bus 671 with training overlay."""
    models = (
        deploy("Llama-3.1-8B", 720),
        deploy("Llama-3.1-70B", 180),
        deploy("Llama-3.1-405B", 90),
        deploy("Qwen3-30B-A3B", 480),
        deploy("Qwen3-235B-A22B", 210),
    )

    replica_schedules = {
        "Llama-3.1-8B": ReplicaSchedule(initial=720).ramp_to(360, t_start=2500, t_end=3000),
        "Llama-3.1-70B": ReplicaSchedule(initial=180).ramp_to(90, t_start=2500, t_end=3000),
        "Llama-3.1-405B": ReplicaSchedule(initial=90).ramp_to(45, t_start=2500, t_end=3000),
        "Qwen3-30B-A3B": ReplicaSchedule(initial=480).ramp_to(240, t_start=2500, t_end=3000),
        "Qwen3-235B-A22B": ReplicaSchedule(initial=210).ramp_to(105, t_start=2500, t_end=3000),
    }

    dc_sites = {
        "default": _DCSiteConfig(
            bus="671",
            base_kw_per_phase=500.0,
            models=models,
            seed=0,
            total_gpu_capacity=7200,
            replica_schedules=replica_schedules,
        ),
    }

    training = (
        TrainingRun(
            n_gpus=2400,
            trace=training_trace,
            target_peak_W_per_gpu=400.0,
        ).at(t_start=1000.0, t_end=2000.0)
        if training_trace is not None
        else None
    )

    ofo_config = OFOConfig(
        primal_step_size=0.1,
        w_throughput=0.001,
        w_switch=1.0,
        voltage_gradient_scale=1e6,
        v_min=V_MIN,
        v_max=V_MAX,
        voltage_dual_step_size=1.0,
        latency_dual_step_size=1.0,
        sensitivity_update_interval=3600,
        sensitivity_perturbation_kw=100.0,
    )

    tap_schedule_entries = [
        (1500, TapPosition(regulators={"creg1a": tap(16), "creg1b": tap(6), "creg1c": tap(17)})),
        (3300, TapPosition(regulators={"creg1a": tap(10), "creg1b": tap(6), "creg1c": tap(10)})),
    ]

    pv_systems = [("675", SyntheticPV(peak_kw=10.0))]
    time_varying_loads = [("680", SyntheticLoad(peak_kw=10.0))]

    return dict(
        dc_sites=dc_sites,
        training=training,
        ofo_config=ofo_config,
        tap_schedule_entries=tap_schedule_entries,
        training_t_start=1000.0,
        training_t_end=2000.0,
        ramp_t_end=3000.0,
        pv_systems=pv_systems,
        time_varying_loads=time_varying_loads,
    )


def setup_ieee34(sys_const, training_trace):
    """IEEE 34-bus: two DC sites (upstream + downstream)."""
    upstream_models = (
        deploy("Llama-3.1-8B", 720),
        deploy("Llama-3.1-70B", 180),
        deploy("Llama-3.1-405B", 90),
    )
    downstream_models = (
        deploy("Qwen3-30B-A3B", 480),
        deploy("Qwen3-235B-A22B", 210),
    )

    dc_sites = {
        "upstream": _DCSiteConfig(
            bus="850",
            base_kw_per_phase=120.0,
            models=upstream_models,
            seed=0,
            total_gpu_capacity=520,
        ),
        "downstream": _DCSiteConfig(
            bus="834",
            base_kw_per_phase=80.0,
            models=downstream_models,
            seed=42,
            total_gpu_capacity=600,
        ),
    }

    ofo_config = OFOConfig(
        primal_step_size=0.05,
        w_throughput=0.001,
        w_switch=1.0,
        voltage_gradient_scale=1e6,
        v_min=V_MIN,
        v_max=V_MAX,
        voltage_dual_step_size=20.0,
        latency_dual_step_size=1.0,
        sensitivity_update_interval=3600,
        sensitivity_perturbation_kw=10.0,
    )

    pv_systems = [
        ("848", SyntheticPV(peak_kw=130.0, site_idx=0)),
        ("830", SyntheticPV(peak_kw=65.0, site_idx=1)),
    ]
    time_varying_loads = [
        ("860", SyntheticLoad(peak_kw=80.0, site_idx=0)),
        ("844", SyntheticLoad(peak_kw=120.0, site_idx=1)),
        ("840", SyntheticLoad(peak_kw=60.0, site_idx=2)),
        ("858", SyntheticLoad(peak_kw=50.0, site_idx=3)),
        ("854", SyntheticLoad(peak_kw=40.0, site_idx=4)),
    ]

    return dict(
        dc_sites=dc_sites,
        training=None,
        ofo_config=ofo_config,
        pv_systems=pv_systems,
        time_varying_loads=time_varying_loads,
    )


def setup_ieee123(sys_const, training_trace):
    """IEEE 123-bus: four DC zones with per-site ramps."""
    dc_sites = {
        "z1_sw": _DCSiteConfig(
            bus="8",
            base_kw_per_phase=310.0,
            models=(deploy("Llama-3.1-8B", 120),),
            seed=0,
            total_gpu_capacity=120,
            replica_schedules={
                "Llama-3.1-8B": ReplicaSchedule(initial=120).ramp_to(180, t_start=500, t_end=1000),
            },
        ),
        "z2_nw": _DCSiteConfig(
            bus="23",
            base_kw_per_phase=265.0,
            models=(deploy("Qwen3-30B-A3B", 80),),
            seed=17,
            total_gpu_capacity=160,
            replica_schedules={
                "Qwen3-30B-A3B": ReplicaSchedule(initial=80).ramp_to(104, t_start=1500, t_end=2500),
            },
        ),
        "z3_se": _DCSiteConfig(
            bus="60",
            base_kw_per_phase=295.0,
            models=(deploy("Llama-3.1-70B", 30), deploy("Llama-3.1-405B", 35)),
            seed=34,
            total_gpu_capacity=400,
            replica_schedules={
                "Llama-3.1-70B": ReplicaSchedule(initial=30).ramp_to(45, t_start=700, t_end=1100),
                "Llama-3.1-405B": ReplicaSchedule(initial=35).ramp_to(52, t_start=700, t_end=1100),
            },
        ),
        "z4_ne": _DCSiteConfig(
            bus="105",
            base_kw_per_phase=325.0,
            models=(deploy("Qwen3-235B-A22B", 55),),
            seed=51,
            total_gpu_capacity=440,
            replica_schedules={
                "Qwen3-235B-A22B": ReplicaSchedule(initial=55).ramp_to(27, t_start=2000, t_end=2500),
            },
        ),
    }

    ofo_config = OFOConfig(
        primal_step_size=0.05,
        w_throughput=0.001,
        w_switch=1.0,
        voltage_gradient_scale=1e6,
        v_min=V_MIN,
        v_max=V_MAX,
        voltage_dual_step_size=0.3,
        latency_dual_step_size=1.0,
        sensitivity_update_interval=3600,
        sensitivity_perturbation_kw=10.0,
    )

    pv_systems = [
        ("1", SyntheticPV(peak_kw=333.3, site_idx=0)),
        ("48", SyntheticPV(peak_kw=333.3, site_idx=1)),
        ("99", SyntheticPV(peak_kw=333.3, site_idx=2)),
    ]

    return dict(
        dc_sites=dc_sites,
        training=None,
        ofo_config=ofo_config,
        pv_systems=pv_systems,
        time_varying_loads=[],
        zones=sys_const.get("zones"),
    )


SETUPS = {
    "ieee13": setup_ieee13,
    "ieee34": setup_ieee34,
    "ieee123": setup_ieee123,
}


# MAIN ENTRY POINT


def main(
    *,
    system: str,
    buses: list[str] | None = None,
    dt_override: str | None = None,
    dt_screening: str | None = None,
    output_dir: Path | None = None,
    top_k: int = 4,
    refine: bool = False,
) -> None:
    sys_const = SYSTEMS[system]()

    # Load data pipeline
    data_sources, data_dir = load_data_sources()

    # Pre-load data using all model specs (data pipeline generates for all data_sources)
    all_models = ALL_MODEL_SPECS

    logger.info("Loading data for %s...", system)
    inference_data = InferenceData.ensure(
        data_dir,
        all_models,
        data_sources,
        plot=False,
        dt_s=float(DT_DC),
    )
    training_trace = TrainingTrace.ensure(data_dir / "training_trace.csv")
    logistic_models = LogisticModelStore.ensure(
        data_dir / "logistic_fits.csv",
        all_models,
        data_sources,
        plot=False,
    )

    # Build experiment config
    setup_fn = SETUPS[system]
    config = setup_fn(sys_const, training_trace)

    dc_sites = config["dc_sites"]
    n_sites = len(dc_sites)
    has_zones = config.get("zones") is not None and len(config.get("zones", {})) > 0

    # Collect all model specs from DC sites for the experiment
    exp_all_models: list[InferenceModelSpec] = []
    seen_labels: set[str] = set()
    for site in dc_sites.values():
        for md, _ in site.models:
            if md.spec.model_label not in seen_labels:
                exp_all_models.append(md.spec)
                seen_labels.add(md.spec.model_label)
    exp_all_models_tuple = tuple(exp_all_models)

    if has_zones:
        logger.info(
            "Config has %d DC site(s) with %d zone(s) -> zone-constrained sweep",
            n_sites,
            len(config["zones"]),
        )
        main_zoned(
            sys=sys_const,
            system=system,
            dc_sites=dc_sites,
            all_models=exp_all_models_tuple,
            inference_data=inference_data,
            training_trace=training_trace,
            training=config.get("training"),
            logistic_models=logistic_models,
            ofo_config=config["ofo_config"],
            zones=config["zones"],
            pv_systems=config.get("pv_systems"),
            time_varying_loads=config.get("time_varying_loads"),
            dt_override=dt_override,
            dt_screening=dt_screening,
            output_dir=output_dir,
            top_k=top_k,
            refine=refine,
        )
    elif n_sites <= 1:
        logger.info("Config has %d DC site(s) -> 1-D sweep", n_sites)
        dc_site = next(iter(dc_sites.values()))
        main_1d(
            sys=sys_const,
            system=system,
            dc_site=dc_site,
            all_models=exp_all_models_tuple,
            inference_data=inference_data,
            training_trace=training_trace,
            logistic_models=logistic_models,
            ofo_config=config["ofo_config"],
            training=config.get("training"),
            replica_schedules=dc_site.replica_schedules,
            tap_schedule_entries=config.get("tap_schedule_entries"),
            training_t_start=config.get("training_t_start", 0),
            training_t_end=config.get("training_t_end", 0),
            ramp_t_end=config.get("ramp_t_end", 0),
            pv_systems=config.get("pv_systems"),
            time_varying_loads=config.get("time_varying_loads"),
            buses=buses,
            dt_override=dt_override,
            output_dir=output_dir,
        )
    else:
        logger.info("Config has %d DC site(s) -> 2-D sweep", n_sites)
        main_2d(
            sys=sys_const,
            system=system,
            dc_sites=dc_sites,
            all_models=exp_all_models_tuple,
            inference_data=inference_data,
            training_trace=training_trace,
            logistic_models=logistic_models,
            ofo_config=config["ofo_config"],
            training=config.get("training"),
            pv_systems=config.get("pv_systems"),
            time_varying_loads=config.get("time_varying_loads"),
            dt_override=dt_override,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    from dataclasses import dataclass

    import tyro

    @dataclass
    class Args:
        """Command-line arguments.

        Attributes:
            system: System name (ieee13, ieee34, ieee123).
            buses: Comma-separated list of buses to test (overrides auto-discovery, 1-D only).
            dt: Override time step for all components (e.g. '60' for 60s resolution).
            dt_screening: Coarser time step for Phase 1 screening only (e.g. '60').
                Phases 2/3 use --dt or config.
            top_k: Number of top candidates per zone to keep after screening
                (zone-constrained mode).
            refine: Enable Phase 3 iterative refinement (zone-constrained mode).
            output_dir: Override output directory path.
            log_level: Logging verbosity (DEBUG, INFO, WARNING).
        """

        system: str = "ieee13"
        buses: str | None = None
        dt: str | None = None
        dt_screening: str | None = None
        top_k: int = 4
        refine: bool = False
        output_dir: str | None = None
        log_level: str = "INFO"

    args = tyro.cli(Args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    bus_list = [b.strip() for b in args.buses.split(",")] if args.buses else None
    out_dir = Path(args.output_dir) if args.output_dir else None
    main(
        system=args.system,
        buses=bus_list,
        dt_override=args.dt,
        dt_screening=args.dt_screening,
        output_dir=out_dir,
        top_k=args.top_k,
        refine=args.refine,
    )
