"""Hosting capacity analysis: find max GPUs per bus within voltage violation budget.

Feasibility is defined by the *integral* of voltage violation (pu * s)
across all bus-phase pairs: a replica count is feasible when
``integral_violation_pu_s <= max_integral``.  This captures both the
duration and severity of out-of-bounds voltage, giving OFO credit for
reducing deviation magnitude (not just duration).

For each candidate bus and each LLM model, binary-searches on num_replicas
using load steps at 20%, 40%, 60%, 80%, 100% activation.  Each step is
tested independently with its own optimised tap position (per-phase,
range -16 to +16, ``dss_controls=False``).  If *any* step exceeds the
integral threshold after tap optimisation, that replica count is infeasible.

A second pass (OFO + tap change) runs the full staircase with OFO
batch-size control and a per-step tap schedule, reporting higher capacity
when OFO can compensate for voltage stress.

Modes:
  - **1-D** (default): sweeps individual buses independently.
  - **2-D** (``--mode 2d``): sweeps all unordered bus pairs, placing
    identical DCs at both locations simultaneously. The hosting capacity of
    a pair is the min across models. Outputs heatmaps showing which bus
    pairs can support the most GPU capacity.

Usage:
    # 1-D (per-bus)
    python sweep_hosting_capacities.py --system ieee13
    python sweep_hosting_capacities.py --system ieee13 --max-power-mw 15
    # 2-D (bus-pair heatmap)
    python sweep_hosting_capacities.py --system ieee13 --mode 2d
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import logging
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np

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
    InferenceRamp,
    ModelDeployment,
    PowerAugmentationConfig,
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData, MLEnergySource
from openg2g.datacenter.workloads.training import TrainingTraceParams
from openg2g.grid.config import TapPosition, TapSchedule
from openg2g.grid.generator import SyntheticPV
from openg2g.grid.load import SyntheticLoad
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.voltage import (
    VoltageStats,
    compute_allbus_voltage_stats,
    discover_candidate_buses,
    extract_all_voltages,
    find_violations,
)

from systems import SYSTEMS, TAP_STEP

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

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
    return ModelDeployment(spec=MODEL_SPECS[label], num_replicas=num_replicas, initial_batch_size=initial_batch_size)


def load_data_sources(config_path=None):
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    sources_raw = cfg["data_sources"]
    data_sources = {s["model_label"]: MLEnergySource(**s) for s in sources_raw}
    ttp = TrainingTraceParams(**(cfg.get("training_trace_params") or {}))
    blob = json.dumps(
        (sorted(sources_raw, key=lambda s: s["model_label"]), cfg.get("training_trace_params") or {}), sort_keys=True
    ).encode()
    data_dir = _REPO_ROOT / "data" / "offline" / hashlib.sha256(blob).hexdigest()[:16]
    return data_sources, ttp, data_dir


logger = logging.getLogger("hosting_capacity")

# Load-step fractions to test (each gets its own optimal tap position)
LOAD_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
STEP_DURATION_S = 60
STEP_DT = Fraction(1)

# Staircase constants for the OFO full-simulation pass
STAIRCASE_DURATION_S = 360
STAIRCASE_DT = Fraction(1)

MIN_TAP = -16
MAX_TAP = 16
MAX_TAP_ITERATIONS = 20

# Default power ceiling for upper-bound computation (MW)
DEFAULT_MAX_POWER_MW = 10.0


# Hosting capacity config


@dataclass
class HostingConfig:
    """All parameters needed for hosting capacity analysis.

    Replaces the JSON-based SweepConfig with inline experiment definitions.
    """

    # Grid
    dss_case_dir: Path
    dss_master_file: str
    bus_kv: float
    source_pu: float
    initial_taps: TapPosition
    exclude_buses: tuple[str, ...]
    regulator_zones: dict[str, list[str]] | None = None

    # Scenario extras (zeroed for hosting capacity, but kept for grid init)
    # Each entry is a (bus, Generator) or (bus, ExternalLoad) tuple.
    pv_systems: list = field(default_factory=list)
    time_varying_loads: list = field(default_factory=list)

    # DC site template
    connection_type: str = "wye"

    # OFO parameters
    ofo_primal_step_size: float = 0.1
    ofo_voltage_dual_step_size: float = 1.0
    ofo_w_throughput: float = 0.001
    ofo_w_switch: float = 1.0
    ofo_voltage_gradient_scale: float = 1e6
    ofo_latency_dual_step_size: float = 1.0
    ofo_sensitivity_update_interval: int = 3600
    ofo_sensitivity_perturbation_kw: float = 100.0

    # Voltage limits
    v_min: float = V_MIN
    v_max: float = V_MAX


def _tap_pu(step: int) -> float:
    """Convert integer tap step to per-unit tap ratio."""
    return 1.0 + step * TAP_STEP


def _tap_step(tp: TapPosition, reg_name: str) -> int:
    """Extract integer tap step from a TapPosition for a given regulator name."""
    val = tp.regulators.get(reg_name, 1.0)
    return round((val - 1.0) / TAP_STEP)


def _max_replicas_for_power(model_spec: InferenceModelSpec, max_power_mw: float) -> int:
    """Compute upper-bound replicas from a total-power ceiling.

    hosting_capacity_kw = max_gpus * 700 * 2 / 1000
    max_gpus = max_replicas * gpus_per_replica
    => max_replicas = max_power_mw * 1e3 / (gpus_per_replica * 700 * 2 / 1000)
                    = max_power_mw * 1e6 / (gpus_per_replica * 1400)
    """
    return int(max_power_mw * 1e6 / (model_spec.gpus_per_replica * 1400))


# Data classes


@dataclass
class StepResult:
    """Result for a single load-fraction step."""

    fraction: float
    stats: VoltageStats
    taps: TapPosition
    feasible: bool


@dataclass
class PerStepResult:
    """Aggregated result across all load steps."""

    feasible: bool
    worst_stats: VoltageStats
    steps: list[StepResult]


# Base (tap-only) primitives


def run_step_check(
    config: HostingConfig,
    dc_buses: list[str],
    model_spec: InferenceModelSpec,
    num_replicas: int,
    inference_data: InferenceData,
    fraction: float,
    tap_position: TapPosition,
) -> tuple[VoltageStats, list]:
    """Run a steady-state simulation at a fixed activation fraction.

    Args:
        dc_buses: List of bus names. Each bus gets an identical DC with the
            same model and num_replicas.

    Returns (VoltageStats, raw_grid_states).
    """
    total_gpu_power_w = num_replicas * model_spec.gpus_per_replica * 700
    base_kw_per_phase = total_gpu_power_w / 3.0 / 1000.0
    total_gpus = num_replicas * model_spec.gpus_per_replica
    replica_counts = {model_spec.model_label: num_replicas}

    single_inference = InferenceData(
        models=(model_spec,),
        power_templates=inference_data.power_templates,
        itl_fits=inference_data.itl_fits,
    )

    dc_config = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=base_kw_per_phase)

    abs_target = round(fraction * num_replicas)

    dc_list: list[OfflineDatacenter] = []
    for i in range(len(dc_buses)):
        sid = f"site_{i}" if len(dc_buses) > 1 else "dc"
        ramps = InferenceRamp(target=abs_target, model=model_spec.model_label).at(t_start=0, t_end=0)
        workload = OfflineWorkload(
            inference_data=single_inference, replica_counts=replica_counts, inference_ramps=ramps
        )
        dc_list.append(
            OfflineDatacenter(
                dc_config,
                workload,
                name=sid,
                dt_s=STEP_DT,
                seed=0,
                power_augmentation=PowerAugmentationConfig(),
                total_gpu_capacity=total_gpus,
            )
        )

    grid = OpenDSSGrid(
        dss_case_dir=config.dss_case_dir,
        dss_master_file=config.dss_master_file,
        source_pu=config.source_pu,
        dt_s=STEP_DT,
        dss_controls=False,
        initial_tap_position=tap_position,
    )
    for dc, bus in zip(dc_list, dc_buses, strict=False):
        grid.attach_dc(dc, bus=bus, connection_type=config.connection_type, power_factor=dc_config.power_factor)
    for bus, gen in config.pv_systems:
        grid.attach_generator(gen, bus=bus)
    for bus, ld in config.time_varying_loads:
        grid.attach_load(ld, bus=bus)

    coord = Coordinator(
        datacenters=dc_list,
        grid=grid,
        controllers=[],
        total_duration_s=STEP_DURATION_S,
    )

    log = coord.run()

    stats = compute_allbus_voltage_stats(
        log.grid_states,
        v_min=config.v_min,
        v_max=config.v_max,
        exclude_buses=config.exclude_buses,
    )
    return stats, log.grid_states


def _reg_phase(reg_name: str) -> str:
    """Map a regulator name to its phase (a/b/c) based on its suffix."""
    rn = reg_name.lower()
    if rn.endswith("a"):
        return "a"
    if rn.endswith("b"):
        return "b"
    return "c"


def _reg_bank(reg_name: str, zone_keys: list[str]) -> str:
    """Map a regulator name to its bank prefix using zone config keys.

    E.g. "creg1a" with zone_keys=["creg1","creg2"] -> "creg1".
    Falls back to the first zone key if no match.
    """
    rn = reg_name.lower()
    for key in sorted(zone_keys, key=len, reverse=True):  # longest match first
        if rn.startswith(key.lower()):
            return key
    return zone_keys[0] if zone_keys else "default"


def _build_zone_bus_sets(
    regulator_zones: dict[str, list[str]] | None,
) -> dict[str, set[str]] | None:
    """Convert zone config to lowercase bus sets for fast lookup."""
    if regulator_zones is None:
        return None
    return {bank: {b.lower() for b in buses} for bank, buses in regulator_zones.items()}


def optimize_taps_for_step(
    config: HostingConfig,
    dc_buses: list[str],
    model_spec: InferenceModelSpec,
    num_replicas: int,
    inference_data: InferenceData,
    fraction: float,
    initial_taps: TapPosition,
) -> tuple[VoltageStats, TapPosition]:
    """Find optimal tap position for a single load-fraction step.

    When ``config.regulator_zones`` is set, each regulator bank is adjusted
    independently based only on violations within its downstream zone.
    Otherwise all regulators are adjusted together by phase.

    Returns (best VoltageStats, best TapPosition).
    """
    v_min = config.v_min
    v_max = config.v_max

    reg_names = list(initial_taps.regulators.keys())
    tap_steps = {rn: _tap_step(initial_taps, rn) for rn in reg_names}

    zone_bus_sets = _build_zone_bus_sets(config.regulator_zones)
    zone_keys = list(config.regulator_zones.keys()) if config.regulator_zones else []

    best_stats = None
    best_tap_pos = initial_taps

    for _iteration in range(MAX_TAP_ITERATIONS):
        tap_pos = TapPosition(regulators={rn: _tap_pu(tap_steps[rn]) for rn in reg_names})

        stats, grid_states = run_step_check(
            config,
            dc_buses,
            model_spec,
            num_replicas,
            inference_data,
            fraction,
            tap_pos,
        )

        if best_stats is None or stats.integral_violation_pu_s < best_stats.integral_violation_pu_s:
            best_stats = stats
            best_tap_pos = tap_pos

        if stats.integral_violation_pu_s == 0.0:
            return stats, tap_pos

        voltages = extract_all_voltages(grid_states, exclude_buses=config.exclude_buses)
        violations = find_violations(voltages, v_min=v_min, v_max=v_max)

        if zone_bus_sets and zone_keys:
            # Zone-aware: adjust each bank independently
            has_conflict = False
            bank_adjust: dict[str, dict[str, int]] = {bank: {"a": 0, "b": 0, "c": 0} for bank in zone_keys}

            for bank, bus_set in zone_bus_sets.items():
                zone_viols = [v for v in violations if v[0].lower() in bus_set]
                for phase in ("a", "b", "c"):
                    under = any(v[1] == phase and v[2] == "under" for v in zone_viols)
                    over = any(v[1] == phase and v[2] == "over" for v in zone_viols)
                    if under and over:
                        has_conflict = True
                    elif under:
                        bank_adjust[bank][phase] = 1
                    elif over:
                        bank_adjust[bank][phase] = -1

            if has_conflict:
                break

            prev_steps = dict(tap_steps)
            for rn in reg_names:
                bank = _reg_bank(rn, zone_keys)
                phase = _reg_phase(rn)
                adj = bank_adjust.get(bank, {}).get(phase, 0)
                tap_steps[rn] = max(MIN_TAP, min(MAX_TAP, tap_steps[rn] + adj))
        else:
            # Legacy: all regulators share phase-level adjustment
            phase_adjust = {"a": 0, "b": 0, "c": 0}
            has_conflict = False
            for phase in ("a", "b", "c"):
                under = any(v[1] == phase and v[2] == "under" for v in violations)
                over = any(v[1] == phase and v[2] == "over" for v in violations)
                if under and over:
                    has_conflict = True
                elif under:
                    phase_adjust[phase] = 1
                elif over:
                    phase_adjust[phase] = -1

            if has_conflict:
                break

            prev_steps = dict(tap_steps)
            for rn in reg_names:
                phase = _reg_phase(rn)
                tap_steps[rn] = max(MIN_TAP, min(MAX_TAP, tap_steps[rn] + phase_adjust[phase]))

        if tap_steps == prev_steps:
            break

    return best_stats, best_tap_pos


def check_all_steps(
    config: HostingConfig,
    dc_buses: list[str],
    model_spec: InferenceModelSpec,
    num_replicas: int,
    inference_data: InferenceData,
    initial_taps: TapPosition,
    integral_threshold: float = 0.0,
) -> PerStepResult:
    """Optimise taps independently for each load step and check feasibility.

    A step is feasible when its integral voltage violation (pu * s) is at
    or below ``integral_threshold``.

    Returns a PerStepResult: feasible=True only if every step passes.
    """
    steps: list[StepResult] = []
    worst_vmin = 1.0
    worst_vmax = 1.0
    total_violation_s = 0.0
    total_integral = 0.0
    all_feasible = True

    for frac in LOAD_FRACTIONS:
        stats, taps = optimize_taps_for_step(
            config,
            dc_buses,
            model_spec,
            num_replicas,
            inference_data,
            frac,
            initial_taps,
        )
        feasible = stats.integral_violation_pu_s <= integral_threshold
        steps.append(StepResult(fraction=frac, stats=stats, taps=taps, feasible=feasible))

        worst_vmin = min(worst_vmin, stats.worst_vmin)
        worst_vmax = max(worst_vmax, stats.worst_vmax)
        total_violation_s += stats.violation_time_s
        total_integral += stats.integral_violation_pu_s
        if not feasible:
            all_feasible = False

    worst_stats = VoltageStats(
        worst_vmin=worst_vmin,
        worst_vmax=worst_vmax,
        violation_time_s=total_violation_s,
        integral_violation_pu_s=total_integral,
    )
    return PerStepResult(feasible=all_feasible, worst_stats=worst_stats, steps=steps)


# OFO + tap-change staircase


def _build_staircase_ramps(model_label: str, num_replicas: int):
    """Build InferenceRampSchedule for the staircase activation pattern.

    Args:
        model_label: Model label string (e.g. ``"Llama-3.1-8B"``).
        num_replicas: Total number of replicas; fractions are converted to
            absolute counts via ``round(fraction * num_replicas)``.
    """
    ramps = InferenceRamp(target=round(0.0 * num_replicas), model=model_label).at(t_start=0, t_end=0)
    ramps = ramps | InferenceRamp(target=round(0.2 * num_replicas), model=model_label).at(t_start=0, t_end=60)
    ramps = ramps | InferenceRamp(target=round(0.4 * num_replicas), model=model_label).at(t_start=60, t_end=120)
    ramps = ramps | InferenceRamp(target=round(0.6 * num_replicas), model=model_label).at(t_start=120, t_end=180)
    ramps = ramps | InferenceRamp(target=round(0.8 * num_replicas), model=model_label).at(t_start=180, t_end=240)
    ramps = ramps | InferenceRamp(target=round(1.0 * num_replicas), model=model_label).at(t_start=240, t_end=300)
    return ramps


def _build_tap_schedule_from_steps(steps: list[StepResult]) -> TapSchedule:
    """Build a TapSchedule that switches taps at each staircase boundary.

    step fraction -> staircase time:
      20% -> t=0, 40% -> t=60, 60% -> t=120, 80% -> t=180, 100% -> t=240
    """
    frac_to_time = {0.2: 0.0, 0.4: 60.0, 0.6: 120.0, 0.8: 180.0, 1.0: 240.0}
    entries: list[tuple[float, TapPosition]] = []
    for sr in steps:
        t = frac_to_time.get(sr.fraction)
        if t is not None:
            entries.append((t, sr.taps))
    return TapSchedule(tuple(entries))


def run_ofo_staircase(
    config: HostingConfig,
    dc_buses: list[str],
    model_spec: InferenceModelSpec,
    num_replicas: int,
    inference_data: InferenceData,
    logistic_models: LogisticModelStore,
    tap_schedule: TapSchedule,
    initial_taps: TapPosition,
    voltage_dual_step_size: float | None = None,
) -> VoltageStats:
    """Run the full 360 s staircase with OFO + tap schedule.

    Args:
        dc_buses: List of bus names. Each bus gets an identical DC with OFO.
    """
    total_gpu_power_w = num_replicas * model_spec.gpus_per_replica * 700
    base_kw_per_phase = total_gpu_power_w / 3.0 / 1000.0
    total_gpus = num_replicas * model_spec.gpus_per_replica
    replica_counts = {model_spec.model_label: num_replicas}

    single_inference = InferenceData(
        models=(model_spec,),
        power_templates=inference_data.power_templates,
        itl_fits=inference_data.itl_fits,
    )

    dc_config = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=base_kw_per_phase)

    vdss = voltage_dual_step_size if voltage_dual_step_size is not None else config.ofo_voltage_dual_step_size
    ofo_config = OFOConfig(
        primal_step_size=config.ofo_primal_step_size,
        w_throughput=config.ofo_w_throughput,
        w_switch=config.ofo_w_switch,
        voltage_gradient_scale=config.ofo_voltage_gradient_scale,
        v_min=config.v_min,
        v_max=config.v_max,
        voltage_dual_step_size=vdss,
        latency_dual_step_size=config.ofo_latency_dual_step_size,
        sensitivity_update_interval=config.ofo_sensitivity_update_interval,
        sensitivity_perturbation_kw=config.ofo_sensitivity_perturbation_kw,
    )

    tap_ctrl = TapScheduleController(schedule=tap_schedule, dt_s=STAIRCASE_DT)

    dc_list: list[OfflineDatacenter] = []
    controllers = [tap_ctrl]
    for i in range(len(dc_buses)):
        sid = f"site_{i}" if len(dc_buses) > 1 else "dc"
        ramps = _build_staircase_ramps(model_spec.model_label, num_replicas)
        workload = OfflineWorkload(
            inference_data=single_inference, replica_counts=replica_counts, inference_ramps=ramps
        )
        dc = OfflineDatacenter(
            dc_config,
            workload,
            name=sid,
            dt_s=STAIRCASE_DT,
            seed=0,
            power_augmentation=PowerAugmentationConfig(),
            total_gpu_capacity=total_gpus,
        )
        dc_list.append(dc)
        controllers.append(
            OFOBatchSizeController(
                (model_spec,),
                datacenter=dc,
                models=logistic_models,
                config=ofo_config,
                dt_s=STAIRCASE_DT,
            )
        )

    grid = OpenDSSGrid(
        dss_case_dir=config.dss_case_dir,
        dss_master_file=config.dss_master_file,
        source_pu=config.source_pu,
        dt_s=STAIRCASE_DT,
        dss_controls=False,
        initial_tap_position=initial_taps,
    )
    for dc, bus in zip(dc_list, dc_buses, strict=False):
        grid.attach_dc(dc, bus=bus, connection_type=config.connection_type, power_factor=dc_config.power_factor)
    for bus, gen in config.pv_systems:
        grid.attach_generator(gen, bus=bus)
    for bus, ld in config.time_varying_loads:
        grid.attach_load(ld, bus=bus)

    coord = Coordinator(
        datacenters=dc_list,
        grid=grid,
        controllers=controllers,
        total_duration_s=STAIRCASE_DURATION_S,
    )

    log = coord.run()

    return compute_allbus_voltage_stats(
        log.grid_states,
        v_min=config.v_min,
        v_max=config.v_max,
        exclude_buses=config.exclude_buses,
    )


# Binary search


def find_max_replicas(
    config: HostingConfig,
    dc_buses: list[str],
    model_spec: InferenceModelSpec,
    inference_data: InferenceData,
    max_replicas_upper: int,
    initial_taps: TapPosition,
    search_tol: int = 5,
    integral_threshold: float = 0.0,
) -> tuple[int, PerStepResult, int]:
    """Binary search for the maximum num_replicas (tap-only, no OFO).

    Args:
        dc_buses: List of bus names. Each bus gets an identical DC.
        integral_threshold: Maximum allowed integral violation (pu * s)
            per load step.

    Handles pre-existing overvoltage: DC load pulls voltage down, so adding
    replicas can *help* at overvoltage buses.  If the initial midpoint fails,
    we probe a small number of replicas before giving up.

    Returns (max_replicas, per_step_result, search_iterations).
    """
    lo = 0
    hi = max_replicas_upper
    best_replicas = 0
    best_result: PerStepResult | None = None
    iterations = 0
    bus_label = "+".join(dc_buses)

    def _try(n: int) -> PerStepResult:
        nonlocal iterations
        iterations += 1
        logger.info(
            "    [%s @ %s] iter %d: trying num_replicas=%d (lo=%d, hi=%d)",
            model_spec.model_label,
            bus_label,
            iterations,
            n,
            lo,
            hi,
        )
        result = check_all_steps(
            config,
            dc_buses,
            model_spec,
            n,
            inference_data,
            initial_taps,
            integral_threshold=integral_threshold,
        )
        for sr in result.steps:
            status = "OK" if sr.feasible else "VIOL"
            taps_str = ",".join(f"{_tap_step(sr.taps, rn):+d}" for rn in sr.taps.regulators)
            logger.info(
                "        %3.0f%%: %s vmin=%.4f vmax=%.4f taps=(%s)",
                sr.fraction * 100,
                status,
                sr.stats.worst_vmin,
                sr.stats.worst_vmax,
                taps_str,
            )
        return result

    # First probe: try a small number to check if DC load helps with
    # pre-existing overvoltage (the load pulls voltage down).
    probe_n = max(1, search_tol)
    probe_result = _try(probe_n)
    if not probe_result.feasible:
        # Even a small DC load can't be hosted (after tap optimisation)
        logger.info("      -> VIOLATION even at %d replicas, capacity = 0", probe_n)
        return (
            0,
            PerStepResult(
                feasible=True,
                worst_stats=VoltageStats(
                    worst_vmin=1.0, worst_vmax=1.0, violation_time_s=0.0, integral_violation_pu_s=0.0
                ),
                steps=[],
            ),
            iterations,
        )
    else:
        logger.info("      -> ALL OK at probe")
        best_replicas = probe_n
        best_result = probe_result
        lo = probe_n

    # Standard binary search
    while hi - lo > search_tol:
        mid = (lo + hi) // 2
        if mid <= lo:
            break
        result = _try(mid)
        if result.feasible:
            logger.info("      -> ALL OK")
            best_replicas = mid
            best_result = result
            lo = mid
        else:
            failed = [sr for sr in result.steps if not sr.feasible]
            logger.info("      -> VIOLATION at %s", ", ".join(f"{sr.fraction * 100:.0f}%" for sr in failed))
            hi = mid

    if best_result is None:
        best_result = PerStepResult(
            feasible=True,
            worst_stats=VoltageStats(worst_vmin=1.0, worst_vmax=1.0, violation_time_s=0.0, integral_violation_pu_s=0.0),
            steps=[],
        )

    return best_replicas, best_result, iterations


def find_max_replicas_ofo(
    config: HostingConfig,
    dc_buses: list[str],
    model_spec: InferenceModelSpec,
    inference_data: InferenceData,
    logistic_models: LogisticModelStore,
    max_replicas_upper: int,
    initial_taps: TapPosition,
    base_max_replicas: int,
    base_steps: list[StepResult],
    search_tol: int = 5,
    integral_threshold: float = 0.0,
) -> tuple[int, VoltageStats, int]:
    """Binary search for max num_replicas with OFO + tap schedule.

    Args:
        dc_buses: List of bus names. Each bus gets an identical DC with OFO.
        integral_threshold: Maximum allowed integral violation (pu * s)
            for the full staircase simulation.

    Starts from the base (tap-only) result. For each candidate, first
    optimises per-step taps, then runs the full 360 s staircase with OFO
    and the resulting tap schedule.

    Returns (max_replicas, stats_at_max, search_iterations).
    """
    lo = base_max_replicas
    hi = max_replicas_upper
    best_replicas = base_max_replicas
    best_stats = VoltageStats(worst_vmin=1.0, worst_vmax=1.0, violation_time_s=0.0, integral_violation_pu_s=0.0)
    iterations = 0

    # Use base taps as initial tap schedule; if lo == hi already, nothing to do
    list(base_steps)

    while hi - lo > search_tol:
        mid = (lo + hi) // 2
        if mid <= base_max_replicas:
            lo = base_max_replicas
            break
        iterations += 1
        logger.info(
            "    [OFO %s @ %s] iter %d: trying num_replicas=%d (lo=%d, hi=%d)",
            model_spec.model_label,
            "+".join(dc_buses),
            iterations,
            mid,
            lo,
            hi,
        )

        # Find per-step optimal taps for this replica count
        step_result = check_all_steps(
            config,
            dc_buses,
            model_spec,
            mid,
            inference_data,
            initial_taps,
        )
        tap_schedule = _build_tap_schedule_from_steps(step_result.steps)
        initial_step_taps = step_result.steps[0].taps if step_result.steps else initial_taps

        # Run full staircase with OFO + tap schedule
        stats = run_ofo_staircase(
            config,
            dc_buses,
            model_spec,
            mid,
            inference_data,
            logistic_models,
            tap_schedule=tap_schedule,
            initial_taps=initial_step_taps,
        )

        if stats.integral_violation_pu_s > integral_threshold:
            logger.info(
                "      -> VIOLATION: integral=%.4f pu*s, vmin=%.4f, vmax=%.4f",
                stats.integral_violation_pu_s,
                stats.worst_vmin,
                stats.worst_vmax,
            )
            hi = mid
        else:
            logger.info("      -> OK: vmin=%.4f, vmax=%.4f", stats.worst_vmin, stats.worst_vmax)
            best_replicas = mid
            best_stats = stats
            lo = mid

    return best_replicas, best_stats, iterations


# Plotting


def _plot_hosting_capacity_combined(
    wo_ofo_summary: list[dict],
    w_ofo_summary: list[dict],
    save_path: Path,
    title: str = "Datacenter Hosting Capacity by Bus",
) -> None:
    """Grouped bar chart comparing hosting capacity w/ and w/o OFO."""
    import matplotlib.pyplot as plt

    buses = [r["dc_bus"] for r in wo_ofo_summary]
    wo_mw = [r["hosting_capacity_MW"] for r in wo_ofo_summary]
    w_mw = [r["hosting_capacity_MW"] for r in w_ofo_summary]

    x = np.arange(len(buses))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(buses) * 1.5), 6))
    ax.bar(x - width / 2, wo_mw, width, label="Tap Only (W/O OFO)", color="#4C72B0")
    ax.bar(x + width / 2, w_mw, width, label="OFO + Tap Change", color="#DD8452")
    ax.set_xlabel("DC Bus")
    ax.set_ylabel("Hosting Capacity (MW)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(buses, rotation=45, ha="right")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved to: %s", save_path)


def _plot_integral_threshold_comparison(
    all_results: dict[float, tuple[list[dict], list[dict]]],
    save_path: Path,
    system: str,
) -> None:
    """Line plot comparing hosting capacity across integral thresholds.

    Args:
        all_results: {threshold_pu_s: (wo_ofo_summary_rows, w_ofo_summary_rows)}
    """
    import matplotlib.pyplot as plt

    thresholds = sorted(all_results.keys())
    buses = [r["dc_bus"] for r in next(iter(all_results.values()))[0]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(buses), 1))

    threshold_labels = [f"{t:.2f}" for t in thresholds]

    for i, bus in enumerate(buses):
        wo_vals = [all_results[t][0][i]["hosting_capacity_MW"] for t in thresholds]
        w_vals = [all_results[t][1][i]["hosting_capacity_MW"] for t in thresholds]
        color = cmap(i)
        ax1.plot(threshold_labels, wo_vals, "o-", color=color, label=bus)
        ax2.plot(threshold_labels, w_vals, "s--", color=color, label=bus)

    ax1.set_title("Tap Only (W/O OFO)")
    ax1.set_xlabel("Max Integral Violation (pu*s)")
    ax1.set_ylabel("Hosting Capacity (MW)")
    ax1.legend(title="Bus", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_title("OFO + Tap Change")
    ax2.set_xlabel("Max Integral Violation (pu*s)")
    ax2.legend(title="Bus", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Hosting Capacity vs. Integral Threshold — {system}", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison plot saved to: %s", save_path)


# CSV helpers


def _save_csv(rows: list[dict], path: Path) -> None:
    """Save a list of dicts as CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved: %s", path)


def _log_summary(summary_rows: list[dict], label: str) -> None:
    logger.info("")
    logger.info("=" * 80)
    logger.info("HOSTING CAPACITY SUMMARY — %s", label)
    logger.info("=" * 80)
    logger.info("%-10s %15s %15s %20s %12s", "Bus", "Capacity (kW)", "Capacity (MW)", "Limiting Model", "Max GPUs")
    logger.info("-" * 80)
    for row in summary_rows:
        logger.info(
            "%-10s %15.1f %15.3f %20s %12d",
            row["dc_bus"],
            row["hosting_capacity_kw"],
            row["hosting_capacity_MW"],
            row["limiting_model"],
            row["limiting_max_gpus"],
        )


# Main


def _run_pass1(
    config: HostingConfig,
    candidate_buses: list[str],
    all_models: tuple,
    inference_data: InferenceData,
    initial_taps: TapPosition,
    max_power_mw: float,
    search_tol: int,
    integral_threshold: float,
) -> tuple[list[dict], list[dict], dict[tuple[str, str], tuple[int, list[StepResult]]]]:
    """Pass 1: tap-only binary search. Returns (rows, summary, base_results)."""
    base_rows: list[dict] = []
    base_summary: list[dict] = []
    base_results: dict[tuple[str, str], tuple[int, list[StepResult]]] = {}

    for dc_bus in candidate_buses:
        logger.info("")
        logger.info("Bus %s:", dc_bus)
        bus_min_kw = float("inf")
        bus_lim_model = ""
        bus_lim_gpus = 0

        for ms in all_models:
            upper = _max_replicas_for_power(ms, max_power_mw)
            logger.info("  Model %s (gpus_per_replica=%d, upper=%d):", ms.model_label, ms.gpus_per_replica, upper)

            try:
                max_reps, result, iters = find_max_replicas(
                    config,
                    [dc_bus],
                    ms,
                    inference_data,
                    max_replicas_upper=upper,
                    initial_taps=initial_taps,
                    search_tol=search_tol,
                    integral_threshold=integral_threshold,
                )
            except Exception as e:
                logger.error("    FAILED: %s", e)
                max_reps, iters = 0, 0
                result = PerStepResult(feasible=True, worst_stats=VoltageStats(1.0, 1.0, 0.0, 0.0), steps=[])

            base_results[(dc_bus, ms.model_label)] = (max_reps, result.steps)

            max_gpus = max_reps * ms.gpus_per_replica
            hosting_kw = max_gpus * 700 * 2 / 1000
            hosting_mw = hosting_kw / 1000

            logger.info(
                "  -> %s: max_replicas=%d, max_gpus=%d, hosting=%.3f MW (%d iters)",
                ms.model_label,
                max_reps,
                max_gpus,
                hosting_mw,
                iters,
            )

            row: dict = {
                "dc_bus": dc_bus,
                "model": ms.model_label,
                "max_replicas": max_reps,
                "max_gpus": max_gpus,
                "hosting_capacity_kw": hosting_kw,
                "hosting_capacity_MW": hosting_mw,
                "search_iterations": iters,
            }
            if result.steps:
                for sr in result.steps:
                    pct = int(sr.fraction * 100)
                    for rn in sr.taps.regulators:
                        row[f"tap_{rn}_{pct}pct"] = _tap_step(sr.taps, rn)
            base_rows.append(row)

            if hosting_kw < bus_min_kw:
                bus_min_kw = hosting_kw
                bus_lim_model = ms.model_label
                bus_lim_gpus = max_gpus

        if bus_min_kw == float("inf"):
            bus_min_kw = 0.0
        base_summary.append(
            {
                "dc_bus": dc_bus,
                "hosting_capacity_kw": bus_min_kw,
                "hosting_capacity_MW": bus_min_kw / 1000,
                "limiting_model": bus_lim_model,
                "limiting_max_gpus": bus_lim_gpus,
            }
        )

    return base_rows, base_summary, base_results


def _run_pass2(
    config: HostingConfig,
    candidate_buses: list[str],
    all_models: tuple,
    inference_data: InferenceData,
    logistic_models: LogisticModelStore,
    initial_taps: TapPosition,
    max_power_mw: float,
    search_tol: int,
    integral_threshold: float,
    base_results: dict[tuple[str, str], tuple[int, list[StepResult]]],
) -> tuple[list[dict], list[dict]]:
    """Pass 2: OFO + tap change binary search. Returns (rows, summary)."""
    ofo_rows: list[dict] = []
    ofo_summary: list[dict] = []

    for dc_bus in candidate_buses:
        logger.info("")
        logger.info("Bus %s:", dc_bus)
        bus_min_kw = float("inf")
        bus_lim_model = ""
        bus_lim_gpus = 0

        for ms in all_models:
            upper = _max_replicas_for_power(ms, max_power_mw)
            base_max, base_steps = base_results[(dc_bus, ms.model_label)]
            logger.info("  Model %s (base=%d, upper=%d):", ms.model_label, base_max, upper)

            if base_max >= upper - search_tol:
                logger.info("    already at power ceiling, skipping OFO search")
                max_reps = base_max
                iters = 0
            else:
                try:
                    max_reps, _stats, iters = find_max_replicas_ofo(
                        config,
                        [dc_bus],
                        ms,
                        inference_data,
                        logistic_models,
                        max_replicas_upper=upper,
                        initial_taps=initial_taps,
                        base_max_replicas=base_max,
                        base_steps=base_steps,
                        search_tol=search_tol,
                        integral_threshold=integral_threshold,
                    )
                except Exception as e:
                    logger.error("    FAILED: %s", e)
                    max_reps, iters = base_max, 0

            max_gpus = max_reps * ms.gpus_per_replica
            hosting_kw = max_gpus * 700 * 2 / 1000
            hosting_mw = hosting_kw / 1000

            logger.info(
                "  -> %s: max_replicas=%d, max_gpus=%d, hosting=%.3f MW (%d OFO iters)",
                ms.model_label,
                max_reps,
                max_gpus,
                hosting_mw,
                iters,
            )

            ofo_rows.append(
                {
                    "dc_bus": dc_bus,
                    "model": ms.model_label,
                    "max_replicas": max_reps,
                    "max_gpus": max_gpus,
                    "hosting_capacity_kw": hosting_kw,
                    "hosting_capacity_MW": hosting_mw,
                    "search_iterations": iters,
                }
            )

            if hosting_kw < bus_min_kw:
                bus_min_kw = hosting_kw
                bus_lim_model = ms.model_label
                bus_lim_gpus = max_gpus

        if bus_min_kw == float("inf"):
            bus_min_kw = 0.0
        ofo_summary.append(
            {
                "dc_bus": dc_bus,
                "hosting_capacity_kw": bus_min_kw,
                "hosting_capacity_MW": bus_min_kw / 1000,
                "limiting_model": bus_lim_model,
                "limiting_max_gpus": bus_lim_gpus,
            }
        )

    return ofo_rows, ofo_summary


# Per-system experiment definitions


def _hosting_config_ieee13(sys: dict) -> tuple[HostingConfig, tuple[ModelDeployment, ...]]:
    """IEEE 13-bus hosting capacity config."""
    models = (
        deploy("Llama-3.1-8B", 720),
        deploy("Llama-3.1-70B", 180),
        deploy("Llama-3.1-405B", 90),
        deploy("Qwen3-30B-A3B", 480),
        deploy("Qwen3-235B-A22B", 210),
    )
    config = HostingConfig(
        dss_case_dir=sys["dss_case_dir"],
        dss_master_file=sys["dss_master_file"],
        bus_kv=sys["bus_kv"],
        source_pu=sys["source_pu"],
        initial_taps=sys["initial_taps"],
        exclude_buses=sys["exclude_buses"],
        regulator_zones=sys.get("regulator_zones"),
        # PV/loads zeroed for hosting capacity analysis
        pv_systems=[("675", SyntheticPV(peak_kw=0.0))],
        time_varying_loads=[("680", SyntheticLoad(peak_kw=0.0))],
        # OFO parameters
        ofo_primal_step_size=0.1,
        ofo_voltage_dual_step_size=1.0,
    )
    return config, models


def _hosting_config_ieee34(sys: dict) -> tuple[HostingConfig, tuple[ModelDeployment, ...]]:
    """IEEE 34-bus hosting capacity config."""
    models = (
        deploy("Llama-3.1-8B", 720),
        deploy("Llama-3.1-70B", 180),
        deploy("Llama-3.1-405B", 90),
        deploy("Qwen3-30B-A3B", 480),
        deploy("Qwen3-235B-A22B", 210),
    )
    config = HostingConfig(
        dss_case_dir=sys["dss_case_dir"],
        dss_master_file=sys["dss_master_file"],
        bus_kv=sys["bus_kv"],
        source_pu=sys["source_pu"],
        initial_taps=sys["initial_taps"],
        exclude_buses=sys["exclude_buses"],
        regulator_zones=sys.get("regulator_zones"),
        # PV/loads zeroed for hosting capacity analysis
        pv_systems=[
            ("848", SyntheticPV(peak_kw=0.0, site_idx=0)),
            ("830", SyntheticPV(peak_kw=0.0, site_idx=1)),
        ],
        time_varying_loads=[
            ("860", SyntheticLoad(peak_kw=0.0, site_idx=0)),
            ("844", SyntheticLoad(peak_kw=0.0, site_idx=1)),
            ("840", SyntheticLoad(peak_kw=0.0, site_idx=2)),
            ("858", SyntheticLoad(peak_kw=0.0, site_idx=3)),
            ("854", SyntheticLoad(peak_kw=0.0, site_idx=4)),
        ],
        # OFO parameters
        ofo_primal_step_size=0.05,
        ofo_voltage_dual_step_size=20.0,
        ofo_sensitivity_perturbation_kw=10.0,
    )
    return config, models


def _hosting_config_ieee123(sys: dict) -> tuple[HostingConfig, tuple[ModelDeployment, ...]]:
    """IEEE 123-bus hosting capacity config."""
    models = (
        deploy("Llama-3.1-8B", 720),
        deploy("Llama-3.1-70B", 180),
        deploy("Llama-3.1-405B", 90),
        deploy("Qwen3-30B-A3B", 480),
        deploy("Qwen3-235B-A22B", 210),
    )
    config = HostingConfig(
        dss_case_dir=sys["dss_case_dir"],
        dss_master_file=sys["dss_master_file"],
        bus_kv=sys["bus_kv"],
        source_pu=sys["source_pu"],
        initial_taps=sys["initial_taps"],
        exclude_buses=sys["exclude_buses"],
        regulator_zones=sys.get("regulator_zones"),
        # PV/loads zeroed for hosting capacity analysis
        pv_systems=[
            ("1", SyntheticPV(peak_kw=0.0, site_idx=0)),
            ("48", SyntheticPV(peak_kw=0.0, site_idx=1)),
            ("99", SyntheticPV(peak_kw=0.0, site_idx=2)),
        ],
        # OFO parameters
        ofo_primal_step_size=0.05,
        ofo_voltage_dual_step_size=0.3,
        ofo_sensitivity_perturbation_kw=10.0,
    )
    return config, models


_HOSTING_CONFIGS = {
    "ieee13": _hosting_config_ieee13,
    "ieee34": _hosting_config_ieee34,
    "ieee123": _hosting_config_ieee123,
}


def main(
    *,
    system: str,
    buses: list[str] | None = None,
    max_power_mw: float = DEFAULT_MAX_POWER_MW,
    search_tol: int = 5,
    max_integrals: list[float] | None = None,
) -> None:
    if max_integrals is None:
        max_integrals = [1.0]

    sys = SYSTEMS[system]()
    config, all_deployments = _HOSTING_CONFIGS[system](sys)
    all_specs = tuple(d.spec for d in all_deployments)

    # Load shared data
    data_sources, _training_trace_params, data_dir = load_data_sources()

    save_dir = Path(__file__).resolve().parent / "outputs" / system / "hosting_capacity_1d"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading inference data...")
    inference_data = InferenceData.ensure(
        data_dir,
        all_specs,
        data_sources,
        plot=False,
        dt_s=float(DT_DC),
    )

    logger.info("Loading logistic fits...")
    logistic_models = LogisticModelStore.ensure(
        data_dir / "logistic_fits.csv",
        all_specs,
        data_sources,
        plot=False,
    )

    # Discover candidate buses
    if buses:
        candidate_buses = buses
        logger.info("Using user-specified buses: %s", candidate_buses)
    else:
        logger.info("Discovering candidate 3-phase buses at %.2f kV...", config.bus_kv)
        candidate_buses = discover_candidate_buses(
            config.dss_case_dir,
            config.dss_master_file,
            config.bus_kv,
            exclude=set(config.exclude_buses),
        )
        logger.info("Found %d candidate buses: %s", len(candidate_buses), candidate_buses)

    if not candidate_buses:
        logger.error("No candidate buses found!")
        return

    initial_taps = config.initial_taps

    logger.info("")
    logger.info("=" * 70)
    logger.info("HOSTING CAPACITY ANALYSIS")
    logger.info("Models: %s", [m.model_label for m in all_specs])
    logger.info("Buses: %s", candidate_buses)
    logger.info("Max integral violations (pu*s): %s", max_integrals)
    logger.info("PV peak_kw: 0.0 (disabled), Time-varying load peak_kw: 0.0 (disabled)")
    logger.info("Max power ceiling: %.1f MW, Search tolerance: %d replicas", max_power_mw, search_tol)
    logger.info("Tap range: %+d to %+d, dss_controls=False", MIN_TAP, MAX_TAP)
    logger.info("=" * 70)

    # Collect results across all integral thresholds for comparison plot
    all_threshold_results: dict[float, tuple[list[dict], list[dict]]] = {}

    for mi in max_integrals:
        tag = f"int{mi:.2f}"

        logger.info("")
        logger.info("=" * 70)
        logger.info("MAX INTEGRAL VIOLATION: %.4f pu*s", mi)
        logger.info("=" * 70)

        # Pass 1: tap-only
        logger.info("")
        logger.info("=== PASS 1: Tap-only (no OFO) — %s ===", tag)
        t0 = time.time()

        base_rows, base_summary, base_results = _run_pass1(
            config,
            candidate_buses,
            all_specs,
            inference_data,
            initial_taps,
            max_power_mw,
            search_tol,
            mi,
        )
        logger.info("Pass 1 complete in %.1f s", time.time() - t0)

        _save_csv(base_rows, save_dir / f"results_{system}_hosting_capacity_WO_OFO_{tag}.csv")
        _save_csv(base_summary, save_dir / f"summary_{system}_hosting_capacity_WO_OFO_{tag}.csv")
        _log_summary(base_summary, f"Tap Only ({tag})")

        # Pass 2: OFO + tap change
        logger.info("")
        logger.info("=== PASS 2: OFO + Tap Change — %s ===", tag)
        t0 = time.time()

        ofo_rows, ofo_summary = _run_pass2(
            config,
            candidate_buses,
            all_specs,
            inference_data,
            logistic_models,
            initial_taps,
            max_power_mw,
            search_tol,
            mi,
            base_results,
        )
        logger.info("Pass 2 complete in %.1f s", time.time() - t0)

        _save_csv(ofo_rows, save_dir / f"results_{system}_hosting_capacity_W_OFO_{tag}.csv")
        _save_csv(ofo_summary, save_dir / f"summary_{system}_hosting_capacity_W_OFO_{tag}.csv")
        _log_summary(ofo_summary, f"OFO + Tap Change ({tag})")

        # Combined plot for this threshold
        _plot_hosting_capacity_combined(
            base_summary,
            ofo_summary,
            save_dir / f"hosting_capacity_{system}_{tag}.png",
            title=f"Hosting Capacity — max integral {mi:.2f} pu*s",
        )

        all_threshold_results[mi] = (base_summary, ofo_summary)

    # Comparison plot across all thresholds
    if len(max_integrals) > 1:
        _plot_integral_threshold_comparison(
            all_threshold_results,
            save_dir / f"hosting_capacity_{system}_comparison.png",
            system=system,
        )

    logger.info("")
    logger.info("All outputs saved to: %s", save_dir)


# 2-D hosting capacity (bus-pair sweep)


def _run_pass1_2d(
    config: HostingConfig,
    candidate_buses: list[str],
    all_models: tuple,
    inference_data: InferenceData,
    initial_taps: TapPosition,
    max_power_mw: float,
    search_tol: int,
    integral_threshold: float,
) -> tuple[list[dict], dict[str, float]]:
    """Pass 1 (tap-only) for all bus pairs. Returns (rows, {pair_key: capacity_MW})."""
    rows: list[dict] = []
    pair_capacity: dict[str, float] = {}

    pairs = list(itertools.combinations(candidate_buses, 2))
    total = len(pairs)

    for idx, (bus_a, bus_b) in enumerate(pairs, 1):
        logger.info("")
        logger.info("[%d/%d] Pair (%s, %s):", idx, total, bus_a, bus_b)
        pair_min_kw = float("inf")
        pair_lim_model = ""

        for ms in all_models:
            upper = _max_replicas_for_power(ms, max_power_mw)
            logger.info("  Model %s (upper=%d):", ms.model_label, upper)

            try:
                max_reps, _result, iters = find_max_replicas(
                    config,
                    [bus_a, bus_b],
                    ms,
                    inference_data,
                    max_replicas_upper=upper,
                    initial_taps=initial_taps,
                    search_tol=search_tol,
                    integral_threshold=integral_threshold,
                )
            except Exception as e:
                logger.error("    FAILED: %s", e)
                max_reps, iters = 0, 0

            max_gpus = max_reps * ms.gpus_per_replica
            hosting_kw = max_gpus * 700 * 2 / 1000
            hosting_mw = hosting_kw / 1000
            logger.info(
                "  -> %s: max_replicas=%d, hosting=%.3f MW (%d iters)", ms.model_label, max_reps, hosting_mw, iters
            )

            rows.append(
                {
                    "bus_a": bus_a,
                    "bus_b": bus_b,
                    "model": ms.model_label,
                    "max_replicas": max_reps,
                    "max_gpus": max_gpus,
                    "hosting_capacity_kw": hosting_kw,
                    "hosting_capacity_MW": hosting_mw,
                }
            )

            if hosting_kw < pair_min_kw:
                pair_min_kw = hosting_kw
                pair_lim_model = ms.model_label

        cap_mw = pair_min_kw / 1000 if pair_min_kw < float("inf") else 0.0
        pair_capacity[f"{bus_a},{bus_b}"] = cap_mw
        logger.info("  Pair capacity: %.3f MW (limited by %s)", cap_mw, pair_lim_model)

    return rows, pair_capacity


def _run_pass2_2d(
    config: HostingConfig,
    candidate_buses: list[str],
    all_models: tuple,
    inference_data: InferenceData,
    logistic_models: LogisticModelStore,
    initial_taps: TapPosition,
    max_power_mw: float,
    search_tol: int,
    integral_threshold: float,
    base_pair_rows: list[dict],
) -> tuple[list[dict], dict[str, float]]:
    """Pass 2 (OFO + tap) for all bus pairs."""
    # Index base results by (bus_a, bus_b, model)
    base_index: dict[tuple[str, str, str], int] = {}
    for r in base_pair_rows:
        base_index[(r["bus_a"], r["bus_b"], r["model"])] = r["max_replicas"]

    rows: list[dict] = []
    pair_capacity: dict[str, float] = {}

    pairs = list(itertools.combinations(candidate_buses, 2))
    total = len(pairs)

    for idx, (bus_a, bus_b) in enumerate(pairs, 1):
        logger.info("")
        logger.info("[%d/%d] OFO Pair (%s, %s):", idx, total, bus_a, bus_b)
        pair_min_kw = float("inf")

        for ms in all_models:
            upper = _max_replicas_for_power(ms, max_power_mw)
            base_max = base_index.get((bus_a, bus_b, ms.model_label), 0)
            logger.info("  Model %s (base=%d, upper=%d):", ms.model_label, base_max, upper)

            if base_max >= upper - search_tol:
                logger.info("    already at power ceiling, skipping OFO search")
                max_reps = base_max
            else:
                try:
                    # Get base steps for tap schedule
                    base_step_result = check_all_steps(
                        config,
                        [bus_a, bus_b],
                        ms,
                        base_max,
                        inference_data,
                        initial_taps,
                    )
                    max_reps, _stats, _iters = find_max_replicas_ofo(
                        config,
                        [bus_a, bus_b],
                        ms,
                        inference_data,
                        logistic_models,
                        max_replicas_upper=upper,
                        initial_taps=initial_taps,
                        base_max_replicas=base_max,
                        base_steps=base_step_result.steps,
                        search_tol=search_tol,
                        integral_threshold=integral_threshold,
                    )
                except Exception as e:
                    logger.error("    FAILED: %s", e)
                    max_reps = base_max

            max_gpus = max_reps * ms.gpus_per_replica
            hosting_kw = max_gpus * 700 * 2 / 1000
            hosting_mw = hosting_kw / 1000
            logger.info("  -> %s: max_replicas=%d, hosting=%.3f MW", ms.model_label, max_reps, hosting_mw)

            rows.append(
                {
                    "bus_a": bus_a,
                    "bus_b": bus_b,
                    "model": ms.model_label,
                    "max_replicas": max_reps,
                    "max_gpus": max_gpus,
                    "hosting_capacity_kw": hosting_kw,
                    "hosting_capacity_MW": hosting_mw,
                }
            )

            if hosting_kw < pair_min_kw:
                pair_min_kw = hosting_kw

        cap_mw = pair_min_kw / 1000 if pair_min_kw < float("inf") else 0.0
        pair_capacity[f"{bus_a},{bus_b}"] = cap_mw

    return rows, pair_capacity


def _plot_hosting_heatmap(
    pair_capacity: dict[str, float],
    candidate_buses: list[str],
    save_path: Path,
    title: str = "Pairwise Hosting Capacity (MW)",
) -> None:
    """Plot a symmetric heatmap of hosting capacity for all bus pairs."""
    import matplotlib.pyplot as plt

    n = len(candidate_buses)
    grid = np.full((n, n), np.nan)
    bus_idx = {b: i for i, b in enumerate(candidate_buses)}

    for key, cap_mw in pair_capacity.items():
        ba, bb = key.split(",")
        i, j = bus_idx[ba], bus_idx[bb]
        grid[i, j] = cap_mw
        grid[j, i] = cap_mw  # symmetric

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(7, n * 0.7)), dpi=150)
    im = ax.imshow(grid, origin="lower", aspect="auto", extent=[-0.5, n - 0.5, -0.5, n - 0.5])
    ax.set_xticks(range(n))
    ax.set_xticklabels(candidate_buses, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(candidate_buses, fontsize=8)
    ax.set_xlabel("Bus A", fontsize=11)
    ax.set_ylabel("Bus B", fontsize=11)
    ax.set_title(title, fontsize=13)
    fig.colorbar(im, ax=ax, label="Hosting Capacity (MW)", shrink=0.8)

    # Annotate cells
    median = np.nanmedian(grid)
    for i in range(n):
        for j in range(n):
            val = grid[i, j]
            if np.isfinite(val):
                ax.text(
                    j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if val > median else "black"
                )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap saved to: %s", save_path)


def main_2d(
    *,
    system: str,
    buses: list[str] | None = None,
    max_power_mw: float = DEFAULT_MAX_POWER_MW,
    search_tol: int = 5,
    max_integrals: list[float] | None = None,
) -> None:
    """2-D hosting capacity: sweep all bus pairs with identical DCs at both."""
    if max_integrals is None:
        max_integrals = [1.0]

    sys = SYSTEMS[system]()
    config, all_deployments = _HOSTING_CONFIGS[system](sys)
    all_specs = tuple(d.spec for d in all_deployments)

    # Load shared data
    data_sources, _training_trace_params, data_dir = load_data_sources()

    save_dir = Path(__file__).resolve().parent / "outputs" / system / "hosting_capacity_2d"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading inference data...")
    inference_data = InferenceData.ensure(
        data_dir,
        all_specs,
        data_sources,
        plot=False,
        dt_s=float(DT_DC),
    )

    logger.info("Loading logistic fits...")
    logistic_models = LogisticModelStore.ensure(
        data_dir / "logistic_fits.csv",
        all_specs,
        data_sources,
        plot=False,
    )

    if buses:
        candidate_buses = buses
    else:
        logger.info("Discovering candidate 3-phase buses at %.2f kV...", config.bus_kv)
        candidate_buses = discover_candidate_buses(
            config.dss_case_dir,
            config.dss_master_file,
            config.bus_kv,
            exclude=set(config.exclude_buses),
        )
        logger.info("Found %d candidate buses: %s", len(candidate_buses), candidate_buses)

    if len(candidate_buses) < 2:
        logger.error("Need at least 2 candidate buses for 2-D hosting capacity!")
        return

    initial_taps = config.initial_taps

    n_pairs = len(candidate_buses) * (len(candidate_buses) - 1) // 2
    logger.info("")
    logger.info("=" * 70)
    logger.info("2-D HOSTING CAPACITY ANALYSIS")
    logger.info("Models: %s", [m.model_label for m in all_specs])
    logger.info("Buses: %s (%d pairs)", candidate_buses, n_pairs)
    logger.info("Max power ceiling: %.1f MW", max_power_mw)
    logger.info("=" * 70)

    for mi in max_integrals:
        tag = f"int{mi:.2f}"

        logger.info("")
        logger.info("=== PASS 1 (tap-only) -- %s ===", tag)
        t0 = time.time()
        base_rows, base_cap = _run_pass1_2d(
            config,
            candidate_buses,
            all_specs,
            inference_data,
            initial_taps,
            max_power_mw,
            search_tol,
            mi,
        )
        logger.info("Pass 1 complete in %.1f s", time.time() - t0)

        _save_csv(base_rows, save_dir / f"results_2d_{system}_WO_OFO_{tag}.csv")
        _plot_hosting_heatmap(
            base_cap,
            candidate_buses,
            save_dir / f"heatmap_2d_{system}_WO_OFO_{tag}.png",
            title=f"Pairwise Hosting Capacity - Tap Only ({tag})",
        )

        logger.info("")
        logger.info("=== PASS 2 (OFO + tap) -- %s ===", tag)
        t0 = time.time()
        ofo_rows, ofo_cap = _run_pass2_2d(
            config,
            candidate_buses,
            all_specs,
            inference_data,
            logistic_models,
            initial_taps,
            max_power_mw,
            search_tol,
            mi,
            base_rows,
        )
        logger.info("Pass 2 complete in %.1f s", time.time() - t0)

        _save_csv(ofo_rows, save_dir / f"results_2d_{system}_W_OFO_{tag}.csv")
        _plot_hosting_heatmap(
            ofo_cap,
            candidate_buses,
            save_dir / f"heatmap_2d_{system}_W_OFO_{tag}.png",
            title=f"Pairwise Hosting Capacity - OFO + Tap ({tag})",
        )

    logger.info("")
    logger.info("All outputs saved to: %s", save_dir)


if __name__ == "__main__":
    import tyro

    @dataclass
    class Args:
        system: str = "ieee13"
        """System name (ieee13, ieee34, ieee123)."""
        mode: Literal["1d", "2d"] = "1d"
        """Analysis mode: '1d' sweeps individual buses, '2d' sweeps all bus pairs
        with identical DCs at both locations."""
        buses: str | None = None
        """Comma-separated list of buses to test (overrides auto-discovery)."""
        max_power_mw: float = DEFAULT_MAX_POWER_MW
        """Power ceiling (MW) for binary-search upper bound."""
        search_tol: int = 5
        """Binary search tolerance in replicas."""
        max_integrals: str = "1.0"
        """Comma-separated max integral violation thresholds in pu*s (e.g. '0.5,1.0,2.0')."""
        log_level: str = "INFO"
        """Logging verbosity (DEBUG, INFO, WARNING)."""

    args = tyro.cli(Args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openg2g.controller.ofo").setLevel(logging.WARNING)

    bus_list = [b.strip() for b in args.buses.split(",")] if args.buses else None
    mi_list = [float(v.strip()) for v in args.max_integrals.split(",")]

    if args.mode == "2d":
        main_2d(
            system=args.system,
            buses=bus_list,
            max_power_mw=args.max_power_mw,
            search_tol=args.search_tol,
            max_integrals=mi_list,
        )
    else:
        main(
            system=args.system,
            buses=bus_list,
            max_power_mw=args.max_power_mw,
            search_tol=args.search_tol,
            max_integrals=mi_list,
        )
