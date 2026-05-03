"""Shared building blocks for the model-insights experiments.

`SPECS` is the model catalog. `load_shared_data` hydrates inference data,
training trace, and logistic fits. `build_scenario` wraps a list of
`(ModelDeployment, ReplicaSchedule)` pairs into a scenario dict that mirrors
the IEEE 13 preset (training overlay, replica ramp, tap schedule, OFO
config); pass `ScenarioOverrides` to deviate. `run_scenario` runs one
controller mode and returns voltage + performance stats.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import numpy as np

from openg2g.controller.ofo import (
    LogisticModelStore,
    OFOBatchSizeController,
    OFOConfig,
)
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import Coordinator, SimulationLog
from openg2g.datacenter.config import (
    DatacenterConfig,
    InferenceModelSpec,
    ModelDeployment,
    PowerAugmentationConfig,
    ReplicaSchedule,
    TrainingRun,
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.datacenter.workloads.training import TrainingTrace
from openg2g.grid.config import TapPosition, TapSchedule
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.performance import PerformanceStats, compute_performance_stats
from openg2g.metrics.voltage import VoltageStats, compute_allbus_voltage_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "offline"))
from systems import SYSTEMS, tap

logger = logging.getLogger("model_insights")

# IEEE 13 scenario preset. Do not tweak per experiment.

DT_DC = Fraction(1, 10)
DT_GRID = Fraction(1, 10)
DT_CTRL = Fraction(1)
V_MIN = 0.95
V_MAX = 1.05
TOTAL_DURATION_S = 3600
POWER_AUG = PowerAugmentationConfig(amplitude_scale_range=(0.98, 1.02), noise_fraction=0.005)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# Model specifications catalog. Each entry binds a simulation label to an
# `InferenceModelSpec`. The HuggingFace `model_id` is matched against
# ML.ENERGY benchmark runs; `feasible_batch_sizes` is the OFO-allowed subset
# of `batch_sizes`.


def _spec(
    label: str,
    model_id: str,
    *,
    gpu_model: str,
    task: str,
    precision: str = "bfloat16",
    gpus_per_replica: int = 1,
    tensor_parallel: int | None = None,
    expert_parallel: int = 1,
    itl_deadline_s: float,
    batch_sizes: tuple[int, ...],
    feasible_batch_sizes: tuple[int, ...] | None = None,
) -> InferenceModelSpec:
    return InferenceModelSpec(
        model_label=label,
        model_id=model_id,
        gpu_model=gpu_model,
        task=task,
        precision=precision,
        gpus_per_replica=gpus_per_replica,
        tensor_parallel=tensor_parallel if tensor_parallel is not None else gpus_per_replica,
        expert_parallel=expert_parallel,
        itl_deadline_s=itl_deadline_s,
        batch_sizes=batch_sizes,
        feasible_batch_sizes=feasible_batch_sizes,
    )


SPECS: dict[str, InferenceModelSpec] = {
    # Llama 3.1
    "Llama-3.1-70B": _spec(
        "Llama-3.1-70B",
        "meta-llama/Llama-3.1-70B-Instruct",
        gpu_model="H100",
        task="lm-arena-chat",
        gpus_per_replica=4,
        itl_deadline_s=0.10,
        batch_sizes=(8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024),
        feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512, 1024),
    ),
    "Llama-3.1-405B": _spec(
        "Llama-3.1-405B",
        "meta-llama/Llama-3.1-405B-Instruct-FP8",
        gpu_model="H100",
        task="lm-arena-chat",
        precision="fp8",
        gpus_per_replica=8,
        itl_deadline_s=0.12,
        batch_sizes=(8, 16, 32, 64, 96, 128, 192, 256),
        feasible_batch_sizes=(8, 16, 32, 64, 128, 256),
    ),
    # Qwen 3 dense
    "Qwen3-8B": _spec(
        "Qwen3-8B",
        "Qwen/Qwen3-8B",
        gpu_model="H100",
        task="lm-arena-chat",
        gpus_per_replica=1,
        itl_deadline_s=0.08,
        batch_sizes=(8, 16, 32, 64, 96, 128, 192, 256, 384, 512),
        feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512),
    ),
    "Qwen3-8B-B200": _spec(
        "Qwen3-8B-B200",
        "Qwen/Qwen3-8B",
        gpu_model="B200",
        task="lm-arena-chat",
        gpus_per_replica=1,
        itl_deadline_s=0.08,
        batch_sizes=(8, 16, 32, 64, 128, 256, 512, 1024, 1536),
        feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512, 1024, 1536),
    ),
    "Qwen3-32B-1GPU-H100": _spec(
        "Qwen3-32B-1GPU-H100",
        "Qwen/Qwen3-32B",
        gpu_model="H100",
        task="lm-arena-chat",
        gpus_per_replica=1,
        itl_deadline_s=0.10,
        batch_sizes=(8, 16, 32),
        feasible_batch_sizes=(8, 16, 32),
    ),
    "Qwen3-32B-1GPU-B200": _spec(
        "Qwen3-32B-1GPU-B200",
        "Qwen/Qwen3-32B",
        gpu_model="B200",
        task="lm-arena-chat",
        gpus_per_replica=1,
        itl_deadline_s=0.10,
        batch_sizes=(8, 16, 32, 64, 96, 128, 192, 256, 384, 512),
        feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512),
    ),
    # Qwen 3 30B A3B Instruct (MoE)
    "Qwen3-30B-A3B-Instruct-1GPU": _spec(
        "Qwen3-30B-A3B-Instruct-1GPU",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        gpu_model="H100",
        task="lm-arena-chat",
        gpus_per_replica=1,
        itl_deadline_s=0.10,
        batch_sizes=(8, 16, 32, 64, 96),
        feasible_batch_sizes=(8, 16, 32, 64, 96),
    ),
    "Qwen3-30B-A3B-Instruct-2GPU": _spec(
        "Qwen3-30B-A3B-Instruct-2GPU",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        gpu_model="H100",
        task="lm-arena-chat",
        gpus_per_replica=2,
        itl_deadline_s=0.10,
        batch_sizes=(8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024),
        feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512, 1024),
    ),
    "Qwen3-30B-A3B-Instruct-1GPU-B200": _spec(
        "Qwen3-30B-A3B-Instruct-1GPU-B200",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        gpu_model="B200",
        task="lm-arena-chat",
        gpus_per_replica=1,
        itl_deadline_s=0.10,
        batch_sizes=(8, 16, 32, 64, 96, 128, 192, 256),
        feasible_batch_sizes=(8, 16, 32, 64, 128, 256),
    ),
    # Qwen 3 235B A22B Instruct (MoE)
    "Qwen3-235B-A22B-Instruct-FP8-8GPU": _spec(
        "Qwen3-235B-A22B-Instruct-FP8-8GPU",
        "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        gpu_model="H100",
        task="lm-arena-chat",
        precision="fp8",
        gpus_per_replica=8,
        itl_deadline_s=0.14,
        batch_sizes=(8, 16, 32, 64, 96, 128, 192, 256, 384, 512),
        feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512),
    ),
    "Qwen3-235B-A22B-Instruct-8GPU": _spec(
        "Qwen3-235B-A22B-Instruct-8GPU",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        gpu_model="H100",
        task="lm-arena-chat",
        gpus_per_replica=8,
        itl_deadline_s=0.14,
        batch_sizes=(8, 16, 32, 64, 96, 128, 192, 256),
        feasible_batch_sizes=(8, 16, 32, 64, 128, 256),
    ),
    "Qwen3-235B-A22B-Instruct-8GPU-B200": _spec(
        "Qwen3-235B-A22B-Instruct-8GPU-B200",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        gpu_model="B200",
        task="lm-arena-chat",
        gpus_per_replica=8,
        itl_deadline_s=0.14,
        batch_sizes=(8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 4096),
        feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048),
    ),
    # Qwen 3 235B A22B Thinking (MoE, reasoning)
    "Qwen3-235B-A22B-Thinking-8GPU": _spec(
        "Qwen3-235B-A22B-Thinking-8GPU",
        "Qwen/Qwen3-235B-A22B-Thinking-2507",
        gpu_model="H100",
        task="gpqa",
        gpus_per_replica=8,
        itl_deadline_s=0.14,
        batch_sizes=(8, 16, 32),
        feasible_batch_sizes=(8, 16, 32),
    ),
    "Qwen3-235B-A22B-Thinking-FP8-8GPU": _spec(
        "Qwen3-235B-A22B-Thinking-FP8-8GPU",
        "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
        gpu_model="H100",
        task="gpqa",
        precision="fp8",
        gpus_per_replica=8,
        itl_deadline_s=0.14,
        batch_sizes=(8, 16, 32, 64, 96, 128),
        feasible_batch_sizes=(8, 16, 32, 64, 96, 128),
    ),
    "Qwen3-235B-A22B-Thinking-4GPU-B200": _spec(
        "Qwen3-235B-A22B-Thinking-4GPU-B200",
        "Qwen/Qwen3-235B-A22B-Thinking-2507",
        gpu_model="B200",
        task="gpqa",
        gpus_per_replica=4,
        itl_deadline_s=0.20,
        batch_sizes=(8, 16, 32, 64, 128, 256),
        feasible_batch_sizes=(8, 16, 32, 64, 128, 256),
    ),
    "Qwen3-235B-A22B-Thinking-8GPU-B200": _spec(
        "Qwen3-235B-A22B-Thinking-8GPU-B200",
        "Qwen/Qwen3-235B-A22B-Thinking-2507",
        gpu_model="B200",
        task="gpqa",
        gpus_per_replica=8,
        itl_deadline_s=0.20,
        batch_sizes=(8, 16, 32, 64, 128, 256, 384, 512, 768),
        feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 384, 512, 768),
    ),
    "Qwen3-235B-A22B-Thinking-FP8-4GPU-B200": _spec(
        "Qwen3-235B-A22B-Thinking-FP8-4GPU-B200",
        "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
        gpu_model="B200",
        task="gpqa",
        precision="fp8",
        gpus_per_replica=4,
        itl_deadline_s=0.14,
        batch_sizes=(8, 16, 32, 64, 128, 256, 384, 512),
        feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 384, 512),
    ),
    # GPT-OSS 120B (MoE, mxfp4)
    "GPT-OSS-120B-H100-2GPU": _spec(
        "GPT-OSS-120B-H100-2GPU",
        "openai/gpt-oss-120b",
        gpu_model="H100",
        task="gpqa",
        precision="mxfp4",
        gpus_per_replica=2,
        itl_deadline_s=0.10,
        batch_sizes=(8, 16, 32, 64, 96, 128, 192, 256),
        feasible_batch_sizes=(8, 16, 32, 64, 128, 256),
    ),
    "GPT-OSS-120B-B200-1GPU": _spec(
        "GPT-OSS-120B-B200-1GPU",
        "openai/gpt-oss-120b",
        gpu_model="B200",
        task="gpqa",
        precision="mxfp4",
        gpus_per_replica=1,
        itl_deadline_s=0.10,
        batch_sizes=(8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072),
        feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512, 1024),
    ),
    "GPT-OSS-120B-B200-2GPU": _spec(
        "GPT-OSS-120B-B200-2GPU",
        "openai/gpt-oss-120b",
        gpu_model="B200",
        task="gpqa",
        precision="mxfp4",
        gpus_per_replica=2,
        itl_deadline_s=0.10,
        batch_sizes=(8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096),
        feasible_batch_sizes=(8, 16, 32, 64, 128, 256, 512, 1024),
    ),
}


# Data pipeline — per-spec content-addressed cache (see InferenceModelSpec.cache_hash)

SPECS_CACHE_DIR = _PROJECT_ROOT / "data" / "specs"
TRAINING_TRACE_PATH = _PROJECT_ROOT / "data" / "training_trace.csv"


@dataclass
class SharedData:
    inference_data: InferenceData
    logistic_models: LogisticModelStore
    training_trace: TrainingTrace | None
    data_dir: Path


def load_shared_data(*, specs_used: tuple[InferenceModelSpec, ...] | None = None) -> SharedData:
    """Load / generate the data backing all model-insights experiments.

    Triggers the ML.ENERGY download the first time it runs; subsequent calls
    hit the per-spec content-addressed cache.

    Args:
        specs_used: Subset of specs to materialize. When `None`, loads the
            full SPECS catalog.
    """
    if specs_used is None:
        specs_used = tuple(SPECS.values())
    SPECS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Model-insights specs cache: %s (%d specs)", SPECS_CACHE_DIR, len(specs_used))
    inference_data = InferenceData.ensure(SPECS_CACHE_DIR, specs_used, plot=False, dt_s=float(DT_DC))
    training_trace = TrainingTrace.ensure(TRAINING_TRACE_PATH)
    logistic_models = LogisticModelStore.ensure(SPECS_CACHE_DIR, specs_used, plot=False)
    return SharedData(inference_data, logistic_models, training_trace, SPECS_CACHE_DIR)


# Scenario factory


_IEEE13_OFO_CONFIG = OFOConfig(
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

_IEEE13_TAP_SCHEDULE = TapSchedule(
    (
        (1500, TapPosition(regulators={"creg1a": tap(16), "creg1b": tap(6), "creg1c": tap(17)})),
        (3300, TapPosition(regulators={"creg1a": tap(10), "creg1b": tap(6), "creg1c": tap(10)})),
    )
)


@dataclass
class ScenarioOverrides:
    """Per-experiment deviations from the IEEE 13 preset.

    Each `None` field inherits the preset; set a field to deviate.
    """

    ofo_config: OFOConfig | None = None
    training_n_gpus: int | None = None  # None = 2400, 0 = disable training overlay
    ramp_factor: float | None = None  # None = 0.5 (ramp to 50% of initial replicas)
    enable_tap_schedule: bool = True
    total_gpu_capacity: int | None = None  # None = 7200
    base_kw_per_phase: float | None = None  # None = 500.0
    seed: int = 0


def build_scenario(
    deployments: list[tuple[ModelDeployment, ReplicaSchedule]],
    *,
    shared: SharedData,
    overrides: ScenarioOverrides | None = None,
) -> dict:
    """Build the IEEE 13 scenario around a custom model deployment."""
    overrides = overrides or ScenarioOverrides()
    sys = SYSTEMS["ieee13"]()

    models = tuple(m for m, _ in deployments)
    specs = tuple(m.spec for m in models)
    raw_schedules = {m.spec.model_label: s for m, s in deployments}

    ramp_factor = 0.5 if overrides.ramp_factor is None else overrides.ramp_factor
    replica_schedules: dict[str, ReplicaSchedule] = {}
    for label, sched in raw_schedules.items():
        if len(sched) > 0:  # caller already set a ramp; take it as-is
            replica_schedules[label] = sched
        else:
            target = max(1, int(round(sched.initial * ramp_factor)))
            replica_schedules[label] = sched.ramp_to(target, t_start=2500.0, t_end=3000.0)

    training_n_gpus = 2400 if overrides.training_n_gpus is None else overrides.training_n_gpus
    training = (
        TrainingRun(n_gpus=training_n_gpus, trace=shared.training_trace, target_peak_W_per_gpu=400.0).at(
            t_start=1000.0, t_end=2000.0
        )
        if (shared.training_trace is not None and training_n_gpus > 0)
        else None
    )

    site_inference = shared.inference_data.filter_models(specs)
    workload_kwargs: dict[str, Any] = {
        "inference_data": site_inference,
        "replica_schedules": replica_schedules,
        "initial_batch_sizes": {m.spec.model_label: m.initial_batch_size for m in models},
    }
    if training is not None:
        workload_kwargs["training"] = training

    total_gpu_capacity = 7200 if overrides.total_gpu_capacity is None else overrides.total_gpu_capacity
    base_kw_per_phase = 500.0 if overrides.base_kw_per_phase is None else overrides.base_kw_per_phase

    dc = OfflineDatacenter(
        DatacenterConfig(gpus_per_server=8, base_kw_per_phase=base_kw_per_phase),
        OfflineWorkload(**workload_kwargs),
        name="default",
        dt_s=DT_DC,
        seed=overrides.seed,
        power_augmentation=POWER_AUG,
        total_gpu_capacity=total_gpu_capacity,
    )

    grid = OpenDSSGrid(
        dss_case_dir=sys["dss_case_dir"],
        dss_master_file=sys["dss_master_file"],
        source_pu=sys["source_pu"],
        dt_s=DT_GRID,
        initial_tap_position=sys["initial_taps"],
        exclude_buses=sys["exclude_buses"],
    )
    grid.attach_dc(dc, bus="671")

    return dict(
        datacenters=[dc],
        dc_info={
            dc: {"specs": specs, "initial_batch_sizes": {m.spec.model_label: m.initial_batch_size for m in models}}
        },
        grid=grid,
        ofo_config=overrides.ofo_config or _IEEE13_OFO_CONFIG,
        tap_schedule=_IEEE13_TAP_SCHEDULE if overrides.enable_tap_schedule else None,
        generators=[],
        time_varying_loads=[],
        zones=None,
        exclude_buses=tuple(sys["exclude_buses"]),
    )


# Runner


Mode = Literal["baseline-no-tap", "baseline-tap-change", "ofo-no-tap", "ofo-tap-change"]


@dataclass
class VariantResult:
    scenario_label: str
    mode: Mode
    seed: int
    voltage: VoltageStats
    performance: PerformanceStats
    log: SimulationLog | None = field(default=None, repr=False)


def run_scenario(
    scenario: dict,
    logistic_models: LogisticModelStore,
    *,
    mode: Mode,
    total_duration_s: int = TOTAL_DURATION_S,
) -> VariantResult:
    """Run one mode against a built scenario. Returns voltage + performance stats."""
    datacenters = scenario["datacenters"]
    dc_info = scenario["dc_info"]
    grid = scenario["grid"]
    ofo_config = scenario["ofo_config"]
    tap_schedule = scenario["tap_schedule"] if "tap_change" in mode else None
    exclude_buses = scenario["exclude_buses"]

    controllers: list = [
        TapScheduleController(
            schedule=tap_schedule if tap_schedule is not None else TapSchedule(()),
            dt_s=DT_CTRL,
        ),
    ]
    if mode.startswith("ofo") and ofo_config is not None:
        for dc, info in dc_info.items():
            controllers.append(
                OFOBatchSizeController(
                    info["specs"],
                    datacenter=dc,
                    models=logistic_models,
                    config=ofo_config,
                    dt_s=DT_CTRL,
                    initial_batch_sizes=info["initial_batch_sizes"],
                    grid=grid,
                )
            )

    coord = Coordinator(
        datacenters=datacenters,
        grid=grid,
        controllers=controllers,
        total_duration_s=total_duration_s,
    )
    log = coord.run()

    vstats = compute_allbus_voltage_stats(log.grid_states, v_min=V_MIN, v_max=V_MAX, exclude_buses=exclude_buses)
    itl_deadlines: dict[str, float] = {}
    for info in dc_info.values():
        for ms in info["specs"]:
            itl_deadlines[ms.model_label] = ms.itl_deadline_s
    pstats = compute_performance_stats(log.dc_states, itl_deadline_s_by_model=itl_deadlines)

    return VariantResult(
        scenario_label="",
        mode=mode,
        seed=0,
        voltage=vstats,
        performance=pstats,
        log=log,
    )


# Achievable power range


def compute_achievable_power_range(
    *,
    deployments: list[tuple[ModelDeployment, ReplicaSchedule]],
    logistic_models: LogisticModelStore,
    at_t_s: float = 0.0,
) -> float:
    """Per-variant controller-free power-flexibility metric (MW).

    For each deployment, evaluates the per-replica power fit at the min and
    max feasible batch sizes, scales by the active replica count at
    `at_t_s` (defaults to the initial replica count), and sums across
    models. Returns the difference between the resulting max and min total
    DC power.

    The logistic is fit to `avg_power_watts`, the per-run average power for
    the full `num_gpus` bench configuration — `model.eval(batch)` already
    covers the whole replica, so do NOT multiply by `gpus_per_replica`.
    """
    total_max_w = 0.0
    total_min_w = 0.0
    for deployment, schedule in deployments:
        spec = deployment.spec
        fit = logistic_models.power(spec.model_label)
        feasible = sorted(spec.feasible_batch_sizes)
        b_lo, b_hi = feasible[0], feasible[-1]
        p_lo = float(fit.eval(b_lo))
        p_hi = float(fit.eval(b_hi))
        n = schedule.count_at(at_t_s)
        total_min_w += p_lo * n
        total_max_w += p_hi * n
    return (total_max_w - total_min_w) / 1e6


def compute_matched_peak_replicas(
    spec: InferenceModelSpec,
    target_peak_kw: float,
    logistic_models: LogisticModelStore,
) -> int:
    """Replica count whose peak inference power matches `target_peak_kw`.

    Peak is evaluated at `max(spec.feasible_batch_sizes)` using the logistic
    power fit. Returns `ceil(target_peak_kw * 1e3 / per-replica-peak-w)`,
    floored at 1. Use this to compare variants on a fixed peak DC-footprint
    budget when GPU counts per replica differ.

    Args:
        spec: The model spec — `feasible_batch_sizes` must already be capped
            to batches that meet the SLO; pass the output of
            `restrict_spec_by_deadline` if an SLO is in play.
        target_peak_kw: Target peak inference power in kW.
        logistic_models: Populated `LogisticModelStore` covering
            `spec.model_label`.
    """
    per_rep_w = float(logistic_models.power(spec.model_label).eval(max(spec.feasible_batch_sizes)))
    if per_rep_w <= 0:
        raise ValueError(f"non-positive peak power for {spec.model_label}: {per_rep_w}")
    return max(1, math.ceil(target_peak_kw * 1e3 / per_rep_w))


def compute_pareto_curve(
    deployment: tuple[ModelDeployment, ReplicaSchedule],
    logistic_models: LogisticModelStore,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(batches, power_MW, throughput_Mtps)` for one deployment.

    Power and throughput come from the logistic fits, evaluated at every
    `spec.feasible_batch_sizes` and scaled by the schedule's initial replica
    count.
    """
    spec = deployment[0].spec
    n = deployment[1].initial
    batches = np.array(sorted(spec.feasible_batch_sizes), dtype=int)
    p_fit = logistic_models.power(spec.model_label)
    t_fit = logistic_models.throughput(spec.model_label)
    p_mw = np.array([float(p_fit.eval(int(b))) * n / 1e6 for b in batches])
    t_mtps = np.array([float(t_fit.eval(int(b))) * n / 1e6 for b in batches])
    return batches, p_mw, t_mtps


# Convenience


def deploy(
    label: str,
    num_replicas: int,
    *,
    initial_batch_size: int | None = None,
) -> tuple[ModelDeployment, ReplicaSchedule]:
    """Shorthand for `(ModelDeployment, ReplicaSchedule(initial=...))`.

    `initial_batch_size=None` (the default) starts the scenario at the
    largest feasible batch — i.e., maximum DC power stress, leaving the
    controller full downward range.

    Args:
        label: Key into `SPECS`.
        num_replicas: Initial replica count for the schedule.
        initial_batch_size: Starting batch.
    """
    spec = SPECS[label]
    feasible_batch_sizes = spec.feasible_batch_sizes
    if feasible_batch_sizes is None:
        raise ValueError(f"{label} has no feasible_batch_sizes")
    feasible = sorted(feasible_batch_sizes)
    if initial_batch_size is None:
        initial_batch_size = feasible[-1]
    elif initial_batch_size not in feasible_batch_sizes:
        initial_batch_size = (
            max(b for b in feasible if b <= initial_batch_size) if feasible[0] <= initial_batch_size else feasible[0]
        )
    return (
        ModelDeployment(spec=spec, initial_batch_size=initial_batch_size),
        ReplicaSchedule(initial=num_replicas),
    )


def max_feasible_batch_under_deadline(
    spec: InferenceModelSpec,
    logistic_models: LogisticModelStore,
    deadline_s: float,
) -> int:
    """Largest element of `spec.feasible_batch_sizes` whose predicted ITL
    (from the logistic latency fit) stays at or under `deadline_s`.

    Returns the smallest feasible batch if no batch meets the deadline.
    """
    lat_fit = logistic_models.latency(spec.model_label)
    feasible = sorted(spec.feasible_batch_sizes)
    for b in reversed(feasible):
        if float(lat_fit.eval(b)) <= deadline_s:
            return b
    return feasible[0]


def restrict_spec_by_deadline(
    spec: InferenceModelSpec,
    logistic_models: LogisticModelStore,
    deadline_s: float,
) -> InferenceModelSpec:
    """Return a copy of `spec` with `feasible_batch_sizes` capped so every
    remaining batch yields predicted ITL ≤ `deadline_s`, and with
    `itl_deadline_s` set to `deadline_s`.
    """
    lat_fit = logistic_models.latency(spec.model_label)
    capped = tuple(b for b in sorted(spec.feasible_batch_sizes) if float(lat_fit.eval(b)) <= deadline_s)
    if not capped:
        capped = (min(spec.feasible_batch_sizes),)
    return spec.model_copy(update={"feasible_batch_sizes": capped, "itl_deadline_s": deadline_s})
