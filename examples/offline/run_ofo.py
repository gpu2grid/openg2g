"""Baseline + OFO simulation for any IEEE test system.

Constructs datacenters, grid, and controllers programmatically for each
system, then runs the specified simulation case. All experiment parameters
(models, DC sites, controller tuning, workload scenarios) are defined
inline.

Modes:
    baseline-no-tap       Baseline (no OFO) without tap schedule.
    baseline-tap-change   Baseline (no OFO) with tap schedule.
    ofo-no-tap            OFO batch-size control without tap schedule.
    ofo-tap-change        OFO batch-size control with tap schedule.
    all                   All four cases above.

Usage:
    python run_ofo.py --system ieee13 --mode baseline-no-tap
    python run_ofo.py --system ieee13 --mode ofo-no-tap
    python run_ofo.py --system ieee34 --mode all
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from openg2g.grid.generator import Generator, SyntheticPV
from openg2g.grid.load import ExternalLoad, SyntheticLoad
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.performance import PerformanceStats, compute_performance_stats
from openg2g.metrics.voltage import VoltageStats, compute_allbus_voltage_stats

from plotting import (
    plot_allbus_voltages_per_phase,
    plot_model_timeseries_4panel,
    plot_power_and_itl_2panel,
    plot_zone_voltage_envelope,
)
from systems import SYSTEMS, TAP_STEP, tap

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MAX_BUSES_FOR_INDIVIDUAL_LINES = 30

logger = logging.getLogger("run_ofo")

# Simulation defaults

DT_DC = Fraction(1, 10)
DT_GRID = Fraction(1, 10)
DT_CTRL = Fraction(1)
V_MIN, V_MAX = 0.95, 1.05
TOTAL_DURATION_S = 3600
POWER_AUG = PowerAugmentationConfig(amplitude_scale_range=(0.98, 1.02), noise_fraction=0.005)

# Model specifications

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
MODEL_SPECS: dict[str, InferenceModelSpec] = {s.model_label: s for s in ALL_MODEL_SPECS}


def deploy(label: str, num_replicas: int, initial_batch_size: int = 128) -> tuple[ModelDeployment, ReplicaSchedule]:
    """Shorthand: `deploy("Llama-3.1-8B", 720, 128)` -> `(ModelDeployment, ReplicaSchedule)`."""
    return (
        ModelDeployment(spec=MODEL_SPECS[label], initial_batch_size=initial_batch_size),
        ReplicaSchedule(initial=num_replicas),
    )


def unpack_deployments(
    *deployments: tuple[ModelDeployment, ReplicaSchedule],
) -> tuple[tuple[ModelDeployment, ...], dict[str, ReplicaSchedule]]:
    """Split `deploy()` output pairs into a models tuple and schedules dict."""
    models = tuple(m for m, _ in deployments)
    schedules = {m.spec.model_label: s for m, s in deployments}
    return models, schedules


# Data pipeline


def load_data_sources(
    config_path: Path | None = None,
) -> tuple[dict[str, MLEnergySource], Path]:
    """Load ML.ENERGY data sources from `data_sources.json`.

    Returns:
        (data_sources, data_dir) where *data_dir* is a hash-based cache
        directory under `<repo_root>/data/offline`.
    """
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


# Plotting helpers


def _plot_voltage_envelope(
    grid_states,
    time_s,
    save_dir,
    *,
    v_min=0.95,
    v_max=1.05,
    exclude_buses=(),
    title="Voltage Envelope",
) -> None:
    """Plot min/max voltage envelope across all monitored buses."""
    drop = {b.lower() for b in exclude_buses}
    v_min_arr = np.full(len(grid_states), np.inf)
    v_max_arr = np.full(len(grid_states), -np.inf)

    for t_idx, gs in enumerate(grid_states):
        for bus in gs.voltages.buses():
            if bus.lower() in drop:
                continue
            pv = gs.voltages[bus]
            for v in (pv.a, pv.b, pv.c):
                if not math.isnan(v):
                    v_min_arr[t_idx] = min(v_min_arr[t_idx], v)
                    v_max_arr[t_idx] = max(v_max_arr[t_idx], v)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(time_s, v_min_arr, v_max_arr, alpha=0.3, color="steelblue")
    ax.plot(time_s, v_min_arr, color="steelblue", linewidth=0.8, label="Vmin")
    ax.plot(time_s, v_max_arr, color="coral", linewidth=0.8, label="Vmax")
    ax.axhline(v_min, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(v_max, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (pu)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_dir / "voltage_envelope.png", dpi=150, bbox_inches="tight", metadata={"Creation Time": None})
    plt.close(fig)


def _plot_tap_positions(grid_states, time_s, save_dir, *, title="Regulator Tap Positions"):
    """Plot regulator tap positions over time as step lines."""
    if not grid_states or grid_states[0].tap_positions is None:
        return
    if not grid_states[0].tap_positions.regulators:
        return

    t_min = np.asarray(time_s) / 60.0
    reg_names = list(grid_states[0].tap_positions.regulators.keys())

    tap_cmap = plt.colormaps.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(12, 4))

    for r_idx, reg_name in enumerate(reg_names):
        tap_pu = np.array(
            [gs.tap_positions.regulators.get(reg_name, 1.0) if gs.tap_positions else 1.0 for gs in grid_states]
        )
        tap_int = np.round((tap_pu - 1.0) / TAP_STEP).astype(int)
        ax.step(t_min, tap_int, where="post", color=tap_cmap(r_idx % 10), linewidth=2.0, alpha=0.8, label=reg_name)

    ax.set_xlabel("Time (minutes)", fontsize=11)
    ax.set_ylabel("Tap Position (steps)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, ncol=max(1, len(reg_names) // 2), loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "tap_positions.png", dpi=150, bbox_inches="tight", metadata={"Creation Time": None})
    plt.close(fig)


def _plot_pv_load_profiles(grid_states, time_s, generators, time_varying_loads, save_dir):
    """Plot PV output and time-varying load curves over the simulation."""
    if not grid_states:
        return

    n_gen = len(generators)
    n_load = len(time_varying_loads)
    n_panels = n_gen + n_load
    if n_panels == 0:
        return

    t_arr = np.linspace(0, float(time_s[-1]), 4000)
    t_min = t_arr / 60.0

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for i, (bus, gen) in enumerate(generators):
        kw_arr = np.array([gen.power_kw(t) for t in t_arr])
        axes[i].plot(t_min, kw_arr, "orange", linewidth=1.5)
        axes[i].set_ylabel("kW/phase", fontsize=12)
        axes[i].set_title(f"Generator @ bus {bus}", fontsize=13)
        axes[i].axhline(0, color="gray", linewidth=0.5)
        axes[i].set_ylim(bottom=-2)
        axes[i].grid(True, alpha=0.3)

    for i, (bus, load) in enumerate(time_varying_loads):
        kw_arr = np.array([load.power_kw(t) for t in t_arr])
        ax = axes[n_gen + i]
        ax.plot(t_min, kw_arr, "steelblue", linewidth=1.5)
        ax.set_ylabel("kW/phase", fontsize=12)
        ax.set_title(f"Load @ bus {bus}", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Time (min)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_dir / "pv_load_profiles.png", dpi=150, bbox_inches="tight", metadata={"Creation Time": None})
    plt.close(fig)


def _plot_comparison(results: dict[str, VoltageStats], save_dir: Path, system: str) -> None:
    modes = list(results.keys())
    n = len(modes)
    fig, axes = plt.subplots(1, 3, figsize=(5 * n, 5))

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:n]

    vals = [results[m].violation_time_s for m in modes]
    axes[0].bar(modes, vals, color=colors)
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Violation Time")
    axes[0].tick_params(axis="x", rotation=30)

    vals = [results[m].worst_vmin for m in modes]
    axes[1].bar(modes, vals, color=colors)
    axes[1].axhline(0.95, color="red", linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Per Unit")
    axes[1].set_title("Worst Vmin")
    axes[1].tick_params(axis="x", rotation=30)

    vals = [results[m].integral_violation_pu_s for m in modes]
    axes[2].bar(modes, vals, color=colors)
    axes[2].set_ylabel("pu * s")
    axes[2].set_title("Integral Violation")
    axes[2].tick_params(axis="x", rotation=30)

    fig.suptitle(f"{system.upper()}: Baseline vs OFO", fontsize=14)
    fig.tight_layout()
    fig.savefig(
        save_dir / f"results_{system}_comparison.png", dpi=150, bbox_inches="tight", metadata={"Creation Time": None}
    )
    plt.close(fig)
    logger.info("Comparison plot saved.")


# Per-system experiment definitions
#
# Each function takes the loaded data pipeline results and returns a dict with
# all constructed objects needed for simulation:
#   datacenters      - list[OfflineDatacenter]
#   dc_info          - dict[OfflineDatacenter, _DCInfo] per-DC metadata
#   grid             - OpenDSSGrid (with DCs, generators, loads attached)
#   ofo_config       - OFOConfig
#   tap_schedule     - TapSchedule | None
#   generators       - list[(bus, Generator)]
#   time_varying_loads - list[(bus, ExternalLoad)]
#   zones            - dict | None
#   exclude_buses    - tuple[str, ...]


class _DCInfo:
    """Per-datacenter metadata for controller construction."""

    __slots__ = ("initial_batch_sizes", "specs")

    def __init__(self, models: tuple[ModelDeployment, ...]) -> None:
        self.specs = tuple(m.spec for m in models)
        self.initial_batch_sizes = {m.spec.model_label: m.initial_batch_size for m in models}


def _build_dc(
    name: str,
    *,
    models: tuple[ModelDeployment, ...],
    replica_schedules: dict[str, ReplicaSchedule],
    inference_data: InferenceData,
    base_kw_per_phase: float,
    total_gpu_capacity: int,
    seed: int = 0,
    training: TrainingRun | None = None,
) -> OfflineDatacenter:
    """Create a single OfflineDatacenter from model deployments."""
    specs = tuple(m.spec for m in models)
    site_inference = inference_data.filter_models(specs)
    batch_sizes = {m.spec.model_label: m.initial_batch_size for m in models}

    workload_kwargs: dict[str, Any] = {
        "inference_data": site_inference,
        "replica_schedules": replica_schedules,
        "initial_batch_sizes": batch_sizes,
    }
    if training is not None:
        workload_kwargs["training"] = training

    return OfflineDatacenter(
        DatacenterConfig(gpus_per_server=8, base_kw_per_phase=base_kw_per_phase),
        OfflineWorkload(**workload_kwargs),
        name=name,
        dt_s=DT_DC,
        seed=seed,
        power_augmentation=POWER_AUG,
        total_gpu_capacity=total_gpu_capacity,
    )


def setup_ieee13(inference_data, training_trace, logistic_models):
    """IEEE 13-bus: single DC at bus 671 with training overlay."""
    sys = SYSTEMS["ieee13"]()

    models, _ = unpack_deployments(
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

    training = (
        TrainingRun(n_gpus=2400, trace=training_trace, target_peak_W_per_gpu=400.0).at(t_start=1000.0, t_end=2000.0)
        if training_trace is not None
        else None
    )

    dc = _build_dc(
        "default",
        models=models,
        inference_data=inference_data,
        base_kw_per_phase=500.0,
        total_gpu_capacity=7200,
        seed=0,
        replica_schedules=replica_schedules,
        training=training,
    )

    generators = [("675", SyntheticPV(peak_kw=10.0))]
    time_varying_loads = [("680", SyntheticLoad(peak_kw=10.0))]

    grid = OpenDSSGrid(
        dss_case_dir=sys["dss_case_dir"],
        dss_master_file=sys["dss_master_file"],
        source_pu=sys["source_pu"],
        dt_s=DT_GRID,
        initial_tap_position=sys["initial_taps"],
        exclude_buses=sys["exclude_buses"],
    )
    grid.attach_dc(dc, bus="671")
    for bus, gen in generators:
        grid.attach_generator(gen, bus=bus)
    for bus, load in time_varying_loads:
        grid.attach_load(load, bus=bus)

    return dict(
        datacenters=[dc],
        dc_info={dc: _DCInfo(models)},
        grid=grid,
        ofo_config=OFOConfig(
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
        ),
        tap_schedule=TapSchedule(
            (
                (1500, TapPosition(regulators={"creg1a": tap(16), "creg1b": tap(6), "creg1c": tap(17)})),
                (3300, TapPosition(regulators={"creg1a": tap(10), "creg1b": tap(6), "creg1c": tap(10)})),
            )
        ),
        generators=generators,
        time_varying_loads=time_varying_loads,
        zones=None,
        exclude_buses=tuple(sys["exclude_buses"]),
    )


def setup_ieee34(inference_data, training_trace, logistic_models):
    """IEEE 34-bus: two DC sites (upstream + downstream)."""
    sys = SYSTEMS["ieee34"]()

    upstream_models, upstream_schedules = unpack_deployments(
        deploy("Llama-3.1-8B", 720),
        deploy("Llama-3.1-70B", 180),
        deploy("Llama-3.1-405B", 90),
    )
    downstream_models, downstream_schedules = unpack_deployments(
        deploy("Qwen3-30B-A3B", 480),
        deploy("Qwen3-235B-A22B", 210),
    )

    dc_upstream = _build_dc(
        "upstream",
        models=upstream_models,
        replica_schedules=upstream_schedules,
        inference_data=inference_data,
        base_kw_per_phase=120.0,
        total_gpu_capacity=2400,
        seed=0,
    )
    dc_downstream = _build_dc(
        "downstream",
        models=downstream_models,
        replica_schedules=downstream_schedules,
        inference_data=inference_data,
        base_kw_per_phase=80.0,
        total_gpu_capacity=2880,
        seed=42,
    )

    generators = [
        ("848", SyntheticPV(peak_kw=130.0)),
        ("830", SyntheticPV(peak_kw=65.0, site_idx=1)),
    ]
    time_varying_loads = [
        ("860", SyntheticLoad(peak_kw=80.0)),
        ("844", SyntheticLoad(peak_kw=120.0, site_idx=1)),
        ("840", SyntheticLoad(peak_kw=60.0, site_idx=2)),
        ("858", SyntheticLoad(peak_kw=50.0, site_idx=3)),
        ("854", SyntheticLoad(peak_kw=40.0, site_idx=4)),
    ]

    grid = OpenDSSGrid(
        dss_case_dir=sys["dss_case_dir"],
        dss_master_file=sys["dss_master_file"],
        source_pu=sys["source_pu"],
        dt_s=DT_GRID,
        initial_tap_position=sys["initial_taps"],
        exclude_buses=sys["exclude_buses"],
    )
    grid.attach_dc(dc_upstream, bus="850")
    grid.attach_dc(dc_downstream, bus="834")
    for bus, gen in generators:
        grid.attach_generator(gen, bus=bus)
    for bus, load in time_varying_loads:
        grid.attach_load(load, bus=bus)

    return dict(
        datacenters=[dc_upstream, dc_downstream],
        dc_info={
            dc_upstream: _DCInfo(upstream_models),
            dc_downstream: _DCInfo(downstream_models),
        },
        grid=grid,
        ofo_config=OFOConfig(
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
        ),
        tap_schedule=TapSchedule(
            ((1800, TapPosition(regulators={"creg2a": tap(10), "creg2b": tap(10), "creg2c": tap(10)})),)
        ),
        generators=generators,
        time_varying_loads=time_varying_loads,
        zones=None,
        exclude_buses=tuple(sys["exclude_buses"]),
    )


def setup_ieee123(inference_data, training_trace, logistic_models):
    """IEEE 123-bus: four DC zones with per-site ramps."""
    sys = SYSTEMS["ieee123"]()

    z1_models, _ = unpack_deployments(deploy("Llama-3.1-8B", 120))
    z2_models, _ = unpack_deployments(deploy("Qwen3-30B-A3B", 80))
    z3_models, _ = unpack_deployments(deploy("Llama-3.1-70B", 30), deploy("Llama-3.1-405B", 35))
    z4_models, _ = unpack_deployments(deploy("Qwen3-235B-A22B", 55))

    dc_z1 = _build_dc(
        "z1_sw",
        models=z1_models,
        replica_schedules={
            "Llama-3.1-8B": ReplicaSchedule(initial=120).ramp_to(180, t_start=500, t_end=1000),
        },
        inference_data=inference_data,
        base_kw_per_phase=310.0,
        total_gpu_capacity=180,
        seed=0,
    )
    dc_z2 = _build_dc(
        "z2_nw",
        models=z2_models,
        replica_schedules={
            "Qwen3-30B-A3B": ReplicaSchedule(initial=80).ramp_to(104, t_start=1500, t_end=2500),
        },
        inference_data=inference_data,
        base_kw_per_phase=265.0,
        total_gpu_capacity=208,
        seed=17,
    )
    dc_z3 = _build_dc(
        "z3_se",
        models=z3_models,
        replica_schedules={
            "Llama-3.1-70B": ReplicaSchedule(initial=30).ramp_to(45, t_start=700, t_end=1100),
            "Llama-3.1-405B": ReplicaSchedule(initial=35).ramp_to(52, t_start=700, t_end=1100),
        },
        inference_data=inference_data,
        base_kw_per_phase=295.0,
        total_gpu_capacity=600,
        seed=34,
    )
    dc_z4 = _build_dc(
        "z4_ne",
        models=z4_models,
        replica_schedules={
            "Qwen3-235B-A22B": ReplicaSchedule(initial=55).ramp_to(27, t_start=2000, t_end=2500),
        },
        inference_data=inference_data,
        base_kw_per_phase=325.0,
        total_gpu_capacity=440,
        seed=51,
    )

    generators = [
        ("1", SyntheticPV(peak_kw=333.3)),
        ("48", SyntheticPV(peak_kw=333.3, site_idx=1)),
        ("99", SyntheticPV(peak_kw=333.3, site_idx=2)),
    ]
    time_varying_loads: list[tuple[str, ExternalLoad]] = []

    grid = OpenDSSGrid(
        dss_case_dir=sys["dss_case_dir"],
        dss_master_file=sys["dss_master_file"],
        source_pu=sys["source_pu"],
        dt_s=DT_GRID,
        initial_tap_position=sys["initial_taps"],
        exclude_buses=sys["exclude_buses"],
    )
    grid.attach_dc(dc_z1, bus="8")
    grid.attach_dc(dc_z2, bus="23")
    grid.attach_dc(dc_z3, bus="60")
    grid.attach_dc(dc_z4, bus="105")
    for bus, gen in generators:
        grid.attach_generator(gen, bus=bus)

    return dict(
        datacenters=[dc_z1, dc_z2, dc_z3, dc_z4],
        dc_info={
            dc_z1: _DCInfo(z1_models),
            dc_z2: _DCInfo(z2_models),
            dc_z3: _DCInfo(z3_models),
            dc_z4: _DCInfo(z4_models),
        },
        grid=grid,
        ofo_config=OFOConfig(
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
        ),
        tap_schedule=TapSchedule(((1800, TapPosition(regulators={"creg4a": tap(16)})),)),
        generators=generators,
        time_varying_loads=time_varying_loads,
        zones=sys.get("zones"),
        exclude_buses=tuple(sys["exclude_buses"]),
    )


SETUPS = {"ieee13": setup_ieee13, "ieee34": setup_ieee34, "ieee123": setup_ieee123}


# Main


def main(*, system: str, mode: str = "baseline-no-tap") -> None:
    # Load data pipeline
    data_sources, data_dir = load_data_sources()

    logger.info("Loading data for %s...", system)
    inference_data = InferenceData.ensure(data_dir, ALL_MODEL_SPECS, data_sources, plot=False, dt_s=float(DT_DC))
    training_trace = TrainingTrace.ensure(data_dir / "training_trace.csv")
    logistic_models = LogisticModelStore.ensure(
        data_dir / "logistic_fits.csv", ALL_MODEL_SPECS, data_sources, plot=False
    )

    if system not in SETUPS:
        raise ValueError(f"Unknown system {system!r}. Choose from: {list(SETUPS)}")
    setup_fn = SETUPS[system]

    # Probe experiment once for case planning (tap schedule, exclude_buses)
    config = setup_fn(inference_data, training_trace, logistic_models)
    exp_tap_schedule: TapSchedule | None = config.get("tap_schedule")
    has_tap_schedule = bool(exp_tap_schedule)
    exclude_buses: tuple[str, ...] = config["exclude_buses"]

    # Build cases: each mode runs exactly one case, "all" runs all four
    _ALL_CASES: list[tuple[str, str, TapSchedule | None]] = [
        ("baseline", "baseline_no-tap", None),
        ("ofo", "ofo_no-tap", None),
    ]
    if has_tap_schedule:
        _ALL_CASES += [
            ("baseline", "baseline_tap-change", exp_tap_schedule),
            ("ofo", "ofo_tap-change", exp_tap_schedule),
        ]

    if mode == "all":
        cases = list(_ALL_CASES)
    else:
        # CLI uses hyphens (baseline-no-tap), case names use underscores (baseline_no-tap)
        case_name = mode.replace("baseline-", "baseline_").replace("ofo-", "ofo_")
        cases = [c for c in _ALL_CASES if c[1] == case_name]
    if not cases:
        logger.warning("No cases to run (mode=%s, has_tap_schedule=%s).", mode, has_tap_schedule)
        return

    save_dir = Path(__file__).resolve().parent / "outputs" / system
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running %d cases for %s: %s", len(cases), system, [c[1] for c in cases])

    results: dict[str, VoltageStats] = {}
    perf_results: dict[str, PerformanceStats] = {}
    for ctrl_mode, case_name, sched in cases:
        # Fresh DCs and grid per case (no state leakage between runs)
        config = setup_fn(inference_data, training_trace, logistic_models)
        datacenters: list[OfflineDatacenter] = config["datacenters"]
        dc_info: dict[OfflineDatacenter, _DCInfo] = config["dc_info"]
        grid: OpenDSSGrid = config["grid"]
        ofo_config: OFOConfig = config["ofo_config"]
        generators: list[tuple[str, Generator]] = config.get("generators", [])
        time_varying_loads: list[tuple[str, ExternalLoad]] = config.get("time_varying_loads", [])
        zones: dict[str, list[str]] | None = config.get("zones")
        run_dir = save_dir / case_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Build controllers
        controllers: list = []
        controllers.append(
            TapScheduleController(schedule=sched if sched is not None else TapSchedule(()), dt_s=DT_CTRL)
        )

        if ctrl_mode == "ofo" and ofo_config is not None:
            for dc, info in dc_info.items():
                controllers.append(
                    OFOBatchSizeController(
                        info.specs,
                        datacenter=dc,
                        models=logistic_models,
                        config=ofo_config,
                        dt_s=DT_CTRL,
                        initial_batch_sizes=info.initial_batch_sizes,
                        grid=grid,
                    )
                )

        # Run
        logger.info("=== %s (%s) ===", ctrl_mode.upper(), case_name)

        coord = Coordinator(
            datacenters=datacenters,
            grid=grid,
            controllers=controllers,
            total_duration_s=TOTAL_DURATION_S,
        )
        log = coord.run()

        vstats = compute_allbus_voltage_stats(log.grid_states, v_min=V_MIN, v_max=V_MAX, exclude_buses=exclude_buses)
        # Throughput + ITL across every model served by any DC in this case
        itl_deadlines: dict[str, float] = {}
        for info in dc_info.values():
            for ms in info.specs:
                itl_deadlines[ms.model_label] = ms.itl_deadline_s
        pstats = compute_performance_stats(log.dc_states, itl_deadline_s_by_model=itl_deadlines)
        logger.info(
            "  %s: viol=%.1fs  integral=%.4f  vmin=%.4f  vmax=%.4f  thpt=%.1f k tok/s  itl_over_deadline=%.2f%%",
            case_name,
            vstats.violation_time_s,
            vstats.integral_violation_pu_s,
            vstats.worst_vmin,
            vstats.worst_vmax,
            pstats.mean_throughput_tps / 1e3,
            pstats.itl_deadline_fraction * 100.0,
        )
        results[case_name] = vstats
        perf_results[case_name] = pstats

        # Plots
        time_s = np.array(log.time_s)

        all_buses: set[str] = set()
        if log.grid_states:
            drop = {b.lower() for b in exclude_buses}
            for bus in log.grid_states[0].voltages.buses():
                if bus.lower() not in drop:
                    all_buses.add(bus)

        if zones:
            plot_zone_voltage_envelope(
                log.grid_states,
                time_s,
                zones=zones,
                save_dir=run_dir,
                v_min=V_MIN,
                v_max=V_MAX,
                drop_buses=exclude_buses,
                title_template=f"{case_name} -- Voltage Envelope (Phase {{label}})",
            )
        elif len(all_buses) > MAX_BUSES_FOR_INDIVIDUAL_LINES:
            _plot_voltage_envelope(
                log.grid_states,
                time_s,
                run_dir,
                v_min=V_MIN,
                v_max=V_MAX,
                exclude_buses=exclude_buses,
                title=f"{case_name} -- Voltage Envelope",
            )
        else:
            plot_allbus_voltages_per_phase(
                log.grid_states,
                time_s,
                save_dir=run_dir,
                v_min=V_MIN,
                v_max=V_MAX,
                title_template=f"{case_name} -- Voltage (Phase {{label}})",
                drop_buses=exclude_buses,
            )

        site_ids = [dc.name for dc in datacenters]
        site_specs_map = {dc.name: info.specs for dc, info in dc_info.items()}

        if ctrl_mode == "ofo":
            for site_id in site_ids:
                site_states = log.dc_states_by_site.get(site_id, log.dc_states)
                if not site_states:
                    continue
                suffix = f"_{site_id}" if len(site_ids) > 1 else ""
                model_labels = [m.model_label for m in site_specs_map[site_id]]
                plot_model_timeseries_4panel(
                    site_states,
                    model_labels=model_labels,
                    regime_shading=False,
                    save_path=run_dir / f"OFO_results{suffix}.png",
                )

        for site_id in site_ids:
            site_states = log.dc_states_by_site.get(site_id, log.dc_states)
            if not site_states:
                continue
            suffix = f"_{site_id}" if len(site_ids) > 1 else ""
            plot_power_and_itl_2panel(
                site_states,
                show_regimes=False,
                save_path=run_dir / f"power_latency{suffix}.png",
            )

        _plot_tap_positions(log.grid_states, time_s, run_dir, title=f"{case_name} -- Tap Positions")
        _plot_pv_load_profiles(log.grid_states, time_s, generators, time_varying_loads, run_dir)

        with open(run_dir / f"result_{case_name}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "case",
                    "violation_time_s",
                    "integral_violation_pu_s",
                    "worst_vmin",
                    "worst_vmax",
                    "mean_throughput_tps",
                    "integrated_throughput_tokens",
                    "itl_deadline_fraction",
                ]
            )
            writer.writerow(
                [
                    case_name,
                    vstats.violation_time_s,
                    vstats.integral_violation_pu_s,
                    vstats.worst_vmin,
                    vstats.worst_vmax,
                    pstats.mean_throughput_tps,
                    pstats.integrated_throughput_tokens,
                    pstats.itl_deadline_fraction,
                ]
            )
        logger.info("Per-case CSV: %s", run_dir / f"result_{case_name}.csv")

    # Comparison CSV
    csv_path = save_dir / f"results_{system}_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "violation_time_s",
                "worst_vmin",
                "worst_vmax",
                "integral_violation_pu_s",
                "mean_throughput_tps",
                "integrated_throughput_tokens",
                "itl_deadline_fraction",
            ]
        )
        for case_name, s in results.items():
            p = perf_results[case_name]
            writer.writerow(
                [
                    case_name,
                    s.violation_time_s,
                    s.worst_vmin,
                    s.worst_vmax,
                    s.integral_violation_pu_s,
                    p.mean_throughput_tps,
                    p.integrated_throughput_tokens,
                    p.itl_deadline_fraction,
                ]
            )
    logger.info("Comparison CSV: %s", csv_path)

    _plot_comparison(results, save_dir, system)

    # Summary table
    logger.info("")
    logger.info("=" * 90)
    logger.info("%s Comparison", system.upper())
    logger.info("=" * 90)
    logger.info(
        "%-22s %10s %10s %10s %14s %14s %10s",
        "Mode",
        "Viol(s)",
        "Vmin",
        "Vmax",
        "Integral",
        "Thpt(k tok/s)",
        "ITL_miss%",
    )
    logger.info("-" * 100)
    for case_name, s in results.items():
        p = perf_results[case_name]
        logger.info(
            "%-22s %10.1f %10.4f %10.4f %14.4f %14.1f %9.2f%%",
            case_name,
            s.violation_time_s,
            s.worst_vmin,
            s.worst_vmax,
            s.integral_violation_pu_s,
            p.mean_throughput_tps / 1e3,
            p.itl_deadline_fraction * 100.0,
        )
    logger.info("-" * 90)
    logger.info("Outputs: %s", save_dir)


if __name__ == "__main__":
    from dataclasses import dataclass

    import tyro

    @dataclass
    class Args:
        """Command-line arguments.

        Attributes:
            system: System name (ieee13, ieee34, ieee123).
            mode: Case to run: baseline-no-tap, baseline-tap-change, ofo-no-tap, ofo-tap-change, or all.
            log_level: Logging verbosity (DEBUG, INFO, WARNING).
        """

        system: Literal["ieee13", "ieee34", "ieee123"] = "ieee13"
        mode: Literal["baseline-no-tap", "baseline-tap-change", "ofo-no-tap", "ofo-tap-change", "all"] = (
            "baseline-no-tap"
        )
        log_level: Literal["DEBUG", "INFO", "WARNING"] = "INFO"

    args = tyro.cli(Args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    main(system=args.system, mode=args.mode)
