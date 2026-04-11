"""Compare different batch-size controllers for voltage regulation.

Runs three control strategies on the same system and produces comparison
plots: (1) no batch control (baseline), (2) rule-based proportional
controller, and (3) OFO primal-dual controller.

Usage:
    python analyze_different_controllers.py --system ieee123
    python analyze_different_controllers.py --system ieee34
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openg2g.controller.ofo import (
    LogisticModelStore,
    OFOBatchSizeController,
    OFOConfig,
)
from openg2g.controller.rule_based import RuleBasedBatchSizeController, RuleBasedConfig
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import Coordinator
from openg2g.datacenter.config import (
    DatacenterConfig,
    InferenceModelSpec,
    ModelDeployment,
    PowerAugmentationConfig,
    ReplicaSchedule,
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData, MLEnergySource
from openg2g.datacenter.workloads.training import TrainingTrace
from openg2g.grid.config import TapPosition, TapSchedule
from openg2g.grid.generator import SyntheticPV
from openg2g.grid.load import SyntheticLoad
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.voltage import VoltageStats, compute_allbus_voltage_stats

from systems import SYSTEMS, tap

logger = logging.getLogger("analyze_different_controllers")

# Simulation defaults

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

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
MODEL_SPECS = {s.model_label: s for s in ALL_MODEL_SPECS}


def deploy(label, num_replicas, initial_batch_size=128):
    return ModelDeployment(spec=MODEL_SPECS[label], num_replicas=num_replicas, initial_batch_size=initial_batch_size)


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
    data_dir = _REPO_ROOT / "data" / "offline" / hashlib.sha256(blob).hexdigest()[:16]
    return data_sources, data_dir


# Per-system setup functions


def _setup_ieee13(inference_data, training_trace, logistic_models):
    """IEEE 13-bus: single DC at bus 671."""
    sys = SYSTEMS["ieee13"]()
    exclude_buses = tuple(sys["exclude_buses"])

    models_default = (
        deploy("Llama-3.1-8B", 720),
        deploy("Llama-3.1-70B", 180),
        deploy("Llama-3.1-405B", 90),
        deploy("Qwen3-30B-A3B", 480),
        deploy("Qwen3-235B-A22B", 210),
    )
    specs_default = tuple(md.spec for md in models_default)
    site_inference = inference_data.filter_models(specs_default)

    workload = OfflineWorkload(
        inference_data=site_inference,
        replica_schedules={
            "Llama-3.1-8B": ReplicaSchedule(initial=720).ramp_to(144, t_start=2500, t_end=3000),
            "Llama-3.1-70B": ReplicaSchedule(initial=180).ramp_to(36, t_start=2500, t_end=3000),
            "Llama-3.1-405B": ReplicaSchedule(initial=90).ramp_to(18, t_start=2500, t_end=3000),
            "Qwen3-30B-A3B": ReplicaSchedule(initial=480).ramp_to(96, t_start=2500, t_end=3000),
            "Qwen3-235B-A22B": ReplicaSchedule(initial=210).ramp_to(42, t_start=2500, t_end=3000),
        },
    )
    dc_default = OfflineDatacenter(
        DatacenterConfig(gpus_per_server=8, base_kw_per_phase=500.0),
        workload,
        name="default",
        dt_s=DT_DC,
        seed=0,
        power_augmentation=POWER_AUG,
        total_gpu_capacity=7200,
    )

    datacenters = [dc_default]
    dc_specs = {dc_default: specs_default}

    grid = OpenDSSGrid(
        dss_case_dir=sys["dss_case_dir"],
        dss_master_file=sys["dss_master_file"],
        source_pu=sys["source_pu"],
        dt_s=DT_GRID,
        initial_tap_position=sys["initial_taps"],
        exclude_buses=exclude_buses,
    )
    grid.attach_dc(dc_default, bus="671", connection_type="wye")
    grid.attach_generator(SyntheticPV(peak_kw=10.0), bus="675")
    grid.attach_load(SyntheticLoad(peak_kw=10.0), bus="680")

    tap_schedule = TapSchedule(
        (
            (1500, TapPosition(regulators={"creg1a": tap(16), "creg1b": tap(6), "creg1c": tap(17)})),
            (3300, TapPosition(regulators={"creg1a": tap(10), "creg1b": tap(6), "creg1c": tap(10)})),
        )
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

    return dict(
        sys=sys,
        datacenters=datacenters,
        dc_specs=dc_specs,
        grid=grid,
        ofo_config=ofo_config,
        tap_schedule=tap_schedule,
        exclude_buses=exclude_buses,
    )


def _setup_ieee34(inference_data, training_trace, logistic_models):
    """IEEE 34-bus: two DC sites (upstream/downstream)."""
    sys = SYSTEMS["ieee34"]()
    exclude_buses = tuple(sys["exclude_buses"])

    models_upstream = (deploy("Llama-3.1-8B", 720), deploy("Llama-3.1-70B", 180), deploy("Llama-3.1-405B", 90))
    models_downstream = (deploy("Qwen3-30B-A3B", 480), deploy("Qwen3-235B-A22B", 210))
    specs_upstream = tuple(md.spec for md in models_upstream)
    specs_downstream = tuple(md.spec for md in models_downstream)

    dc_upstream = OfflineDatacenter(
        DatacenterConfig(gpus_per_server=8, base_kw_per_phase=120.0),
        OfflineWorkload(
            inference_data=inference_data.filter_models(specs_upstream),
            replica_schedules={md.spec.model_label: ReplicaSchedule(initial=md.num_replicas) for md in models_upstream},
        ),
        name="upstream",
        dt_s=DT_DC,
        seed=0,
        power_augmentation=POWER_AUG,
        total_gpu_capacity=2400,
    )
    dc_downstream = OfflineDatacenter(
        DatacenterConfig(gpus_per_server=8, base_kw_per_phase=80.0),
        OfflineWorkload(
            inference_data=inference_data.filter_models(specs_downstream),
            replica_schedules={
                md.spec.model_label: ReplicaSchedule(initial=md.num_replicas) for md in models_downstream
            },
        ),
        name="downstream",
        dt_s=DT_DC,
        seed=42,
        power_augmentation=POWER_AUG,
        total_gpu_capacity=2880,
    )

    datacenters = [dc_upstream, dc_downstream]
    dc_specs = {dc_upstream: specs_upstream, dc_downstream: specs_downstream}

    grid = OpenDSSGrid(
        dss_case_dir=sys["dss_case_dir"],
        dss_master_file=sys["dss_master_file"],
        source_pu=sys["source_pu"],
        dt_s=DT_GRID,
        initial_tap_position=sys["initial_taps"],
        exclude_buses=exclude_buses,
    )
    grid.attach_dc(dc_upstream, bus="850", connection_type="wye")
    grid.attach_dc(dc_downstream, bus="834", connection_type="wye")
    grid.attach_generator(SyntheticPV(peak_kw=130.0), bus="848")
    grid.attach_generator(SyntheticPV(peak_kw=65.0, site_idx=1), bus="830")
    grid.attach_load(SyntheticLoad(peak_kw=80.0), bus="860")
    grid.attach_load(SyntheticLoad(peak_kw=120.0, site_idx=1), bus="844")
    grid.attach_load(SyntheticLoad(peak_kw=60.0, site_idx=2), bus="840")
    grid.attach_load(SyntheticLoad(peak_kw=50.0, site_idx=3), bus="858")
    grid.attach_load(SyntheticLoad(peak_kw=40.0, site_idx=4), bus="854")

    tap_schedule = TapSchedule(
        (
            (
                1800,
                TapPosition(
                    regulators={
                        "creg2a": tap(10),
                        "creg2b": tap(10),
                        "creg2c": tap(10),
                    }
                ),
            ),
        )
    )
    ofo_config = OFOConfig(
        primal_step_size=0.05,
        w_throughput=0.001,
        w_switch=1.0,
        voltage_gradient_scale=1e6,
        voltage_dual_step_size=20.0,
        latency_dual_step_size=1.0,
        sensitivity_update_interval=3600,
        sensitivity_perturbation_kw=10.0,
        v_min=V_MIN,
        v_max=V_MAX,
    )

    return dict(
        sys=sys,
        datacenters=datacenters,
        dc_specs=dc_specs,
        grid=grid,
        ofo_config=ofo_config,
        tap_schedule=tap_schedule,
        exclude_buses=exclude_buses,
    )


def _setup_ieee123(inference_data, training_trace, logistic_models):
    """IEEE 123-bus: four DC sites across zones."""
    sys = SYSTEMS["ieee123"]()
    exclude_buses = tuple(sys["exclude_buses"])

    models_z1 = (deploy("Llama-3.1-8B", 120),)
    models_z2 = (deploy("Qwen3-30B-A3B", 80),)
    models_z3 = (deploy("Llama-3.1-70B", 30), deploy("Llama-3.1-405B", 35))
    models_z4 = (deploy("Qwen3-235B-A22B", 55),)
    specs_z1 = tuple(md.spec for md in models_z1)
    specs_z2 = tuple(md.spec for md in models_z2)
    specs_z3 = tuple(md.spec for md in models_z3)
    specs_z4 = tuple(md.spec for md in models_z4)

    dc_z1 = OfflineDatacenter(
        DatacenterConfig(gpus_per_server=8, base_kw_per_phase=310.0),
        OfflineWorkload(
            inference_data=inference_data.filter_models(specs_z1),
            replica_schedules={
                "Llama-3.1-8B": ReplicaSchedule(initial=120).ramp_to(180, t_start=500, t_end=1000),
            },
        ),
        name="z1_sw",
        dt_s=DT_DC,
        seed=0,
        power_augmentation=POWER_AUG,
        total_gpu_capacity=180,
    )
    dc_z2 = OfflineDatacenter(
        DatacenterConfig(gpus_per_server=8, base_kw_per_phase=265.0),
        OfflineWorkload(
            inference_data=inference_data.filter_models(specs_z2),
            replica_schedules={
                "Qwen3-30B-A3B": ReplicaSchedule(initial=80).ramp_to(104, t_start=1500, t_end=2500),
            },
        ),
        name="z2_nw",
        dt_s=DT_DC,
        seed=17,
        power_augmentation=POWER_AUG,
        total_gpu_capacity=208,
    )
    dc_z3 = OfflineDatacenter(
        DatacenterConfig(gpus_per_server=8, base_kw_per_phase=295.0),
        OfflineWorkload(
            inference_data=inference_data.filter_models(specs_z3),
            replica_schedules={
                "Llama-3.1-70B": ReplicaSchedule(initial=30).ramp_to(45, t_start=700, t_end=1100),
                "Llama-3.1-405B": ReplicaSchedule(initial=35),
            },
        ),
        name="z3_se",
        dt_s=DT_DC,
        seed=34,
        power_augmentation=POWER_AUG,
        total_gpu_capacity=460,
    )
    dc_z4 = OfflineDatacenter(
        DatacenterConfig(gpus_per_server=8, base_kw_per_phase=325.0),
        OfflineWorkload(
            inference_data=inference_data.filter_models(specs_z4),
            replica_schedules={
                "Qwen3-235B-A22B": ReplicaSchedule(initial=55).ramp_to(27, t_start=2000, t_end=2500),
            },
        ),
        name="z4_ne",
        dt_s=DT_DC,
        seed=51,
        power_augmentation=POWER_AUG,
        total_gpu_capacity=440,
    )

    datacenters = [dc_z1, dc_z2, dc_z3, dc_z4]
    dc_specs = {dc_z1: specs_z1, dc_z2: specs_z2, dc_z3: specs_z3, dc_z4: specs_z4}

    grid = OpenDSSGrid(
        dss_case_dir=sys["dss_case_dir"],
        dss_master_file=sys["dss_master_file"],
        source_pu=sys["source_pu"],
        dt_s=DT_GRID,
        initial_tap_position=sys["initial_taps"],
        exclude_buses=exclude_buses,
    )
    grid.attach_dc(dc_z1, bus="8", connection_type="wye")
    grid.attach_dc(dc_z2, bus="23", connection_type="wye")
    grid.attach_dc(dc_z3, bus="60", connection_type="wye")
    grid.attach_dc(dc_z4, bus="105", connection_type="wye")
    grid.attach_generator(SyntheticPV(peak_kw=333.3), bus="1")
    grid.attach_generator(SyntheticPV(peak_kw=333.3, site_idx=1), bus="48")
    grid.attach_generator(SyntheticPV(peak_kw=333.3, site_idx=2), bus="99")

    tap_schedule = TapSchedule(((1800, TapPosition(regulators={"creg4a": tap(16)})),))
    ofo_config = OFOConfig(
        primal_step_size=0.05,
        w_throughput=0.001,
        w_switch=1.0,
        voltage_gradient_scale=1e6,
        voltage_dual_step_size=0.3,
        latency_dual_step_size=1.0,
        sensitivity_update_interval=3600,
        sensitivity_perturbation_kw=10.0,
        v_min=V_MIN,
        v_max=V_MAX,
    )

    return dict(
        sys=sys,
        datacenters=datacenters,
        dc_specs=dc_specs,
        grid=grid,
        ofo_config=ofo_config,
        tap_schedule=tap_schedule,
        exclude_buses=exclude_buses,
    )


SETUPS = {"ieee13": _setup_ieee13, "ieee34": _setup_ieee34, "ieee123": _setup_ieee123}


# Plotting


def plot_voltage_comparison(
    logs: dict[str, object],
    save_dir: Path,
    *,
    v_min: float = 0.95,
    v_max: float = 1.05,
    exclude_buses: tuple[str, ...] = (),
) -> None:
    """Side-by-side voltage envelopes for each controller mode."""
    modes = list(logs.keys())
    n = len(modes)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    drop = {b.lower() for b in exclude_buses}

    for ax, mode in zip(axes, modes, strict=False):
        log = logs[mode]
        time_s = np.array(log.time_s)

        v_min_arr = np.full(len(log.grid_states), np.inf)
        v_max_arr = np.full(len(log.grid_states), -np.inf)

        for t_idx, gs in enumerate(log.grid_states):
            for bus in gs.voltages.buses():
                if bus.lower() in drop:
                    continue
                pv = gs.voltages[bus]
                for v in (pv.a, pv.b, pv.c):
                    if not math.isnan(v):
                        v_min_arr[t_idx] = min(v_min_arr[t_idx], v)
                        v_max_arr[t_idx] = max(v_max_arr[t_idx], v)

        ax.fill_between(time_s, v_min_arr, v_max_arr, alpha=0.3, color="steelblue")
        ax.plot(time_s, v_min_arr, color="steelblue", linewidth=0.5, label="Vmin")
        ax.plot(time_s, v_max_arr, color="coral", linewidth=0.5, label="Vmax")
        ax.axhline(v_min, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(v_max, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_title(mode.replace("_", " ").title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Voltage (pu)")
    fig.suptitle("Voltage Envelope Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "voltage_comparison.png", dpi=150, bbox_inches="tight", metadata={"Creation Time": None})
    plt.close(fig)
    logger.info("Saved voltage_comparison.png")


def plot_batch_comparison(
    logs: dict[str, object],
    save_dir: Path,
) -> None:
    """Batch size over time for each controller mode."""
    modes = list(logs.keys())
    # Collect all model labels across all modes
    all_models: list[str] = []
    for log in logs.values():
        for st in log.dc_states:
            for m in st.batch_size_by_model:
                if m not in all_models:
                    all_models.append(m)
        break  # only need one log to get model names

    n_models = len(all_models)
    len(modes)
    fig, axes = plt.subplots(
        n_models,
        1,
        figsize=(12, 3 * n_models),
        sharex=True,
        squeeze=False,
    )

    colors = ["#999999", "#4CAF50", "#2196F3", "#FF9800", "#E91E63"]

    for row, model_label in enumerate(all_models):
        ax = axes[row][0]
        for i, mode in enumerate(modes):
            log = logs[mode]
            times = [s.time_s for s in log.dc_states]
            batches = [s.batch_size_by_model.get(model_label, 0) for s in log.dc_states]
            ax.plot(
                times,
                batches,
                color=colors[i % len(colors)],
                linewidth=1,
                alpha=0.8,
                label=mode.replace("_", " ").title(),
            )
        ax.set_ylabel("Batch Size")
        ax.set_title(model_label, fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2)

    axes[-1][0].set_xlabel("Time (s)")
    fig.suptitle("Batch Size Comparison by Model", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "batch_size_comparison.png", dpi=150, bbox_inches="tight", metadata={"Creation Time": None})
    plt.close(fig)
    logger.info("Saved batch_size_comparison.png")


def plot_summary_bar(
    stats: dict[str, VoltageStats],
    save_dir: Path,
) -> None:
    """Summary bar chart comparing key metrics across modes."""
    modes = list(stats.keys())
    n = len(modes)
    x = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Violation time
    vals = [stats[m].violation_time_s for m in modes]
    axes[0].bar(x, vals, color=["#999999", "#4CAF50", "#2196F3"][:n])
    axes[0].set_ylabel("Violation Time (s)")
    axes[0].set_title("Voltage Violation Time")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace("_", "\n") for m in modes], fontsize=9)

    # Integral violation
    vals = [stats[m].integral_violation_pu_s for m in modes]
    axes[1].bar(x, vals, color=["#999999", "#4CAF50", "#2196F3"][:n])
    axes[1].set_ylabel("Integral Violation (pu·s)")
    axes[1].set_title("Integral Voltage Violation")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.replace("_", "\n") for m in modes], fontsize=9)

    # Worst Vmin
    vals = [stats[m].worst_vmin for m in modes]
    axes[2].bar(x, vals, color=["#999999", "#4CAF50", "#2196F3"][:n])
    axes[2].set_ylabel("Worst Vmin (pu)")
    axes[2].set_title("Worst Minimum Voltage")
    axes[2].axhline(0.95, color="red", linestyle="--", linewidth=1, alpha=0.7)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([m.replace("_", "\n") for m in modes], fontsize=9)

    fig.suptitle("Controller Comparison Summary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "summary_bar_chart.png", dpi=150, bbox_inches="tight", metadata={"Creation Time": None})
    plt.close(fig)
    logger.info("Saved summary_bar_chart.png")


# Main


def main(
    *,
    system: str,
    rule_step_size: float = 0.3,
    rule_deadband: float = 0.005,
) -> None:
    if system not in SETUPS:
        raise ValueError(f"Unknown system {system!r}; choose from {list(SETUPS)}")

    save_dir = Path(__file__).resolve().parent / "outputs" / system / "controller_comparison"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    data_sources, data_dir = load_data_sources()

    # Collect all specs across all sites for data loading (need full set for InferenceData)
    setup_fn = SETUPS[system]
    # We need all model specs to load data before building the setup.
    # Temporarily gather them from the deploy calls used per system.
    all_specs_by_system = {
        "ieee13": (LLAMA_8B, LLAMA_70B, LLAMA_405B, QWEN_30B, QWEN_235B),
        "ieee34": (LLAMA_8B, LLAMA_70B, LLAMA_405B, QWEN_30B, QWEN_235B),
        "ieee123": (LLAMA_8B, QWEN_30B, LLAMA_70B, LLAMA_405B, QWEN_235B),
    }
    all_specs_tuple = all_specs_by_system[system]

    inference_data = InferenceData.ensure(
        data_dir,
        all_specs_tuple,
        data_sources,
        plot=False,
        dt_s=float(DT_DC),
    )
    training_trace = TrainingTrace.ensure(data_dir / "training_trace.csv")
    logistic_models = LogisticModelStore.ensure(
        data_dir / "logistic_fits.csv",
        all_specs_tuple,
        data_sources,
        plot=False,
    )

    # Rule-based config
    rb_config = RuleBasedConfig(
        step_size=rule_step_size,
        deadband=rule_deadband,
        v_min=V_MIN,
        v_max=V_MAX,
    )

    # Run all modes
    modes = ["baseline", "rule_based", "ofo"]
    all_stats: dict[str, VoltageStats] = {}
    all_logs: dict[str, object] = {}

    for mode in modes:
        # Fresh DCs and grid per mode (no state leakage between runs)
        setup = setup_fn(inference_data, training_trace, logistic_models)
        datacenters: list[OfflineDatacenter] = setup["datacenters"]
        dc_specs: dict[OfflineDatacenter, tuple[InferenceModelSpec, ...]] = setup["dc_specs"]
        grid: OpenDSSGrid = setup["grid"]
        ofo_config: OFOConfig = setup["ofo_config"]
        tap_schedule: TapSchedule = setup["tap_schedule"]
        exclude_buses: tuple[str, ...] = setup["exclude_buses"]
        logger.info("")
        logger.info("=" * 60)
        logger.info("MODE: %s", mode.upper())
        logger.info("=" * 60)

        controllers: list = []

        # Tap controller: only baseline gets tap schedule changes;
        # OFO and rule-based use fixed initial taps (no tap schedule)
        if mode == "baseline":
            sched = tap_schedule
        else:
            sched = TapSchedule(())
        controllers.append(TapScheduleController(schedule=sched, dt_s=DT_CTRL))

        # Batch-size controller
        if mode == "ofo":
            for dc, specs in dc_specs.items():
                ofo_ctrl = OFOBatchSizeController(
                    specs,
                    datacenter=dc,
                    models=logistic_models,
                    config=ofo_config,
                    dt_s=DT_CTRL,
                    grid=grid,
                )
                controllers.append(ofo_ctrl)

        elif mode == "rule_based":
            for dc, specs in dc_specs.items():
                rb_ctrl = RuleBasedBatchSizeController(
                    specs,
                    datacenter=dc,
                    config=rb_config,
                    dt_s=DT_CTRL,
                    exclude_buses=exclude_buses,
                    grid=grid,
                )
                controllers.append(rb_ctrl)

        # baseline: no batch controller added

        coord = Coordinator(
            datacenters=datacenters,
            grid=grid,
            controllers=controllers,
            total_duration_s=TOTAL_DURATION_S,
        )

        logger.info("Running %s...", mode)
        log = coord.run()

        vstats = compute_allbus_voltage_stats(
            log.grid_states,
            v_min=V_MIN,
            v_max=V_MAX,
            exclude_buses=exclude_buses,
        )
        logger.info(
            "  %s: viol=%.1fs  integral=%.4f  vmin=%.4f  vmax=%.4f",
            mode,
            vstats.violation_time_s,
            vstats.integral_violation_pu_s,
            vstats.worst_vmin,
            vstats.worst_vmax,
        )

        all_stats[mode] = vstats
        all_logs[mode] = log

    # Summary table
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(
        "%-15s %12s %12s %12s %12s",
        "Mode",
        "Viol(s)",
        "Integral",
        "Worst Vmin",
        "Worst Vmax",
    )
    logger.info("-" * 80)
    for mode in modes:
        s = all_stats[mode]
        logger.info(
            "%-15s %12.1f %12.4f %12.4f %12.4f",
            mode,
            s.violation_time_s,
            s.integral_violation_pu_s,
            s.worst_vmin,
            s.worst_vmax,
        )

    # Save results CSV
    csv_path = save_dir / f"results_{system}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "violation_time_s", "integral_violation_pu_s", "worst_vmin", "worst_vmax"])
        for mode in modes:
            s = all_stats[mode]
            writer.writerow([mode, s.violation_time_s, s.integral_violation_pu_s, s.worst_vmin, s.worst_vmax])
    logger.info("Results CSV: %s", csv_path)

    # Plots
    plot_voltage_comparison(all_logs, save_dir, v_min=V_MIN, v_max=V_MAX, exclude_buses=exclude_buses)
    plot_batch_comparison(all_logs, save_dir)
    plot_summary_bar(all_stats, save_dir)

    logger.info("")
    logger.info("All outputs saved to: %s", save_dir)


if __name__ == "__main__":
    from dataclasses import dataclass

    import tyro

    @dataclass
    class Args:
        """Command-line arguments.

        Attributes:
            system: System name: ieee13, ieee34, or ieee123.
            rule_step_size: Proportional gain for the rule-based controller
                (log2 units per pu violation).
            rule_deadband: Deadband for the rule-based controller (pu).
            log_level: Logging verbosity.
        """

        system: str = "ieee123"
        rule_step_size: float = 0.3
        rule_deadband: float = 0.005
        log_level: str = "INFO"

    args = tyro.cli(Args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openg2g.coordinator").setLevel(logging.WARNING)
    logging.getLogger("openg2g.datacenter").setLevel(logging.WARNING)
    logging.getLogger("openg2g.grid").setLevel(logging.WARNING)
    logging.getLogger("openg2g.controller.ofo").setLevel(logging.WARNING)

    main(
        system=args.system,
        rule_step_size=args.rule_step_size,
        rule_deadband=args.rule_deadband,
    )
