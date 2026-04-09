"""LLM load shifting: compare OFO with and without cross-site replica shifting.

Runs two OFO simulations -- one with load shifting disabled and one enabled --
then produces comparison plots showing voltage improvement and replica movement.

Each DC site must run at least 3 models (warm-start requirement).

Usage:
    python analyze_LLM_load_shifting.py --system ieee123
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from systems import ieee123, tap

from openg2g.controller.load_shift import LoadShiftConfig, LoadShiftController
from openg2g.controller.ofo import (
    LogisticModelStore,
    OFOBatchSizeController,
    OFOConfig,
)
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import Coordinator
from openg2g.datacenter.command import ShiftReplicas
from openg2g.datacenter.config import (
    DatacenterConfig,
    InferenceModelSpec,
    ModelDeployment,
    PowerAugmentationConfig,
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData, MLEnergySource
from openg2g.datacenter.workloads.training import TrainingTrace, TrainingTraceParams
from openg2g.grid.config import TapPosition, TapSchedule
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.voltage import VoltageStats, compute_allbus_voltage_stats

logger = logging.getLogger("load_shift_comparison")

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
MODEL_SPECS: dict[str, InferenceModelSpec] = {s.model_label: s for s in ALL_MODEL_SPECS}


def deploy(label: str, num_replicas: int, initial_batch_size: int = 128) -> ModelDeployment:
    """Shorthand: ``deploy("Llama-3.1-8B", 720, 128)``."""
    return ModelDeployment(spec=MODEL_SPECS[label], num_replicas=num_replicas, initial_batch_size=initial_batch_size)


def load_data_sources(config_path: Path | None = None) -> tuple[dict[str, MLEnergySource], TrainingTraceParams, Path]:
    """Load ML.ENERGY data sources from ``config.json``.

    Returns:
        (data_sources, training_trace_params, data_dir) where *data_dir* is a
        hash-based cache directory under ``<repo_root>/data/offline``.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    sources_raw = cfg["data_sources"]
    data_sources = {s["model_label"]: MLEnergySource(**s) for s in sources_raw}
    ttp = TrainingTraceParams(**(cfg.get("training_trace_params") or {}))
    blob = json.dumps(
        (sorted(sources_raw, key=lambda s: s["model_label"]), cfg.get("training_trace_params") or {}),
        sort_keys=True,
    ).encode()
    data_dir = _REPO_ROOT / "data" / "offline" / hashlib.sha256(blob).hexdigest()[:16]
    return data_sources, ttp, data_dir


# IEEE 123-bus load-shift experiment definition

# Per-site model deployments (name -> (bus, base_kw, models, gpu_cap))
_SITE_DEFS: dict[str, dict[str, Any]] = {
    "z1_sw": dict(
        bus="7",
        base_kw_per_phase=210,
        models=(deploy("Llama-3.1-8B", 120), deploy("Qwen3-30B-A3B", 120), deploy("Llama-3.1-70B", 120)),
        total_gpu_capacity=1092,
    ),
    "z2_nw": dict(
        bus="33",
        base_kw_per_phase=175,
        models=(deploy("Llama-3.1-8B", 120), deploy("Qwen3-30B-A3B", 120), deploy("Llama-3.1-405B", 40)),
        total_gpu_capacity=728,
    ),
    "z3_se": dict(
        bus="76",
        base_kw_per_phase=215,
        models=(
            deploy("Qwen3-30B-A3B", 120),
            deploy("Llama-3.1-70B", 120),
            deploy("Llama-3.1-405B", 40),
            deploy("Qwen3-235B-A22B", 40),
        ),
        total_gpu_capacity=1768,
    ),
    "z4_ne": dict(
        bus="108",
        base_kw_per_phase=235,
        models=(
            deploy("Llama-3.1-8B", 120),
            deploy("Qwen3-30B-A3B", 120),
            deploy("Llama-3.1-405B", 40),
            deploy("Qwen3-235B-A22B", 40),
        ),
        total_gpu_capacity=1300,
    ),
}

LOAD_SHIFT_TAP_SCHEDULE = TapSchedule(((1800, TapPosition(regulators={"creg4a": tap(15)})),))

LOAD_SHIFT_GPUS_PER_SHIFT = 8
LOAD_SHIFT_HEADROOM = 0.3


def _build_experiment(
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
    *,
    load_shift_headroom: float = 0.0,
) -> dict[str, Any]:
    """Construct datacenters and grid for one simulation run.

    Args:
        load_shift_headroom: Fraction of extra GPU capacity for load shifting.
            0.0 disables headroom (no-shift run), >0 enables it.

    Returns:
        Dict with keys: datacenters, dc_specs, dc_bus_map, grid, exclude_buses.
    """
    sys = ieee123()
    sys["initial_taps"] = TapPosition(
        regulators={
            "creg1a": tap(9),
            "creg2a": tap(5),
            "creg3a": tap(5),
            "creg3c": tap(5),
            "creg4a": tap(13),
            "creg4b": tap(1),
            "creg4c": tap(4),
        }
    )
    exclude_buses = tuple(sys["exclude_buses"])

    datacenters: list[OfflineDatacenter] = []
    dc_specs: dict[OfflineDatacenter, tuple[InferenceModelSpec, ...]] = {}
    dc_bus_map: dict[OfflineDatacenter, str] = {}

    for site_name, site_def in _SITE_DEFS.items():
        models: tuple[ModelDeployment, ...] = site_def["models"]
        specs = tuple(m.spec for m in models)
        dc = OfflineDatacenter(
            DatacenterConfig(gpus_per_server=8, base_kw_per_phase=site_def["base_kw_per_phase"]),
            OfflineWorkload(
                inference_data=inference_data.filter_models(specs),
                replica_counts={m.spec.model_label: m.num_replicas for m in models},
            ),
            name=site_name,
            dt_s=DT_DC,
            seed=0,
            total_gpu_capacity=site_def["total_gpu_capacity"],
            load_shift_headroom=load_shift_headroom,
            power_augmentation=POWER_AUG,
        )
        datacenters.append(dc)
        dc_specs[dc] = specs
        dc_bus_map[dc] = site_def["bus"]

    grid = OpenDSSGrid(
        dss_case_dir=sys["dss_case_dir"],
        dss_master_file=sys["dss_master_file"],
        source_pu=sys["source_pu"],
        dt_s=DT_GRID,
        initial_tap_position=sys["initial_taps"],
        exclude_buses=exclude_buses,
    )
    for dc in datacenters:
        grid.attach_dc(dc, bus=dc_bus_map[dc])

    return dict(
        datacenters=datacenters,
        dc_specs=dc_specs,
        dc_bus_map=dc_bus_map,
        grid=grid,
        exclude_buses=exclude_buses,
    )


# Shift timeseries extraction


def _extract_shift_timeseries(
    log: Any,
    site_ids: list[str],
    models_by_site: dict[str, list[str]],
    gpus_per_replica: dict[str, int],
    dt_ctrl_s: float,
) -> dict[str, dict[str, list[int]]]:
    """Extract cumulative replica offset per site per model over time.

    Returns: {site_id: {model_label: [offset_at_t0, offset_at_t1, ...]}}
    """
    # Build time axis from grid states
    time_s = np.array(log.time_s)
    len(time_s)

    # Initialize offset tracking
    {sid: {m: 0 for m in models} for sid, models in models_by_site.items()}

    # Collect all ShiftReplicas commands with their approximate time
    shift_events: list[tuple[float, str, str, int]] = []  # (time, site, model, delta)
    cmd_time = 0.0
    for cmd in log.commands:
        if isinstance(cmd, ShiftReplicas):
            shift_events.append((cmd_time, cmd.target.name if cmd.target else "", cmd.model_label, cmd.replica_delta))
        # Approximate time from command order (dt_ctrl spacing)
        cmd_time += dt_ctrl_s / 2  # rough estimate

    # Build timeseries by replaying shifts
    result: dict[str, dict[str, list[int]]] = {sid: {m: [] for m in models} for sid, models in models_by_site.items()}

    # Use DC states timestamps for time axis
    dc_times: dict[str, list[float]] = {}
    for sid in site_ids:
        states = log.dc_states_by_site.get(sid, [])
        dc_times[sid] = [s.time_s for s in states]

    # Reconstruct per-site active replicas from DC states
    for sid in site_ids:
        states = log.dc_states_by_site.get(sid, [])
        for s in states:
            for m in models_by_site.get(sid, []):
                result[sid][m].append(s.active_replicas_by_model.get(m, 0))

    return result, dc_times


# Plotting


def plot_load_shift_comparison(
    log_no_shift: Any,
    log_with_shift: Any,
    stats_no_shift: VoltageStats,
    stats_with_shift: VoltageStats,
    site_ids: list[str],
    models_by_site: dict[str, list[str]],
    gpus_per_replica: dict[str, int],
    save_dir: Path,
    exclude_buses: tuple[str, ...] = (),
) -> None:
    """Create comparison plots for OFO with and without load shifting."""

    drop = {b.lower() for b in exclude_buses}

    # --- Plot 1: Voltage envelope comparison ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, log, label, stats in [
        (axes[0], log_no_shift, "OFO (no load shift)", stats_no_shift),
        (axes[1], log_with_shift, "OFO + Load Shift", stats_with_shift),
    ]:
        time_s = np.array(log.time_s)
        n_steps = len(time_s)
        vmin_t = np.full(n_steps, 2.0)
        vmax_t = np.full(n_steps, 0.0)
        for i, gs in enumerate(log.grid_states):
            all_v = []
            for bus in gs.voltages.buses():
                if bus.lower() in drop:
                    continue
                pv = gs.voltages[bus]
                for v in (pv.a, pv.b, pv.c):
                    if not math.isnan(v):
                        all_v.append(v)
            if all_v:
                vmin_t[i] = min(all_v)
                vmax_t[i] = max(all_v)

        ax.fill_between(time_s / 60, vmin_t, vmax_t, alpha=0.3, color="steelblue")
        ax.plot(time_s / 60, vmin_t, color="steelblue", linewidth=0.8, label="Vmin")
        ax.plot(time_s / 60, vmax_t, color="coral", linewidth=0.8, label="Vmax")
        ax.axhline(0.95, color="red", linestyle="--", alpha=0.6, linewidth=1)
        ax.axhline(1.05, color="red", linestyle="--", alpha=0.6, linewidth=1)
        ax.set_ylabel("Voltage (pu)", fontsize=12)
        ax.set_title(
            f"{label}  |  Viol={stats.violation_time_s:.0f}s, Integral={stats.integral_violation_pu_s:.2f} pu·s",
            fontsize=13,
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(0.93, 1.06)
        ax.grid(True, alpha=0.2)

    axes[1].set_xlabel("Time (min)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_dir / "voltage_comparison.png", dpi=150, bbox_inches="tight", metadata={"Creation Time": None})
    plt.close(fig)
    logger.info("Saved: %s", save_dir / "voltage_comparison.png")

    # --- Plot 2: Net replica shift per site per model ---
    # Only generated when ShiftReplicas commands were issued
    n_shift_cmds = sum(1 for c in log_with_shift.commands if isinstance(c, ShiftReplicas))
    if n_shift_cmds == 0:
        logger.info("No ShiftReplicas commands found — skipping replica shift plot")
    else:
        # Compute delta = active_replicas(with_shift) - active_replicas(no_shift)
        all_model_labels = sorted(set(m for ms in models_by_site.values() for m in ms))
        model_short = {m: m.replace("Llama-3.1-", "L").replace("Qwen3-", "Q") for m in all_model_labels}
        colors_models = plt.cm.tab10(np.linspace(0, 1, 10))
        model_color = {m: colors_models[i % len(colors_models)] for i, m in enumerate(all_model_labels)}

        fig, axes = plt.subplots(len(site_ids), 1, figsize=(14, 2.5 * len(site_ids)), sharex=True)
        if len(site_ids) == 1:
            axes = [axes]

        for ax_idx, sid in enumerate(site_ids):
            ax = axes[ax_idx]
            states_no = log_no_shift.dc_states_by_site.get(sid, [])
            states_ws = log_with_shift.dc_states_by_site.get(sid, [])
            site_models = models_by_site.get(sid, [])

            n = min(len(states_no), len(states_ws))
            if n == 0:
                continue
            step = max(1, n // 360)

            any_nonzero = False
            for m in site_models:
                r_no = np.array([s.active_replicas_by_model.get(m, 0) for s in states_no[:n]])
                r_ws = np.array([s.active_replicas_by_model.get(m, 0) for s in states_ws[:n]])
                delta = r_ws - r_no
                t = np.array([s.time_s for s in states_no[:n]])

                if np.any(delta != 0):
                    any_nonzero = True

                ax.plot(t[::step] / 60, delta[::step], color=model_color[m], linewidth=1.8, label=model_short[m])

            ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
            ax.set_ylabel("Net Shift\n(replicas)", fontsize=10)
            ax.set_title(f"Site {sid}", fontsize=12)
            if any_nonzero:
                ax.legend(loc="upper right", fontsize=8, ncol=len(site_models))
            ax.grid(True, alpha=0.2)

        axes[-1].set_xlabel("Time (min)", fontsize=12)
        fig.suptitle("Net Replica Shift per Site (with shift − without shift)", fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(save_dir / "net_replica_shift.png", dpi=150, bbox_inches="tight", metadata={"Creation Time": None})
        plt.close(fig)
        logger.info("Saved: %s", save_dir / "net_replica_shift.png")

    # --- Plot 3: DC load with and without load shifting (combined) ---
    fig, ax = plt.subplots(figsize=(14, 5))

    # Aggregate total DC load across all sites
    def _total_dc_load(log, site_ids):
        """Return (time_min, total_kw_per_phase) arrays."""
        all_t = None
        total_p = None
        for sid in site_ids:
            states = log.dc_states_by_site.get(sid, [])
            if not states:
                continue
            t = np.array([s.time_s for s in states])
            p = np.array([(s.power_w.a + s.power_w.b + s.power_w.c) / 3e3 for s in states])
            if all_t is None:
                all_t = t
                total_p = p
            else:
                n = min(len(all_t), len(t))
                all_t = all_t[:n]
                total_p = total_p[:n] + p[:n]
        if all_t is None:
            return np.array([]), np.array([])
        return all_t, total_p

    t_no, p_no = _total_dc_load(log_no_shift, site_ids)
    t_ws, p_ws = _total_dc_load(log_with_shift, site_ids)

    step_no = max(1, len(t_no) // 500)
    step_ws = max(1, len(t_ws) // 500)

    if len(t_no) > 0:
        ax.plot(
            t_no[::step_no] / 60, p_no[::step_no], color="steelblue", linewidth=1.5, alpha=0.8, label="OFO (no shift)"
        )
    if len(t_ws) > 0:
        ax.plot(
            t_ws[::step_ws] / 60, p_ws[::step_ws], color="coral", linewidth=1.5, alpha=0.9, label="OFO + Load Shift"
        )

    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Total DC Load (kW per phase)", fontsize=12)
    ax.set_title("Aggregate DC Load: With vs Without Load Shifting", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_dir / "dc_load_comparison.png", dpi=150, bbox_inches="tight", metadata={"Creation Time": None})
    plt.close(fig)
    logger.info("Saved: %s", save_dir / "dc_load_comparison.png")

    # --- Plot 5: Per-site DC load with and without load shifting ---
    fig, axes_dc = plt.subplots(len(site_ids), 1, figsize=(14, 3 * len(site_ids)), sharex=True)
    if len(site_ids) == 1:
        axes_dc = [axes_dc]

    for ax_idx, sid in enumerate(site_ids):
        ax = axes_dc[ax_idx]
        states_no = log_no_shift.dc_states_by_site.get(sid, [])
        states_ws = log_with_shift.dc_states_by_site.get(sid, [])

        if states_no:
            step = max(1, len(states_no) // 500)
            t = np.array([s.time_s for s in states_no])[::step]
            p = np.array([(s.power_w.a + s.power_w.b + s.power_w.c) / 3e3 for s in states_no])[::step]
            ax.plot(t / 60, p, color="steelblue", linewidth=1.2, alpha=0.8, label="No shift")

        if states_ws:
            step = max(1, len(states_ws) // 500)
            t = np.array([s.time_s for s in states_ws])[::step]
            p = np.array([(s.power_w.a + s.power_w.b + s.power_w.c) / 3e3 for s in states_ws])[::step]
            ax.plot(t / 60, p, color="coral", linewidth=1.2, alpha=0.9, label="With shift")

        ax.set_ylabel("kW/phase", fontsize=10)
        ax.set_title(f"Site {sid}", fontsize=12)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.2)

    axes_dc[-1].set_xlabel("Time (min)", fontsize=12)
    fig.suptitle("Per-Site DC Load: With vs Without Load Shifting", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_dir / "dc_load_per_site.png", dpi=150, bbox_inches="tight", metadata={"Creation Time": None})
    plt.close(fig)
    logger.info("Saved: %s", save_dir / "dc_load_per_site.png")

    # --- Plot 6: Summary bar chart ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    labels = ["OFO", "OFO + Load Shift"]
    colors = ["#4C72B0", "#DD8452"]

    vals = [stats_no_shift.violation_time_s, stats_with_shift.violation_time_s]
    axes[0].bar(labels, vals, color=colors)
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Voltage Violation Time")

    vals = [stats_no_shift.worst_vmin, stats_with_shift.worst_vmin]
    axes[1].bar(labels, vals, color=colors)
    axes[1].axhline(0.95, color="red", linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Per Unit")
    axes[1].set_title("Worst Vmin")

    vals = [stats_no_shift.integral_violation_pu_s, stats_with_shift.integral_violation_pu_s]
    axes[2].bar(labels, vals, color=colors)
    axes[2].set_ylabel("pu·s")
    axes[2].set_title("Integral Violation")

    fig.suptitle("IEEE 123-Bus: OFO vs OFO + Load Shift", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_dir / "summary_bar_chart.png", dpi=150, bbox_inches="tight", metadata={"Creation Time": None})
    plt.close(fig)
    logger.info("Saved: %s", save_dir / "summary_bar_chart.png")

    # Print summary
    n_shifts = sum(1 for c in log_with_shift.commands if isinstance(c, ShiftReplicas)) // 2
    print()
    print("=" * 80)
    print("OFO vs OFO + Load Shift Comparison")
    print("=" * 80)
    print(f"{'Metric':<30s} {'OFO':>15s} {'OFO+Shift':>15s} {'Improvement':>15s}")
    print("-" * 80)
    vt1, vt2 = stats_no_shift.violation_time_s, stats_with_shift.violation_time_s
    iv1, iv2 = stats_no_shift.integral_violation_pu_s, stats_with_shift.integral_violation_pu_s
    print(f"{'Violation time (s)':<30s} {vt1:>15.1f} {vt2:>15.1f} {(1 - vt2 / vt1) * 100 if vt1 else 0:>14.0f}%")
    print(f"{'Integral violation (pu·s)':<30s} {iv1:>15.2f} {iv2:>15.2f} {(1 - iv2 / iv1) * 100 if iv1 else 0:>14.0f}%")
    print(f"{'Worst Vmin (pu)':<30s} {stats_no_shift.worst_vmin:>15.4f} {stats_with_shift.worst_vmin:>15.4f}")
    print(f"{'Worst Vmax (pu)':<30s} {stats_no_shift.worst_vmax:>15.4f} {stats_with_shift.worst_vmax:>15.4f}")
    print(f"{'Load shift events':<30s} {'—':>15s} {n_shifts:>15d}")
    print("-" * 80)
    print(f"Outputs: {save_dir}")


# Main


def _run_experiment(
    *,
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
    ofo_config: OFOConfig,
    load_shift_enabled: bool,
    load_shift_headroom: float,
    load_shift_gpus_per_shift: int,
    save_dir: Path,
    case_name: str,
) -> tuple[VoltageStats, Any]:
    """Build experiment, construct controllers, run simulation, return stats + log."""
    exp = _build_experiment(
        inference_data,
        training_trace,
        logistic_models,
        load_shift_headroom=load_shift_headroom if load_shift_enabled else 0.0,
    )
    datacenters: list[OfflineDatacenter] = exp["datacenters"]
    dc_specs: dict[OfflineDatacenter, tuple[InferenceModelSpec, ...]] = exp["dc_specs"]
    dc_bus_map: dict[OfflineDatacenter, str] = exp["dc_bus_map"]
    grid: OpenDSSGrid = exp["grid"]
    exclude_buses: tuple[str, ...] = exp["exclude_buses"]

    # Build controllers
    controllers: list = []
    controllers.append(TapScheduleController(schedule=LOAD_SHIFT_TAP_SCHEDULE, dt_s=DT_CTRL))

    for dc, specs in dc_specs.items():
        controllers.append(
            OFOBatchSizeController(
                specs,
                datacenter=dc,
                models=logistic_models,
                config=ofo_config,
                dt_s=DT_CTRL,
            )
        )

    if load_shift_enabled:
        models_by_dc = {dc: [s.model_label for s in specs] for dc, specs in dc_specs.items()}
        all_specs_flat = {s.model_label: s for specs in dc_specs.values() for s in specs}
        controllers.append(
            LoadShiftController(
                config=LoadShiftConfig(
                    enabled=True,
                    gpus_per_shift=load_shift_gpus_per_shift,
                    headroom=load_shift_headroom,
                ),
                dt_s=DT_CTRL,
                datacenters=datacenters,
                dc_bus_map=dc_bus_map,
                models_by_dc=models_by_dc,
                gpus_per_replica_by_model={m: s.gpus_per_replica for m, s in all_specs_flat.items()},
                feasible_batch_sizes_by_model={m: s.feasible_batch_sizes for m, s in all_specs_flat.items()},
                v_min=V_MIN,
                v_max=V_MAX,
            )
        )

    coord = Coordinator(
        datacenters=datacenters,
        grid=grid,
        controllers=controllers,
        total_duration_s=TOTAL_DURATION_S,
    )
    log = coord.run()

    vstats = compute_allbus_voltage_stats(log.grid_states, v_min=V_MIN, v_max=V_MAX, exclude_buses=exclude_buses)
    logger.info(
        "  viol=%.1fs  integral=%.4f  vmin=%.4f  vmax=%.4f",
        vstats.violation_time_s,
        vstats.integral_violation_pu_s,
        vstats.worst_vmin,
        vstats.worst_vmax,
    )

    # Per-case plots and CSV
    from plotting import (
        plot_model_timeseries_4panel,
        plot_power_and_itl_2panel,
    )

    run_dir = save_dir / case_name
    run_dir.mkdir(parents=True, exist_ok=True)
    time_s = np.array(log.time_s)
    site_ids = [dc.name for dc in datacenters]

    # Per-site OFO and power/latency plots
    for dc, specs in dc_specs.items():
        site_states = log.dc_states_by_site.get(dc.name, [])
        if not site_states:
            continue
        suffix = f"_{dc.name}" if len(site_ids) > 1 else ""
        model_labels = [s.model_label for s in specs]
        plot_model_timeseries_4panel(
            site_states,
            model_labels=model_labels,
            regime_shading=False,
            save_path=run_dir / f"OFO_results{suffix}.png",
        )
        plot_power_and_itl_2panel(
            site_states,
            show_regimes=False,
            save_path=run_dir / f"power_latency{suffix}.png",
        )

    # Voltage envelope
    v_min_arr = np.full(len(log.grid_states), np.inf)
    v_max_arr = np.full(len(log.grid_states), -np.inf)
    drop = {b.lower() for b in exclude_buses}
    for t_idx, gs in enumerate(log.grid_states):
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
    ax.axhline(V_MIN, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(V_MAX, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (pu)")
    ax.set_title(f"{case_name} — Voltage Envelope")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(run_dir / "voltage_envelope.png", dpi=150, bbox_inches="tight", metadata={"Creation Time": None})
    plt.close(fig)

    # Tap positions
    if log.grid_states and log.grid_states[0].tap_positions and log.grid_states[0].tap_positions.regulators:
        TAP_STEP = 0.00625
        t_min = time_s / 60.0
        reg_names = list(log.grid_states[0].tap_positions.regulators.keys())
        tap_cmap = plt.colormaps.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(12, 4))
        for r_idx, reg_name in enumerate(reg_names):
            tap_pu = np.array(
                [gs.tap_positions.regulators.get(reg_name, 1.0) if gs.tap_positions else 1.0 for gs in log.grid_states]
            )
            tap_int = np.round((tap_pu - 1.0) / TAP_STEP).astype(int)
            ax.step(t_min, tap_int, where="post", color=tap_cmap(r_idx % 10), linewidth=2.0, alpha=0.8, label=reg_name)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Tap Position (steps)")
        ax.set_title(f"{case_name} — Tap Positions")
        ax.legend(fontsize=8, ncol=max(1, len(reg_names) // 2))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "tap_positions.png", dpi=150, bbox_inches="tight", metadata={"Creation Time": None})
        plt.close(fig)

    # Per-case CSV
    with open(run_dir / f"result_{case_name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "violation_time_s", "integral_violation_pu_s", "worst_vmin", "worst_vmax"])
        w.writerow(
            [case_name, vstats.violation_time_s, vstats.integral_violation_pu_s, vstats.worst_vmin, vstats.worst_vmax]
        )

    return vstats, log


def main(*, system: str = "ieee123") -> None:
    # Collect all unique model specs across sites
    all_models: list[ModelDeployment] = []
    seen: set[str] = set()
    for site_def in _SITE_DEFS.values():
        for m in site_def["models"]:
            if m.spec.model_label not in seen:
                all_models.append(m)
                seen.add(m.spec.model_label)
    all_specs = tuple(m.spec for m in all_models)

    # Load data pipeline
    data_sources, training_trace_params, data_dir = load_data_sources()

    logger.info("Loading inference data...")
    inference_data = InferenceData.ensure(
        data_dir,
        all_specs,
        data_sources,
        plot=False,
        dt_s=float(DT_DC),
    )
    logistic_models = LogisticModelStore.ensure(
        data_dir / "logistic_fits.csv",
        all_specs,
        data_sources,
        plot=False,
    )
    training_trace = TrainingTrace.ensure(
        data_dir / "training_trace.csv",
        training_trace_params,
    )

    save_dir = Path(__file__).resolve().parent / "outputs" / system / "load_shift_comparison"
    save_dir.mkdir(parents=True, exist_ok=True)

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

    shared = dict(
        inference_data=inference_data,
        training_trace=training_trace,
        logistic_models=logistic_models,
        ofo_config=ofo_config,
        save_dir=save_dir,
    )

    # --- Run 1: OFO without load shift ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("RUN 1: OFO without load shifting")
    logger.info("=" * 70)

    stats_no_shift, log_no_shift = _run_experiment(
        **shared,
        load_shift_enabled=False,
        load_shift_headroom=0.0,
        load_shift_gpus_per_shift=LOAD_SHIFT_GPUS_PER_SHIFT,
        case_name="ofo_no_shift",
    )

    # --- Run 2: OFO with load shift ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("RUN 2: OFO with load shifting")
    logger.info("=" * 70)

    stats_with_shift, log_with_shift = _run_experiment(
        **shared,
        load_shift_enabled=True,
        load_shift_headroom=LOAD_SHIFT_HEADROOM,
        load_shift_gpus_per_shift=LOAD_SHIFT_GPUS_PER_SHIFT,
        case_name="ofo_with_shift",
    )

    # --- Plots ---
    site_ids = list(_SITE_DEFS.keys())
    models_by_site = {sid: [m.spec.model_label for m in site_def["models"]] for sid, site_def in _SITE_DEFS.items()}
    gpus_per_replica = {m.spec.model_label: m.spec.gpus_per_replica for m in all_models}

    sys_const = ieee123()
    sys_const["initial_taps"] = TapPosition(
        regulators={
            "creg1a": tap(9),
            "creg2a": tap(5),
            "creg3a": tap(5),
            "creg3c": tap(5),
            "creg4a": tap(13),
            "creg4b": tap(1),
            "creg4c": tap(4),
        }
    )

    plot_load_shift_comparison(
        log_no_shift,
        log_with_shift,
        stats_no_shift,
        stats_with_shift,
        site_ids,
        models_by_site,
        gpus_per_replica,
        save_dir,
        exclude_buses=tuple(sys_const["exclude_buses"]),
    )


if __name__ == "__main__":
    from dataclasses import dataclass

    import tyro

    @dataclass
    class Args:
        system: str = "ieee123"
        """System name (used for output directory)."""
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

    main(system=args.system)
