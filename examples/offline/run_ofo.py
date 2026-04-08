"""Baseline + OFO comparison for any IEEE test system.

Constructs datacenters, grid, and controllers programmatically for each
system, then runs baseline and/or OFO simulations and produces comparison
plots.  All experiment parameters (models, DC sites, controller tuning,
workload scenarios) are defined inline — no external JSON config required.

Modes:
    no-tap       Baseline + OFO without tap schedule (2 cases, default).
    tap-change   Baseline + OFO with tap schedule (2 cases).
    both         Baseline only, with and without tap schedule (2 cases).
    all          All 4 cases: baseline + OFO x with/without tap schedule.

Usage:
    python run_ofo.py --system ieee13
    python run_ofo.py --system ieee34
    python run_ofo.py --system ieee123
    python run_ofo.py --system ieee34 --mode tap-change
    python run_ofo.py --system ieee34 --mode all
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from plot_all_figures import (
    extract_per_model_timeseries,
    plot_allbus_voltages_per_phase,
    plot_model_timeseries_4panel,
    plot_power_and_itl_2panel,
    plot_zone_voltage_envelope,
)
from sweep_dc_locations import ScenarioOpenDSSGrid, eval_profile, load_csv_profile, load_profile_kw, pv_profile_kw
from systems import (
    DT_CTRL,
    DT_DC,
    DT_GRID,
    POWER_AUG,
    SYSTEMS,
    TAP_STEP,
    TOTAL_DURATION_S,
    V_MAX,
    V_MIN,
    DCSite,
    PVSystemSpec,
    TimeVaryingLoadSpec,
    all_model_specs,
    deploy,
    load_data_sources,
    tap,
)

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
    PowerAugmentationConfig,
    TrainingRun,
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.datacenter.workloads.training import TrainingTrace
from openg2g.grid.config import DCLoadSpec, TapPosition, TapSchedule
from openg2g.metrics.voltage import VoltageStats, compute_allbus_voltage_stats

MAX_BUSES_FOR_INDIVIDUAL_LINES = 30

logger = logging.getLogger("run_ofo")


# ── Plotting helpers ─────────────────────────────────────────────────────────


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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    fig.savefig(save_dir / "voltage_envelope.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_tap_positions(grid_states, time_s, save_dir, *, title="Regulator Tap Positions"):
    """Plot regulator tap positions over time as step lines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    fig.savefig(save_dir / "tap_positions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pv_load_profiles(grid_states, time_s, pv_systems, time_varying_loads, save_dir):
    """Plot PV output and time-varying load curves over the simulation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not grid_states:
        return

    n_pv = len(pv_systems)
    n_load = len(time_varying_loads)
    n_panels = n_pv + n_load
    if n_panels == 0:
        return

    t_arr = np.linspace(0, float(time_s[-1]), 4000)
    t_min = t_arr / 60.0

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for i, spec in enumerate(pv_systems):
        csv_data = load_csv_profile(spec.csv_path) if spec.csv_path else None
        kw_arr = np.array(
            [
                eval_profile(t, peak_kw=spec.peak_kw, csv_data=csv_data, profile_fn=pv_profile_kw, site_idx=i)
                for t in t_arr
            ]
        )
        axes[i].plot(t_min, kw_arr, "orange", linewidth=1.5)
        axes[i].set_ylabel("kW/phase", fontsize=12)
        axes[i].set_title(f"PV @ bus {spec.bus} (peak={spec.peak_kw:.0f} kW/ph)", fontsize=13)
        axes[i].axhline(0, color="gray", linewidth=0.5)
        axes[i].set_ylim(bottom=-2)
        axes[i].grid(True, alpha=0.3)

    for i, spec in enumerate(time_varying_loads):
        csv_data = load_csv_profile(spec.csv_path) if spec.csv_path else None
        kw_arr = np.array(
            [
                eval_profile(t, peak_kw=spec.peak_kw, csv_data=csv_data, profile_fn=load_profile_kw, site_idx=i)
                for t in t_arr
            ]
        )
        ax = axes[n_pv + i]
        ax.plot(t_min, kw_arr, "steelblue", linewidth=1.5)
        ax.set_ylabel("kW/phase", fontsize=12)
        ax.set_title(f"Load @ bus {spec.bus} (peak={spec.peak_kw:.0f} kW/ph)", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Time (min)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_dir / "pv_load_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_comparison(results: dict[str, VoltageStats], save_dir: Path, system: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    fig.savefig(save_dir / f"results_{system}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison plot saved.")


# ── Single simulation run ────────────────────────────────────────────────────


def run_mode(
    mode: str,
    *,
    # System
    sys: dict,
    # DC sites
    dc_sites: dict[str, DCSite],
    # Data
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
    # Workload (applied to all sites unless site has its own ramps)
    training: TrainingRun | None = None,
    # Controller
    ofo_config: OFOConfig | None = None,
    tap_schedule: TapSchedule | None = None,
    # Scenario grid extras
    pv_systems: list[PVSystemSpec] | None = None,
    time_varying_loads: list[TimeVaryingLoadSpec] | None = None,
    # Simulation
    total_duration_s: int = TOTAL_DURATION_S,
    power_augmentation: PowerAugmentationConfig = POWER_AUG,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    # Load shift
    load_shift_enabled: bool = False,
    load_shift_gpus_per_shift: int = 8,
    load_shift_headroom: float = 0.3,
    # Output
    save_dir: Path,
    folder_name: str,
    # Optional zone info for plotting
    zones: dict[str, list[str]] | None = None,
) -> tuple[VoltageStats, Any]:
    """Run 'baseline' or 'ofo' and return (VoltageStats, SimulationLog)."""
    run_dir = save_dir / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pv_systems = pv_systems or []
    time_varying_loads = time_varying_loads or []
    exclude_buses = tuple(sys["exclude_buses"])
    site_ids = list(dc_sites.keys())
    all_specs: list[InferenceModelSpec] = []
    for site in dc_sites.values():
        all_specs.extend(md.spec for md in site.models)

    dc_loads: dict[str, DCLoadSpec] = {}
    datacenters: dict[str, OfflineDatacenter] = {}
    controllers: list = []
    site_models_map: dict[str, tuple[InferenceModelSpec, ...]] = {}
    primary_bus = ""

    for site_id, site in dc_sites.items():
        site_specs = tuple(md.spec for md in site.models)
        site_models_map[site_id] = site_specs
        site_inference = inference_data.filter_models(site_specs)
        replica_counts = {md.spec.model_label: md.num_replicas for md in site.models}
        batch_sizes = {md.spec.model_label: md.initial_batch_size for md in site.models}

        dc_config = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=site.base_kw_per_phase)

        workload_kwargs: dict[str, Any] = {
            "inference_data": site_inference,
            "replica_counts": replica_counts,
            "initial_batch_sizes": batch_sizes,
        }
        if site.inference_ramps is not None:
            workload_kwargs["inference_ramps"] = site.inference_ramps
        if training is not None:
            workload_kwargs["training"] = training
        workload = OfflineWorkload(**workload_kwargs)

        dc = OfflineDatacenter(
            dc_config,
            workload,
            dt_s=DT_DC,
            seed=site.seed,
            power_augmentation=power_augmentation,
            total_gpu_capacity=site.total_gpu_capacity,
            load_shift_headroom=site.load_shift_headroom,
        )
        datacenters[site_id] = dc
        dc_loads[site_id] = DCLoadSpec(bus=site.bus, bus_kv=site.bus_kv, connection_type=site.connection_type)
        if not primary_bus:
            primary_bus = site.bus

    # Grid
    if len(site_ids) == 1:
        sid = site_ids[0]
        site = dc_sites[sid]
        grid = ScenarioOpenDSSGrid(
            pv_systems=pv_systems,
            time_varying_loads=time_varying_loads,
            source_pu=sys["source_pu"],
            dss_case_dir=sys["dss_case_dir"],
            dss_master_file=sys["dss_master_file"],
            dc_bus=site.bus,
            dc_bus_kv=site.bus_kv,
            power_factor=DatacenterConfig(base_kw_per_phase=0).power_factor,
            dt_s=DT_GRID,
            connection_type=site.connection_type,
            initial_tap_position=sys["initial_taps"],
        )
    else:
        grid = ScenarioOpenDSSGrid(
            pv_systems=pv_systems,
            time_varying_loads=time_varying_loads,
            source_pu=sys["source_pu"],
            dss_case_dir=sys["dss_case_dir"],
            dss_master_file=sys["dss_master_file"],
            dc_loads=dc_loads,
            power_factor=DatacenterConfig(base_kw_per_phase=0).power_factor,
            dt_s=DT_GRID,
            initial_tap_position=sys["initial_taps"],
            exclude_buses=exclude_buses,
        )

    # Tap controller
    sched = tap_schedule if tap_schedule is not None else TapSchedule(())
    controllers.append(TapScheduleController(schedule=sched, dt_s=DT_CTRL))

    # OFO controllers
    if mode == "ofo" and ofo_config is not None:
        for site_id in site_ids:
            site_batch_sizes = {md.spec.model_label: md.initial_batch_size for md in dc_sites[site_id].models}
            controllers.append(
                OFOBatchSizeController(
                    site_models_map[site_id],
                    models=logistic_models,
                    config=ofo_config,
                    dt_s=DT_CTRL,
                    site_id=site_id if len(site_ids) > 1 else None,
                    initial_batch_sizes=site_batch_sizes,
                )
            )

    # Load shift controller
    if mode == "ofo" and load_shift_enabled and len(site_ids) > 1:
        from openg2g.controller.load_shift import LoadShiftConfig, LoadShiftController

        ls_cfg = LoadShiftConfig(enabled=True, gpus_per_shift=load_shift_gpus_per_shift, headroom=load_shift_headroom)
        controllers.append(
            LoadShiftController(
                config=ls_cfg,
                dt_s=DT_CTRL,
                datacenters=datacenters,
                site_bus_map={sid: dc_sites[sid].bus for sid in site_ids},
                models_by_site={sid: [m.model_label for m in ms] for sid, ms in site_models_map.items()},
                gpus_per_replica_by_model={m.model_label: m.gpus_per_replica for m in all_specs},
                feasible_batch_sizes_by_model={m.model_label: list(m.feasible_batch_sizes) for m in all_specs},
                v_min=v_min,
                v_max=v_max,
            )
        )
        logger.info(
            "LoadShiftController enabled: gpus_per_shift=%d, headroom=%.1f",
            load_shift_gpus_per_shift,
            load_shift_headroom,
        )

    # Run
    logger.info("=== %s (%s) ===", mode.upper(), folder_name)

    coord = Coordinator(
        datacenters=datacenters,
        grid=grid,
        controllers=controllers,
        total_duration_s=total_duration_s,
        dc_bus=primary_bus,
    )
    log = coord.run()

    vstats = compute_allbus_voltage_stats(log.grid_states, v_min=v_min, v_max=v_max, exclude_buses=exclude_buses)
    logger.info(
        "  %s: viol=%.1fs  integral=%.4f  vmin=%.4f  vmax=%.4f",
        folder_name,
        vstats.violation_time_s,
        vstats.integral_violation_pu_s,
        vstats.worst_vmin,
        vstats.worst_vmax,
    )

    # Plots
    time_s = np.array(log.time_s)

    # Count monitored buses
    all_buses = set()
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
            v_min=v_min,
            v_max=v_max,
            drop_buses=exclude_buses,
            title_template=f"{folder_name} — Voltage Envelope (Phase {{label}})",
        )
    elif len(all_buses) > MAX_BUSES_FOR_INDIVIDUAL_LINES:
        _plot_voltage_envelope(
            log.grid_states,
            time_s,
            run_dir,
            v_min=v_min,
            v_max=v_max,
            exclude_buses=exclude_buses,
            title=f"{folder_name} — Voltage Envelope",
        )
    else:
        plot_allbus_voltages_per_phase(
            log.grid_states,
            time_s,
            save_dir=run_dir,
            v_min=v_min,
            v_max=v_max,
            title_template=f"{folder_name} — Voltage (Phase {{label}})",
            drop_buses=exclude_buses,
        )

    if mode == "ofo":
        for site_id in site_ids:
            site_states = log.dc_states_by_site.get(site_id, log.dc_states)
            if not site_states:
                continue
            suffix = f"_{site_id}" if len(site_ids) > 1 else ""
            per_model = extract_per_model_timeseries(site_states)
            model_labels = [m.model_label for m in site_models_map[site_id]]
            plot_model_timeseries_4panel(
                per_model.time_s,
                per_model,
                model_labels=model_labels,
                regime_shading=False,
                save_path=run_dir / f"OFO_results{suffix}.png",
            )

    for site_id in site_ids:
        site_states = log.dc_states_by_site.get(site_id, log.dc_states)
        if not site_states:
            continue
        suffix = f"_{site_id}" if len(site_ids) > 1 else ""
        per_model = extract_per_model_timeseries(site_states)
        kW_A = np.array([s.power_w.a / 1e3 for s in site_states])
        kW_B = np.array([s.power_w.b / 1e3 for s in site_states])
        kW_C = np.array([s.power_w.c / 1e3 for s in site_states])
        plot_power_and_itl_2panel(
            per_model.time_s,
            kW_A,
            kW_B,
            kW_C,
            avg_itl_by_model=per_model.itl_s,
            show_regimes=False,
            save_path=run_dir / f"power_latency{suffix}.png",
        )

    _plot_tap_positions(log.grid_states, time_s, run_dir, title=f"{folder_name} — Tap Positions")
    _plot_pv_load_profiles(log.grid_states, time_s, pv_systems, time_varying_loads, run_dir)

    with open(run_dir / f"result_{folder_name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "violation_time_s", "integral_violation_pu_s", "worst_vmin", "worst_vmax"])
        writer.writerow(
            [folder_name, vstats.violation_time_s, vstats.integral_violation_pu_s, vstats.worst_vmin, vstats.worst_vmax]
        )
    logger.info("Per-case CSV: %s", run_dir / f"result_{folder_name}.csv")

    return vstats, log


# ══════════════════════════════════════════════════════════════════════════════
# Per-system experiment definitions
# ══════════════════════════════════════════════════════════════════════════════


def _experiment_ieee13(sys, inference_data, training_trace, logistic_models):
    """IEEE 13-bus: single DC at bus 671 with training overlay."""
    models = (
        deploy("Llama-3.1-8B", 720),
        deploy("Llama-3.1-70B", 180),
        deploy("Llama-3.1-405B", 90),
        deploy("Qwen3-30B-A3B", 480),
        deploy("Qwen3-235B-A22B", 210),
    )

    # Ramp all 5 models down to 20% of their initial replicas at t=2500..3000
    ramps = (
        InferenceRamp(target=144, model="Llama-3.1-8B").at(t_start=2500, t_end=3000)
        | InferenceRamp(target=36, model="Llama-3.1-70B").at(t_start=2500, t_end=3000)
        | InferenceRamp(target=18, model="Llama-3.1-405B").at(t_start=2500, t_end=3000)
        | InferenceRamp(target=96, model="Qwen3-30B-A3B").at(t_start=2500, t_end=3000)
        | InferenceRamp(target=42, model="Qwen3-235B-A22B").at(t_start=2500, t_end=3000)
    )

    dc_sites = {
        "default": DCSite(
            bus="671",
            bus_kv=sys["bus_kv"],
            base_kw_per_phase=500.0,
            total_gpu_capacity=7200,
            models=models,
            seed=0,
            inference_ramps=ramps,
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

    tap_schedule = TapSchedule(
        (
            (1500, TapPosition(regulators={"creg1a": tap(16), "creg1b": tap(6), "creg1c": tap(17)})),
            (3300, TapPosition(regulators={"creg1a": tap(10), "creg1b": tap(6), "creg1c": tap(10)})),
        )
    )

    pv_systems = [PVSystemSpec(bus="675", bus_kv=4.16, peak_kw=10.0)]
    time_varying_loads = [TimeVaryingLoadSpec(bus="680", bus_kv=4.16, peak_kw=10.0)]

    return dict(
        dc_sites=dc_sites,
        training=training,
        ofo_config=ofo_config,
        tap_schedule=tap_schedule,
        pv_systems=pv_systems,
        time_varying_loads=time_varying_loads,
    )


def _experiment_ieee34(sys, inference_data, training_trace, logistic_models):
    """IEEE 34-bus: two DC sites (upstream + downstream)."""
    dc_sites = {
        "upstream": DCSite(
            bus="850",
            bus_kv=sys["bus_kv"],
            base_kw_per_phase=120.0,
            total_gpu_capacity=520,
            models=(
                deploy("Llama-3.1-8B", 720),
                deploy("Llama-3.1-70B", 180),
                deploy("Llama-3.1-405B", 90),
            ),
            seed=0,
        ),
        "downstream": DCSite(
            bus="834",
            bus_kv=sys["bus_kv"],
            base_kw_per_phase=80.0,
            total_gpu_capacity=600,
            models=(
                deploy("Qwen3-30B-A3B", 480),
                deploy("Qwen3-235B-A22B", 210),
            ),
            seed=42,
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

    tap_schedule = TapSchedule(
        ((1800, TapPosition(regulators={"creg2a": tap(10), "creg2b": tap(10), "creg2c": tap(10)})),)
    )

    pv_systems = [
        PVSystemSpec(bus="848", bus_kv=24.9, peak_kw=130.0),
        PVSystemSpec(bus="830", bus_kv=24.9, peak_kw=65.0),
    ]
    time_varying_loads = [
        TimeVaryingLoadSpec(bus="860", bus_kv=24.9, peak_kw=80.0),
        TimeVaryingLoadSpec(bus="844", bus_kv=24.9, peak_kw=120.0),
        TimeVaryingLoadSpec(bus="840", bus_kv=24.9, peak_kw=60.0),
        TimeVaryingLoadSpec(bus="858", bus_kv=24.9, peak_kw=50.0),
        TimeVaryingLoadSpec(bus="854", bus_kv=24.9, peak_kw=40.0),
    ]

    return dict(
        dc_sites=dc_sites,
        training=None,
        ofo_config=ofo_config,
        tap_schedule=tap_schedule,
        pv_systems=pv_systems,
        time_varying_loads=time_varying_loads,
    )


def _experiment_ieee123(sys, inference_data, training_trace, logistic_models):
    """IEEE 123-bus: four DC zones with per-site ramps."""
    dc_sites = {
        "z1_sw": DCSite(
            bus="8",
            bus_kv=sys["bus_kv"],
            base_kw_per_phase=310.0,
            total_gpu_capacity=180,  # 120 initial + headroom for 1.5× ramp
            models=(deploy("Llama-3.1-8B", 120),),
            seed=0,
            inference_ramps=InferenceRamp(target=180, model="Llama-3.1-8B").at(t_start=500, t_end=1000),
        ),
        "z2_nw": DCSite(
            bus="23",
            bus_kv=sys["bus_kv"],
            base_kw_per_phase=265.0,
            total_gpu_capacity=208,  # 160 initial + headroom for 1.3× ramp
            models=(deploy("Qwen3-30B-A3B", 80),),
            seed=17,
            inference_ramps=InferenceRamp(target=104, model="Qwen3-30B-A3B").at(t_start=1500, t_end=2500),
        ),
        "z3_se": DCSite(
            bus="60",
            bus_kv=sys["bus_kv"],
            base_kw_per_phase=295.0,
            total_gpu_capacity=600,  # 400 initial + headroom for 1.5× ramp
            models=(deploy("Llama-3.1-70B", 30), deploy("Llama-3.1-405B", 35)),
            seed=34,
            inference_ramps=(
                InferenceRamp(target=45, model="Llama-3.1-70B").at(t_start=700, t_end=1100)
                | InferenceRamp(target=52, model="Llama-3.1-405B").at(t_start=700, t_end=1100)
            ),
        ),
        "z4_ne": DCSite(
            bus="105",
            bus_kv=sys["bus_kv"],
            base_kw_per_phase=325.0,
            total_gpu_capacity=440,
            models=(deploy("Qwen3-235B-A22B", 55),),
            seed=51,
            inference_ramps=InferenceRamp(target=27, model="Qwen3-235B-A22B").at(t_start=2000, t_end=2500),
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

    tap_schedule = TapSchedule(((1800, TapPosition(regulators={"creg4a": tap(16)})),))

    pv_systems = [
        PVSystemSpec(bus="1", bus_kv=4.16, peak_kw=333.3),
        PVSystemSpec(bus="48", bus_kv=4.16, peak_kw=333.3),
        PVSystemSpec(bus="99", bus_kv=4.16, peak_kw=333.3),
    ]

    return dict(
        dc_sites=dc_sites,
        training=None,
        ofo_config=ofo_config,
        tap_schedule=tap_schedule,
        pv_systems=pv_systems,
        time_varying_loads=[],
        zones=sys.get("zones"),
    )


_EXPERIMENTS = {
    "ieee13": _experiment_ieee13,
    "ieee34": _experiment_ieee34,
    "ieee123": _experiment_ieee123,
}


# ── Main ─────────────────────────────────────────────────────────────────────


def main(*, system: str, mode: str = "no-tap") -> None:
    sys = SYSTEMS[system]()

    # Load data pipeline
    data_sources, training_trace_params, data_dir = load_data_sources()

    # Build experiment — this defines models inline, so we collect them for data loading
    # We need a temporary call to get models for data loading
    experiment_fn = _EXPERIMENTS[system]

    # Pre-load data using all model specs (data pipeline generates for all data_sources)
    specs = all_model_specs()

    logger.info("Loading data for %s...", system)
    inference_data = InferenceData.ensure(
        data_dir,
        specs,
        data_sources,
        plot=False,
        dt_s=float(DT_DC),
    )
    training_trace = TrainingTrace.ensure(data_dir / "training_trace.csv", training_trace_params)
    logistic_models = LogisticModelStore.ensure(
        data_dir / "logistic_fits.csv",
        specs,
        data_sources,
        plot=False,
    )

    # Build experiment config
    experiment = experiment_fn(sys, inference_data, training_trace, logistic_models)
    has_tap_schedule = bool(experiment.get("tap_schedule"))

    # Build cases
    cases: list[tuple[str, str, TapSchedule | None]] = []
    if mode in ("no-tap", "both", "all"):
        cases.append(("baseline", "baseline_no-tap", None))
    if mode in ("no-tap", "all"):
        cases.append(("ofo", "ofo_no-tap", None))
    if mode in ("tap-change", "both", "all") and has_tap_schedule:
        cases.append(("baseline", "baseline_tap-change", experiment["tap_schedule"]))
    if mode in ("tap-change", "all") and has_tap_schedule:
        cases.append(("ofo", "ofo_tap-change", experiment["tap_schedule"]))
    if not cases:
        logger.warning("No cases to run (mode=%s, has_tap_schedule=%s).", mode, has_tap_schedule)
        return

    save_dir = Path(__file__).resolve().parent / "outputs" / system
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running %d cases for %s: %s", len(cases), system, [c[1] for c in cases])

    results: dict[str, VoltageStats] = {}
    for ctrl_mode, folder, sched in cases:
        stats, _log = run_mode(
            ctrl_mode,
            sys=sys,
            dc_sites=experiment["dc_sites"],
            inference_data=inference_data,
            training_trace=training_trace,
            logistic_models=logistic_models,
            training=experiment.get("training"),
            ofo_config=experiment["ofo_config"],
            tap_schedule=sched,
            pv_systems=experiment.get("pv_systems"),
            time_varying_loads=experiment.get("time_varying_loads"),
            zones=experiment.get("zones"),
            save_dir=save_dir,
            folder_name=folder,
        )
        results[folder] = stats

    # Comparison CSV
    csv_path = save_dir / f"results_{system}_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "violation_time_s", "worst_vmin", "worst_vmax", "integral_violation_pu_s"])
        for case_name, s in results.items():
            writer.writerow([case_name, s.violation_time_s, s.worst_vmin, s.worst_vmax, s.integral_violation_pu_s])
    logger.info("Comparison CSV: %s", csv_path)

    _plot_comparison(results, save_dir, system)

    # Summary table
    logger.info("")
    logger.info("=" * 90)
    logger.info("%s Comparison", system.upper())
    logger.info("=" * 90)
    logger.info("%-22s %10s %10s %10s %14s", "Mode", "Viol(s)", "Vmin", "Vmax", "Integral")
    logger.info("-" * 90)
    for case_name, s in results.items():
        logger.info(
            "%-22s %10.1f %10.4f %10.4f %14.4f",
            case_name,
            s.violation_time_s,
            s.worst_vmin,
            s.worst_vmax,
            s.integral_violation_pu_s,
        )
    logger.info("-" * 90)
    logger.info("Outputs: %s", save_dir)


if __name__ == "__main__":
    from dataclasses import dataclass as dc

    import tyro

    @dc
    class Args:
        system: str = "ieee13"
        """System name (ieee13, ieee34, ieee123)."""
        mode: str = "no-tap"
        """Run mode: 'no-tap', 'tap-change', 'both', or 'all'."""
        log_level: str = "INFO"
        """Logging verbosity (DEBUG, INFO, WARNING)."""

    args = tyro.cli(Args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    main(system=args.system, mode=args.mode)
