"""Unified OFO simulation for any IEEE test system.

Reads the config JSON, detects the system structure (single or multi-DC,
with or without PV/loads/tap schedule), and runs baseline + OFO comparison.

Modes:
    no-tap       Baseline + OFO without tap schedule (2 cases, default).
    tap-change   Baseline + OFO with tap schedule from config (2 cases).
    both         Baseline only, with and without tap schedule (2 cases).
    all          All 4 cases: baseline + OFO × with/without tap schedule.

Default outputs per case folder:
    allbus_voltages_phase_{A,B,C}.png   Per-phase voltage trajectories
    tap_positions.png                   Regulator tap positions over time
    power_latency_{site}.png            Per-site 3-phase power + ITL
    pv_load_profiles.png                PV and time-varying load curves
    result_{case}.csv                   Voltage violation summary
    OFO_results_{site}.png              (OFO cases only) batch/power/ITL/throughput

Comparison outputs (in parent directory):
    results_{system}_comparison.csv     All cases side-by-side
    results_{system}_comparison.png     Bar chart comparing violation metrics

Usage:
    # IEEE 13 — baseline + OFO, no tap schedule
    python run_ofo.py --config config_ieee13.json --system ieee13

    # IEEE 34 — baseline + OFO, no tap schedule
    python run_ofo.py --config config_ieee34.json --system ieee34

    # IEEE 123 — baseline + OFO, no tap schedule
    python run_ofo.py --config config_ieee123.json --system ieee123

    # Baseline + OFO with tap schedule
    python run_ofo.py --config config_ieee34.json --system ieee34 --mode tap-change

    # Baseline only, with and without tap schedule
    python run_ofo.py --config config_ieee34.json --system ieee34 --mode both

    # All 4 cases
    python run_ofo.py --config config_ieee34.json --system ieee34 --mode all
"""

from __future__ import annotations

import csv
import logging
import math
from fractions import Fraction
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
from sweep_dc_locations import (
    TAP_STEP,
    ScenarioOpenDSSGrid,
    SweepConfig,
    _build_workload_kwargs,
    _parse_fraction,
    _resolve_models_for_site,
    _taps_dict_to_position,
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
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.datacenter.workloads.training import TrainingTrace
from openg2g.grid.config import DCLoadSpec, TapPosition, TapSchedule
from openg2g.metrics.voltage import VoltageStats, compute_allbus_voltage_stats

MAX_BUSES_FOR_INDIVIDUAL_LINES = 30

logger = logging.getLogger("run_ofo")


# ── Build tap schedule from config entries ────────────────────────────────────


def _build_tap_schedule(
    entries: list,
    initial_taps: dict[str, float | str] | None,
) -> TapSchedule:
    """Build a TapSchedule from config tap_schedule entries."""
    schedule_entries = []
    for entry in sorted(entries, key=lambda e: e.t):
        t = entry.t
        extras = entry.model_extra or {}
        regs = {}
        for k, v in extras.items():
            if isinstance(v, str):
                regs[k] = 1.0 + int(v) * TAP_STEP
            else:
                regs[k] = float(v)
        schedule_entries.append((t, TapPosition(regulators=regs)))
    return TapSchedule(tuple(schedule_entries))


# ── Simple voltage envelope (for systems with many buses, no zones) ───────────


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


# ── Tap position plot ────────────────────────────────────────────────────────


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


# ── PV & load profile plot ───────────────────────────────────────────────────


def _plot_pv_load_profiles(grid_states, time_s, config, save_dir):
    """Plot actual PV output and time-varying load curves over the simulation.

    Each PV system and each time-varying load gets its own subplot, matching
    the style of the original ``run_ieee34_ofo._plot_event_profiles``.
    Evaluates the same profile functions used by ``ScenarioOpenDSSGrid``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sweep_dc_locations import eval_profile, load_csv_profile, load_profile_kw, pv_profile_kw

    if not grid_states:
        return

    pv_configs = config.pv_systems or []
    tvl_configs = config.time_varying_loads or []
    n_pv = len(pv_configs)
    n_load = len(tvl_configs)
    n_panels = n_pv + n_load
    if n_panels == 0:
        return

    t_arr = np.linspace(0, float(time_s[-1]), 4000)
    t_min = t_arr / 60.0

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for i, spec in enumerate(pv_configs):
        csv_data = load_csv_profile(spec.csv_path) if getattr(spec, "csv_path", None) else None
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

    for i, spec in enumerate(tvl_configs):
        csv_data = load_csv_profile(spec.csv_path) if getattr(spec, "csv_path", None) else None
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


# ── Single simulation run ─────────────────────────────────────────────────────


def run_mode(
    mode: str,
    *,
    config: SweepConfig,
    all_models: tuple[InferenceModelSpec, ...],
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
    dt_dc: Fraction,
    dt_grid: Fraction,
    dt_ctrl: Fraction,
    save_dir: Path,
    tap_schedule: TapSchedule | None = None,
    folder_name: str | None = None,
) -> tuple[VoltageStats, Any]:
    """Run 'baseline' or 'ofo' and return (VoltageStats, SimulationLog)."""
    sim = config.simulation
    if folder_name is None:
        folder_name = "baseline_no-tap" if mode == "baseline" else mode
    run_dir = save_dir / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    site_ids = list(config.dc_sites.keys())
    dc_loads: dict[str, DCLoadSpec] = {}
    datacenters: dict[str, OfflineDatacenter] = {}
    controllers: list = []
    site_models_map: dict[str, tuple[InferenceModelSpec, ...]] = {}
    primary_bus = ""

    for site_id, site_cfg in config.dc_sites.items():
        site_models = _resolve_models_for_site(site_cfg, all_models)
        site_models_map[site_id] = site_models
        site_inference = inference_data.filter_models(site_models) if site_cfg.models is not None else inference_data

        dc_config = DatacenterConfig(
            gpus_per_server=8,
            base_kw_per_phase=site_cfg.base_kw_per_phase,
        )
        workload_kwargs = _build_workload_kwargs(
            config,
            site_inference,
            training_trace,
            site_ramps=site_cfg.inference_ramps if site_cfg.inference_ramps else None,
        )
        workload = OfflineWorkload(**workload_kwargs)

        ls_headroom = 0.0
        if config.load_shift and config.load_shift.get("enabled", False):
            ls_headroom = config.load_shift.get("headroom", 0.3)
        dc = OfflineDatacenter(
            dc_config,
            workload,
            dt_s=dt_dc,
            seed=site_cfg.seed,
            power_augmentation=config.power_augmentation,
            total_gpu_capacity=site_cfg.total_gpu_capacity,
            load_shift_headroom=ls_headroom,
        )
        datacenters[site_id] = dc
        dc_loads[site_id] = DCLoadSpec(
            bus=site_cfg.bus,
            bus_kv=site_cfg.bus_kv,
            connection_type=site_cfg.connection_type,
        )
        if not primary_bus:
            primary_bus = site_cfg.bus

    # Grid
    initial_taps = _taps_dict_to_position(config.initial_taps)
    exclude_buses = tuple(config.exclude_buses)

    if len(site_ids) == 1:
        sid = site_ids[0]
        grid = ScenarioOpenDSSGrid(
            pv_systems=config.pv_systems,
            time_varying_loads=config.time_varying_loads,
            source_pu=config.source_pu,
            dss_case_dir=config.ieee_case_dir,
            dss_master_file=config.dss_master_file,
            dc_bus=config.dc_sites[sid].bus,
            dc_bus_kv=config.dc_sites[sid].bus_kv,
            power_factor=DatacenterConfig(base_kw_per_phase=0).power_factor,
            dt_s=dt_grid,
            connection_type=config.dc_sites[sid].connection_type,
            initial_tap_position=initial_taps,
        )
    else:
        grid = ScenarioOpenDSSGrid(
            pv_systems=config.pv_systems,
            time_varying_loads=config.time_varying_loads,
            source_pu=config.source_pu,
            dss_case_dir=config.ieee_case_dir,
            dss_master_file=config.dss_master_file,
            dc_loads=dc_loads,
            power_factor=DatacenterConfig(base_kw_per_phase=0).power_factor,
            dt_s=dt_grid,
            initial_tap_position=initial_taps,
            exclude_buses=exclude_buses,
        )

    # Tap controller
    sched = tap_schedule if tap_schedule is not None else TapSchedule(())
    controllers.append(TapScheduleController(schedule=sched, dt_s=dt_ctrl))

    # OFO controllers (only in ofo mode)
    if mode == "ofo":
        ofo_params = config.ofo
        ofo_config = OFOConfig(
            primal_step_size=ofo_params.primal_step_size,
            w_throughput=ofo_params.w_throughput,
            w_switch=ofo_params.w_switch,
            voltage_gradient_scale=ofo_params.voltage_gradient_scale,
            v_min=sim.v_min,
            v_max=sim.v_max,
            voltage_dual_step_size=ofo_params.voltage_dual_step_size,
            latency_dual_step_size=ofo_params.latency_dual_step_size,
            sensitivity_update_interval=ofo_params.sensitivity_update_interval,
            sensitivity_perturbation_kw=ofo_params.sensitivity_perturbation_kw,
        )
        for site_id in site_ids:
            ofo_ctrl = OFOBatchSizeController(
                site_models_map[site_id],
                models=logistic_models,
                config=ofo_config,
                dt_s=dt_ctrl,
                site_id=site_id if len(site_ids) > 1 else None,
            )
            controllers.append(ofo_ctrl)

    # Load shift controller (must be after OFO controllers)
    if mode == "ofo" and config.load_shift and config.load_shift.get("enabled", False) and len(site_ids) > 1:
        from openg2g.controller.load_shift import LoadShiftConfig, LoadShiftController

        ls_cfg = LoadShiftConfig(
            enabled=True,
            gpus_per_shift=config.load_shift.get("gpus_per_shift", 8),
            headroom=config.load_shift.get("headroom", 0.3),
        )
        site_bus_map = {sid: config.dc_sites[sid].bus for sid in site_ids}
        models_by_site = {sid: [m.model_label for m in ms] for sid, ms in site_models_map.items()}
        gpus_per_replica_by_model = {m.model_label: m.gpus_per_replica for m in all_models}
        feasible_bs_by_model = {m.model_label: list(m.feasible_batch_sizes) for m in all_models}
        controllers.append(
            LoadShiftController(
                config=ls_cfg,
                dt_s=dt_ctrl,
                datacenters=datacenters,
                site_bus_map=site_bus_map,
                models_by_site=models_by_site,
                gpus_per_replica_by_model=gpus_per_replica_by_model,
                feasible_batch_sizes_by_model=feasible_bs_by_model,
                v_min=sim.v_min,
                v_max=sim.v_max,
            )
        )
        logger.info(
            "LoadShiftController enabled: gpus_per_shift=%d, headroom=%.1f", ls_cfg.gpus_per_shift, ls_cfg.headroom
        )

    # Run
    logger.info("=== %s (%s) ===", mode.upper(), folder_name)

    if len(datacenters) == 1 and "default" in datacenters:
        coord = Coordinator(
            datacenter=datacenters["default"],
            grid=grid,
            controllers=controllers,
            total_duration_s=sim.total_duration_s,
            dc_bus=primary_bus,
        )
    else:
        coord = Coordinator(
            datacenters=datacenters,
            grid=grid,
            controllers=controllers,
            total_duration_s=sim.total_duration_s,
            dc_bus=primary_bus,
        )
    log = coord.run()

    vstats = compute_allbus_voltage_stats(
        log.grid_states,
        v_min=sim.v_min,
        v_max=sim.v_max,
        exclude_buses=exclude_buses,
    )
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

    # Count monitored buses for plot decision
    all_buses = set()
    if log.grid_states:
        drop = {b.lower() for b in exclude_buses}
        for bus in log.grid_states[0].voltages.buses():
            if bus.lower() not in drop:
                all_buses.add(bus)

    zones = getattr(config, "zones", None)
    if zones:
        # Zone-based voltage envelope plot
        plot_zone_voltage_envelope(
            log.grid_states,
            time_s,
            zones=zones,
            save_dir=run_dir,
            v_min=sim.v_min,
            v_max=sim.v_max,
            drop_buses=exclude_buses,
            title_template=f"{folder_name} — Voltage Envelope (Phase {{label}})",
        )
    elif len(all_buses) > MAX_BUSES_FOR_INDIVIDUAL_LINES:
        # Too many buses for individual lines — plot min/max envelope
        _plot_voltage_envelope(
            log.grid_states,
            time_s,
            run_dir,
            v_min=sim.v_min,
            v_max=sim.v_max,
            exclude_buses=exclude_buses,
            title=f"{folder_name} — Voltage Envelope",
        )
    else:
        # Few buses — plot individual lines per phase
        plot_allbus_voltages_per_phase(
            log.grid_states,
            time_s,
            save_dir=run_dir,
            v_min=sim.v_min,
            v_max=sim.v_max,
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

    # Power + latency plot for every case (baseline and ofo)
    for site_id in site_ids:
        site_states = log.dc_states_by_site.get(site_id, log.dc_states)
        if not site_states:
            continue
        suffix = f"_{site_id}" if len(site_ids) > 1 else ""
        per_model = extract_per_model_timeseries(site_states)

        # Build 3-phase power arrays from total DC power
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

    # Tap position plot
    _plot_tap_positions(
        log.grid_states,
        time_s,
        run_dir,
        title=f"{folder_name} — Tap Positions",
    )

    # PV & load profiles
    _plot_pv_load_profiles(log.grid_states, time_s, config, run_dir)

    # Per-case voltage violation summary CSV
    with open(run_dir / f"result_{folder_name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case",
                "violation_time_s",
                "integral_violation_pu_s",
                "worst_vmin",
                "worst_vmax",
            ]
        )
        writer.writerow(
            [
                folder_name,
                vstats.violation_time_s,
                vstats.integral_violation_pu_s,
                vstats.worst_vmin,
                vstats.worst_vmax,
            ]
        )
    logger.info("Per-case CSV: %s", run_dir / f"result_{folder_name}.csv")

    return vstats, log


# ── Comparison plotting ──────────────────────────────────────────────────────


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


# ── Main ─────────────────────────────────────────────────────────────────────


def main(*, config_path: Path, system: str, mode: str = "no-tap") -> None:
    config_path = config_path.resolve()
    config = SweepConfig.model_validate_json(config_path.read_bytes())
    config_dir = config_path.parent
    config.ieee_case_dir = (config_dir / config.ieee_case_dir).resolve()

    sim = config.simulation
    all_models = tuple(config.models)
    data_dir = config.data_dir or Path("data/offline") / config.data_hash
    data_dir = (config_dir / data_dir).resolve() if not data_dir.is_absolute() else data_dir

    dt_dc = _parse_fraction(sim.dt_dc)
    dt_grid = _parse_fraction(sim.dt_grid)
    dt_ctrl = _parse_fraction(sim.dt_ctrl)

    save_dir = Path(__file__).resolve().parent / "outputs" / system
    save_dir.mkdir(parents=True, exist_ok=True)

    SAVE_LOG_FILE = False
    if SAVE_LOG_FILE:
        file_handler = logging.FileHandler(save_dir / "console_output_ofo.txt", mode="w")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logging.getLogger().addHandler(file_handler)

    # Load data
    logger.info("Loading data for %s...", system)
    data_sources = {s.model_label: s for s in config.data_sources}
    inference_data = InferenceData.ensure(
        data_dir,
        all_models,
        data_sources,
        mlenergy_data_dir=config.mlenergy_data_dir,
        plot=False,
        dt_s=float(dt_dc),
    )
    training_trace = TrainingTrace.ensure(
        data_dir / "training_trace.csv",
        config.training_trace_params,
    )
    logistic_models = LogisticModelStore.ensure(
        data_dir / "logistic_fits.csv",
        all_models,
        data_sources,
        mlenergy_data_dir=config.mlenergy_data_dir,
        plot=False,
    )

    # Build tap schedule if defined
    has_tap_schedule = config.tap_schedule is not None and len(config.tap_schedule) > 0
    tap_sched = _build_tap_schedule(config.tap_schedule, config.initial_taps) if has_tap_schedule else None

    shared = dict(
        config=config,
        all_models=all_models,
        inference_data=inference_data,
        training_trace=training_trace,
        logistic_models=logistic_models,
        dt_dc=dt_dc,
        dt_grid=dt_grid,
        dt_ctrl=dt_ctrl,
        save_dir=save_dir,
    )

    # Build cases based on --mode
    # Modes: no-tap, tap-change, both (baseline w/ and w/o tap), all (4 cases)
    cases: list[tuple[str, str, TapSchedule | None]] = []
    if mode in ("no-tap", "both", "all"):
        cases.append(("baseline", "baseline_no-tap", None))
    if mode in ("no-tap", "all"):
        cases.append(("ofo", "ofo_no-tap", None))
    if mode in ("tap-change", "both", "all") and has_tap_schedule:
        cases.append(("baseline", "baseline_tap-change", tap_sched))
    if mode in ("tap-change", "all") and has_tap_schedule:
        cases.append(("ofo", "ofo_tap-change", tap_sched))
    if not cases:
        logger.warning(
            "No cases to run (mode=%s, has_tap_schedule=%s). Use --mode no-tap or ensure config has tap_schedule.",
            mode,
            has_tap_schedule,
        )
        return

    logger.info(
        "Running %d cases for %s: %s",
        len(cases),
        system,
        [c[1] for c in cases],
    )

    results: dict[str, VoltageStats] = {}
    for run_mode_name, folder, sched in cases:
        stats, _sim_log = run_mode(
            run_mode_name,
            **shared,
            tap_schedule=sched,
            folder_name=folder,
        )
        results[folder] = stats

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
            ]
        )
        for case_name, s in results.items():
            writer.writerow(
                [
                    case_name,
                    s.violation_time_s,
                    s.worst_vmin,
                    s.worst_vmax,
                    s.integral_violation_pu_s,
                ]
            )
    logger.info("Comparison CSV: %s", csv_path)

    # Comparison plot
    _plot_comparison(results, save_dir, system)

    # Summary table
    logger.info("")
    logger.info("=" * 90)
    logger.info("%s Comparison", system.upper())
    logger.info("=" * 90)
    logger.info(
        "%-22s %10s %10s %10s %14s",
        "Mode",
        "Viol(s)",
        "Vmin",
        "Vmax",
        "Integral",
    )
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
    from dataclasses import dataclass

    import tyro

    @dataclass
    class Args:
        config: str
        """Path to the config JSON file."""
        system: str = "ieee13"
        """System name (ieee13, ieee34, ieee123) for output directory."""
        mode: str = "no-tap"
        """Run mode: 'no-tap' (baseline+OFO, no tap schedule),
        'tap-change' (baseline+OFO, with tap schedule),
        'both' (baseline w/ and w/o tap schedule),
        'all' (all 4 cases: baseline+OFO × w/wo tap)."""
        log_level: str = "INFO"
        """Logging verbosity (DEBUG, INFO, WARNING)."""

    args = tyro.cli(Args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    main(config_path=Path(args.config), system=args.system, mode=args.mode)
