"""Compare different batch-size controllers for voltage regulation.

Runs three control strategies on the same system and produces comparison
plots: (1) no batch control (baseline), (2) rule-based proportional
controller, and (3) OFO primal-dual controller.

Usage:
    python analyze_different_controllers.py --config config_ieee123.json --system ieee123
    python analyze_different_controllers.py --config config_ieee34.json --system ieee34
"""

from __future__ import annotations

import csv
import logging
import math
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sweep_dc_locations import (
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
from openg2g.controller.rule_based import RuleBasedBatchSizeController, RuleBasedConfig
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

logger = logging.getLogger("controller_comparison")


# ── Simulation runner ────────────────────────────────────────────────────────


def run_simulation(
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
    tap_schedule: TapSchedule | None = None,
    rule_based_config: RuleBasedConfig | None = None,
) -> tuple[VoltageStats, object]:
    """Run a simulation with the specified controller mode.

    Modes:
        'baseline': NoopController (no batch control)
        'rule_based': RuleBasedBatchSizeController
        'ofo': OFOBatchSizeController

    Returns (VoltageStats, SimulationLog).
    """
    sim = config.simulation
    site_ids = list(config.dc_sites.keys())
    dc_loads: dict[str, DCLoadSpec] = {}
    datacenters: dict[str, OfflineDatacenter] = {}
    controllers: list = []
    site_models_map: dict[str, tuple[InferenceModelSpec, ...]] = {}
    primary_bus = ""

    for site_id, site_cfg in config.dc_sites.items():
        site_models = _resolve_models_for_site(site_cfg, all_models)
        site_models_map[site_id] = site_models
        site_inference = inference_data.filter_models(site_models) if site_cfg.models else inference_data

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

        dc = OfflineDatacenter(
            dc_config,
            workload,
            dt_s=dt_dc,
            seed=site_cfg.seed,
            power_augmentation=config.power_augmentation,
            total_gpu_capacity=site_cfg.total_gpu_capacity,
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
        # Single-site: use dc_bus/dc_bus_kv interface
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
        # Multi-site: use dc_loads interface
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

    # Tap controller: only baseline gets tap schedule changes;
    # OFO and rule-based use fixed initial taps (no tap schedule)
    if mode == "baseline":
        sched = tap_schedule if tap_schedule is not None else TapSchedule(())
    else:
        sched = TapSchedule(())
    controllers.append(TapScheduleController(schedule=sched, dt_s=dt_ctrl))

    # Batch-size controller
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

    elif mode == "rule_based":
        rb_config = rule_based_config or RuleBasedConfig(
            v_min=sim.v_min,
            v_max=sim.v_max,
        )
        for site_id in site_ids:
            rb_ctrl = RuleBasedBatchSizeController(
                site_models_map[site_id],
                config=rb_config,
                dt_s=dt_ctrl,
                site_id=site_id if len(site_ids) > 1 else None,
                exclude_buses=exclude_buses,
            )
            controllers.append(rb_ctrl)

    # baseline: no batch controller added

    coord = Coordinator(
        datacenters=datacenters,
        grid=grid,
        controllers=controllers,
        total_duration_s=sim.total_duration_s,
        dc_bus=primary_bus,
    )

    logger.info("Running %s...", mode)
    log = coord.run()

    vstats = compute_allbus_voltage_stats(
        log.grid_states,
        v_min=sim.v_min,
        v_max=sim.v_max,
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

    return vstats, log


# ── Plotting ─────────────────────────────────────────────────────────────────


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
    fig.savefig(save_dir / "voltage_comparison.png", dpi=150, bbox_inches="tight")
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
    fig.savefig(save_dir / "batch_size_comparison.png", dpi=150, bbox_inches="tight")
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
    fig.savefig(save_dir / "summary_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved summary_bar_chart.png")


# ── Main ─────────────────────────────────────────────────────────────────────


def main(
    *,
    config_path: Path,
    system: str,
    rule_step_size: float = 0.3,
    rule_deadband: float = 0.005,
) -> None:
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

    save_dir = Path(__file__).resolve().parent / "outputs" / system / "controller_comparison"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    inference_data = InferenceData.ensure(
        data_dir,
        all_models,
        {s.model_label: s for s in config.data_sources},
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
        {s.model_label: s for s in config.data_sources},
        mlenergy_data_dir=config.mlenergy_data_dir,
        plot=False,
    )

    # Build tap schedule from config (if present)
    tap_schedule = None
    if config.tap_schedule:
        from sweep_dc_locations import TAP_STEP

        entries = []
        for entry in config.tap_schedule:
            t = entry.t
            extras = entry.model_extra or {}
            tap_pos = TapPosition(
                regulators={k: (1.0 + int(v) * TAP_STEP if isinstance(v, str) else float(v)) for k, v in extras.items()}
            )
            entries.append((t, tap_pos))
        if entries:
            tap_schedule = TapSchedule(tuple(entries))

    exclude_buses = tuple(config.exclude_buses)

    # Rule-based config
    rb_config = RuleBasedConfig(
        step_size=rule_step_size,
        deadband=rule_deadband,
        v_min=sim.v_min,
        v_max=sim.v_max,
    )

    # ── Run all modes ──
    modes = ["baseline", "rule_based", "ofo"]
    all_stats: dict[str, VoltageStats] = {}
    all_logs: dict[str, object] = {}

    for mode in modes:
        logger.info("")
        logger.info("=" * 60)
        logger.info("MODE: %s", mode.upper())
        logger.info("=" * 60)

        vstats, log = run_simulation(
            mode,
            config=config,
            all_models=all_models,
            inference_data=inference_data,
            training_trace=training_trace,
            logistic_models=logistic_models,
            dt_dc=dt_dc,
            dt_grid=dt_grid,
            dt_ctrl=dt_ctrl,
            tap_schedule=tap_schedule,
            rule_based_config=rb_config if mode == "rule_based" else None,
        )
        all_stats[mode] = vstats
        all_logs[mode] = log

    # ── Summary table ──
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

    # ── Save results CSV ──
    csv_path = save_dir / f"results_{system}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "violation_time_s", "integral_violation_pu_s", "worst_vmin", "worst_vmax"])
        for mode in modes:
            s = all_stats[mode]
            writer.writerow([mode, s.violation_time_s, s.integral_violation_pu_s, s.worst_vmin, s.worst_vmax])
    logger.info("Results CSV: %s", csv_path)

    # ── Plots ──
    plot_voltage_comparison(all_logs, save_dir, v_min=sim.v_min, v_max=sim.v_max, exclude_buses=exclude_buses)
    plot_batch_comparison(all_logs, save_dir)
    plot_summary_bar(all_stats, save_dir)

    logger.info("")
    logger.info("All outputs saved to: %s", save_dir)


if __name__ == "__main__":
    from dataclasses import dataclass

    import tyro

    @dataclass
    class Args:
        config: str
        """Path to the config JSON file."""
        system: str = "ieee123"
        """System name for output directory."""
        rule_step_size: float = 0.3
        """Proportional gain for the rule-based controller (log2 units per pu violation)."""
        rule_deadband: float = 0.005
        """Deadband for the rule-based controller (pu)."""
        log_level: str = "INFO"
        """Logging verbosity."""

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
        config_path=Path(args.config),
        system=args.system,
        rule_step_size=args.rule_step_size,
        rule_deadband=args.rule_deadband,
    )
