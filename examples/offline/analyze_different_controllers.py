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
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sweep_dc_locations import ScenarioOpenDSSGrid
from systems import (
    SYSTEMS,
    DCSite,
    DT_CTRL,
    DT_DC,
    DT_GRID,
    POWER_AUG,
    PVSystemSpec,
    TimeVaryingLoadSpec,
    V_MAX,
    V_MIN,
    deploy,
    load_data_sources,
    tap,
)

from openg2g.controller.ofo import (
    LogisticModelStore,
    OFOConfig,
)
from openg2g.controller.rule_based import RuleBasedBatchSizeController, RuleBasedConfig
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import Coordinator
from openg2g.datacenter.config import (
    DatacenterConfig,
    InferenceModelSpec,
    InferenceRamp,
    ModelDeployment,
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.datacenter.workloads.training import TrainingTrace
from openg2g.grid.config import DCLoadSpec, TapPosition, TapSchedule
from openg2g.metrics.voltage import VoltageStats, compute_allbus_voltage_stats

logger = logging.getLogger("controller_comparison")

TOTAL_DURATION_S = 3600


# ── Per-system experiment definitions ────────────────────────────────────────


def experiment_ieee34() -> dict:
    """IEEE 34-bus experiment: two DC sites (upstream/downstream)."""
    sys = SYSTEMS["ieee34"]()
    return dict(
        sys=sys,
        dc_sites={
            "upstream": DCSite(
                bus="850",
                bus_kv=24.9,
                base_kw_per_phase=120.0,
                models=(deploy("Llama-3.1-8B", 720), deploy("Llama-3.1-70B", 180), deploy("Llama-3.1-405B", 90)),
                seed=0,
                total_gpu_capacity=520,
            ),
            "downstream": DCSite(
                bus="834",
                bus_kv=24.9,
                base_kw_per_phase=80.0,
                models=(deploy("Qwen3-30B-A3B", 480), deploy("Qwen3-235B-A22B", 210)),
                seed=42,
                total_gpu_capacity=600,
            ),
        },
        ofo_config=OFOConfig(
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
        ),
        pv_systems=[
            PVSystemSpec(bus="848", bus_kv=24.9, peak_kw=130.0),
            PVSystemSpec(bus="830", bus_kv=24.9, peak_kw=65.0),
        ],
        time_varying_loads=[
            TimeVaryingLoadSpec(bus="860", bus_kv=24.9, peak_kw=80.0),
            TimeVaryingLoadSpec(bus="844", bus_kv=24.9, peak_kw=120.0),
            TimeVaryingLoadSpec(bus="840", bus_kv=24.9, peak_kw=60.0),
            TimeVaryingLoadSpec(bus="858", bus_kv=24.9, peak_kw=50.0),
            TimeVaryingLoadSpec(bus="854", bus_kv=24.9, peak_kw=40.0),
        ],
        tap_schedule=TapSchedule((
            (1800, TapPosition(regulators={
                "creg2a": tap(10), "creg2b": tap(10), "creg2c": tap(10),
            })),
        )),
    )


def experiment_ieee123() -> dict:
    """IEEE 123-bus experiment: four DC sites across zones."""
    sys = SYSTEMS["ieee123"]()
    return dict(
        sys=sys,
        dc_sites={
            "z1_sw": DCSite(
                bus="8",
                bus_kv=4.16,
                base_kw_per_phase=310.0,
                models=(deploy("Llama-3.1-8B", 120),),
                seed=0,
                total_gpu_capacity=120,
                inference_ramps=InferenceRamp(target=180, model="Llama-3.1-8B").at(t_start=500, t_end=1000),
            ),
            "z2_nw": DCSite(
                bus="23",
                bus_kv=4.16,
                base_kw_per_phase=265.0,
                models=(deploy("Qwen3-30B-A3B", 80),),
                seed=17,
                total_gpu_capacity=160,
                inference_ramps=InferenceRamp(target=104, model="Qwen3-30B-A3B").at(t_start=1500, t_end=2500),
            ),
            "z3_se": DCSite(
                bus="60",
                bus_kv=4.16,
                base_kw_per_phase=295.0,
                models=(deploy("Llama-3.1-70B", 30), deploy("Llama-3.1-405B", 35)),
                seed=34,
                total_gpu_capacity=400,
                inference_ramps=InferenceRamp(target=45, model="Llama-3.1-70B").at(t_start=700, t_end=1100),
            ),
            "z4_ne": DCSite(
                bus="105",
                bus_kv=4.16,
                base_kw_per_phase=325.0,
                models=(deploy("Qwen3-235B-A22B", 55),),
                seed=51,
                total_gpu_capacity=440,
                inference_ramps=InferenceRamp(target=27, model="Qwen3-235B-A22B").at(t_start=2000, t_end=2500),
            ),
        },
        ofo_config=OFOConfig(
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
        ),
        pv_systems=[
            PVSystemSpec(bus="1", bus_kv=4.16, peak_kw=333.3),
            PVSystemSpec(bus="48", bus_kv=4.16, peak_kw=333.3),
            PVSystemSpec(bus="99", bus_kv=4.16, peak_kw=333.3),
        ],
        time_varying_loads=[],
        tap_schedule=TapSchedule((
            (1800, TapPosition(regulators={"creg4a": tap(16)})),
        )),
    )


EXPERIMENTS = {"ieee34": experiment_ieee34, "ieee123": experiment_ieee123}


# ── Simulation runner ────────────────────────────────────────────────────────


def run_simulation(
    mode: str,
    *,
    sys: dict,
    dc_sites: dict[str, DCSite],
    ofo_config: OFOConfig,
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
    pv_systems: list[PVSystemSpec] | None = None,
    time_varying_loads: list[TimeVaryingLoadSpec] | None = None,
    tap_schedule: TapSchedule | None = None,
    rule_based_config: RuleBasedConfig | None = None,
    save_dir: Path,
) -> tuple[VoltageStats, object]:
    """Run a simulation with the specified controller mode.

    Modes:
        'baseline': NoopController (no batch control)
        'rule_based': RuleBasedBatchSizeController
        'ofo': OFOBatchSizeController

    Returns (VoltageStats, SimulationLog).
    """
    pv_systems = pv_systems or []
    time_varying_loads = time_varying_loads or []
    exclude_buses = tuple(sys["exclude_buses"])
    site_ids = list(dc_sites.keys())

    dc_loads: dict[str, DCLoadSpec] = {}
    datacenters: dict[str, OfflineDatacenter] = {}
    controllers: list = []
    site_specs_map: dict[str, tuple[InferenceModelSpec, ...]] = {}
    primary_bus = ""

    for site_id, site in dc_sites.items():
        site_specs = tuple(md.spec for md in site.models)
        site_specs_map[site_id] = site_specs
        site_inference = inference_data.filter_models(site_specs)

        replica_counts = {md.spec.model_label: md.num_replicas for md in site.models}

        dc_config = DatacenterConfig(
            gpus_per_server=8,
            base_kw_per_phase=site.base_kw_per_phase,
        )
        workload_kwargs: dict = {
            "inference_data": site_inference,
            "replica_counts": replica_counts,
        }
        if site.inference_ramps is not None:
            workload_kwargs["inference_ramps"] = site.inference_ramps
        workload = OfflineWorkload(**workload_kwargs)

        dc = OfflineDatacenter(
            dc_config,
            workload,
            dt_s=DT_DC,
            seed=site.seed,
            power_augmentation=POWER_AUG,
            total_gpu_capacity=site.total_gpu_capacity,
        )
        datacenters[site_id] = dc
        dc_loads[site_id] = DCLoadSpec(
            bus=site.bus,
            bus_kv=site.bus_kv,
            connection_type=site.connection_type,
        )
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

    # Tap controller: only baseline gets tap schedule changes;
    # OFO and rule-based use fixed initial taps (no tap schedule)
    if mode == "baseline":
        sched = tap_schedule if tap_schedule is not None else TapSchedule(())
    else:
        sched = TapSchedule(())
    controllers.append(TapScheduleController(schedule=sched, dt_s=DT_CTRL))

    # Batch-size controller
    if mode == "ofo":
        from openg2g.controller.ofo import OFOBatchSizeController

        for site_id in site_ids:
            ofo_ctrl = OFOBatchSizeController(
                site_specs_map[site_id],
                models=logistic_models,
                config=ofo_config,
                dt_s=DT_CTRL,
                site_id=site_id if len(site_ids) > 1 else None,
            )
            controllers.append(ofo_ctrl)

    elif mode == "rule_based":
        rb_config = rule_based_config or RuleBasedConfig(v_min=V_MIN, v_max=V_MAX)
        for site_id in site_ids:
            rb_ctrl = RuleBasedBatchSizeController(
                site_specs_map[site_id],
                config=rb_config,
                dt_s=DT_CTRL,
                site_id=site_id if len(site_ids) > 1 else None,
                exclude_buses=exclude_buses,
            )
            controllers.append(rb_ctrl)

    # baseline: no batch controller added

    coord = Coordinator(
        datacenters=datacenters,
        grid=grid,
        controllers=controllers,
        total_duration_s=TOTAL_DURATION_S,
        dc_bus=primary_bus,
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
    system: str,
    rule_step_size: float = 0.3,
    rule_deadband: float = 0.005,
) -> None:
    if system not in EXPERIMENTS:
        raise ValueError(f"Unknown system {system!r}; choose from {list(EXPERIMENTS)}")

    exp = EXPERIMENTS[system]()
    sys = exp["sys"]
    dc_sites: dict[str, DCSite] = exp["dc_sites"]
    ofo_config: OFOConfig = exp["ofo_config"]
    pv_systems: list[PVSystemSpec] = exp["pv_systems"]
    time_varying_loads: list[TimeVaryingLoadSpec] = exp["time_varying_loads"]
    tap_schedule: TapSchedule | None = exp["tap_schedule"]
    exclude_buses = tuple(sys["exclude_buses"])

    # Collect all models across sites
    all_models: list[ModelDeployment] = []
    for site in dc_sites.values():
        all_models.extend(site.models)
    all_specs_tuple = tuple(m.spec for m in all_models)

    save_dir = Path(__file__).resolve().parent / "outputs" / system / "controller_comparison"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    data_sources, training_trace_params, data_dir = load_data_sources()

    inference_data = InferenceData.ensure(
        data_dir,
        all_specs_tuple,
        data_sources,
        plot=False,
        dt_s=float(DT_DC),
    )
    training_trace = TrainingTrace.ensure(
        data_dir / "training_trace.csv",
        training_trace_params,
    )
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
            sys=sys,
            dc_sites=dc_sites,
            ofo_config=ofo_config,
            inference_data=inference_data,
            training_trace=training_trace,
            logistic_models=logistic_models,
            pv_systems=pv_systems,
            time_varying_loads=time_varying_loads,
            tap_schedule=tap_schedule,
            rule_based_config=rb_config if mode == "rule_based" else None,
            save_dir=save_dir,
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
        system: str = "ieee123"
        """System name: ieee34 or ieee123."""
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
        system=args.system,
        rule_step_size=args.rule_step_size,
        rule_deadband=args.rule_deadband,
    )
