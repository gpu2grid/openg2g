"""Storage-on-OFO datacenter-grid coordination workflow.

This example is intentionally built on top of the standard IEEE 123-bus
controller-comparison setup in `analyze_different_controllers.py`. It reuses
the same datacenter placements, model specs, ML.ENERGY inference traces,
synthetic PV, duration, replica schedules, and OFO tuning, then adds one
co-located storage system at each datacenter bus.

Scenarios:

    baseline            Standard IEEE123 baseline mode from the base example
    ofo                 Standard IEEE123 OFO mode from the base example
    ofo_idle_storage    Standard OFO plus storage attached but not controlled
    ofo_qv_storage      Standard OFO plus local Q-V storage droop
    ofo_pv_storage      Standard OFO plus local P-V storage droop

The main contrast is the incremental effect of storage when datacenter OFO is
already active: idle storage reproduces OFO-only behavior, Q-V droop can
support voltage without materially changing SOC, and P-V droop also supports
voltage while using stored energy.

Usage:
    python examples/offline/analyze_storage_coordination.py
    python examples/offline/analyze_storage_coordination.py --scenarios ofo ofo_qv_storage
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/openg2g-matplotlib"))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import analyze_different_controllers as standard_ieee_examples
import matplotlib.pyplot as plt
import numpy as np

from openg2g.controller.ofo import LogisticModelStore, OFOBatchSizeController, OFOConfig
from openg2g.controller.storage import LocalVoltageStorageDroopController, StorageDroopConfig
from openg2g.controller.tap_schedule import TapScheduleController
from openg2g.coordinator import Coordinator, SimulationLog
from openg2g.datacenter.base import DatacenterState
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.datacenter.workloads.training import TrainingTrace
from openg2g.grid.base import GridState, PhaseVoltages
from openg2g.grid.config import TapSchedule
from openg2g.grid.storage import BatteryStorage, StorageState
from openg2g.metrics.performance import PerformanceStats, compute_performance_stats
from openg2g.metrics.voltage import VoltageStats, compute_allbus_voltage_stats

logger = logging.getLogger("analyze_storage_coordination")

ScenarioName = Literal["baseline", "ofo", "ofo_idle_storage", "ofo_qv_storage", "ofo_pv_storage"]

DEFAULT_SCENARIOS: tuple[ScenarioName, ...] = (
    "baseline",
    "ofo",
    "ofo_idle_storage",
    "ofo_qv_storage",
    "ofo_pv_storage",
)
STORAGE_PLOT_SCENARIOS: tuple[ScenarioName, ...] = (
    "ofo_idle_storage",
    "ofo_pv_storage",
    "ofo_qv_storage",
)

DT_DC = standard_ieee_examples.DT_DC
DT_CTRL = standard_ieee_examples.DT_CTRL
TOTAL_DURATION_S = standard_ieee_examples.TOTAL_DURATION_S
V_MIN = standard_ieee_examples.V_MIN
V_MAX = standard_ieee_examples.V_MAX
SPECS_CACHE_DIR = standard_ieee_examples.SPECS_CACHE_DIR
TRAINING_TRACE_PATH = standard_ieee_examples.TRAINING_TRACE_PATH
STANDARD_IEEE123_MODEL_SPECS = (
    standard_ieee_examples.LLAMA_8B,
    standard_ieee_examples.QWEN_30B,
    standard_ieee_examples.LLAMA_70B,
    standard_ieee_examples.LLAMA_405B,
    standard_ieee_examples.QWEN_235B,
)


@dataclass(frozen=True)
class StorageSite:
    name: str
    bus: str
    rated_power_kw: float
    apparent_power_kva: float
    duration_h: float = 2.0


@dataclass
class ScenarioResult:
    name: ScenarioName
    log: SimulationLog[DatacenterState, GridState]
    stats: VoltageStats
    performance: PerformanceStats
    storages: list[RecordingBatteryStorage] = field(default_factory=list)


class RecordingBatteryStorage(BatteryStorage):
    """BatteryStorage variant that records every OpenDSS state sync."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.history: list[StorageState] = []

    def update_state(self, state: StorageState) -> None:
        super().update_state(state)
        if self.history and abs(self.history[-1].time_s - state.time_s) <= 1e-12:
            self.history[-1] = state
        else:
            self.history.append(state)

    def reset(self) -> None:
        super().reset()
        self.history.clear()


STORAGE_SITES: tuple[StorageSite, ...] = (
    StorageSite("bat_z1", "8", rated_power_kw=250.0, apparent_power_kva=300.0),
    StorageSite("bat_z2", "23", rated_power_kw=250.0, apparent_power_kva=300.0),
    StorageSite("bat_z3", "60", rated_power_kw=300.0, apparent_power_kva=360.0),
    StorageSite("bat_z4", "105", rated_power_kw=350.0, apparent_power_kva=420.0),
)


def build_scenario(
    scenario: ScenarioName,
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
) -> tuple[Coordinator[DatacenterState, GridState], list[RecordingBatteryStorage]]:
    setup = standard_ieee_examples.SETUPS["ieee123"](inference_data, training_trace, logistic_models)
    datacenters = setup["datacenters"]
    dc_specs = setup["dc_specs"]
    grid = setup["grid"]
    ofo_config: OFOConfig = setup["ofo_config"]
    tap_schedule = setup["tap_schedule"]

    storages: list[RecordingBatteryStorage] = []
    storage_enabled = scenario in {"ofo_idle_storage", "ofo_qv_storage", "ofo_pv_storage"}
    if storage_enabled:
        for site in STORAGE_SITES:
            storage = RecordingBatteryStorage(
                name=site.name,
                rated_power_kw=site.rated_power_kw,
                capacity_kwh=site.rated_power_kw * site.duration_h,
                initial_soc=0.5,
                apparent_power_kva=site.apparent_power_kva,
            )
            storages.append(storage)
            grid.attach_storage(storage, bus=site.bus)

    controllers = []
    if scenario == "baseline":
        controllers.append(TapScheduleController(schedule=tap_schedule, dt_s=DT_CTRL))
    else:
        controllers.append(TapScheduleController(schedule=TapSchedule(()), dt_s=DT_CTRL))
        for dc, specs in dc_specs.items():
            controllers.append(
                OFOBatchSizeController(
                    specs,
                    datacenter=dc,
                    grid=grid,
                    models=logistic_models,
                    config=ofo_config,
                    dt_s=DT_CTRL,
                )
            )

    if scenario == "ofo_qv_storage":
        controllers.append(
            LocalVoltageStorageDroopController(
                grid=grid,
                config=StorageDroopConfig(mode="qv"),
                dt_s=DT_CTRL,
            )
        )
    elif scenario == "ofo_pv_storage":
        controllers.append(
            LocalVoltageStorageDroopController(
                grid=grid,
                config=StorageDroopConfig(mode="pv"),
                dt_s=DT_CTRL,
            )
        )

    coordinator = Coordinator(
        datacenters=datacenters,
        grid=grid,
        controllers=controllers,
        total_duration_s=TOTAL_DURATION_S,
    )
    return coordinator, storages


def run_scenario(
    scenario: ScenarioName,
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
) -> ScenarioResult:
    logger.info("Running scenario: %s", scenario)
    coordinator, storages = build_scenario(scenario, inference_data, training_trace, logistic_models)
    log = coordinator.run()
    stats = compute_allbus_voltage_stats(
        log.grid_states,
        v_min=V_MIN,
        v_max=V_MAX,
        exclude_buses=tuple(standard_ieee_examples.SYSTEMS["ieee123"]()["exclude_buses"]),
    )
    performance = compute_performance_stats(
        log.dc_states,
        itl_deadline_s_by_model={spec.model_label: spec.itl_deadline_s for spec in STANDARD_IEEE123_MODEL_SPECS},
    )
    return ScenarioResult(name=scenario, log=log, stats=stats, performance=performance, storages=storages)


def local_voltage_pu(state: GridState, bus: str, bus_case_cache: dict[str, str] | None = None) -> float:
    if bus in state.voltages:
        phases = state.voltages[bus]
    else:
        phases = state.voltages[_resolve_bus_case(state, bus, bus_case_cache)]
    values = finite_phase_voltages(phases)
    if not values:
        return float("nan")
    return min(values)


def finite_phase_voltages(phases: PhaseVoltages) -> list[float]:
    return [float(v) for v in (phases.a, phases.b, phases.c) if not math.isnan(float(v))]


def _resolve_bus_case(state: GridState, bus: str, bus_case_cache: dict[str, str] | None = None) -> str:
    target = bus.lower()
    if bus_case_cache is not None:
        cached = bus_case_cache.get(target)
        if cached is not None and cached in state.voltages:
            return cached
    for candidate in state.voltages.buses():
        if candidate.lower() == target:
            if bus_case_cache is not None:
                bus_case_cache[target] = candidate
            return candidate
    raise KeyError(f"Bus {bus!r} not found in grid state.")


def scenario_voltage_envelope(result: ScenarioResult, exclude_buses: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    drop = {bus.lower() for bus in exclude_buses}
    v_min_arr = np.full(len(result.log.grid_states), np.inf)
    v_max_arr = np.full(len(result.log.grid_states), -np.inf)
    for idx, state in enumerate(result.log.grid_states):
        for bus in state.voltages.buses():
            if bus.lower() in drop:
                continue
            for value in finite_phase_voltages(state.voltages[bus]):
                v_min_arr[idx] = min(v_min_arr[idx], value)
                v_max_arr[idx] = max(v_max_arr[idx], value)
    return v_min_arr, v_max_arr


def write_summary(results: dict[ScenarioName, ScenarioResult], save_dir: Path) -> None:
    path = save_dir / "summary_metrics.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "violation_time_s",
                "integral_violation_pu_s",
                "worst_vmin_pu",
                "worst_vmax_pu",
                "mean_throughput_tps",
                "integrated_throughput_tokens",
                "itl_deadline_fraction",
                "avg_final_soc",
                "max_abs_storage_kw",
                "max_abs_storage_kvar",
            ],
        )
        writer.writeheader()
        for name, result in results.items():
            storage_states = [state for storage in result.storages for state in storage.history]
            final_socs = [storage.history[-1].soc for storage in result.storages if storage.history]
            writer.writerow(
                {
                    "scenario": name,
                    "violation_time_s": result.stats.violation_time_s,
                    "integral_violation_pu_s": result.stats.integral_violation_pu_s,
                    "worst_vmin_pu": result.stats.worst_vmin,
                    "worst_vmax_pu": result.stats.worst_vmax,
                    "mean_throughput_tps": result.performance.mean_throughput_tps,
                    "integrated_throughput_tokens": result.performance.integrated_throughput_tokens,
                    "itl_deadline_fraction": result.performance.itl_deadline_fraction,
                    "avg_final_soc": float(np.mean(final_socs)) if final_socs else "",
                    "max_abs_storage_kw": max((abs(s.power_kw) for s in storage_states), default=0.0),
                    "max_abs_storage_kvar": max((abs(s.reactive_power_kvar) for s in storage_states), default=0.0),
                }
            )
    logger.info("Saved %s", path)


def write_storage_timeseries(results: dict[ScenarioName, ScenarioResult], save_dir: Path) -> None:
    path = save_dir / "storage_timeseries.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "storage_name",
                "time_s",
                "bus",
                "local_voltage_pu",
                "soc",
                "stored_kwh",
                "power_kw",
                "reactive_power_kvar",
                "dss_state",
            ],
        )
        writer.writeheader()
        for scenario, result in results.items():
            state_by_time = {state.time_s: state for state in result.log.grid_states}
            bus_by_storage = {site.name: site.bus for site in STORAGE_SITES}
            bus_case_cache: dict[str, str] = {}
            for storage in result.storages:
                bus = bus_by_storage[storage.name]
                for state in storage.history:
                    grid_state = state_by_time.get(state.time_s)
                    writer.writerow(
                        {
                            "scenario": scenario,
                            "storage_name": storage.name,
                            "time_s": state.time_s,
                            "bus": bus,
                            "local_voltage_pu": ""
                            if grid_state is None
                            else local_voltage_pu(grid_state, bus, bus_case_cache),
                            "soc": state.soc,
                            "stored_kwh": state.stored_kwh,
                            "power_kw": state.power_kw,
                            "reactive_power_kvar": state.reactive_power_kvar,
                            "dss_state": state.dss_state,
                        }
                    )
    logger.info("Saved %s", path)


def write_storage_dispatch_summary(results: dict[ScenarioName, ScenarioResult], save_dir: Path) -> None:
    path = save_dir / "storage_dispatch_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "storage_name",
                "min_power_kw",
                "max_power_kw",
                "max_abs_power_kw",
                "min_reactive_power_kvar",
                "max_reactive_power_kvar",
                "max_abs_reactive_power_kvar",
                "initial_soc",
                "final_soc",
                "delta_soc",
            ],
        )
        writer.writeheader()
        for scenario, result in results.items():
            for storage in result.storages:
                if not storage.history:
                    continue
                power_kw = [state.power_kw for state in storage.history]
                reactive_power_kvar = [state.reactive_power_kvar for state in storage.history]
                initial_soc = storage.history[0].soc
                final_soc = storage.history[-1].soc
                writer.writerow(
                    {
                        "scenario": scenario,
                        "storage_name": storage.name,
                        "min_power_kw": min(power_kw),
                        "max_power_kw": max(power_kw),
                        "max_abs_power_kw": max(abs(value) for value in power_kw),
                        "min_reactive_power_kvar": min(reactive_power_kvar),
                        "max_reactive_power_kvar": max(reactive_power_kvar),
                        "max_abs_reactive_power_kvar": max(abs(value) for value in reactive_power_kvar),
                        "initial_soc": initial_soc,
                        "final_soc": final_soc,
                        "delta_soc": final_soc - initial_soc,
                    }
                )
    logger.info("Saved %s", path)


def write_datacenter_timeseries(results: dict[ScenarioName, ScenarioResult], save_dir: Path) -> None:
    path = save_dir / "datacenter_power_timeseries.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "datacenter", "time_s", "power_kw"])
        writer.writeheader()
        for scenario, result in results.items():
            for name, states in result.log.dc_states_by_site.items():
                for state in states:
                    writer.writerow(
                        {
                            "scenario": scenario,
                            "datacenter": name,
                            "time_s": state.time_s,
                            "power_kw": (state.power_w.a + state.power_w.b + state.power_w.c) / 1e3,
                        }
                    )
    logger.info("Saved %s", path)


def plot_voltage_envelopes(results: dict[ScenarioName, ScenarioResult], save_dir: Path) -> None:
    """Plot feeder-wide voltage envelopes for each storage scenario."""
    exclude = tuple(standard_ieee_examples.SYSTEMS["ieee123"]()["exclude_buses"])
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.0 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (scenario, result) in zip(axes, results.items(), strict=True):
        time_s = np.asarray(result.log.time_s)
        v_min_arr, v_max_arr = scenario_voltage_envelope(result, exclude)
        ax.fill_between(time_s, v_min_arr, v_max_arr, alpha=0.28, color="#4C78A8")
        ax.plot(time_s, v_min_arr, color="#2F5F8F", linewidth=0.8, label="Feeder min")
        ax.plot(time_s, v_max_arr, color="#F58518", linewidth=0.8, label="Feeder max")
        ax.axhline(V_MIN, color="#C44E52", linestyle="--", linewidth=1.0)
        ax.axhline(V_MAX, color="#C44E52", linestyle="--", linewidth=1.0)
        ax.set_title(scenario.replace("_", " ").title())
        ax.set_ylabel("Voltage (pu)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Standard IEEE 123-Bus OFO Case With Storage Scenarios", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "voltage_envelope_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_local_storage_voltages(results: dict[ScenarioName, ScenarioResult], save_dir: Path) -> None:
    """Plot voltage traces at the buses where storage systems are colocated."""
    storage_buses = {site.name: site.bus for site in STORAGE_SITES}
    fig, axes = plt.subplots(len(storage_buses), 1, figsize=(12, 8), sharex=True, sharey=True)
    if len(storage_buses) == 1:
        axes = [axes]

    for ax, (storage_name, bus) in zip(axes, storage_buses.items(), strict=True):
        for scenario, result in results.items():
            time_s = np.asarray(result.log.time_s)
            bus_case_cache: dict[str, str] = {}
            voltages = np.asarray([local_voltage_pu(state, bus, bus_case_cache) for state in result.log.grid_states])
            ax.plot(time_s, voltages, linewidth=1.0, label=scenario.replace("_", " "))
        ax.axhline(V_MIN, color="#C44E52", linestyle="--", linewidth=1.0)
        ax.axhline(V_MAX, color="#C44E52", linestyle="--", linewidth=1.0)
        ax.set_title(f"{storage_name} local voltage at bus {bus}")
        ax.set_ylabel("Voltage (pu)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(save_dir / "local_storage_voltage_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_storage_dispatch(results: dict[ScenarioName, ScenarioResult], save_dir: Path) -> None:
    """Plot storage real power, reactive power, and state of charge."""
    scenarios = tuple(
        scenario for scenario in STORAGE_PLOT_SCENARIOS if scenario in results and results[scenario].storages
    )
    if not scenarios:
        return

    metric_specs = (
        ("power_kw", "P (kW)"),
        ("reactive_power_kvar", "Q (kvar)"),
        ("soc", "SOC"),
    )
    colors = ("#4C78A8", "#F58518", "#54A24B", "#B279A2")
    dispatch_limit = storage_dispatch_axis_limit(results, scenarios)

    fig, axes = plt.subplots(
        len(metric_specs),
        len(scenarios),
        figsize=(4.3 * len(scenarios), 3.0 * len(metric_specs)),
        sharex=True,
        squeeze=False,
    )
    legend_handles = []
    legend_labels = []

    for col, scenario in enumerate(scenarios):
        storage_by_name = {storage.name: storage for storage in results[scenario].storages}
        for row, (attr, ylabel) in enumerate(metric_specs):
            ax = axes[row][col]
            for storage_idx, site in enumerate(STORAGE_SITES):
                storage = storage_by_name.get(site.name)
                if storage and storage.history:
                    time_s = np.asarray([state.time_s for state in storage.history])
                    values = np.asarray([getattr(state, attr) for state in storage.history])
                    (line,) = ax.plot(
                        time_s,
                        values,
                        linewidth=0.9,
                        color=colors[storage_idx % len(colors)],
                        label=site.name.replace("_", " "),
                    )
                    if row == 0 and col == 0:
                        legend_handles.append(line)
                        legend_labels.append(site.name.replace("_", " "))
            if attr != "soc":
                ax.axhline(0.0, color="#555555", linewidth=0.6, alpha=0.55)
                ax.set_ylim(-dispatch_limit, dispatch_limit)
            else:
                ax.set_ylim(-0.02, 1.02)

            if row == 0:
                ax.set_title(storage_scenario_label(scenario), fontsize=10)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == len(metric_specs) - 1:
                ax.set_xlabel("Time (s)")
            ax.grid(True, alpha=0.25)

    fig.suptitle("Storage Dispatch and State of Charge", fontsize=14, fontweight="bold")
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=len(STORAGE_SITES), frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(save_dir / "storage_dispatch_soc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def storage_dispatch_axis_limit(
    results: dict[ScenarioName, ScenarioResult],
    scenarios: Sequence[ScenarioName],
) -> float:
    max_abs = 0.0
    for scenario in scenarios:
        for storage in results[scenario].storages:
            for state in storage.history:
                max_abs = max(max_abs, abs(state.power_kw), abs(state.reactive_power_kvar))
    if max_abs <= 0.0:
        return 1.0
    return max_abs * 1.05


def storage_scenario_label(scenario: ScenarioName) -> str:
    labels = {
        "ofo_idle_storage": "Idle Storage",
        "ofo_pv_storage": "P-V Storage",
        "ofo_qv_storage": "Q-V Storage",
    }
    return labels.get(scenario, scenario.replace("_", " ").title())


def plot_datacenter_fleet_power(results: dict[ScenarioName, ScenarioResult], save_dir: Path) -> None:
    """Plot aggregate datacenter fleet power across the selected scenarios."""
    fig, ax = plt.subplots(figsize=(12, 4))
    for scenario, result in results.items():
        by_time: dict[float, float] = {}
        for states in result.log.dc_states_by_site.values():
            for state in states:
                by_time[state.time_s] = (
                    by_time.get(state.time_s, 0.0) + (state.power_w.a + state.power_w.b + state.power_w.c) / 1e3
                )
        time_s = np.asarray(sorted(by_time))
        power_kw = np.asarray([by_time[t] for t in time_s])
        ax.plot(time_s, power_kw, linewidth=1.2, label=scenario.replace("_", " "))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fleet datacenter power (kW)")
    ax.set_title("OFO-Controlled Datacenter Fleet Power With Storage Scenarios")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_dir / "datacenter_fleet_power_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=DEFAULT_SCENARIOS,
        default=list(DEFAULT_SCENARIOS),
        help="Scenarios to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "ieee123" / "storage_on_ofo",
        help="Directory for CSV and plot outputs.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    save_dir = Path(args.output_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    specs = STANDARD_IEEE123_MODEL_SPECS
    logger.info("Loading inference traces for %d model specs...", len(specs))
    inference_data = InferenceData.ensure(
        SPECS_CACHE_DIR,
        specs,
        plot=False,
        dt_s=float(DT_DC),
    )
    logger.info("Loading standard training trace...")
    training_trace = TrainingTrace.ensure(TRAINING_TRACE_PATH)
    logger.info("Loading OFO logistic models for %d model specs...", len(specs))
    logistic_models = LogisticModelStore.ensure(
        SPECS_CACHE_DIR,
        specs,
        plot=False,
    )

    results: dict[ScenarioName, ScenarioResult] = {}
    for scenario in cast_scenarios(args.scenarios):
        result = run_scenario(scenario, inference_data, training_trace, logistic_models)
        results[scenario] = result
        logger.info(
            (
                "%s: violation_time=%.1fs, integral=%.3f pu*s, worst_vmin=%.4f, "
                "worst_vmax=%.4f, throughput=%.1f k tok/s, itl_miss=%.2f%%"
            ),
            scenario,
            result.stats.violation_time_s,
            result.stats.integral_violation_pu_s,
            result.stats.worst_vmin,
            result.stats.worst_vmax,
            result.performance.mean_throughput_tps / 1e3,
            result.performance.itl_deadline_fraction * 100.0,
        )

    write_summary(results, save_dir)
    write_storage_timeseries(results, save_dir)
    write_storage_dispatch_summary(results, save_dir)
    write_datacenter_timeseries(results, save_dir)
    plot_voltage_envelopes(results, save_dir)
    plot_local_storage_voltages(results, save_dir)
    plot_storage_dispatch(results, save_dir)
    plot_datacenter_fleet_power(results, save_dir)

    logger.info("Saved storage coordination outputs to %s", save_dir)


def cast_scenarios(values: Sequence[str]) -> tuple[ScenarioName, ...]:
    return cast(tuple[ScenarioName, ...], tuple(values))


if __name__ == "__main__":
    main()
