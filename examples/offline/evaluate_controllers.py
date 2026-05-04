"""Evaluate controllers (baseline, OFO, rule-based, PPO) across scenarios.

Combines single-run comparison helpers (plotting, display names) with the
multi-scenario evaluation pipeline (scenario generation, CSV export, aggregate
plots).

Usage:
    python examples/offline/evaluate_controllers.py \
        --ppo-models outputs/ieee13/ppo_ablation_a4_full_episode/ppo_model.zip \
                     outputs/ieee13/ppo_ablation_a9_low_ent/ppo_model.zip \
        --ppo-labels a4_full_ep a9_low_ent \
        --n-scenarios 10 --seed-start 500

    # Quick test with 3 scenarios
    python examples/offline/evaluate_controllers.py \
        --ppo-models outputs/ieee13/ppo_ablation_a5_2M/ppo_model.zip \
        --n-scenarios 3
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from build_scenario_library import run_simulation
from systems import (
    DT_DC,
    EXPERIMENTS,
    SPECS_CACHE_DIR,
    TRAINING_TRACE_PATH,
    V_MAX,
    V_MIN,
    DCSite,
    PVSystemSpec,
    TimeVaryingLoadSpec,
    deploy,
    materialize_scenario,
    randomize_scenario,
)

from openg2g.controller.ofo import LogisticModelStore, OFOConfig
from openg2g.controller.rule_based import RuleBasedConfig
from openg2g.datacenter.config import ModelDeployment
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.datacenter.workloads.training import TrainingTrace

logger = logging.getLogger("evaluate_controllers")


# ── Display helpers ───────────────────────────────────────────────────────────

DISPLAY_NAMES: dict[str, str] = {
    "baseline_no_tap": "No Control",
    "rule_based": "Droop Control",
    "ofo": "OFO Control",
}


def _display_order_key(mode: str) -> int:
    if mode == "baseline_no_tap":
        return 0
    if mode == "rule_based" or mode.startswith("rule_based_s"):
        return 1
    if mode.startswith("ppo_"):
        return 2
    if mode == "ofo" or mode.startswith("ofo_"):
        return 3
    return 99


def _sort_modes(modes: list[str]) -> list[str]:
    return sorted(modes, key=_display_order_key)


def _display_name(mode: str) -> str:
    if mode in DISPLAY_NAMES:
        return DISPLAY_NAMES[mode]
    if mode.startswith("rule_based_s"):
        return "Droop Control"
    if mode.startswith("ppo_"):
        return "PPO Control"
    return mode.replace("_", " ").title()


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_voltage_comparison(
    logs: dict[str, object],
    save_dir: Path,
    *,
    v_min: float = 0.95,
    v_max: float = 1.05,
    exclude_buses: tuple[str, ...] = (),
    scenario_idx: int | None = None,
    use_display_names: bool = False,
) -> None:
    """Side-by-side voltage envelopes for each controller mode."""
    modes = _sort_modes(list(logs.keys()))
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
        ax.set_xlabel("Time (s)", fontsize=13)
        ax.set_title(_display_name(mode) if use_display_names else mode, fontsize=14)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Voltage (pu)", fontsize=13)
    fig.suptitle("Voltage Envelope Comparison", fontsize=16, fontweight="bold")
    fig.tight_layout()
    stem = f"scenario_{scenario_idx:03d}_voltage_comparison" if scenario_idx is not None else "voltage_comparison"
    fig.savefig(save_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s.png", stem)


def plot_violation_bars(
    results: dict[str, dict],
    save_dir: Path,
    *,
    scenario_idx: int | None = None,
    use_display_names: bool = False,
) -> None:
    """Four-panel bar chart for a single scenario:
    violation time, integral violation, mean throughput, batch size changes.
    """
    modes = _sort_modes(list(results.keys()))
    if not modes:
        return

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(modes))]

    viol_s = [float(results[m].get("violation_time_s", 0.0)) for m in modes]
    integ = [float(results[m].get("integral", 0.0)) for m in modes]
    tput = [float(results[m].get("mean_throughput_toks_s", 0.0)) for m in modes]
    batch_chg = [float(results[m].get("batch_changes", 0.0)) for m in modes]
    labels = [_display_name(m) if use_display_names else m for m in modes]

    fig, axes = plt.subplots(1, 4, figsize=(max(18, 2.0 * len(modes) + 12), 6))
    ax_v, ax_i, ax_t, ax_b = axes

    x = np.arange(len(modes))
    for ax, vals, ylabel, title, fmt in [
        (ax_v, viol_s, "Violation time (s)", "Violation time", "{:.0f}"),
        (ax_i, integ, "Integral violation (pu·s)", "Integral violation", "{:.2f}"),
        (ax_t, tput, "Throughput (tok/s)", "Mean throughput", "{:.2e}"),
        (ax_b, batch_chg, "Batch size changes", "Batch size changes", "{:.0f}"),
    ]:
        ax.bar(x, vals, color=colors, alpha=0.88, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.3)
        for xi, val in zip(x, vals):
            ax.text(xi, val, fmt.format(val), ha="center", va="bottom", fontsize=10)

    fig.suptitle("Per-scenario controller metrics", fontsize=16, fontweight="bold")
    fig.tight_layout()
    stem = f"scenario_{scenario_idx:03d}_performance_summary" if scenario_idx is not None else "violation_bars"
    fig.savefig(save_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s.png", stem)


def plot_batch_comparison(
    logs: dict[str, object],
    save_dir: Path,
    *,
    scenario_idx: int | None = None,
    use_display_names: bool = False,
) -> None:
    """Batch size over time for each controller mode, one subplot per (site, model) pair."""
    modes = _sort_modes(list(logs.keys()))
    if not modes:
        logger.info("plot_batch_comparison: no controllers to plot")
        return

    site_models: list[tuple[str, str]] = []
    for log in logs.values():
        for site_id, states in log.dc_states_by_site.items():
            if not states:
                continue
            for m in states[0].batch_size_by_model:
                pair = (site_id, m)
                if pair not in site_models:
                    site_models.append(pair)
        break

    n_rows = len(site_models)
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(13, 4 * n_rows),
        sharex=True,
        squeeze=False,
    )

    for row, (site_id, model_label) in enumerate(site_models):
        ax = axes[row][0]
        for i, mode in enumerate(modes):
            log = logs[mode]
            site_states = log.dc_states_by_site.get(site_id, [])
            times = [s.time_s for s in site_states]
            batches = [s.batch_size_by_model.get(model_label, 0) for s in site_states]
            ax.plot(
                times,
                batches,
                color=cmap(i % 10),
                linewidth=1.5,
                alpha=0.85,
                label=_display_name(mode) if use_display_names else mode,
            )
        ax.set_ylabel("Batch Size", fontsize=13)
        title = f"{model_label} @ {site_id}" if len(log.dc_states_by_site) > 1 else model_label
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=12, loc="upper right")
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.3)

    axes[-1][0].set_xlabel("Time (s)", fontsize=13)
    fig.suptitle("Batch Size Comparison by Model", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    stem = f"scenario_{scenario_idx:03d}_batch_size_comparison" if scenario_idx is not None else "batch_size_comparison"
    fig.savefig(save_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s.png", stem)


# ── Metrics helpers ───────────────────────────────────────────────────────────

def count_batch_changes(log) -> int:
    """Count step-to-step batch-size changes summed across every (site, model) series."""
    total = 0
    states_by_site = getattr(log, "dc_states_by_site", None) or {}
    for states in states_by_site.values():
        if len(states) < 2:
            continue
        labels: set[str] = set()
        for s in states:
            labels.update(s.batch_size_by_model.keys())
        for lab in labels:
            prev = None
            for s in states:
                bs = s.batch_size_by_model.get(lab, 0)
                if prev is not None and bs != prev:
                    total += 1
                prev = bs
    return total


def extract_perf_metrics(log, itl_deadlines: dict[str, float] | None = None) -> dict[str, float]:
    """Pull throughput / latency / power time-series from a SimulationLog and
    return scalar summaries (means, peaks, percentiles).
    """
    dc_states = list(log.dc_states)
    if not dc_states:
        return {
            "mean_throughput_toks_s": 0.0,
            "peak_throughput_toks_s": 0.0,
            "mean_latency_s": 0.0,
            "p99_latency_s": 0.0,
            "mean_power_kw": 0.0,
            "peak_power_kw": 0.0,
            "batch_changes": 0,
            "itl_violation_rate": 0.0,
        }

    labels = set()
    for s in dc_states:
        labels.update(s.batch_size_by_model.keys())
    labels = sorted(labels)

    tps_total = np.zeros(len(dc_states))
    itl_vals: list[float] = []
    for i, s in enumerate(dc_states):
        for lab in labels:
            bs = float(s.batch_size_by_model.get(lab, 0) or 0)
            replicas = float(s.active_replicas_by_model.get(lab, 0) or 0)
            itl = float(s.observed_itl_s_by_model.get(lab, float("nan")))
            if itl > 0 and not math.isnan(itl):
                tps_total[i] += bs * replicas / itl
                itl_vals.append(itl)
    mean_tps = float(np.nanmean(tps_total)) if len(tps_total) else 0.0
    peak_tps = float(np.nanmax(tps_total)) if len(tps_total) else 0.0
    mean_itl = float(np.nanmean(itl_vals)) if itl_vals else 0.0
    p99_itl = float(np.nanpercentile(itl_vals, 99)) if itl_vals else 0.0

    dc_kw_series: list[float] = []
    for s in dc_states:
        p = getattr(s, "power_w", None)
        if p is None:
            continue
        try:
            total_w = float(p.a) + float(p.b) + float(p.c)
        except Exception:
            total_w = 0.0
        dc_kw_series.append(total_w / 1000.0)
    mean_kw = float(np.mean(dc_kw_series)) if dc_kw_series else 0.0
    peak_kw = float(np.max(dc_kw_series)) if dc_kw_series else 0.0

    itl_viol_count = 0
    itl_total_count = 0
    if itl_deadlines:
        for s in dc_states:
            for label, deadline in itl_deadlines.items():
                itl = s.observed_itl_s_by_model.get(label, float("nan"))
                itl = float(itl)
                if not math.isnan(itl) and itl > 0:
                    itl_total_count += 1
                    if itl > deadline:
                        itl_viol_count += 1
    itl_violation_rate = itl_viol_count / itl_total_count if itl_total_count > 0 else 0.0

    return {
        "mean_throughput_toks_s": mean_tps,
        "peak_throughput_toks_s": peak_tps,
        "mean_latency_s": mean_itl,
        "p99_latency_s": p99_itl,
        "mean_power_kw": mean_kw,
        "peak_power_kw": peak_kw,
        "batch_changes": count_batch_changes(log),
        "itl_violation_rate": itl_violation_rate,
    }


# ── Scenario generation ───────────────────────────────────────────────────────

def generate_test_scenarios(
    exp: dict,
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
    *,
    n_scenarios: int,
    seed_start: int,
    min_baseline_integral: float = 0.2,
    min_recovery_frac: float = 0.7,
    save_dir: Path,
    randomize_ramps: bool = True,
) -> list[dict]:
    """Generate randomized scenarios and filter using the library recovery rule.

    A seed is accepted only if:
        baseline_no_tap integral >= min_baseline_integral  AND
        (base - ofo) / base >= min_recovery_frac
    """
    dc_sites_base = exp["dc_sites"]
    pv_base = exp.get("pv_systems", [])
    tvl_base = exp.get("time_varying_loads", [])
    training_base = exp.get("training_base")

    sys_cfg = exp["sys"]
    ofo_config = exp["ofo_config"]

    accepted: list[dict] = []
    seed = seed_start
    attempts = 0
    max_attempts = n_scenarios * 20
    tried: list[str] = []

    while len(accepted) < n_scenarios and attempts < max_attempts:
        effective_seed = seed * 1000 + 7
        sc = randomize_scenario(
            seed=effective_seed,
            dc_sites_base=dc_sites_base,
            pv_systems_base=pv_base,
            tvl_base=tvl_base,
            training_base=training_base,
            randomize_ramps=randomize_ramps,
        )

        training_overlay = sc["params"]["training_overlay"]

        bl_stats, bl_log = run_simulation(
            "baseline_no_tap",
            sys=sys_cfg,
            dc_sites=sc["dc_sites"],
            ofo_config=ofo_config,
            inference_data=inference_data,
            training_trace=training_trace,
            logistic_models=logistic_models,
            pv_systems=sc["pv_systems"],
            time_varying_loads=sc["tvl"],
            tap_schedule=exp.get("tap_schedule"),
            training_overlay=training_overlay,
            save_dir=save_dir,
        )
        ofo_stats, ofo_log = run_simulation(
            "ofo",
            sys=sys_cfg,
            dc_sites=sc["dc_sites"],
            ofo_config=ofo_config,
            inference_data=inference_data,
            training_trace=training_trace,
            logistic_models=logistic_models,
            pv_systems=sc["pv_systems"],
            time_varying_loads=sc["tvl"],
            tap_schedule=exp.get("tap_schedule"),
            training_overlay=training_overlay,
            save_dir=save_dir,
        )
        base_int = float(bl_stats.integral_violation_pu_s)
        ofo_int = float(ofo_stats.integral_violation_pu_s)
        recovery = (base_int - ofo_int) / base_int if base_int > 0 else 0.0
        passes = base_int >= min_baseline_integral and recovery >= min_recovery_frac
        verdict = "ACCEPT" if passes else "reject"
        line = (
            f"  seed={effective_seed} base_int={base_int:.3f} ofo_int={ofo_int:.4f} "
            f"recovery={100*recovery:.1f}%  {verdict}"
        )
        logger.info(line)
        tried.append(line)

        if passes:
            sc["_filter_results"] = {
                "baseline_no_tap": {"stats": bl_stats, "log": bl_log},
                "ofo": {"stats": ofo_stats, "log": ofo_log},
            }
            accepted.append(sc)
        seed += 1
        attempts += 1

    logger.info(
        "Filter complete: %d accepted out of %d attempted seeds (%d-%d)",
        len(accepted), attempts, seed_start, seed - 1,
    )
    if len(accepted) < n_scenarios:
        logger.warning(
            "Only accepted %d/%d scenarios — consider lowering min_recovery_frac or expanding seed range",
            len(accepted), n_scenarios,
        )
    return accepted


def load_scenarios_from_library(
    library_path: str,
    *,
    exp: dict,
    n_scenarios: int,
) -> list[dict]:
    """Load pre-screened scenarios from a ScenarioLibrary pickle.

    Each record carries its resolved ``dc_sites`` / ``pv_systems`` / ``tvl``
    directly, so loading is a pure pickle read — no rng replay, no
    distribution mismatch when records from different build-time ranges are
    mixed in one library.
    """
    from openg2g.rl.env import ScenarioLibrary

    lib = ScenarioLibrary(library_path)
    training_base = exp.get("training_base")

    logger.info("Loaded library with %d records from %s", len(lib), library_path)

    n_take = min(n_scenarios, len(lib))
    if n_take < n_scenarios:
        logger.warning(
            "Library has only %d scenarios — capping n_scenarios from %d to %d",
            len(lib), n_scenarios, n_take,
        )

    return [materialize_scenario(rec, training_base=training_base) for rec in lib.scenarios[:n_take]]


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_scenario(
    scenario: dict,
    *,
    exp: dict,
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
    ppo_models: list[str],
    ppo_labels: list[str],
    save_dir: Path,
    scenario_idx: int,
    obs_mode: str = "full-voltage",
    ofo_variants: list[tuple[str, "OFOConfig"]] | None = None,
    include_rule_based: bool = False,
    rule_based_config: "RuleBasedConfig | None" = None,
    rule_step_sizes: tuple[float, ...] = (10.0,),
    rule_zone_local: bool = False,
    no_per_scenario_plots: bool = False,
    no_default_ofo: bool = False,
    use_display_names: bool = False,
) -> dict:
    """Run baseline, OFO (+variants), rule-based, and PPO models on a single scenario."""
    sys_cfg = exp["sys"]
    ofo_config = exp["ofo_config"]
    exclude_buses = tuple(sys_cfg["exclude_buses"])
    ofo_variants = ofo_variants or []

    itl_deadlines: dict[str, float] = {
        md.spec.model_label: md.spec.itl_deadline_s
        for site in exp["dc_sites"].values()
        for md, _ in site.models
    }

    dc_sites = scenario["dc_sites"]
    pv_systems = scenario["pv_systems"]
    tvl = scenario["tvl"]
    training_overlay = scenario["params"]["training_overlay"]

    sc_save = save_dir / f"scenario_{scenario_idx:03d}"
    sc_save.mkdir(parents=True, exist_ok=True)

    results = {}
    all_logs: dict[str, object] = {}

    cached = scenario.get("_filter_results", {})
    default_modes = ["baseline_no_tap"] if no_default_ofo else ["baseline_no_tap", "ofo"]
    for mode in default_modes:
        if mode in cached:
            vstats = cached[mode]["stats"]
            log = cached[mode]["log"]
        else:
            vstats, log = run_simulation(
                mode,
                sys=sys_cfg,
                dc_sites=dc_sites,
                ofo_config=ofo_config,
                inference_data=inference_data,
                training_trace=training_trace,
                logistic_models=logistic_models,
                pv_systems=pv_systems,
                time_varying_loads=tvl,
                tap_schedule=exp.get("tap_schedule"),
                training_overlay=training_overlay,
                save_dir=sc_save,
            )
        perf = extract_perf_metrics(log, itl_deadlines)
        results[mode] = {
            "violation_time_s": vstats.violation_time_s,
            "integral": vstats.integral_violation_pu_s,
            "worst_vmin": vstats.worst_vmin,
            "worst_vmax": vstats.worst_vmax,
            **perf,
        }
        all_logs[mode] = log
        logger.info(
            "  scenario %d %s: viol=%.0fs integral=%.4f vmin=%.4f vmax=%.4f  "
            "tput=%.1f p99_lat=%.3fs power=%.1fkW batch_chg=%d",
            scenario_idx, mode, vstats.violation_time_s,
            vstats.integral_violation_pu_s, vstats.worst_vmin, vstats.worst_vmax,
            perf["mean_throughput_toks_s"], perf["p99_latency_s"], perf["mean_power_kw"],
            perf["batch_changes"],
        )

    if include_rule_based:
        for step_size in rule_step_sizes:
            label = "rule_based" if len(rule_step_sizes) == 1 else f"rule_based_s{step_size:g}"
            rb_config = rule_based_config or RuleBasedConfig(v_min=V_MIN, v_max=V_MAX, step_size=step_size)
            if rule_based_config is not None:
                rb_config = rule_based_config
            vstats, log = run_simulation(
                label,
                sys=sys_cfg,
                dc_sites=dc_sites,
                ofo_config=ofo_config,
                inference_data=inference_data,
                training_trace=training_trace,
                logistic_models=logistic_models,
                pv_systems=pv_systems,
                time_varying_loads=tvl,
                tap_schedule=exp.get("tap_schedule"),
                rule_based_config=rb_config,
                rule_zone_local=rule_zone_local,
                training_overlay=training_overlay,
                save_dir=sc_save,
            )
            perf = extract_perf_metrics(log, itl_deadlines)
            results[label] = {
                "violation_time_s": vstats.violation_time_s,
                "integral": vstats.integral_violation_pu_s,
                "worst_vmin": vstats.worst_vmin,
                "worst_vmax": vstats.worst_vmax,
                **perf,
            }
            all_logs[label] = log
            logger.info(
                "  scenario %d %s: viol=%.0fs integral=%.4f vmin=%.4f vmax=%.4f  "
                "tput=%.1f p99_lat=%.3fs power=%.1fkW batch_chg=%d",
                scenario_idx, label, vstats.violation_time_s,
                vstats.integral_violation_pu_s, vstats.worst_vmin, vstats.worst_vmax,
                perf["mean_throughput_toks_s"], perf["p99_latency_s"], perf["mean_power_kw"],
                perf["batch_changes"],
            )
            if rule_based_config is not None:
                break

    for variant_label, variant_cfg in ofo_variants:
        vstats, log = run_simulation(
            "ofo",
            sys=sys_cfg,
            dc_sites=dc_sites,
            ofo_config=variant_cfg,
            inference_data=inference_data,
            training_trace=training_trace,
            logistic_models=logistic_models,
            pv_systems=pv_systems,
            time_varying_loads=tvl,
            tap_schedule=exp.get("tap_schedule"),
            training_overlay=training_overlay,
            save_dir=sc_save,
        )
        perf = extract_perf_metrics(log, itl_deadlines)
        mode_key = f"ofo_{variant_label}"
        results[mode_key] = {
            "violation_time_s": vstats.violation_time_s,
            "integral": vstats.integral_violation_pu_s,
            "worst_vmin": vstats.worst_vmin,
            "worst_vmax": vstats.worst_vmax,
            **perf,
        }
        all_logs[mode_key] = log
        logger.info(
            "  scenario %d %s: viol=%.0fs integral=%.4f vmin=%.4f vmax=%.4f  "
            "tput=%.1f p99_lat=%.3fs power=%.1fkW batch_chg=%d",
            scenario_idx, mode_key, vstats.violation_time_s,
            vstats.integral_violation_pu_s, vstats.worst_vmin, vstats.worst_vmax,
            perf["mean_throughput_toks_s"], perf["p99_latency_s"], perf["mean_power_kw"],
            perf["batch_changes"],
        )

    for ppo_path, label in zip(ppo_models, ppo_labels):
        vstats, log = run_simulation(
            "ppo",
            sys=sys_cfg,
            dc_sites=dc_sites,
            ofo_config=ofo_config,
            inference_data=inference_data,
            training_trace=training_trace,
            logistic_models=logistic_models,
            pv_systems=pv_systems,
            time_varying_loads=tvl,
            tap_schedule=exp.get("tap_schedule"),
            ppo_model=ppo_path,
            obs_mode=obs_mode,
            training_overlay=training_overlay,
            save_dir=sc_save,
        )
        perf = extract_perf_metrics(log, itl_deadlines)
        results[f"ppo_{label}"] = {
            "violation_time_s": vstats.violation_time_s,
            "integral": vstats.integral_violation_pu_s,
            "worst_vmin": vstats.worst_vmin,
            "worst_vmax": vstats.worst_vmax,
            **perf,
        }
        all_logs[f"ppo_{label}"] = log
        logger.info(
            "  scenario %d ppo_%s: viol=%.0fs integral=%.4f vmin=%.4f vmax=%.4f  "
            "tput=%.1f p99_lat=%.3fs power=%.1fkW batch_chg=%d",
            scenario_idx, label, vstats.violation_time_s,
            vstats.integral_violation_pu_s, vstats.worst_vmin, vstats.worst_vmax,
            perf["mean_throughput_toks_s"], perf["p99_latency_s"], perf["mean_power_kw"],
            perf["batch_changes"],
        )

    if not no_per_scenario_plots:
        plot_voltage_comparison(
            all_logs, sc_save,
            v_min=V_MIN, v_max=V_MAX, exclude_buses=exclude_buses,
            scenario_idx=scenario_idx, use_display_names=use_display_names,
        )
        plot_batch_comparison(all_logs, sc_save, scenario_idx=scenario_idx, use_display_names=use_display_names)
        plot_violation_bars(results, sc_save, scenario_idx=scenario_idx, use_display_names=use_display_names)

    return results


def plot_aggregate(
    all_results: list[dict],
    scenario_params: list[dict],
    save_dir: Path,
    modes: list[str],
    *,
    system: str = "",
    use_display_names: bool = False,
) -> None:
    """2×3 aggregate bar chart (means) + per-scenario breakdown + normalized integral + CDF + scatter."""
    n_sc = len(all_results)
    prefix = f"{system}_" if system else ""

    colors = ["#999999", "#4CAF50", "#2196F3", "#FF9800", "#E91E63",
              "#9C27B0", "#00BCD4", "#795548", "#607D8B"]

    display_labels = [_display_name(m) if use_display_names else m for m in modes]

    metrics = [
        ("violation_time_s",       "Mean Violation Time (s)"),
        ("integral",                "Mean Integral Violation (pu·s)"),
        ("batch_changes",           "Mean Batch Size Changes"),
        ("mean_throughput_toks_s",  "Mean Throughput (tok/s)"),
        ("mean_power_kw",           "Mean Data Center Power (kW)"),
        ("itl_violation_rate",      "Mean ITL Violation Rate"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(max(15, len(modes) * 3.0), 10))
    x = np.arange(len(modes))

    for ax, (metric, title) in zip(axes.flat, metrics):
        means = []
        for mode in modes:
            vals = [r[mode].get(metric, 0) for r in all_results if mode in r]
            means.append(np.mean(vals) if vals else 0.0)

        ax.bar(x, means, color=colors[: len(modes)], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, rotation=30, ha="right", fontsize=12)
        ax.set_ylabel(title, fontsize=13)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Aggregate Controller Metrics — {n_sc} Scenarios", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f"{prefix}controller_evaluation.png"
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)

    # ── Per-scenario integral (absolute) ──
    fig, ax = plt.subplots(figsize=(max(10, n_sc * 0.8), 6))
    x = np.arange(n_sc)
    width = 0.8 / len(modes)

    for i, mode in enumerate(modes):
        vals = [r[mode]["integral"] if mode in r else 0 for r in all_results]
        ax.bar(x + i * width, vals, width,
               label=_display_name(mode) if use_display_names else mode,
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xlabel("Scenario", fontsize=13)
    ax.set_ylabel("Integral Violation (pu·s)", fontsize=13)
    ax.set_title("Per-Scenario Integral Violation", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels([f"S{i}" for i in range(n_sc)], fontsize=10)
    ax.legend(fontsize=10, loc="upper right")
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fname = f"{prefix}scenario_summary.png"
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)

    # ── Per-scenario normalized integral (relative to baseline_no_tap) ──
    baseline_key = "baseline_no_tap"
    if baseline_key in modes:
        fig, ax = plt.subplots(figsize=(max(10, n_sc * 0.8), 6))
        x = np.arange(n_sc)
        non_baseline = [m for m in modes if m != baseline_key]
        width = 0.8 / len(non_baseline)

        for i, mode in enumerate(non_baseline):
            norm_vals = []
            for r in all_results:
                base = r.get(baseline_key, {}).get("integral", 0.0)
                val = r.get(mode, {}).get("integral", 0.0)
                norm_vals.append(val / base if base > 0 else 0.0)
            ax.bar(x + i * width, norm_vals, width,
                   label=_display_name(mode) if use_display_names else mode,
                   color=colors[(modes.index(mode)) % len(colors)], alpha=0.85)

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6, label="Baseline")
        ax.set_xlabel("Scenario", fontsize=13)
        ax.set_ylabel("Normalized Integral (relative to No Control)", fontsize=13)
        ax.set_title("Per-Scenario Normalized Integral Violation", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (len(non_baseline) - 1) / 2)
        ax.set_xticklabels([f"S{i}" for i in range(n_sc)], fontsize=10)
        ax.legend(fontsize=10, loc="upper right")
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fname = f"{prefix}scenario_normalized_integral.png"
        fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", fname)

    # ── CDF of integral violation ──
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, mode in enumerate(modes):
        vals = sorted([r[mode].get("integral", 0.0) for r in all_results if mode in r])
        if not vals:
            continue
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, color=colors[i % len(colors)], linewidth=2,
                label=_display_name(mode) if use_display_names else mode)
    ax.set_xlabel("Integral Violation (pu·s)", fontsize=13)
    ax.set_ylabel("Cumulative Fraction", fontsize=13)
    ax.set_title("CDF of Integral Violation Across Scenarios", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f"{prefix}cdf_integral.png"
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)

    # ── Throughput vs. voltage violation scatter ──
    scatter_data = {}
    for i, mode in enumerate(modes):
        integrals = [r[mode].get("integral", 0.0) for r in all_results if mode in r]
        tputs = [r[mode].get("mean_throughput_toks_s", 0.0) for r in all_results if mode in r]
        scatter_data[mode] = (integrals, tputs)

    n_modes = len(modes)
    ncols = 2
    nrows = math.ceil(n_modes / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows),
                             sharex=True, sharey=True, squeeze=False)

    for idx, mode in enumerate(modes):
        ax = axes[idx // ncols][idx % ncols]
        integrals, tputs = scatter_data[mode]
        label = _display_name(mode) if use_display_names else mode

        for other_mode, (oi, ot) in scatter_data.items():
            if other_mode != mode:
                ax.scatter(oi, ot, color="lightgrey", s=40, alpha=0.6,
                           edgecolors="none", zorder=1)

        ax.scatter(integrals, tputs, color=colors[idx % len(colors)],
                   s=80, alpha=0.9, edgecolors="black", linewidths=0.5, zorder=2)
        ax.scatter(np.mean(integrals), np.mean(tputs),
                   color=colors[idx % len(colors)], s=220, marker="*",
                   edgecolors="black", linewidths=0.8, zorder=3)

        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_xlabel("Integral Violation (pu·s)", fontsize=12)
        ax.set_ylabel("Mean Throughput (tok/s)", fontsize=12)
        ax.tick_params(labelsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2e}"))
        ax.grid(True, alpha=0.3)

    for idx in range(n_modes, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Throughput vs. Voltage Violation by Controller", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f"{prefix}throughput_vs_violation.png"
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", fname)

    logger.info("Saved aggregate figures to %s", save_dir)


def main(
    *,
    ppo_models: tuple[str, ...] = (),
    ppo_labels: tuple[str, ...] = (),
    system: str = "ieee13",
    n_scenarios: int = 10,
    seed_start: int = 500,
    output_dir: str = "",
    obs_mode: str = "full-voltage",
    min_baseline_integral: float = 0.2,
    min_recovery_frac: float = 0.7,
    ofo_w_throughputs: tuple[float, ...] = (),
    ofo_w_switches: tuple[float, ...] = (),
    ofo_primal_steps: tuple[float, ...] = (),
    ofo_extra_variants: tuple[str, ...] = (),
    no_default_ofo: bool = False,
    include_rule_based: bool = False,
    rule_step_sizes: tuple[float, ...] = (10.0,),
    rule_zone_local: bool = False,
    no_per_scenario_plots: bool = False,
    no_aggregate_plots: bool = False,
    randomize_ramps: bool = True,
    scenario_library: str = "",
    use_display_names: bool = False,
    log_level: str = "INFO",
) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openg2g.coordinator").setLevel(logging.WARNING)
    logging.getLogger("openg2g.datacenter").setLevel(logging.WARNING)
    logging.getLogger("openg2g.grid").setLevel(logging.WARNING)
    logging.getLogger("openg2g.controller.ofo").setLevel(logging.WARNING)

    ppo_models_resolved = [str(Path(p).resolve()) for p in ppo_models]

    if not ppo_labels:
        ppo_labels = tuple(
            Path(p).parent.name.replace("ppo_ablation_", "") for p in ppo_models
        )

    if system not in EXPERIMENTS:
        raise ValueError(f"Unknown system {system!r}. Valid: {sorted(EXPERIMENTS)}")

    training_trace = TrainingTrace.ensure(TRAINING_TRACE_PATH)

    exp = EXPERIMENTS[system](training_trace=training_trace)
    all_models = []
    for site in exp["dc_sites"].values():
        # site.models is now tuple[(ModelDeployment, ReplicaSchedule), ...]
        all_models.extend(md for md, _ in site.models)
    all_specs = tuple(m.spec for m in all_models)

    inference_data = InferenceData.ensure(
        SPECS_CACHE_DIR, all_specs, plot=False, dt_s=float(DT_DC),
    )
    logistic_models = LogisticModelStore.ensure(
        SPECS_CACHE_DIR, all_specs, plot=False,
    )

    base_ofo = exp["ofo_config"]
    ofo_variants: list[tuple[str, OFOConfig]] = []

    def _w_tag(w: float) -> str:
        if w == 0:
            return "w0"
        if w >= 1e-3:
            return f"w{w:g}".replace(".", "p")
        return f"w{w:.0e}".replace("-0", "-")

    def _fmt(v: float) -> str:
        return f"{v:g}".replace(".", "p")

    if ofo_w_throughputs or ofo_w_switches or ofo_primal_steps:
        w_list = ofo_w_throughputs or (base_ofo.w_throughput,)
        s_list = ofo_w_switches or (base_ofo.w_switch,)
        p_list = ofo_primal_steps or (base_ofo.primal_step_size,)
        for w in w_list:
            for s in s_list:
                for p in p_list:
                    overrides: dict = {}
                    parts: list[str] = []
                    if ofo_w_throughputs:
                        overrides["w_throughput"] = float(w)
                        parts.append(_w_tag(float(w)))
                    if ofo_w_switches:
                        overrides["w_switch"] = float(s)
                        parts.append(f"ws{_fmt(float(s))}")
                    if ofo_primal_steps:
                        overrides["primal_step_size"] = float(p)
                        parts.append(f"ps{_fmt(float(p))}")
                    cfg = base_ofo.model_copy(update=overrides)
                    label = "_".join(parts) if parts else "variant"
                    ofo_variants.append((label, cfg))

    for spec in ofo_extra_variants:
        overrides: dict = {}
        parts: list[str] = []
        for kv in spec.split(","):
            k, _, v = kv.strip().partition("=")
            if not k:
                continue
            fv = float(v)
            overrides[k] = fv
            if k == "w_throughput":
                parts.append(_w_tag(fv))
            elif k == "w_switch":
                parts.append(f"ws{_fmt(fv)}")
            elif k == "primal_step_size":
                parts.append(f"ps{_fmt(fv)}")
            else:
                parts.append(f"{k}{_fmt(fv)}")
        cfg = base_ofo.model_copy(update=overrides)
        ofo_variants.append(("_".join(parts) if parts else "extra", cfg))

    if ofo_variants:
        logger.info(
            "OFO variants (%d): %s",
            len(ofo_variants),
            [(l, c.w_throughput, c.w_switch, c.primal_step_size) for l, c in ofo_variants],
        )

    save_dir = Path(__file__).resolve().parent / "outputs" / system / (
        output_dir or f"eval_multi_seed{seed_start}_n{n_scenarios}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    if scenario_library:
        test_scenarios = load_scenarios_from_library(
            scenario_library,
            exp=exp,
            n_scenarios=n_scenarios,
        )
    else:
        test_scenarios = generate_test_scenarios(
            exp,
            inference_data=inference_data,
            training_trace=training_trace,
            logistic_models=logistic_models,
            n_scenarios=n_scenarios,
            seed_start=seed_start,
            min_baseline_integral=min_baseline_integral,
            min_recovery_frac=min_recovery_frac,
            save_dir=save_dir,
            randomize_ramps=randomize_ramps,
        )

    all_results = []
    scenario_params = []
    modes_set = set()

    for i, scenario in enumerate(test_scenarios):
        params = scenario["params"]
        logger.info("")
        logger.info("=" * 70)
        logger.info(
            "SCENARIO %d/%d: seed=%d pv_scale=%.2f load_scale=%.2f",
            i + 1, n_scenarios, scenario["seed"],
            params["pv_scale"], params["load_scale"],
        )
        if params["training_overlay"]:
            to = params["training_overlay"]
            logger.info(
                "  training: t=[%.0f, %.0f] n_gpus=%d",
                to["t_start"], to["t_end"], to["n_gpus"],
            )
        logger.info("=" * 70)

        results = run_scenario(
            scenario,
            exp=exp,
            inference_data=inference_data,
            training_trace=training_trace,
            logistic_models=logistic_models,
            ppo_models=ppo_models_resolved,
            ppo_labels=list(ppo_labels),
            save_dir=save_dir,
            scenario_idx=i,
            obs_mode=obs_mode,
            ofo_variants=ofo_variants,
            include_rule_based=include_rule_based,
            rule_step_sizes=rule_step_sizes,
            rule_zone_local=rule_zone_local,
            no_per_scenario_plots=no_per_scenario_plots,
            no_default_ofo=no_default_ofo,
            use_display_names=use_display_names,
        )
        all_results.append(results)
        scenario_params.append(params)
        modes_set.update(results.keys())

    rb_labels = (
        ["rule_based"] if len(rule_step_sizes) == 1
        else [f"rule_based_s{s:g}" for s in rule_step_sizes]
    )
    mode_order = (
        ["baseline_no_tap"] + rb_labels
        + [f"ppo_{l}" for l in ppo_labels]
        + ["ofo"] + [f"ofo_{label}" for label, _ in ofo_variants]
    )
    modes = [m for m in mode_order if m in modes_set]

    logger.info("")
    logger.info("=" * 90)
    logger.info("AGGREGATE RESULTS (%d scenarios)", n_scenarios)
    logger.info("=" * 90)
    header = (
        f"{'Mode':<20s} {'Viol(s)':>10s} {'±':>8s} {'Integral':>10s} {'±':>8s} "
        f"{'Worst Vmin':>12s} {'Worst Vmax':>12s} {'Batch Δ':>10s} {'±':>8s}"
    )
    logger.info(header)
    logger.info("-" * 104)

    for mode in modes:
        viol = [r[mode]["violation_time_s"] for r in all_results if mode in r]
        intg = [r[mode]["integral"] for r in all_results if mode in r]
        vmin = [r[mode]["worst_vmin"] for r in all_results if mode in r]
        vmax = [r[mode]["worst_vmax"] for r in all_results if mode in r]
        bchg = [r[mode].get("batch_changes", 0) for r in all_results if mode in r]
        if viol:
            logger.info(
                "%-20s %10.1f %8.1f %10.4f %8.4f %12.4f %12.4f %10.1f %8.1f",
                mode,
                np.mean(viol), np.std(viol),
                np.mean(intg), np.std(intg),
                np.mean(vmin), np.mean(vmax),
                np.mean(bchg), np.std(bchg),
            )

    extra_cols = [
        "mean_throughput_toks_s", "peak_throughput_toks_s",
        "mean_latency_s", "p99_latency_s",
        "mean_power_kw", "peak_power_kw",
        "batch_changes",
        "itl_violation_rate",
    ]
    csv_path = save_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario", "seed", "pv_scale", "load_scale",
            "mode", "violation_time_s", "integral", "worst_vmin", "worst_vmax",
            *extra_cols,
        ])
        for i, (results, params) in enumerate(zip(all_results, scenario_params)):
            for mode, stats in results.items():
                sc = test_scenarios[i]
                writer.writerow([
                    i, sc["seed"], params["pv_scale"], params["load_scale"],
                    mode,
                    stats["violation_time_s"], stats["integral"],
                    stats["worst_vmin"], stats["worst_vmax"],
                    *[stats.get(k, "") for k in extra_cols],
                ])
    logger.info("Results CSV: %s", csv_path)

    if not no_aggregate_plots:
        plot_aggregate(all_results, scenario_params, save_dir, modes, system=system, use_display_names=use_display_names)
    logger.info("All outputs saved to: %s", save_dir)


if __name__ == "__main__":
    import tyro

    @dataclass
    class Args:
        ppo_models: tuple[str, ...] = ()
        """Paths to trained PPO model .zip files. Empty = run only baseline + OFO variants."""
        ppo_labels: tuple[str, ...] = ()
        """Legend labels (one per model). Defaults to parent dir names."""
        system: str = "ieee13"
        """Which feeder experiment to use. Valid: ieee13, ieee34, ieee123."""
        n_scenarios: int = 10
        """Number of held-out scenarios to evaluate on."""
        seed_start: int = 500
        """Starting seed offset for test scenarios (seeds = seed_start*1000+7, ...)."""
        output_dir: str = ""
        """Output directory name under outputs/<system>/. Auto-generated if empty."""
        obs_mode: str = "full-voltage"
        """Observation mode used during PPO training: full-voltage, per-zone-summary, per-bus-summary, or system-summary-only."""
        min_baseline_integral: float = 0.2
        """Minimum baseline_no_tap integral (pu*s) for a scenario seed to be accepted."""
        min_recovery_frac: float = 0.7
        """Minimum (base-ofo)/base recovery fraction for a seed to be accepted."""
        ofo_w_throughputs: tuple[float, ...] = ()
        """Extra OFO variants, given as throughput weight values (e.g. 0.0001 0.00001 0)."""
        ofo_w_switches: tuple[float, ...] = ()
        """Extra OFO variants, given as switching-cost weight values (e.g. 1.0 3.0 10.0)."""
        ofo_primal_steps: tuple[float, ...] = ()
        """Extra OFO variants, given as primal-step-size values (e.g. 0.02 0.05 0.1)."""
        ofo_extra_variants: tuple[str, ...] = ()
        """Explicit OFO variants as 'k=v,k=v' (e.g. 'w_throughput=0,w_switch=1.0')."""
        no_default_ofo: bool = False
        """Skip emitting the default OFO as 'ofo' in results. Filter/cache path still uses it."""
        include_rule_based: bool = False
        """Also evaluate the rule-based controller."""
        rule_step_sizes: tuple[float, ...] = (10.0,)
        """Step size(s) for the rule-based controller."""
        rule_zone_local: bool = False
        """When True AND sys defines `zones` AND there are >1 DC sites, each rule-based controller observes only buses in its own zone (decentralized credit assignment for ieee123)."""
        no_per_scenario_plots: bool = False
        """Skip per-scenario voltage and batch plots (saves disk + time)."""
        no_aggregate_plots: bool = False
        """Skip aggregate comparison and per-scenario integral plots."""
        randomize_ramps: bool = True
        """Synthesize per-episode inference ramps. Set --no-randomize-ramps for ieee34."""
        scenario_library: str = ""
        """Path to a pre-screened scenario library .pkl (from build_scenario_library.py)."""
        use_display_names: bool = False
        """Use human-readable display names in all plots."""
        log_level: str = "INFO"
        """Logging verbosity."""

    args = tyro.cli(Args)
    main(
        ppo_models=args.ppo_models,
        ppo_labels=args.ppo_labels,
        system=args.system,
        n_scenarios=args.n_scenarios,
        seed_start=args.seed_start,
        output_dir=args.output_dir,
        obs_mode=args.obs_mode,
        min_baseline_integral=args.min_baseline_integral,
        min_recovery_frac=args.min_recovery_frac,
        ofo_w_throughputs=args.ofo_w_throughputs,
        ofo_w_switches=args.ofo_w_switches,
        ofo_primal_steps=args.ofo_primal_steps,
        ofo_extra_variants=args.ofo_extra_variants,
        no_default_ofo=args.no_default_ofo,
        include_rule_based=args.include_rule_based,
        rule_step_sizes=args.rule_step_sizes,
        rule_zone_local=args.rule_zone_local,
        no_per_scenario_plots=args.no_per_scenario_plots,
        no_aggregate_plots=args.no_aggregate_plots,
        randomize_ramps=args.randomize_ramps,
        scenario_library=args.scenario_library,
        use_display_names=args.use_display_names,
        log_level=args.log_level,
    )
