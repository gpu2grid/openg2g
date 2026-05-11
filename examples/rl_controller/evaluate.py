"""Evaluate controllers (baseline, OFO, rule-based, PPO) across scenarios.

Combines single-run comparison helpers (plotting, display names) with the
multi-scenario evaluation pipeline (scenario generation, CSV export, aggregate
plots).

Usage:
    python examples/rl_controller/evaluate.py \
        --ppo-models outputs/ieee13/ppo_seed1/ppo_model.zip \
                     outputs/ieee13/ppo_seed2/ppo_model.zip \
        --ppo-labels seed1 seed2 \
        --n-scenarios 10 --seed-start 500

    # Quick test with 3 scenarios
    python examples/rl_controller/evaluate.py \
        --ppo-models outputs/ieee13/ppo/ppo_model.zip \
        --n-scenarios 3
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from build_library import run_simulation
from env import ScenarioLibrary
from scenarios import (
    EXPERIMENTS,
    randomize_scenario,
)

from openg2g.controller.ofo import LogisticModelStore, OFOConfig
from openg2g.controller.rule_based import RuleBasedConfig
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.datacenter.workloads.training import TrainingTrace

from plotting import (
    plot_aggregate,
    plot_batch_comparison,
    plot_violation_bars,
    plot_voltage_comparison,
)
from systems import (
    DT_DC,
    SPECS_CACHE_DIR,
    TRAINING_TRACE_PATH,
    V_MAX,
    V_MIN,
)

logger = logging.getLogger("evaluate_controllers")


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
            f"recovery={100 * recovery:.1f}%  {verdict}"
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
        len(accepted),
        attempts,
        seed_start,
        seed - 1,
    )
    if len(accepted) < n_scenarios:
        logger.warning(
            "Only accepted %d/%d scenarios: consider lowering min_recovery_frac or expanding seed range",
            len(accepted),
            n_scenarios,
        )
    return accepted


def load_scenarios_from_library(
    library_path: str,
    *,
    n_scenarios: int,
    training_trace: TrainingTrace,
) -> list[dict]:
    """Load pre-screened scenarios from a `ScenarioLibrary` directory.

    Replays `randomize_scenario(seed)` for each record (deterministic, since
    the RNG is seeded) to rebuild the per-episode dict. `training_trace` is
    needed because libraries built with `--use-training-overlay` reference
    a TrainingTrace at materialization time; for libraries without overlay
    it can be `None`.
    """
    lib = ScenarioLibrary(library_path, training_trace=training_trace)

    logger.info("Loaded library with %d records from %s", len(lib), library_path)

    n_take = min(n_scenarios, len(lib))
    if n_take < n_scenarios:
        logger.warning(
            "Library has only %d scenarios; capping n_scenarios from %d to %d",
            len(lib),
            n_scenarios,
            n_take,
        )

    return [lib.materialize(rec) for rec in lib.scenarios[:n_take]]


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
    ofo_variants: list[tuple[str, OFOConfig]] | None = None,
    include_rule_based: bool = False,
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
        md.spec.model_label: md.spec.itl_deadline_s for site in exp["dc_sites"].values() for md, _ in site.models
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
            scenario_idx,
            mode,
            vstats.violation_time_s,
            vstats.integral_violation_pu_s,
            vstats.worst_vmin,
            vstats.worst_vmax,
            perf["mean_throughput_toks_s"],
            perf["p99_latency_s"],
            perf["mean_power_kw"],
            perf["batch_changes"],
        )

    if include_rule_based:
        for step_size in rule_step_sizes:
            label = "rule_based" if len(rule_step_sizes) == 1 else f"rule_based_s{step_size:g}"
            rb_config = RuleBasedConfig(v_min=V_MIN, v_max=V_MAX, step_size=step_size)
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
                scenario_idx,
                label,
                vstats.violation_time_s,
                vstats.integral_violation_pu_s,
                vstats.worst_vmin,
                vstats.worst_vmax,
                perf["mean_throughput_toks_s"],
                perf["p99_latency_s"],
                perf["mean_power_kw"],
                perf["batch_changes"],
            )

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
            scenario_idx,
            mode_key,
            vstats.violation_time_s,
            vstats.integral_violation_pu_s,
            vstats.worst_vmin,
            vstats.worst_vmax,
            perf["mean_throughput_toks_s"],
            perf["p99_latency_s"],
            perf["mean_power_kw"],
            perf["batch_changes"],
        )

    for ppo_path, label in zip(ppo_models, ppo_labels, strict=False):
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
            scenario_idx,
            label,
            vstats.violation_time_s,
            vstats.integral_violation_pu_s,
            vstats.worst_vmin,
            vstats.worst_vmax,
            perf["mean_throughput_toks_s"],
            perf["p99_latency_s"],
            perf["mean_power_kw"],
            perf["batch_changes"],
        )

    if not no_per_scenario_plots:
        plot_voltage_comparison(
            all_logs,
            sc_save,
            v_min=V_MIN,
            v_max=V_MAX,
            exclude_buses=exclude_buses,
            scenario_idx=scenario_idx,
            use_display_names=use_display_names,
        )
        plot_batch_comparison(all_logs, sc_save, scenario_idx=scenario_idx, use_display_names=use_display_names)
        plot_violation_bars(results, sc_save, scenario_idx=scenario_idx, use_display_names=use_display_names)

    return results


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
        ppo_labels = tuple(str(i) for i in range(len(ppo_models)))

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
        SPECS_CACHE_DIR,
        all_specs,
        plot=False,
        dt_s=float(DT_DC),
    )
    logistic_models = LogisticModelStore.ensure(
        SPECS_CACHE_DIR,
        all_specs,
        plot=False,
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
            [(lbl, c.w_throughput, c.w_switch, c.primal_step_size) for lbl, c in ofo_variants],
        )

    save_dir = (
        Path(__file__).resolve().parent
        / "outputs"
        / system
        / (output_dir or f"eval_multi_seed{seed_start}_n{n_scenarios}")
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    if scenario_library:
        test_scenarios = load_scenarios_from_library(
            scenario_library,
            n_scenarios=n_scenarios,
            training_trace=training_trace,
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
            i + 1,
            n_scenarios,
            scenario["seed"],
            params["pv_scale"],
            params["load_scale"],
        )
        if params["training_overlay"]:
            to = params["training_overlay"]
            logger.info(
                "  training: t=[%.0f, %.0f] n_gpus=%d",
                to["t_start"],
                to["t_end"],
                to["n_gpus"],
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

    rb_labels = ["rule_based"] if len(rule_step_sizes) == 1 else [f"rule_based_s{s:g}" for s in rule_step_sizes]
    mode_order = (
        ["baseline_no_tap"]
        + rb_labels
        + [f"ppo_{lbl}" for lbl in ppo_labels]
        + ["ofo"]
        + [f"ofo_{label}" for label, _ in ofo_variants]
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
                np.mean(viol),
                np.std(viol),
                np.mean(intg),
                np.std(intg),
                np.mean(vmin),
                np.mean(vmax),
                np.mean(bchg),
                np.std(bchg),
            )

    extra_cols = [
        "mean_throughput_toks_s",
        "peak_throughput_toks_s",
        "mean_latency_s",
        "p99_latency_s",
        "mean_power_kw",
        "peak_power_kw",
        "batch_changes",
        "itl_violation_rate",
    ]
    csv_path = save_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "seed",
                "pv_scale",
                "load_scale",
                "mode",
                "violation_time_s",
                "integral",
                "worst_vmin",
                "worst_vmax",
                *extra_cols,
            ]
        )
        for i, (results, params) in enumerate(zip(all_results, scenario_params, strict=False)):
            for mode, stats in results.items():
                sc = test_scenarios[i]
                writer.writerow(
                    [
                        i,
                        sc["seed"],
                        params["pv_scale"],
                        params["load_scale"],
                        mode,
                        stats["violation_time_s"],
                        stats["integral"],
                        stats["worst_vmin"],
                        stats["worst_vmax"],
                        *[stats.get(k, "") for k in extra_cols],
                    ]
                )
    logger.info("Results CSV: %s", csv_path)

    if not no_aggregate_plots:
        plot_aggregate(
            all_results, scenario_params, save_dir, modes, system=system, use_display_names=use_display_names
        )
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
        """Observation mode used during PPO training: full-voltage, per-zone-summary, per-bus-summary, or system-summary-only."""  # noqa: E501
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
        """When True AND sys defines `zones` AND there are >1 DC sites, each rule-based controller observes only buses in its own zone (decentralized credit assignment for ieee123)."""  # noqa: E501
        no_per_scenario_plots: bool = False
        """Skip per-scenario voltage and batch plots (saves disk + time)."""
        no_aggregate_plots: bool = False
        """Skip aggregate comparison and per-scenario integral plots."""
        randomize_ramps: bool = True
        """Synthesize per-episode inference ramps. Set --no-randomize-ramps for ieee34."""
        scenario_library: str = ""
        """Path to a pre-screened scenario library directory (from build_library.py)."""
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
