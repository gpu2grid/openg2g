"""LLM load shifting: compare OFO with and without cross-site replica shifting.

Runs two OFO simulations — one with load shifting disabled and one enabled —
then produces comparison plots showing voltage improvement and replica movement.

The config must have a ``load_shift`` section with ``enabled: true`` and
each DC site must run at least 3 models (warm-start requirement).

Usage:
    python analyze_LLM_load_shifting.py --config config_ieee123_load_shift.json --system ieee123
"""

from __future__ import annotations

import copy
import json
import logging
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sweep_dc_locations import (
    SweepConfig,
    ScenarioOpenDSSGrid,
    _parse_fraction,
    _resolve_models_for_site,
    _taps_dict_to_position,
)
from run_ofo import run_mode, _build_tap_schedule
from openg2g.datacenter.command import ShiftReplicas
from openg2g.datacenter.config import InferenceModelSpec
from openg2g.datacenter.workloads.inference import InferenceData
from openg2g.controller.ofo import LogisticModelStore
from openg2g.metrics.voltage import VoltageStats

logger = logging.getLogger("load_shift_comparison")


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
    n_steps = len(time_s)

    # Initialize offset tracking
    offsets: dict[str, dict[str, int]] = {
        sid: {m: 0 for m in models} for sid, models in models_by_site.items()
    }

    # Collect all ShiftReplicas commands with their approximate time
    shift_events: list[tuple[float, str, str, int]] = []  # (time, site, model, delta)
    cmd_time = 0.0
    for cmd in log.commands:
        if isinstance(cmd, ShiftReplicas):
            shift_events.append((cmd_time, cmd.target_site_id or "", cmd.model_label, cmd.replica_delta))
        # Approximate time from command order (dt_ctrl spacing)
        cmd_time += dt_ctrl_s / 2  # rough estimate

    # Build timeseries by replaying shifts
    result: dict[str, dict[str, list[int]]] = {
        sid: {m: [] for m in models} for sid, models in models_by_site.items()
    }

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
        ax.set_title(f"{label}  |  Viol={stats.violation_time_s:.0f}s, Integral={stats.integral_violation_pu_s:.2f} pu·s",
                      fontsize=13)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(0.93, 1.06)
        ax.grid(True, alpha=0.2)

    axes[1].set_xlabel("Time (min)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_dir / "voltage_comparison.png", dpi=150, bbox_inches="tight")
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

                ax.plot(t[::step] / 60, delta[::step], color=model_color[m],
                        linewidth=1.8, label=model_short[m])

            ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
            ax.set_ylabel("Net Shift\n(replicas)", fontsize=10)
            ax.set_title(f"Site {sid}", fontsize=12)
            if any_nonzero:
                ax.legend(loc="upper right", fontsize=8, ncol=len(site_models))
            ax.grid(True, alpha=0.2)

        axes[-1].set_xlabel("Time (min)", fontsize=12)
        fig.suptitle("Net Replica Shift per Site (with shift − without shift)", fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(save_dir / "net_replica_shift.png", dpi=150, bbox_inches="tight")
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
        ax.plot(t_no[::step_no] / 60, p_no[::step_no],
                color="steelblue", linewidth=1.5, alpha=0.8, label="OFO (no shift)")
    if len(t_ws) > 0:
        ax.plot(t_ws[::step_ws] / 60, p_ws[::step_ws],
                color="coral", linewidth=1.5, alpha=0.9, label="OFO + Load Shift")

    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Total DC Load (kW per phase)", fontsize=12)
    ax.set_title("Aggregate DC Load: With vs Without Load Shifting", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_dir / "dc_load_comparison.png", dpi=150, bbox_inches="tight")
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
    fig.savefig(save_dir / "dc_load_per_site.png", dpi=150, bbox_inches="tight")
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
    fig.savefig(save_dir / "summary_bar_chart.png", dpi=150, bbox_inches="tight")
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
    print(f"{'Violation time (s)':<30s} {vt1:>15.1f} {vt2:>15.1f} {(1-vt2/vt1)*100 if vt1 else 0:>14.0f}%")
    print(f"{'Integral violation (pu·s)':<30s} {iv1:>15.2f} {iv2:>15.2f} {(1-iv2/iv1)*100 if iv1 else 0:>14.0f}%")
    print(f"{'Worst Vmin (pu)':<30s} {stats_no_shift.worst_vmin:>15.4f} {stats_with_shift.worst_vmin:>15.4f}")
    print(f"{'Worst Vmax (pu)':<30s} {stats_no_shift.worst_vmax:>15.4f} {stats_with_shift.worst_vmax:>15.4f}")
    print(f"{'Load shift events':<30s} {'—':>15s} {n_shifts:>15d}")
    print("-" * 80)
    print(f"Outputs: {save_dir}")


def main(*, config_path: Path, system: str = "ieee123") -> None:
    config_path = config_path.resolve()
    config = SweepConfig.model_validate_json(config_path.read_bytes())
    sim = config.simulation

    config_dir = config_path.parent
    config.ieee_case_dir = (config_dir / config.ieee_case_dir).resolve()

    all_models = tuple(config.models)
    data_sources = {s.model_label: s for s in config.data_sources}
    data_dir = config.data_dir or Path("data/offline") / config.data_hash

    dt_dc = _parse_fraction(sim.dt_dc)
    dt_grid = _parse_fraction(sim.dt_grid)
    dt_ctrl = _parse_fraction(sim.dt_ctrl)

    logger.info("Loading inference data...")
    inference_data = InferenceData.ensure(
        data_dir, all_models, data_sources,
        mlenergy_data_dir=config.mlenergy_data_dir, plot=False, dt_s=float(dt_dc),
    )
    logistic_models = LogisticModelStore.ensure(
        data_dir / "logistic_fits.csv", all_models, data_sources,
        mlenergy_data_dir=config.mlenergy_data_dir, plot=False,
    )

    save_dir = Path(__file__).resolve().parent / "outputs" / system / "load_shift_comparison"
    save_dir.mkdir(parents=True, exist_ok=True)

    from openg2g.datacenter.workloads.training import TrainingTrace
    training_trace = TrainingTrace.ensure(
        data_dir / "training_trace.csv", config.training_trace_params,
    )

    shared = dict(
        config=config,
        all_models=all_models,
        inference_data=inference_data,
        training_trace=training_trace,
        logistic_models=logistic_models,
        dt_dc=dt_dc, dt_grid=dt_grid, dt_ctrl=dt_ctrl,
        save_dir=save_dir,
    )

    # --- Run 1: OFO without load shift ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("RUN 1: OFO without load shifting")
    logger.info("=" * 70)
    # Temporarily disable load_shift
    orig_ls = config.load_shift
    config.load_shift = {"enabled": False}
    stats_no_shift, log_no_shift = run_mode("ofo", **shared, folder_name="ofo_no_shift")

    # --- Run 2: OFO with load shift ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("RUN 2: OFO with load shifting")
    logger.info("=" * 70)
    config.load_shift = orig_ls if orig_ls else {"enabled": True, "gpus_per_shift": 8, "headroom": 0.3}
    stats_with_shift, log_with_shift = run_mode("ofo", **shared, folder_name="ofo_with_shift")

    # --- Plots ---
    site_ids = list(config.dc_sites.keys())
    models_by_site = {sid: list(scfg.models or []) for sid, scfg in config.dc_sites.items()}
    gpus_per_replica = {m.model_label: m.gpus_per_replica for m in all_models}

    plot_load_shift_comparison(
        log_no_shift, log_with_shift,
        stats_no_shift, stats_with_shift,
        site_ids, models_by_site, gpus_per_replica,
        save_dir,
        exclude_buses=tuple(config.exclude_buses),
    )


if __name__ == "__main__":
    import tyro
    from dataclasses import dataclass

    @dataclass
    class Args:
        config: str = "config_ieee123_load_shift.json"
        """Path to config JSON with load_shift section."""
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

    main(config_path=Path(args.config), system=args.system)
