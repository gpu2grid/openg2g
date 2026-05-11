"""Plotting helpers for the RL pipeline.

Used by `build_library.py` (per-scenario voltage envelopes, batch traces,
acceptance summary across the library) and `evaluate.py` (per-scenario
controller comparison plots and aggregate roll-up across the test set).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scenarios import ScenarioRecord

from systems import V_MAX, V_MIN

logger = logging.getLogger(__name__)


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


def _voltage_envelope(grid_states, *, exclude_buses: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return (vmin_t, vmax_t) per step for plotting."""
    drop = {b.lower() for b in exclude_buses}
    vmin = np.full(len(grid_states), np.inf)
    vmax = np.full(len(grid_states), -np.inf)
    for i, gs in enumerate(grid_states):
        for bus in gs.voltages.buses():
            if bus.lower() in drop:
                continue
            pv = gs.voltages[bus]
            for v in (pv.a, pv.b, pv.c):
                if math.isnan(v):
                    continue
                if v < vmin[i]:
                    vmin[i] = v
                if v > vmax[i]:
                    vmax[i] = v
    return vmin, vmax


def _voltage_envelope_by_zone(
    grid_states,
    *,
    zones: dict[str, list[str]],
    exclude_buses: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {zone_name: (vmin_t, vmax_t)} per step, one array pair per zone."""
    drop = {b.lower() for b in exclude_buses}
    zone_sets = {z: {b.lower() for b in buses} for z, buses in zones.items()}
    n = len(grid_states)
    vmin = {z: np.full(n, np.inf) for z in zones}
    vmax = {z: np.full(n, -np.inf) for z in zones}
    for i, gs in enumerate(grid_states):
        for bus in gs.voltages.buses():
            bl = bus.lower()
            if bl in drop:
                continue
            pv = gs.voltages[bus]
            for z, bset in zone_sets.items():
                if bl not in bset:
                    continue
                for v in (pv.a, pv.b, pv.c):
                    if math.isnan(v):
                        continue
                    if v < vmin[z][i]:
                        vmin[z][i] = v
                    if v > vmax[z][i]:
                        vmax[z][i] = v
    return {z: (vmin[z], vmax[z]) for z in zones}


def _plot_batch_sizes(
    records: list[ScenarioRecord],
    batch_data: dict[int, dict],
    save_path: Path,
    *,
    max_rows: int = 40,
) -> None:
    """Plot batch size over time per accepted scenario, baseline vs OFO, one row per scenario.

    When the library has more than `max_rows` scenarios, only the first
    `max_rows` are shown. A single tall figure of hundreds of rows quickly
    exceeds matplotlib's 65535-pixel dimension limit, so we cap here.
    """
    n = len(records)
    if n == 0:
        return
    if n > max_rows:
        logger.info("_plot_batch_sizes: capping at first %d of %d records", max_rows, n)
        records = records[:max_rows]
        n = max_rows

    # Collect all (site_id, label) columns from the first scenario. For
    # single-DC feeders (ieee13) there's one site; multi-DC feeders
    # (ieee34) get one column per (site, model) pair.
    first_seed = records[0].seed
    ofo_by_site = batch_data[first_seed]["ofo"]
    cols_meta: list[tuple[str, str]] = []
    for site_id, sdata in ofo_by_site.items():
        for label in sdata["batch_by_model"]:
            cols_meta.append((site_id, label))
    n_cols = len(cols_meta)

    fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 2.5 * n), sharex=True, squeeze=False)

    for row, rec in enumerate(records):
        bd = batch_data[rec.seed]
        for col, (site_id, label) in enumerate(cols_meta):
            ax = axes[row][col]
            bl_site = bd["baseline"][site_id]
            ofo_site = bd["ofo"][site_id]
            ax.plot(
                bl_site["time_s"],
                bl_site["batch_by_model"][label],
                color="#888",
                linewidth=0.7,
                alpha=0.7,
                label="baseline",
            )
            ax.plot(
                ofo_site["time_s"],
                ofo_site["batch_by_model"][label],
                color="#2196F3",
                linewidth=0.7,
                alpha=0.9,
                label="OFO",
            )
            if row == 0:
                short = label.split("/")[-1] if "/" in label else label
                title = f"{site_id}:{short}" if len(ofo_by_site) > 1 else short
                ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"seed={rec.seed}\nBatch", fontsize=8)
            ax.grid(True, alpha=0.2)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")

    for col in range(n_cols):
        axes[-1][col].set_xlabel("Time (s)")
    fig.suptitle("Accepted scenarios: batch size (baseline vs OFO)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _extract_batch_data(log) -> dict:
    """Extract time_s and per-model batch sizes from a simulation log.

    Returns {site_id: {"time_s": [...], "batch_by_model": {label: [bs]}}}.
    Uses `log.dc_states_by_site` (per-site lists) so multi-DC feeders
    like ieee34 don't get interleaved timestamps or alternating zeros.
    """
    per_site: dict[str, dict] = {}
    for site_id, states in log.dc_states_by_site.items():
        time_s = [s.time_s for s in states]
        labels: list[str] = []
        if states:
            for m in states[0].batch_size_by_model:
                if m not in labels:
                    labels.append(m)
        batch_by_model = {m: [s.batch_size_by_model.get(m, 0) for s in states] for m in labels}
        per_site[site_id] = {"time_s": time_s, "batch_by_model": batch_by_model}
    return per_site


def _plot_envelopes(
    records: list[ScenarioRecord],
    envelopes: dict,
    save_path: Path,
    *,
    total_duration_s: int,
    zones: dict[str, list[str]] | None = None,
    max_rows: int = 40,
) -> None:
    """Plot voltage envelope per accepted scenario, baseline vs OFO.

    When `zones` is provided (multi-zone feeders like ieee123), each scenario
    gets one subplot per zone showing the per-zone vmin/vmax band. Otherwise a
    single subplot with the global envelope is used.

    Caps at `max_rows * 2` records (global mode) or `max_rows` records
    (per-zone mode) to stay under matplotlib's 65535-pixel dimension limit.
    """
    n = len(records)
    if n == 0:
        return

    t = np.arange(total_duration_s)

    if zones:
        zone_names = list(zones.keys())
        n_zones = len(zone_names)
        cap = max_rows
        if n > cap:
            logger.info("_plot_envelopes: capping at first %d of %d records", cap, n)
            records = records[:cap]
            n = cap
        zone_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
        fig, axes = plt.subplots(n, n_zones, figsize=(5 * n_zones, 3 * n), sharex=True, squeeze=False)
        for row, rec in enumerate(records):
            env = envelopes[rec.seed]
            for col, z in enumerate(zone_names):
                ax = axes[row][col]
                bl_z = env["baseline_zones"].get(z)
                of_z = env["ofo_zones"].get(z)
                color = zone_colors[col % len(zone_colors)]
                if bl_z is not None:
                    ax.fill_between(t, bl_z[0], bl_z[1], alpha=0.25, color="#888", label="baseline")
                if of_z is not None:
                    ax.fill_between(t, of_z[0], of_z[1], alpha=0.4, color=color, label="OFO")
                ax.axhline(V_MIN, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
                ax.axhline(V_MAX, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
                ax.grid(True, alpha=0.2)
                if row == 0:
                    ax.set_title(z, fontsize=10, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(
                        f"seed={rec.seed}\npv×{rec.pv_scale:.2f} ld×{rec.load_scale:.2f}\n"
                        f"bl={rec.baseline_integral:.1f} ofo={rec.ofo_integral:.1f} "
                        f"rec={rec.recovery_frac:.0%}",
                        fontsize=7,
                    )
                else:
                    ax.set_ylabel("V (pu)", fontsize=8)
                if row == 0 and col == 0:
                    ax.legend(loc="lower right", fontsize=7)
        for col in range(n_zones):
            axes[-1][col].set_xlabel("Time (s)", fontsize=8)
        fig.suptitle(
            "Accepted scenarios: per-zone voltage envelope (baseline vs OFO)",
            fontsize=13,
            fontweight="bold",
        )
    else:
        cap = max_rows * 2
        if n > cap:
            logger.info("_plot_envelopes: capping at first %d of %d records", cap, n)
            records = records[:cap]
            n = cap
        cols = 2
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3 * rows), sharex=True)
        axes = np.atleast_2d(axes)

        for idx, rec in enumerate(records):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            bmin, bmax = envelopes[rec.seed]["baseline"]
            omin, omax = envelopes[rec.seed]["ofo"]
            ax.fill_between(t, bmin, bmax, alpha=0.25, color="#888", label="baseline")
            ax.fill_between(t, omin, omax, alpha=0.4, color="#2196F3", label="OFO")
            ax.axhline(V_MIN, color="red", linestyle="--", linewidth=1, alpha=0.6)
            ax.axhline(V_MAX, color="red", linestyle="--", linewidth=1, alpha=0.6)
            ax.set_title(
                f"seed={rec.seed} pv×{rec.pv_scale:.2f} load×{rec.load_scale:.2f}\n"
                f"int: bl={rec.baseline_integral:.2f} ofo={rec.ofo_integral:.2f} "
                f"recov={rec.recovery_frac:.0%}",
                fontsize=9,
            )
            ax.set_ylabel("V (pu)", fontsize=9)
            ax.grid(True, alpha=0.2)
            if idx == 0:
                ax.legend(loc="lower right", fontsize=8)

        for k in range(n, rows * cols):
            r, c = divmod(k, cols)
            axes[r][c].axis("off")

        for c in range(cols):
            axes[-1][c].set_xlabel("Time (s)")
        fig.suptitle("Accepted scenarios: voltage envelope (baseline vs OFO)", fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(all_stats: list[dict], save_path: Path) -> None:
    """Bar chart of baseline vs OFO integral for every candidate (accepted + rejected)."""
    n = len(all_stats)
    if n == 0:
        return
    seeds = [s["seed"] for s in all_stats]
    bl = [s["baseline_integral"] for s in all_stats]
    of = [s["ofo_integral"] for s in all_stats]
    accepted = [s["accepted"] for s in all_stats]

    x = np.arange(n)
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * n), 5))
    ax.bar(x - w / 2, bl, w, color="#888", label="baseline integral")
    ax.bar(x + w / 2, of, w, color="#2196F3", label="OFO integral")
    for i, ok in enumerate(accepted):
        marker = "✓" if ok else "✗"
        color = "green" if ok else "red"
        ax.annotate(marker, xy=(i, max(bl[i], of[i])), ha="center", va="bottom", color=color, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds], rotation=45)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Integral voltage violation (pu·s)")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_title("Candidate scenarios: baseline vs OFO integral violation")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


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
        for xi, val in zip(x, vals, strict=False):
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

    colors = ["#999999", "#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4", "#795548", "#607D8B"]

    display_labels = [_display_name(m) if use_display_names else m for m in modes]

    metrics = [
        ("violation_time_s", "Mean Violation Time (s)"),
        ("integral", "Mean Integral Violation (pu·s)"),
        ("batch_changes", "Mean Batch Size Changes"),
        ("mean_throughput_toks_s", "Mean Throughput (tok/s)"),
        ("mean_power_kw", "Mean Data Center Power (kW)"),
        ("itl_violation_rate", "Mean ITL Violation Rate"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(max(15, len(modes) * 3.0), 10))
    x = np.arange(len(modes))

    for ax, (metric, title) in zip(axes.flat, metrics, strict=False):
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

    fig.suptitle(f"Aggregate Controller Metrics: {n_sc} Scenarios", fontsize=16, fontweight="bold")
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
        ax.bar(
            x + i * width,
            vals,
            width,
            label=_display_name(mode) if use_display_names else mode,
            color=colors[i % len(colors)],
            alpha=0.85,
        )

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
            ax.bar(
                x + i * width,
                norm_vals,
                width,
                label=_display_name(mode) if use_display_names else mode,
                color=colors[(modes.index(mode)) % len(colors)],
                alpha=0.85,
            )

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
        ax.plot(
            vals,
            cdf,
            color=colors[i % len(colors)],
            linewidth=2,
            label=_display_name(mode) if use_display_names else mode,
        )
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
    for _i, mode in enumerate(modes):
        integrals = [r[mode].get("integral", 0.0) for r in all_results if mode in r]
        tputs = [r[mode].get("mean_throughput_toks_s", 0.0) for r in all_results if mode in r]
        scatter_data[mode] = (integrals, tputs)

    n_modes = len(modes)
    ncols = 2
    nrows = math.ceil(n_modes / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), sharex=True, sharey=True, squeeze=False)

    for idx, mode in enumerate(modes):
        ax = axes[idx // ncols][idx % ncols]
        integrals, tputs = scatter_data[mode]
        label = _display_name(mode) if use_display_names else mode

        for other_mode, (oi, ot) in scatter_data.items():
            if other_mode != mode:
                ax.scatter(oi, ot, color="lightgrey", s=40, alpha=0.6, edgecolors="none", zorder=1)

        ax.scatter(
            integrals,
            tputs,
            color=colors[idx % len(colors)],
            s=80,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.5,
            zorder=2,
        )
        ax.scatter(
            np.mean(integrals),
            np.mean(tputs),
            color=colors[idx % len(colors)],
            s=220,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )

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
