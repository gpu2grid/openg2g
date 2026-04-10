"""OFO hyperparameter sweep.

Defines OFO baseline parameters and system configuration programmatically
for each IEEE test system, builds a sweep grid centred on those baseline
values, and runs OFO for each combination.

Sweep mode auto-selects based on the number of DC sites in the config:
  - 1 DC site  -> 1-D sweep: varies each parameter one-at-a-time.
  - 2+ DC sites -> 2-D sweep: sweeps all per-site parameter combinations
                   independently, producing heatmap visualizations.

Outputs (in outputs/<system>/sweep_ofo_parameters/):
  results_<system>_sweep_ofo_parameters.csv   -- one row per run
  <param>__<value>/                           -- per-run plots (1-D mode)
  <param>__<site1>_<v1>__<site2>_<v2>/        -- per-run plots (2-D mode)

Usage:
    # IEEE 13 (single DC, 1-D sweep)
    python sweep_ofo_parameters.py --system ieee13

    # IEEE 34 (two DCs, 2-D sweep)
    python sweep_ofo_parameters.py --system ieee34

    # IEEE 123 (four DCs, 2-D sweep)
    python sweep_ofo_parameters.py --system ieee123 --dt 60

    # Force 1-D sweep on multi-DC system (shared parameters)
    python sweep_ofo_parameters.py --system ieee34 --sweep-mode 1d

    # Override time resolution
    python sweep_ofo_parameters.py --system ieee34 --dt 60
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from openg2g.controller.ofo import (
    LogisticModelStore,
    OFOBatchSizeController,
    OFOConfig,
)
from openg2g.coordinator import Coordinator
from openg2g.datacenter.config import (
    DatacenterConfig,
    InferenceModelSpec,
    InferenceRamp,
    InferenceRampSchedule,
    ModelDeployment,
    PowerAugmentationConfig,
    TrainingRun,
)
from openg2g.datacenter.offline import OfflineDatacenter, OfflineWorkload
from openg2g.datacenter.workloads.inference import InferenceData, MLEnergySource
from openg2g.datacenter.workloads.training import TrainingTrace, TrainingTraceParams
from openg2g.grid.generator import SyntheticPV
from openg2g.grid.load import SyntheticLoad
from openg2g.grid.opendss import OpenDSSGrid
from openg2g.metrics.voltage import compute_allbus_voltage_stats

from plotting import (
    plot_allbus_voltages_per_phase,
    plot_model_timeseries_4panel,
)
from systems import SYSTEMS

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DT_DC = Fraction(1, 10)
DT_GRID = Fraction(1, 10)
DT_CTRL = Fraction(1)
V_MIN, V_MAX = 0.95, 1.05
TOTAL_DURATION_S = 3600
POWER_AUG = PowerAugmentationConfig(amplitude_scale_range=(0.98, 1.02), noise_fraction=0.005)

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
MODEL_SPECS = {s.model_label: s for s in ALL_MODEL_SPECS}


def deploy(label, num_replicas, initial_batch_size=128):
    return ModelDeployment(spec=MODEL_SPECS[label], num_replicas=num_replicas, initial_batch_size=initial_batch_size)


def load_data_sources(config_path=None):
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    sources_raw = cfg["data_sources"]
    data_sources = {s["model_label"]: MLEnergySource(**s) for s in sources_raw}
    ttp = TrainingTraceParams(**(cfg.get("training_trace_params") or {}))
    blob = json.dumps(
        (sorted(sources_raw, key=lambda s: s["model_label"]), cfg.get("training_trace_params") or {}), sort_keys=True
    ).encode()
    data_dir = _REPO_ROOT / "data" / "offline" / hashlib.sha256(blob).hexdigest()[:16]
    return data_sources, ttp, data_dir


@dataclass
class _DCSiteConfig:
    bus: str
    base_kw_per_phase: float
    total_gpu_capacity: int
    models: tuple[ModelDeployment, ...] = ()
    seed: int = 0
    connection_type: str = "wye"
    inference_ramps: InferenceRampSchedule | None = None
    load_shift_headroom: float = 0.0


logger = logging.getLogger("sweep_ofo")

# Default sweep multipliers (centred on baseline value)

DEFAULT_SWEEP_MULTIPLIERS: dict[str, list[float]] = {
    "primal_step_size": [0.2, 0.5, 1.0, 2.0, 5.0],
    "voltage_dual_step_size": [0.2, 0.5, 1.0, 2.0, 5.0],
    "latency_dual_step_size": [0.1, 1.0, 10.0, 50.0, 100.0],
    "w_throughput": [0.0, 0.1, 0.5, 1.0, 5.0, 10.0],
    "w_switch": [0.2, 0.5, 1.0, 2.0, 5.0],
}


def build_sweep_grid(
    baseline: OFOConfig,
    multipliers: dict[str, list[float]] | None = None,
) -> list[tuple[str, float, OFOConfig]]:
    """Build a one-at-a-time sweep grid from baseline OFO config.

    For each parameter, the baseline value is multiplied by each multiplier
    to produce the sweep values.  Multiplier 1.0 = baseline value.
    Special case: if baseline value is 0 (e.g. w_throughput=0), the multiplier
    is treated as an absolute value instead.
    """
    mults = multipliers or DEFAULT_SWEEP_MULTIPLIERS
    runs: list[tuple[str, float, OFOConfig]] = []
    seen: set[str] = set()

    for param_name, mult_list in mults.items():
        base_val = getattr(baseline, param_name)
        for mult in mult_list:
            if base_val == 0:
                value = mult  # absolute value when baseline is 0
            else:
                value = round(base_val * mult, 10)
            rid = f"{param_name}__{value:.6g}"
            if rid in seen:
                continue
            seen.add(rid)
            ofo_cfg = baseline.model_copy(update={param_name: value})
            runs.append((param_name, value, ofo_cfg))

    return runs


def _compute_sweep_values(
    baseline: OFOConfig,
    multipliers: dict[str, list[float]] | None = None,
) -> dict[str, list[float]]:
    """Compute absolute sweep values for each parameter from multipliers."""
    mults = multipliers or DEFAULT_SWEEP_MULTIPLIERS
    result: dict[str, list[float]] = {}
    for param_name, mult_list in mults.items():
        base_val = getattr(baseline, param_name)
        seen: set[float] = set()
        values: list[float] = []
        for mult in mult_list:
            if base_val == 0:
                value = mult
            else:
                value = round(base_val * mult, 10)
            if value not in seen:
                seen.add(value)
                values.append(value)
        result[param_name] = values
    return result


def build_sweep_grid_2d(
    baseline: OFOConfig,
    site_ids: list[str],
    multipliers: dict[str, list[float]] | None = None,
) -> list[tuple[str, dict[str, float], dict[str, OFOConfig]]]:
    """Build a 2-D sweep grid for multi-DC systems.

    For each parameter, creates all combinations of values across sites.
    Returns list of (param_name, {site_id: value}, {site_id: OFOConfig}).
    """
    param_values = _compute_sweep_values(baseline, multipliers)
    runs: list[tuple[str, dict[str, float], dict[str, OFOConfig]]] = []

    for param_name, values in param_values.items():
        for combo in itertools.product(values, repeat=len(site_ids)):
            site_values = dict(zip(site_ids, combo, strict=False))
            site_configs = {sid: baseline.model_copy(update={param_name: val}) for sid, val in site_values.items()}
            runs.append((param_name, site_values, site_configs))

    return runs


# Helpers


def _run_id(param: str, value: float) -> str:
    return f"{param}__{value:.6g}"


def _run_id_2d(param: str, site_values: dict[str, float]) -> str:
    parts = [param]
    for sid, val in site_values.items():
        parts.append(f"{sid}_{val:.6g}")
    return "__".join(parts)


def _collect_metrics(
    log,
    *,
    models: tuple[InferenceModelSpec, ...],
    wall_time_s: float,
    param_name: str,
    param_value: float,
    v_min: float,
    v_max: float,
    exclude_buses: tuple[str, ...] = (),
) -> dict:
    """Extract scalar metrics from a SimulationLog into a flat dict."""
    vstats = compute_allbus_voltage_stats(log.grid_states, v_min=v_min, v_max=v_max, exclude_buses=exclude_buses)

    row: dict = {
        "param_name": param_name,
        "param_value": param_value,
        "violation_time_s": vstats.violation_time_s,
        "integral_violation_pu_s": vstats.integral_violation_pu_s,
        "worst_vmin": vstats.worst_vmin,
        "worst_vmax": vstats.worst_vmax,
        "wall_time_s": wall_time_s,
    }

    # Per-site metrics (use per-site states to avoid interleaving artefacts)
    deadlines = {ms.model_label: ms.itl_deadline_s for ms in models}
    for site_id, site_states in log.dc_states_by_site.items():
        if not site_states:
            continue
        kW = np.array([(s.power_w.a + s.power_w.b + s.power_w.c) / 3e3 for s in site_states])
        row[f"avg_power_kw_per_phase__{site_id}"] = float(kW.mean())

        model_labels = sorted(site_states[0].batch_size_by_model.keys())
        for label in model_labels:
            batches = np.array([s.batch_size_by_model.get(label, 0) for s in site_states])
            row[f"avg_batch__{label}"] = float(np.mean(batches)) if batches.size else float("nan")
            row[f"n_batch_changes__{label}"] = int(np.sum(np.diff(batches) != 0)) if batches.size > 1 else 0

            itl = np.array([s.observed_itl_s_by_model.get(label, float("nan")) for s in site_states])
            deadline = deadlines.get(label, float("nan"))
            finite = itl[np.isfinite(itl)]
            row[f"itl_violation_frac__{label}"] = float(np.mean(finite > deadline)) if finite.size > 0 else float("nan")

            # Average throughput (tokens/s) = batch_size * active_replicas / itl
            wrep = np.array([s.active_replicas_by_model.get(label, 0) for s in site_states])
            if itl.size > 0 and wrep.size > 0 and batches.size > 0:
                with np.errstate(divide="ignore", invalid="ignore"):
                    throughput = np.where(itl > 0, batches.astype(float) * wrep.astype(float) / itl, np.nan)
                finite_tp = throughput[np.isfinite(throughput)]
                row[f"avg_throughput_tokens_per_s__{label}"] = (
                    float(np.mean(finite_tp)) if finite_tp.size > 0 else float("nan")
                )
            else:
                row[f"avg_throughput_tokens_per_s__{label}"] = float("nan")

    return row


def _plot_sweep_summary(df: pd.DataFrame, save_dir: Path, model_labels: list[str]) -> None:
    """Generate 4-panel summary plots for each swept parameter."""
    import matplotlib.pyplot as plt

    param_names = df["param_name"].unique()

    for param in param_names:
        sub = df[df["param_name"] == param].sort_values("param_value")
        x = sub["param_value"].values

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

        # (1) Violation time (left y) and integral violation (right y)
        ax1 = axes[0, 0]
        color1 = "tab:blue"
        ax1.plot(x, sub["violation_time_s"].values, "o-", color=color1, lw=2, ms=6)
        ax1.set_ylabel("Violation Time (s)", color=color1, fontsize=11)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_xlabel(param, fontsize=11)
        ax1.set_title("(a) Voltage Violation Metrics", fontsize=12)
        ax1.grid(True, alpha=0.3)

        ax1r = ax1.twinx()
        color2 = "tab:red"
        ax1r.plot(x, sub["integral_violation_pu_s"].values, "s--", color=color2, lw=2, ms=6)
        ax1r.set_ylabel("Integral Violation (pu*s)", color=color2, fontsize=11)
        ax1r.tick_params(axis="y", labelcolor=color2)

        # Use log scale for x if values span > 2 orders of magnitude
        if len(x) > 1 and x.min() > 0 and x.max() / x.min() > 100:
            ax1.set_xscale("log")

        # (2) Batch changes per model
        ax2 = axes[0, 1]
        for label in model_labels:
            col = f"n_batch_changes__{label}"
            if col in sub.columns:
                ax2.plot(x, sub[col].values, "o-", lw=1.5, ms=5, label=label)
        ax2.set_xlabel(param, fontsize=11)
        ax2.set_ylabel("Number of Batch Changes", fontsize=11)
        ax2.set_title("(b) Batch Size Changes", fontsize=12)
        ax2.legend(fontsize=8, loc="best")
        ax2.grid(True, alpha=0.3)
        if len(x) > 1 and x.min() > 0 and x.max() / x.min() > 100:
            ax2.set_xscale("log")

        # (3) ITL violation fraction per model
        ax3 = axes[1, 0]
        for label in model_labels:
            col = f"itl_violation_frac__{label}"
            if col in sub.columns:
                ax3.plot(x, sub[col].values, "o-", lw=1.5, ms=5, label=label)
        ax3.set_xlabel(param, fontsize=11)
        ax3.set_ylabel("ITL Violation Fraction", fontsize=11)
        ax3.set_title("(c) ITL Deadline Violation Fraction", fontsize=12)
        ax3.legend(fontsize=8, loc="best")
        ax3.grid(True, alpha=0.3)
        if len(x) > 1 and x.min() > 0 and x.max() / x.min() > 100:
            ax3.set_xscale("log")

        # (4) Average throughput per model
        ax4 = axes[1, 1]
        for label in model_labels:
            col = f"avg_throughput_tokens_per_s__{label}"
            if col in sub.columns:
                ax4.plot(x, sub[col].values, "o-", lw=1.5, ms=5, label=label)
        ax4.set_xlabel(param, fontsize=11)
        ax4.set_ylabel("Avg Throughput (tokens/s)", fontsize=11)
        ax4.set_title("(d) Average Throughput per Model", fontsize=12)
        tp_cols = [
            sub[f"avg_throughput_tokens_per_s__{ml}"]
            for ml in model_labels
            if f"avg_throughput_tokens_per_s__{ml}" in sub.columns
        ]
        if tp_cols and pd.concat(tp_cols, ignore_index=True).dropna().gt(0).any():
            ax4.set_yscale("log")
        ax4.legend(fontsize=8, loc="best")
        ax4.grid(True, alpha=0.3)
        if len(x) > 1 and x.min() > 0 and x.max() / x.min() > 100:
            ax4.set_xscale("log")

        fig.suptitle(f"OFO Sweep: {param}", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(save_dir / f"sweep_{param}.png", bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved sweep plot: sweep_%s.png", param)


def _build_pairwise_grid(
    sub: pd.DataFrame,
    col0: str,
    col1: str,
    metric_col: str,
    agg: str = "mean",
) -> tuple[list[float], list[float], np.ndarray]:
    """Build a 2-D grid for one metric, aggregating over other dimensions.

    Groups by (col0, col1), applies `agg` (e.g. "mean") to collapse any
    remaining site dimensions, and returns (vals0, vals1, grid[y, x]).
    """
    grouped = sub.groupby([col0, col1])[metric_col].agg(agg).reset_index()
    vals0 = sorted(grouped[col0].unique())
    vals1 = sorted(grouped[col1].unique())
    grid = np.full((len(vals1), len(vals0)), np.nan)
    for _, row in grouped.iterrows():
        i0 = vals0.index(row[col0])
        i1 = vals1.index(row[col1])
        grid[i1, i0] = row[metric_col]
    return vals0, vals1, grid


def _plot_sweep_summary_2d(
    df: pd.DataFrame,
    save_dir: Path,
    site_ids: list[str],
    model_labels: list[str],
) -> None:
    """Generate pairwise heatmap plots for per-site parameter sweeps.

    For each pair of sites (s_i, s_j), produces heatmaps with s_i on the
    x-axis and s_j on the y-axis.  When there are more than 2 sites, each
    heatmap cell is the **mean** over the remaining sites' parameter values.
    """
    import matplotlib.pyplot as plt

    param_names = df["param_name"].unique()

    # Generate all ordered pairs so that every pair appears once
    site_pairs = list(itertools.combinations(site_ids, 2))

    voltage_metrics = [
        ("violation_time_s", "Violation Time (s)"),
        ("integral_violation_pu_s", "Integral Violation (pu*s)"),
        ("worst_vmin", "Worst Vmin (pu)"),
        ("worst_vmax", "Worst Vmax (pu)"),
    ]

    for param in param_names:
        sub = df[df["param_name"] == param]

        for s0, s1 in site_pairs:
            col0 = f"param_value__{s0}"
            col1 = f"param_value__{s1}"
            if col0 not in sub.columns or col1 not in sub.columns:
                continue

            pair_tag = f"{s0}_vs_{s1}"
            agg_note = " (mean over other sites)" if len(site_ids) > 2 else ""

            # Voltage metrics heatmap
            fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)

            for ax, (metric_col, metric_label) in zip(axes.flat, voltage_metrics, strict=False):
                if metric_col not in sub.columns:
                    ax.set_visible(False)
                    continue
                vals0, vals1, hgrid = _build_pairwise_grid(sub, col0, col1, metric_col)
                if len(vals0) < 2 or len(vals1) < 2:
                    ax.set_visible(False)
                    continue

                im = ax.imshow(
                    hgrid,
                    origin="lower",
                    aspect="auto",
                    extent=[-0.5, len(vals0) - 0.5, -0.5, len(vals1) - 0.5],
                )
                ax.set_xticks(range(len(vals0)))
                ax.set_xticklabels([f"{v:.4g}" for v in vals0], fontsize=8, rotation=45)
                ax.set_yticks(range(len(vals1)))
                ax.set_yticklabels([f"{v:.4g}" for v in vals1], fontsize=8)
                ax.set_xlabel(f"{param} ({s0})", fontsize=10)
                ax.set_ylabel(f"{param} ({s1})", fontsize=10)
                ax.set_title(metric_label, fontsize=11)
                fig.colorbar(im, ax=ax, shrink=0.8)

                median = np.nanmedian(hgrid)
                for i1_idx in range(len(vals1)):
                    for i0_idx in range(len(vals0)):
                        val = hgrid[i1_idx, i0_idx]
                        if np.isfinite(val):
                            ax.text(
                                i0_idx,
                                i1_idx,
                                f"{val:.2g}",
                                ha="center",
                                va="center",
                                fontsize=7,
                                color="white" if val > median else "black",
                            )

            fig.suptitle(
                f"OFO Sweep: {param} ({s0} vs {s1}){agg_note}",
                fontsize=14,
                fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(save_dir / f"sweep_2d_{param}_{pair_tag}.png", bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved heatmap: sweep_2d_%s_%s.png", param, pair_tag)

            # Per-model heatmap
            model_metrics = []
            for label in model_labels:
                itl_col = f"itl_violation_frac__{label}"
                tp_col = f"avg_throughput_tokens_per_s__{label}"
                if itl_col in sub.columns:
                    model_metrics.append((itl_col, f"ITL Violation Frac ({label})"))
                if tp_col in sub.columns:
                    model_metrics.append((tp_col, f"Avg Throughput ({label})"))

            if not model_metrics:
                continue

            n_plots = len(model_metrics)
            ncols = min(n_plots, 2)
            nrows = math.ceil(n_plots / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), dpi=150, squeeze=False)

            for idx, (metric_col, metric_label) in enumerate(model_metrics):
                ax = axes[idx // ncols, idx % ncols]
                vals0, vals1, hgrid = _build_pairwise_grid(sub, col0, col1, metric_col)
                if len(vals0) < 2 or len(vals1) < 2:
                    ax.set_visible(False)
                    continue

                im = ax.imshow(
                    hgrid,
                    origin="lower",
                    aspect="auto",
                    extent=[-0.5, len(vals0) - 0.5, -0.5, len(vals1) - 0.5],
                )
                ax.set_xticks(range(len(vals0)))
                ax.set_xticklabels([f"{v:.4g}" for v in vals0], fontsize=8, rotation=45)
                ax.set_yticks(range(len(vals1)))
                ax.set_yticklabels([f"{v:.4g}" for v in vals1], fontsize=8)
                ax.set_xlabel(f"{param} ({s0})", fontsize=10)
                ax.set_ylabel(f"{param} ({s1})", fontsize=10)
                ax.set_title(metric_label, fontsize=11)
                fig.colorbar(im, ax=ax, shrink=0.8)

            for idx in range(n_plots, nrows * ncols):
                axes[idx // ncols, idx % ncols].set_visible(False)

            fig.suptitle(
                f"OFO Sweep (per-model): {param} ({s0} vs {s1}){agg_note}",
                fontsize=14,
                fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(save_dir / f"sweep_2d_{param}_{pair_tag}_models.png", bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved model heatmap: sweep_2d_%s_%s_models.png", param, pair_tag)


def _make_bus_color_map(buses: list[str]) -> dict[str, tuple[float, ...]]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    color_map = {}
    for i, bus in enumerate(sorted(buses, key=lambda b: b.lower())):
        color_map[bus] = cmap(i / max(len(buses) - 1, 1))
    return color_map


def _save_plots(
    log,
    *,
    run_dir: Path,
    site_models_map: dict[str, tuple[InferenceModelSpec, ...]],
    site_bus_map: dict[str, str],
    v_min: float,
    v_max: float,
    exclude_buses: tuple[str, ...] = (),
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    time_s = np.array(log.time_s)

    # Per-site OFO result plots (avoids interleaving artefacts)
    for site_id, site_states in log.dc_states_by_site.items():
        if not site_states:
            continue
        site_models = site_models_map.get(site_id)
        if not site_models:
            continue
        bus = site_bus_map.get(site_id, site_id)
        plot_model_timeseries_4panel(
            site_states,
            model_labels=[ms.model_label for ms in site_models],
            regime_shading=False,
            save_path=run_dir / f"OFO_results_{bus}.png",
        )

    # Build dynamic bus color map (consistent with run_ieee34_ofo.py)
    drop = {b.lower() for b in exclude_buses}
    all_buses = [b for b in log.grid_states[0].voltages.buses() if b.lower() not in drop]
    bus_colors = _make_bus_color_map(all_buses)

    plot_allbus_voltages_per_phase(
        log.grid_states,
        time_s,
        save_dir=run_dir,
        v_min=v_min,
        v_max=v_max,
        bus_color_map=bus_colors,
        drop_buses=exclude_buses,
        title_template="Voltage trajectories (Phase {label})",
    )


# Per-system experiment definitions


def _experiment_ieee13(
    sys: dict,
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
) -> dict[str, Any]:
    """IEEE 13-bus: single DC at bus 671 with training overlay."""
    models = (
        deploy("Llama-3.1-8B", 720),
        deploy("Llama-3.1-70B", 180),
        deploy("Llama-3.1-405B", 90),
        deploy("Qwen3-30B-A3B", 480),
        deploy("Qwen3-235B-A22B", 210),
    )

    # Ramp target=0.2 of initial replicas: 144, 36, 18, 96, 42
    inference_ramps = (
        InferenceRamp(target=144, model="Llama-3.1-8B").at(t_start=2500, t_end=3000)
        | InferenceRamp(target=36, model="Llama-3.1-70B").at(t_start=2500, t_end=3000)
        | InferenceRamp(target=18, model="Llama-3.1-405B").at(t_start=2500, t_end=3000)
        | InferenceRamp(target=96, model="Qwen3-30B-A3B").at(t_start=2500, t_end=3000)
        | InferenceRamp(target=42, model="Qwen3-235B-A22B").at(t_start=2500, t_end=3000)
    )

    dc_sites = {
        "default": _DCSiteConfig(
            bus="671",
            base_kw_per_phase=500.0,
            models=models,
            seed=0,
            total_gpu_capacity=7200,
            inference_ramps=inference_ramps,
        ),
    }

    training = TrainingRun(
        n_gpus=2400,
        trace=training_trace,
        target_peak_W_per_gpu=400.0,
    ).at(t_start=1000.0, t_end=2000.0)

    baseline_ofo = OFOConfig(
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

    pv_systems = [("675", SyntheticPV(peak_kw=10.0))]
    time_varying_loads = [("680", SyntheticLoad(peak_kw=10.0))]

    return dict(
        dc_sites=dc_sites,
        training=training,
        baseline_ofo=baseline_ofo,
        pv_systems=pv_systems,
        time_varying_loads=time_varying_loads,
    )


def _experiment_ieee34(
    sys: dict,
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
) -> dict[str, Any]:
    """IEEE 34-bus: two DC sites (upstream + downstream)."""
    upstream_models = (
        deploy("Llama-3.1-8B", 720),
        deploy("Llama-3.1-70B", 180),
        deploy("Llama-3.1-405B", 90),
    )
    downstream_models = (
        deploy("Qwen3-30B-A3B", 480),
        deploy("Qwen3-235B-A22B", 210),
    )

    dc_sites = {
        "upstream": _DCSiteConfig(
            bus="850",
            base_kw_per_phase=120.0,
            models=upstream_models,
            seed=0,
            total_gpu_capacity=520,
        ),
        "downstream": _DCSiteConfig(
            bus="834",
            base_kw_per_phase=80.0,
            models=downstream_models,
            seed=42,
            total_gpu_capacity=600,
        ),
    }

    baseline_ofo = OFOConfig(
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

    pv_systems = [
        ("848", SyntheticPV(peak_kw=130.0, site_idx=0)),
        ("830", SyntheticPV(peak_kw=65.0, site_idx=1)),
    ]
    time_varying_loads = [
        ("860", SyntheticLoad(peak_kw=80.0, site_idx=0)),
        ("844", SyntheticLoad(peak_kw=120.0, site_idx=1)),
        ("840", SyntheticLoad(peak_kw=60.0, site_idx=2)),
        ("858", SyntheticLoad(peak_kw=50.0, site_idx=3)),
        ("854", SyntheticLoad(peak_kw=40.0, site_idx=4)),
    ]

    return dict(
        dc_sites=dc_sites,
        training=None,
        baseline_ofo=baseline_ofo,
        pv_systems=pv_systems,
        time_varying_loads=time_varying_loads,
    )


def _experiment_ieee123(
    sys: dict,
    inference_data: InferenceData,
    training_trace: TrainingTrace,
    logistic_models: LogisticModelStore,
) -> dict[str, Any]:
    """IEEE 123-bus: four DC zones with per-site ramps."""
    dc_sites = {
        "z1_sw": _DCSiteConfig(
            bus="8",
            base_kw_per_phase=310.0,
            models=(deploy("Llama-3.1-8B", 120),),
            seed=0,
            total_gpu_capacity=120,
            inference_ramps=InferenceRamp(target=180, model="Llama-3.1-8B").at(t_start=500, t_end=1000),
        ),
        "z2_nw": _DCSiteConfig(
            bus="23",
            base_kw_per_phase=265.0,
            models=(deploy("Qwen3-30B-A3B", 80),),
            seed=17,
            total_gpu_capacity=160,
            inference_ramps=InferenceRamp(target=104, model="Qwen3-30B-A3B").at(t_start=1500, t_end=2500),
        ),
        "z3_se": _DCSiteConfig(
            bus="60",
            base_kw_per_phase=295.0,
            models=(deploy("Llama-3.1-70B", 30), deploy("Llama-3.1-405B", 35)),
            seed=34,
            total_gpu_capacity=400,
            inference_ramps=(
                InferenceRamp(target=45, model="Llama-3.1-70B").at(t_start=700, t_end=1100)
                | InferenceRamp(target=52, model="Llama-3.1-405B").at(t_start=700, t_end=1100)
            ),
        ),
        "z4_ne": _DCSiteConfig(
            bus="105",
            base_kw_per_phase=325.0,
            models=(deploy("Qwen3-235B-A22B", 55),),
            seed=51,
            total_gpu_capacity=440,
            inference_ramps=InferenceRamp(target=27, model="Qwen3-235B-A22B").at(t_start=2000, t_end=2500),
        ),
    }

    baseline_ofo = OFOConfig(
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

    pv_systems = [
        ("1", SyntheticPV(peak_kw=333.3, site_idx=0)),
        ("48", SyntheticPV(peak_kw=333.3, site_idx=1)),
        ("99", SyntheticPV(peak_kw=333.3, site_idx=2)),
    ]

    return dict(
        dc_sites=dc_sites,
        training=None,
        baseline_ofo=baseline_ofo,
        pv_systems=pv_systems,
        time_varying_loads=[],
    )


_EXPERIMENTS = {
    "ieee13": _experiment_ieee13,
    "ieee34": _experiment_ieee34,
    "ieee123": _experiment_ieee123,
}


# Shared setup


def _setup(
    *,
    system: str,
    dt_override: str | None = None,
    output_dir: str | None = None,
):
    """Build experiment config, load shared data, build datacenters and grid.

    Returns a namespace-like dict with all objects needed by sweep runners.
    """
    sys = SYSTEMS[system]()
    experiment_fn = _EXPERIMENTS[system]

    dt_dc = DT_DC
    dt_grid = DT_GRID
    dt_ctrl = DT_CTRL

    if dt_override is not None:
        if "/" in dt_override:
            num, den = dt_override.split("/", 1)
            frac = Fraction(int(num), int(den))
        else:
            frac = Fraction(int(dt_override))
        dt_dc = frac
        dt_grid = frac
        dt_ctrl = frac

    if output_dir is not None:
        save_dir = Path(output_dir).resolve()
    else:
        save_dir = Path(__file__).resolve().parent / "outputs" / system / "sweep_ofo_parameters"
    # Don't mkdir yet -- main() may append _1d/_2d suffix before creating

    # Load shared data (done once)

    data_sources, training_trace_params, data_dir = load_data_sources()

    all_models = ALL_MODEL_SPECS

    logger.info("Loading inference data...")
    inference_data = InferenceData.ensure(
        data_dir,
        all_models,
        data_sources,
        plot=False,
        dt_s=float(dt_dc),
    )

    logger.info("Loading training trace...")
    training_trace = TrainingTrace.ensure(data_dir / "training_trace.csv", training_trace_params)

    logger.info("Loading logistic fits...")
    logistic_models = LogisticModelStore.ensure(
        data_dir / "logistic_fits.csv",
        all_models,
        data_sources,
        plot=False,
    )

    # Build experiment config

    experiment = experiment_fn(sys, inference_data, training_trace, logistic_models)
    dc_sites: dict[str, _DCSiteConfig] = experiment["dc_sites"]
    baseline_ofo: OFOConfig = experiment["baseline_ofo"]
    training: TrainingRun | None = experiment.get("training")
    pv_systems: list = experiment.get("pv_systems", [])
    time_varying_loads: list = experiment.get("time_varying_loads", [])

    logger.info("Baseline OFO config: %s", baseline_ofo)

    # Build shared datacenters and grid

    site_ids = list(dc_sites.keys())
    datacenters: dict[str, OfflineDatacenter] = {}
    site_models_map: dict[str, tuple[InferenceModelSpec, ...]] = {}

    for site_id, site in dc_sites.items():
        site_specs = tuple(md.spec for md in site.models)
        site_models_map[site_id] = site_specs
        site_inference = inference_data.filter_models(site_specs)
        replica_counts = {md.spec.model_label: md.num_replicas for md in site.models}
        batch_sizes = {md.spec.model_label: md.initial_batch_size for md in site.models}

        dc_config = DatacenterConfig(gpus_per_server=8, base_kw_per_phase=site.base_kw_per_phase)

        workload_kwargs: dict = {
            "inference_data": site_inference,
            "replica_counts": replica_counts,
            "initial_batch_sizes": batch_sizes,
        }
        if training is not None:
            workload_kwargs["training"] = training
        if site.inference_ramps is not None:
            workload_kwargs["inference_ramps"] = site.inference_ramps
        workload = OfflineWorkload(**workload_kwargs)

        dc = OfflineDatacenter(
            dc_config,
            workload,
            name=site_id,
            dt_s=dt_dc,
            seed=site.seed,
            power_augmentation=POWER_AUG,
            total_gpu_capacity=site.total_gpu_capacity,
            load_shift_headroom=site.load_shift_headroom,
        )
        datacenters[site_id] = dc

    site_bus_map = {sid: dc_sites[sid].bus for sid in dc_sites}
    exclude_buses = tuple(sys["exclude_buses"])

    grid = OpenDSSGrid(
        dss_case_dir=sys["dss_case_dir"],
        dss_master_file=sys["dss_master_file"],
        source_pu=sys["source_pu"],
        dt_s=dt_grid,
        initial_tap_position=sys["initial_taps"],
        exclude_buses=exclude_buses,
    )
    pf = DatacenterConfig(base_kw_per_phase=0).power_factor
    for site_id, site in dc_sites.items():
        grid.attach_dc(datacenters[site_id], bus=site.bus, connection_type=site.connection_type, power_factor=pf)
    for bus, gen in pv_systems:
        grid.attach_generator(gen, bus=bus)
    for bus, ld in time_varying_loads:
        grid.attach_load(ld, bus=bus)

    return dict(
        all_models=all_models,
        baseline_ofo=baseline_ofo,
        logistic_models=logistic_models,
        site_ids=site_ids,
        site_models_map=site_models_map,
        site_bus_map=site_bus_map,
        datacenters=datacenters,
        grid=grid,
        exclude_buses=exclude_buses,
        save_dir=save_dir,
        dt_ctrl=dt_ctrl,
        system=system,
    )


# 1-D sweep (single DC site)


def _run_single_sim(
    *,
    site_ids: list[str],
    site_ofo_cfgs: dict[str, OFOConfig],
    logistic_models,
    site_models_map,
    site_bus_map,
    datacenters,
    grid,
    exclude_buses,
    dt_ctrl,
    all_models,
    run_dir: Path,
    total_duration_s: int = TOTAL_DURATION_S,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
):
    """Run one simulation with per-site OFO configs. Returns (log, wall_time_s)."""
    controllers = []
    for site_id in site_ids:
        site_models = site_models_map[site_id]
        ofo_cfg = site_ofo_cfgs[site_id]
        ofo_ctrl = OFOBatchSizeController(
            site_models,
            datacenter=datacenters[site_id],
            models=logistic_models,
            config=ofo_cfg,
            dt_s=dt_ctrl,
        )
        controllers.append(ofo_ctrl)

    coord = Coordinator(
        datacenters=list(datacenters.values()),
        grid=grid,
        controllers=controllers,
        total_duration_s=total_duration_s,
    )

    t0 = time.monotonic()
    log = coord.run()
    wall_time_s = time.monotonic() - t0

    _save_plots(
        log,
        run_dir=run_dir,
        site_models_map=site_models_map,
        site_bus_map=site_bus_map,
        v_min=v_min,
        v_max=v_max,
        exclude_buses=exclude_buses,
    )

    return log, wall_time_s


def _run_sweep_1d(ctx: dict) -> None:
    """One-at-a-time sweep: same OFO config for all sites."""
    runs = build_sweep_grid(ctx["baseline_ofo"])
    total = len(runs)
    save_dir = ctx["save_dir"]
    all_models = ctx["all_models"]
    system = ctx["system"]

    logger.info("1-D sweep: %d runs across %d parameters", total, len(DEFAULT_SWEEP_MULTIPLIERS))

    rows: list[dict] = []
    for i, (param_name, param_value, ofo_cfg) in enumerate(runs, start=1):
        rid = _run_id(param_name, param_value)
        run_dir = save_dir / rid
        logger.info("[%d/%d] %s = %s", i, total, param_name, param_value)

        try:
            # All sites use the same OFO config in 1-D mode
            site_ofo_cfgs = {sid: ofo_cfg for sid in ctx["site_ids"]}
            log, wall_time_s = _run_single_sim(
                site_ids=ctx["site_ids"],
                site_ofo_cfgs=site_ofo_cfgs,
                logistic_models=ctx["logistic_models"],
                site_models_map=ctx["site_models_map"],
                site_bus_map=ctx["site_bus_map"],
                datacenters=ctx["datacenters"],
                grid=ctx["grid"],
                exclude_buses=ctx["exclude_buses"],
                dt_ctrl=ctx["dt_ctrl"],
                all_models=all_models,
                run_dir=run_dir,
            )

            row = _collect_metrics(
                log,
                models=all_models,
                wall_time_s=wall_time_s,
                param_name=param_name,
                param_value=param_value,
                v_min=V_MIN,
                v_max=V_MAX,
                exclude_buses=ctx["exclude_buses"],
            )
            rows.append(row)
            logger.info(
                "  -> violation_time=%.1fs  integral_viol=%.4f pu*s  worst_vmin=%.4f  wall=%.1fs",
                row["violation_time_s"],
                row["integral_violation_pu_s"],
                row["worst_vmin"],
                row["wall_time_s"],
            )
        except Exception:
            logger.exception("Run %s failed; skipping.", rid)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = save_dir / f"results_{system}_sweep_ofo_parameters.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved %d rows to %s", len(df), csv_path)
        print(df.to_string(index=False))

        model_labels = [m.model_label for m in all_models]
        _plot_sweep_summary(df, save_dir, model_labels)
    else:
        logger.warning("No successful runs; no CSV written.")

    logger.info("All outputs in: %s", save_dir)


# 2-D sweep (multiple DC sites)


def _run_sweep_2d(ctx: dict) -> None:
    """Per-site parameter sweep: independent values for each site."""
    site_ids = ctx["site_ids"]
    runs = build_sweep_grid_2d(ctx["baseline_ofo"], site_ids)
    total = len(runs)
    save_dir = ctx["save_dir"]
    all_models = ctx["all_models"]
    system = ctx["system"]

    logger.info(
        "2-D sweep: %d runs across %d parameters, %d sites (%s)",
        total,
        len(DEFAULT_SWEEP_MULTIPLIERS),
        len(site_ids),
        ", ".join(site_ids),
    )

    rows: list[dict] = []
    for i, (param_name, site_values, site_ofo_cfgs) in enumerate(runs, start=1):
        rid = _run_id_2d(param_name, site_values)
        run_dir = save_dir / rid
        vals_str = ", ".join(f"{sid}={v:.6g}" for sid, v in site_values.items())
        logger.info("[%d/%d] %s: %s", i, total, param_name, vals_str)

        try:
            log, wall_time_s = _run_single_sim(
                site_ids=site_ids,
                site_ofo_cfgs=site_ofo_cfgs,
                logistic_models=ctx["logistic_models"],
                site_models_map=ctx["site_models_map"],
                site_bus_map=ctx["site_bus_map"],
                datacenters=ctx["datacenters"],
                grid=ctx["grid"],
                exclude_buses=ctx["exclude_buses"],
                dt_ctrl=ctx["dt_ctrl"],
                all_models=all_models,
                run_dir=run_dir,
            )

            row = _collect_metrics(
                log,
                models=all_models,
                wall_time_s=wall_time_s,
                param_name=param_name,
                param_value=0.0,  # placeholder
                v_min=V_MIN,
                v_max=V_MAX,
                exclude_buses=ctx["exclude_buses"],
            )
            # Add per-site parameter values
            for sid, val in site_values.items():
                row[f"param_value__{sid}"] = val
            # Remove the flat param_value (not meaningful for 2-D)
            row.pop("param_value", None)
            rows.append(row)

            logger.info(
                "  -> violation_time=%.1fs  integral_viol=%.4f pu*s  worst_vmin=%.4f  wall=%.1fs",
                row["violation_time_s"],
                row["integral_violation_pu_s"],
                row["worst_vmin"],
                row["wall_time_s"],
            )
        except Exception:
            logger.exception("Run %s failed; skipping.", rid)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = save_dir / f"results_{system}_sweep_ofo_parameters_2d.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved %d rows to %s", len(df), csv_path)
        print(df.to_string(index=False))

        model_labels = [m.model_label for m in all_models]
        _plot_sweep_summary_2d(df, save_dir, site_ids, model_labels)
    else:
        logger.warning("No successful runs; no CSV written.")

    logger.info("All outputs in: %s", save_dir)


# Main


def main(
    *,
    system: str,
    sweep_mode: Literal["auto", "1d", "2d"] = "auto",
    dt_override: str | None = None,
    output_dir: str | None = None,
) -> None:
    ctx = _setup(
        system=system,
        dt_override=dt_override,
        output_dir=output_dir,
    )

    n_sites = len(ctx["site_ids"])

    if sweep_mode == "auto":
        use_2d = n_sites >= 2
    elif sweep_mode == "2d":
        use_2d = True
    else:
        use_2d = False

    if use_2d and n_sites < 2:
        logger.warning("2-D sweep requested but only %d DC site(s); falling back to 1-D.", n_sites)
        use_2d = False

    # Append _1d or _2d to output directory if using default path
    if output_dir is None:
        suffix = "_2d" if use_2d else "_1d"
        ctx["save_dir"] = ctx["save_dir"].parent / (ctx["save_dir"].name + suffix)
    ctx["save_dir"].mkdir(parents=True, exist_ok=True)

    if use_2d:
        logger.info("%d DC sites, sweep mode: 2-D (independent parameters per site)", n_sites)
        _run_sweep_2d(ctx)
    else:
        logger.info("%d DC site(s), sweep mode: 1-D (shared parameters across sites)", n_sites)
        _run_sweep_1d(ctx)


if __name__ == "__main__":
    from dataclasses import dataclass

    import tyro

    @dataclass
    class Args:
        system: str = "ieee13"
        """System name (e.g. ieee13, ieee34, ieee123). Used for output directory and CSV naming."""
        sweep_mode: Literal["auto", "1d", "2d"] = "auto"
        """Sweep mode: 'auto' selects based on number of DC sites (1 site -> 1-D, 2+ sites -> 2-D).
        '1d' forces one-at-a-time sweep with shared parameters across all sites.
        '2d' forces independent per-site parameter sweep (requires 2+ DC sites)."""
        dt: str | None = None
        """Override simulation time step (e.g. '60' for 60s resolution)."""
        output_dir: str | None = None
        """Override output directory (default: outputs/<system>/sweep_ofo_parameters/)."""
        log_level: str = "INFO"
        """Logging verbosity (DEBUG, INFO, WARNING)."""

    args = tyro.cli(Args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    main(
        system=args.system,
        sweep_mode=args.sweep_mode,
        dt_override=args.dt,
        output_dir=args.output_dir,
    )
