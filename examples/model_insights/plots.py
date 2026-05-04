"""Cross-variant plots for the model-insights experiments.

Each panel saves as its own PDF, SVG, and PNG; the paper composes subfloats
from these. Style comes from the local `paper_plots.mplstyle`. Legends are
color-only and horizontal; display labels drop precision / parallelism
decorations — those live in captions.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tyro
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.use("Agg")
import common as mi
import matplotlib.pyplot as plt

_MPLSTYLE_PATH = Path(__file__).resolve().parent / "paper_plots.mplstyle"
if not _MPLSTYLE_PATH.exists():
    raise FileNotFoundError(
        f"Paper mplstyle not found at {_MPLSTYLE_PATH}. "
        "This file is the paper-wide plot style and is load-bearing; "
        "do not regenerate figures without it."
    )
plt.style.use(str(_MPLSTYLE_PATH))

logger = logging.getLogger("model_insights.plots")

OUTDIR = Path(__file__).resolve().parent / "outputs"

OFO_MODE = "ofo-no-tap"
BASELINE_MODE = "baseline-no-tap"

COLORS = {
    "baseline": "#9A9A9A",
    "ofo": "#4C72B0",
    "h100": "#4C72B0",
    "b200": "#C44E52",
    "hardware": "#C44E52",
}

DISPLAY_LABELS = {
    "Qwen3-8B": "Qwen 3 8B",
    "Qwen3-30B-A3B-Instruct-1GPU": "Qwen 3 30B A3B",
    "Qwen3-30B-A3B-Instruct-2GPU": "Qwen 3 30B A3B",
    "Llama-3.1-70B": "Llama 3.1 70B",
    "Llama-3.1-405B": "Llama 3.1 405B",
    "GPT-OSS-120B-H100-2GPU": "GPT-OSS 120B",
    "GPT-OSS-120B-B200-1GPU": "GPT-OSS 120B",
    "GPT-OSS-120B-B200-2GPU": "GPT-OSS 120B",
    "Qwen3-8B-B200": "Qwen 3 8B",
    "Qwen3-32B-1GPU-H100": "Qwen 3 32B",
    "Qwen3-32B-1GPU-B200": "Qwen 3 32B",
    "Qwen3-30B-A3B-Instruct-1GPU-B200": "Qwen 3 30B A3B",
    "Qwen3-235B-A22B-Instruct-FP8-8GPU": "Qwen 3 235B A22B",
    "Qwen3-235B-A22B-Instruct-8GPU": "Qwen 3 235B A22B",
    "Qwen3-235B-A22B-Instruct-8GPU-B200": "Qwen 3 235B A22B",
    "Qwen3-235B-A22B-Thinking-8GPU": "Qwen 3 235B A22B Thinking",
    "Qwen3-235B-A22B-Thinking-4GPU-B200": "Qwen 3 235B A22B Thinking",
    "Qwen3-235B-A22B-Thinking-8GPU-B200": "Qwen 3 235B A22B Thinking",
    "Qwen3-235B-A22B-Thinking-FP8-8GPU": "Qwen 3 235B A22B Thinking",
    "Qwen3-235B-A22B-Thinking-FP8-4GPU-B200": "Qwen 3 235B A22B Thinking",
    "qwen-8b-1gpu": "Qwen 3 8B",
    "qwen-32b-1gpu": "Qwen 3 32B",
    "qwen-30b-a3b-instruct-1gpu": "Qwen 3 30B A3B",
    "qwen-30b-a3b-instruct": "Qwen 3 30B A3B",
    "qwen-235b-a22b-thinking": "Qwen 3 235B A22B Thinking",
    "gpt-oss-120b": "GPT-OSS 120B",
}


def _pretty(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)


def _line_handle(color: str, label: str) -> Line2D:
    """Legend handle for a line-marker plot (Pareto curves)."""
    return Line2D(
        [0],
        [0],
        color=color,
        marker="o",
        markersize=5.5,
        markeredgecolor="black",
        markeredgewidth=0.4,
        lw=1.6,
        label=label,
    )


def _combo_handle(color: str, label: str):
    """Composite legend handle — a bar swatch + a line marker, both in
    `color`, rendered side-by-side under one label. Use with
    `HandlerTuple` so one legend entry covers both the (a) bars and the
    (b) Pareto panels at once.
    """
    bar = Patch(facecolor=color, edgecolor="white")
    line = Line2D(
        [0], [0], color=color, marker="o", markersize=5.5, markeredgecolor="black", markeredgewidth=0.4, lw=1.6
    )
    return ((bar, line), label)


def _save_legend(handles, stem: Path, *, ncol: int, width: float, tight: bool = True) -> None:
    """Save a standalone legend PDF.

    Handles may be plain artists OR `(tuple_of_artists, label)` pairs
    (produced by `_combo_handle`) that render as side-by-side glyphs
    under one label via `HandlerTuple`. When `tight=False` the full
    figsize is preserved (useful for padding a narrow legend so it
    renders at larger physical font in LaTeX).
    """
    if not handles:
        return
    from matplotlib.legend_handler import HandlerTuple

    plain_handles, tuple_handles, tuple_labels = [], [], []
    for h in handles:
        if isinstance(h, tuple) and len(h) == 2 and isinstance(h[0], tuple):
            tuple_handles.append(h[0])
            tuple_labels.append(h[1])
        else:
            plain_handles.append(h)

    fig, ax = plt.subplots(figsize=(width, 0.35))
    ax.axis("off")
    legend_fontsize = 11
    if tuple_handles:
        ax.legend(
            handles=tuple_handles + plain_handles,
            labels=tuple_labels + [h.get_label() for h in plain_handles],
            ncol=ncol,
            loc="center",
            frameon=False,
            fontsize=legend_fontsize,
            handlelength=2.6,
            columnspacing=1.2,
            handletextpad=0.5,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)},
        )
    else:
        ax.legend(
            handles=plain_handles,
            ncol=ncol,
            loc="center",
            frameon=False,
            fontsize=legend_fontsize,
            handlelength=1.2,
            columnspacing=1.2,
            handletextpad=0.4,
        )
    stem.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        kw = {"bbox_inches": "tight", "pad_inches": 0.02}
    else:
        kw = {"bbox_inches": None, "pad_inches": 0.0}
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "Creator": None, "Producer": None}, **kw)
    fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None}, **kw)
    fig.savefig(stem.with_suffix(".png"), dpi=200, metadata={"Creation Time": None}, **kw)
    plt.close(fig)


def _grouped_bars(
    ax,
    base_vals,
    ofo_vals,
    colors,
    *,
    base_errs=None,
    ofo_errs=None,
    ylog: bool = True,
    ylabel: str = "Integral violation\n(pu·s)",
    bar_width: float = 0.38,
) -> None:
    """Bars grouped by variant. Each variant has two bars: hatched
    (uncoordinated) and solid (coordinated), both colored by variant.
    No x tick labels — identity comes from the shared legend + caption."""
    n = len(colors)
    xs = np.arange(n)
    offset = bar_width / 2 + 0.012
    for i, color in enumerate(colors):
        be = base_errs[i] if base_errs is not None else None
        oe = ofo_errs[i] if ofo_errs is not None else None
        ax.bar(
            xs[i] - offset,
            base_vals[i],
            bar_width,
            color=color,
            hatch="///",
            edgecolor="white",
            linewidth=0.6,
            yerr=be,
            capsize=1.8,
            error_kw={"lw": 0.7},
        )
        ax.bar(
            xs[i] + offset,
            ofo_vals[i],
            bar_width,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            yerr=oe,
            capsize=1.8,
            error_kw={"lw": 0.7},
        )
    ax.set_xticks([])
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylabel(ylabel, fontsize=9)
    if ylog:
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 1.2)
    ax.grid(True, axis="y", alpha=0.25)


# Fixed axes rectangle so all (a)/(b) panel PDFs share dimensions and
# x-baselines line up when composed side-by-side in LaTeX.
_PANEL_FIGSIZE = (2.6, 1.8)
_PANEL_AXES = (0.23, 0.26, 0.74, 0.70)  # (left, bottom, width, height)


def _make_panel():
    fig, ax = plt.subplots(figsize=_PANEL_FIGSIZE)
    ax.set_position(_PANEL_AXES)
    return fig, ax


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    # No tight_layout — it overrides set_position. bbox_inches=None
    # preserves the exact figsize, guaranteeing matched dimensions.
    kwargs = {"bbox_inches": None, "pad_inches": 0.0}
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "Creator": None, "Producer": None}, **kwargs)
    fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None}, **kwargs)
    fig.savefig(stem.with_suffix(".png"), dpi=200, metadata={"Creation Time": None}, **kwargs)
    plt.close(fig)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["variant", "mode"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        integral_mean=("integral_violation_pu_s", "mean"),
        integral_std=("integral_violation_pu_s", "std"),
        vmin_mean=("worst_vmin", "mean"),
        thpt_mean=("mean_throughput_tps", "mean"),
        itl_miss_mean=("itl_deadline_fraction", "mean"),
        apr_mw=("achievable_power_range_mw", "mean"),
    )


def _pareto_from_row(row, logistic_models) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (batches, power MW, throughput M tok/s) for one variant."""
    variant = row["variant"]
    base = variant.split("_dl")[0]
    spec = mi.SPECS[base]
    fmax = int(row["feasible_batch_max"])
    fmin = int(row["feasible_batch_min"])
    batches = np.array([b for b in sorted(spec.feasible_batch_sizes) if fmin <= b <= fmax], dtype=int)
    p_fit = logistic_models.power(base)
    t_fit = logistic_models.throughput(base)
    n = int(row["num_replicas"])
    p_mw = np.array([float(p_fit.eval(int(b))) * n / 1e6 for b in batches])
    t_mtps = np.array([float(t_fit.eval(int(b))) * n / 1e6 for b in batches])  # tokens/s → M tokens/s
    return batches, p_mw, t_mtps


def _apply_pareto_axes(ax, with_ylabel: bool = True) -> None:
    ax.set_xlabel("Datacenter power (MW)", fontsize=9)
    if with_ylabel:
        ax.set_ylabel("Token throughput\n(M tok/s)", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=max(0, ax.get_xlim()[0]))
    ax.grid(True, alpha=0.3)


def _plot_pareto(
    ax,
    entries,
    logistic_models,
    color_map,
    label_map,
    *,
    marker_size: float = 5.5,
) -> None:
    """Color-only Pareto curves. Each entry is a dict with variant,
    num_replicas, feasible_batch_min, feasible_batch_max."""
    for entry in entries:
        batches, p_mw, t_m = _pareto_from_row(entry, logistic_models)
        if len(batches) == 0:
            continue
        key = entry.get("color_key", entry["variant"])
        color = color_map.get(key, COLORS["ofo"])
        label = label_map.get(entry["variant"], _pretty(entry["variant"]))
        ax.plot(
            p_mw,
            t_m,
            "-o",
            color=color,
            markersize=marker_size,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=label,
            alpha=0.95,
            lw=1.6,
        )


def _pair_bars(
    ax,
    x,
    base_vals,
    ofo_vals,
    *,
    ylog: bool,
    ylabel: str,
    base_errs=None,
    ofo_errs=None,
    tick_labels=None,
    rotation: float = 25.0,
) -> None:
    width = 0.38
    ax.bar(x - width / 2, base_vals, width, yerr=base_errs, color=COLORS["baseline"], label="Baseline", capsize=2.5)
    ax.bar(x + width / 2, ofo_vals, width, yerr=ofo_errs, color=COLORS["ofo"], label="OFO", capsize=2.5)
    ax.set_xticks(x)
    if tick_labels is not None:
        ax.set_xticklabels(tick_labels, rotation=rotation, ha="right" if rotation else "center")
    ax.set_ylabel(ylabel)
    if ylog:
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 2.5)
    ax.grid(True, axis="y", alpha=0.25)


# ──────────────────────────────────────────────────────────────────────
# Model size & architecture


def plot_model_size(
    df,
    out,
    logistic_models,
    *,
    suffix: str = "",
    pareto_xlim: tuple[float, float] | None = None,
) -> tuple[float, float]:
    agg = _aggregate(df)
    all_variants = df["variant"].drop_duplicates().tolist()

    # Sort variants by power-swing range (least flexible → most flexible).
    def _swing(variant: str) -> float:
        row = df[(df["variant"] == variant) & (df["mode"] == OFO_MODE)].iloc[0].to_dict()
        _, p_mw, _ = _pareto_from_row(row, logistic_models)
        return float(p_mw.max() - p_mw.min()) if len(p_mw) else 0.0

    variants = sorted(all_variants, key=_swing)
    palette = plt.get_cmap("tab10").colors
    color_map = {v: palette[i % len(palette)] for i, v in enumerate(variants)}

    def _val(v, mode, col):
        row = agg[(agg.variant == v) & (agg["mode"] == mode)]
        return float(row[col].iloc[0]) if not row.empty else math.nan

    # (a) integral violation — colored bars, hatched (uncoord.) vs solid (coord.)
    fig, ax = _make_panel()
    _grouped_bars(
        ax,
        [_val(v, BASELINE_MODE, "integral_mean") for v in variants],
        [_val(v, OFO_MODE, "integral_mean") for v in variants],
        [color_map[v] for v in variants],
        base_errs=[_val(v, BASELINE_MODE, "integral_std") for v in variants],
        ofo_errs=[_val(v, OFO_MODE, "integral_std") for v in variants],
    )
    _save(fig, out / f"model_size_a{suffix}")

    # (b) Pareto frontier — same color per variant
    entries = [df[(df["variant"] == v) & (df["mode"] == OFO_MODE)].iloc[0].to_dict() for v in variants]
    fig, ax = _make_panel()
    _plot_pareto(ax, entries, logistic_models, color_map=color_map, label_map={v: _pretty(v) for v in variants})
    _apply_pareto_axes(ax)
    resolved_pareto_xlim = ax.get_xlim()
    if pareto_xlim is not None:
        ax.set_xlim(pareto_xlim)
        resolved_pareto_xlim = pareto_xlim
    _save(fig, out / f"model_size_b{suffix}")

    # Shared legend: model colors only (hatch/solid explained in caption).
    handles = [_combo_handle(color_map[v], _pretty(v)) for v in variants]
    _save_legend(handles, out / f"model_size_legend{suffix}", ncol=len(variants), width=4.6)
    return resolved_pareto_xlim


# ──────────────────────────────────────────────────────────────────────
# Parallelism


def plot_parallelism(df, out, logistic_models, *, suffix: str = "") -> None:
    agg = _aggregate(df).merge(
        df[["variant", "pair", "parallelism_label"]].drop_duplicates(),
        on="variant",
        how="left",
    )
    pairs = sorted(agg["pair"].unique())
    for pair in pairs:
        sub = agg[agg["pair"] == pair].copy()
        variants = sub[["variant", "parallelism_label"]].drop_duplicates().sort_values("variant")["variant"].tolist()
        labels = [sub[sub["variant"] == v]["parallelism_label"].iloc[0] for v in variants]
        stem = "parallelism" if len(pairs) == 1 else f"parallelism_{pair}"

        def _v(v, mode, col, sub=sub):
            r = sub[(sub["variant"] == v) & (sub["mode"] == mode)]
            return float(r[col].iloc[0]) if not r.empty else math.nan

        color_map = {labels[i]: (COLORS["h100"] if i == 0 else COLORS["b200"]) for i in range(len(variants))}

        # (a) integral violation — colored bars, hatch = baseline vs OFO
        fig, ax = _make_panel()
        _grouped_bars(
            ax,
            [_v(v, BASELINE_MODE, "integral_mean") for v in variants],
            [_v(v, OFO_MODE, "integral_mean") for v in variants],
            [color_map[lbl] for lbl in labels],
            base_errs=[_v(v, BASELINE_MODE, "integral_std") for v in variants],
            ofo_errs=[_v(v, OFO_MODE, "integral_std") for v in variants],
        )
        _save(fig, out / f"{stem}_a{suffix}")

        # (b) Pareto — same colors.
        entries = []
        for vv, lbl in zip(variants, labels, strict=True):
            r = df[(df["variant"] == vv) & (df["mode"] == OFO_MODE)].iloc[0].to_dict()
            r["color_key"] = lbl
            entries.append(r)
        fig, ax = _make_panel()
        _plot_pareto(
            ax,
            entries,
            logistic_models,
            color_map=color_map,
            label_map={vv: lbl for vv, lbl in zip(variants, labels, strict=True)},
        )
        _apply_pareto_axes(ax)
        _save(fig, out / f"{stem}_b{suffix}")

        # Pareto-only legend (line markers); width tuned to match the
        # rendered font scale of the bar+line legends in other experiments.
        handles = [_line_handle(color_map[lbl], lbl) for lbl in labels]
        _save_legend(handles, out / f"{stem}_legend{suffix}", ncol=len(labels), width=2.3)


# ──────────────────────────────────────────────────────────────────────
# Hardware (per-pair + shared legend)


_HARDWARE_DROP_PAIRS = {"qwen-30b-a3b-instruct-1gpu"}  # MoE finding redundant with model_size


def plot_hardware(df, out, logistic_models, *, suffix: str = "") -> None:
    agg = _aggregate(df).merge(
        df[["variant", "pair", "hardware"]].drop_duplicates(),
        on="variant",
        how="left",
    )
    pairs = [p for p in sorted(agg["pair"].unique()) if p not in _HARDWARE_DROP_PAIRS]
    hardware_order = ["H100", "B200"]

    variants_by_pair_hw: dict[tuple[str, str], str] = {}
    for _, row in agg.drop_duplicates(["pair", "hardware", "variant"]).iterrows():
        variants_by_pair_hw[(row["pair"], row["hardware"])] = row["variant"]

    def _v(pair_, hw, mode, col):
        v = variants_by_pair_hw.get((pair_, hw))
        if v is None:
            return math.nan
        r = agg[(agg["variant"] == v) & (agg["mode"] == mode)]
        return float(r[col].iloc[0]) if not r.empty else math.nan

    # Two-color encoding (blue for the first model pair, red for the
    # second), matching the parallelism and precision figures.
    pair_colors = {p: (COLORS["h100"] if i == 0 else COLORS["b200"]) for i, p in enumerate(pairs)}

    # (a) — integral violation. X-axis = hardware. Within each hardware
    # group, 4 bars: 2 models × (uncoord, coord). Color = model, hatch =
    # uncoord (hatched) / coord (solid).
    bar_width = 0.18
    x = np.arange(len(hardware_order))
    fig, ax = _make_panel()
    n = len(pairs)
    for pi, pair in enumerate(pairs):
        color = pair_colors[pair]
        for mode_idx, mode in enumerate([BASELINE_MODE, OFO_MODE]):
            hatch = "///" if mode == BASELINE_MODE else ""
            offset = (pi - (n - 1) / 2) * 2 * bar_width + (mode_idx - 0.5) * bar_width
            vals = [_v(pair, hw, mode, "integral_mean") for hw in hardware_order]
            ax.bar(x + offset, vals, bar_width, color=color, hatch=hatch, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(hardware_order)
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.5)
    ax.set_ylabel("Integral violation\n(pu·s)", fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)
    _save(fig, out / f"hardware_a{suffix}")

    # (b) — Pareto for both models on BOTH hardware. Color = model
    # (shared with (a) and the main legend). Marker fill = hardware:
    # open = H100, filled = B200. An inline mini-legend inside the axes
    # explains the marker encoding.
    fig, ax = _make_panel()
    for pair in pairs:
        color = pair_colors[pair]
        for hw in hardware_order:
            variant = variants_by_pair_hw.get((pair, hw))
            if variant is None:
                continue
            r = df[(df["variant"] == variant) & (df["mode"] == OFO_MODE)].iloc[0].to_dict()
            batches, p_mw, t_m = _pareto_from_row(r, logistic_models)
            if len(batches) == 0:
                continue
            if hw == "H100":
                ax.plot(
                    p_mw,
                    t_m,
                    "-o",
                    color=color,
                    markersize=5.5,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.3,
                    lw=1.6,
                    alpha=0.95,
                )
            else:
                ax.plot(
                    p_mw,
                    t_m,
                    "-o",
                    color=color,
                    markersize=5.5,
                    markeredgecolor="black",
                    markeredgewidth=0.4,
                    lw=1.6,
                    alpha=0.95,
                )
    _apply_pareto_axes(ax)
    # Inline hardware-marker key (small, inside the axes so it doesn't
    # add to the shared top legend).
    hw_key = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="#555",
            markerfacecolor="white",
            markeredgecolor="#555",
            markeredgewidth=1.3,
            lw=1.0,
            label="H100",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#555",
            markerfacecolor="#555",
            markeredgecolor="black",
            markeredgewidth=0.4,
            lw=1.0,
            label="B200",
        ),
    ]
    ax.legend(
        handles=hw_key,
        loc="upper left",
        frameon=False,
        fontsize=8,
        handlelength=1.2,
        handletextpad=0.4,
        borderaxespad=0.2,
        labelspacing=0.25,
    )
    _save(fig, out / f"hardware_b{suffix}")

    # Shared legend: model colors only.
    handles = [_combo_handle(pair_colors[p], _pretty(p)) for p in pairs]
    _save_legend(handles, out / f"hardware_legend{suffix}", ncol=len(pairs), width=2.5)


# ──────────────────────────────────────────────────────────────────────
# Precision


MAIN_PRECISION_PAIR = "qwen-235b-a22b-instruct-h100-8gpu"


def plot_precision(df, out, logistic_models, *, suffix: str = "") -> None:
    # Main paper figure uses only MAIN_PRECISION_PAIR; the other pairs
    # appear in the appendix.
    df = df[df["pair"] == MAIN_PRECISION_PAIR].copy()
    agg = _aggregate(df).merge(
        df[["variant", "precision_label"]].drop_duplicates(),
        on="variant",
        how="left",
    )
    variants = agg.drop_duplicates("variant").sort_values("variant")["variant"].tolist()
    labels = [agg[agg["variant"] == v]["precision_label"].iloc[0] for v in variants]

    def _v(v, mode, col):
        r = agg[(agg["variant"] == v) & (agg["mode"] == mode)]
        return float(r[col].iloc[0]) if not r.empty else math.nan

    color_map = {"bf16": COLORS["ofo"], "FP8": COLORS["hardware"]}

    # (a) violation — colored bars, hatch = uncoord., solid = coord.
    fig, ax = _make_panel()
    _grouped_bars(
        ax,
        [_v(v, BASELINE_MODE, "integral_mean") for v in variants],
        [_v(v, OFO_MODE, "integral_mean") for v in variants],
        [color_map[lbl] for lbl in labels],
        base_errs=[_v(v, BASELINE_MODE, "integral_std") for v in variants],
        ofo_errs=[_v(v, OFO_MODE, "integral_std") for v in variants],
    )
    _save(fig, out / f"precision_a{suffix}")

    # (b) Pareto — same colors
    entries = []
    for v, lbl in zip(variants, labels, strict=True):
        r = df[(df["variant"] == v) & (df["mode"] == OFO_MODE)].iloc[0].to_dict()
        r["color_key"] = lbl
        entries.append(r)
    fig, ax = _make_panel()
    _plot_pareto(
        ax,
        entries,
        logistic_models,
        color_map=color_map,
        label_map={vv: lbl for vv, lbl in zip(variants, labels, strict=True)},
    )
    _apply_pareto_axes(ax)
    _save(fig, out / f"precision_b{suffix}")

    # Shared legend — colors only; hatch/solid explained in the caption.
    # Display "BF16" (not "bf16") per paper style.
    def _display_prec(label: str) -> str:
        return "BF16" if label == "bf16" else label

    handles = [_combo_handle(color_map[lbl], _display_prec(lbl)) for lbl in sorted(set(labels))]
    _save_legend(handles, out / f"precision_legend{suffix}", ncol=len(set(labels)), width=1.8)


# ──────────────────────────────────────────────────────────────────────


def main(
    indir: Path = OUTDIR,
    outdir: Path = OUTDIR / "figures",
) -> None:
    """Generate figures for every CSV present in `indir`.

    `model_size` and `parallelism` each write one CSV per GPU
    (`<name>_b200.csv`, `<name>_h100.csv`); one figure set per GPU is
    emitted with a matching suffix. `hardware` and `precision` span both
    GPUs in their PAIRS list and write a single CSV (kept under
    `<name>_b200.csv` for filename consistency).

    Args:
        indir: Directory holding per-experiment CSVs.
        outdir: Where to write PDF/SVG/PNG per panel.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%H:%M:%S"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    shared = mi.load_shared_data()
    lm = shared.logistic_models

    dual_gpu = (
        ("model_size", plot_model_size),
        ("parallelism", plot_parallelism),
    )
    model_size_b200_xlim: tuple[float, float] | None = None
    for gpu_tag in ("b200", "h100"):
        suffix = f"_{gpu_tag}"
        for name, plotter in dual_gpu:
            path = indir / f"{name}{suffix}.csv"
            if not path.exists():
                logger.warning("Missing %s — skipping", path)
                continue
            df = pd.read_csv(path)
            logger.info("Loaded %s: %d rows", path.name, len(df))
            if name == "model_size":
                if gpu_tag == "h100" and model_size_b200_xlim is not None:
                    plot_model_size(df, outdir, lm, suffix=suffix, pareto_xlim=model_size_b200_xlim)
                else:
                    resolved = plot_model_size(df, outdir, lm, suffix=suffix)
                    if gpu_tag == "b200":
                        model_size_b200_xlim = resolved
            else:
                plotter(df, outdir, lm, suffix=suffix)

    cross_gpu = (("hardware", plot_hardware), ("precision", plot_precision))
    for name, plotter in cross_gpu:
        path = indir / f"{name}_b200.csv"
        if not path.exists():
            logger.warning("Missing %s — skipping", path)
            continue
        df = pd.read_csv(path)
        logger.info("Loaded %s: %d rows", path.name, len(df))
        plotter(df, outdir, lm, suffix="_b200")

    logger.info("Wrote figures to %s", outdir)


if __name__ == "__main__":
    tyro.cli(main)
