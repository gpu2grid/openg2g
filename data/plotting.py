"""Data characterization plots for the OpenG2G paper.

Ported from Fit_curves.ipynb and Read_data_Llama.ipynb.
These plots visualize the benchmark data used as simulation inputs:
  - Fig. 2: Power trajectories across batch sizes
  - Fig. 3 / Fig. 9: Logistic fits (power, latency, throughput)
  - Fig. 4 / Fig. 10: ITL mixture distributions
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mlenergy_data.modeling import LogisticModel


def plot_power_trajectories(
    trace_dir: Path,
    model_labels: list[str],
    num_gpus_by_model: dict[str, int],
    *,
    batch_sizes: list[int] | None = None,
    xlim_s: tuple[float, float] | None = None,
    rolling_window: int = 10,
    figsize: tuple[float, float] = (10, 5),
    dpi: int = 160,
    save_path: Path | str | None = None,
) -> Figure:
    """Plot total GPU power trajectories per batch size (Fig. 2).

    One subplot per model. Each curve is a different batch size,
    color-coded using the tab10 colormap.

    Args:
        trace_dir: Directory containing per-model trace CSVs with columns
            ``relative_time_s`` and ``power_total_W``.
        model_labels: Models to plot (one subplot each).
        num_gpus_by_model: Number of GPUs for each model label.
        batch_sizes: If given, only plot these batch sizes. Otherwise plot all found.
        xlim_s: If given, (x_lo, x_hi) in seconds to clip the x-axis view.
        rolling_window: Rolling average window for smoothing.
        figsize: Total figure size (width, height) for all subplots combined.
        dpi: Figure DPI.
        save_path: If given, save the figure to this path.
    """
    n_models = len(model_labels)
    fig, axes = plt.subplots(
        n_models,
        1,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
    )

    panel_labels = "abcdefghij"

    for row, model_label in enumerate(model_labels):
        ax: Axes = axes[row, 0]
        num_gpus = num_gpus_by_model[model_label]

        pattern = f"{model_label}_num_gpus_{num_gpus}_max_num_seqs_*.csv"
        csv_files = sorted(
            trace_dir.glob(pattern),
            key=lambda p: _extract_batch_from_filename(p),
        )

        if not csv_files:
            ax.set_title(f"{model_label} (no traces found)")
            continue

        batches_found: list[int] = []
        times: list[np.ndarray] = []
        powers_kw: list[np.ndarray] = []
        for csv_path in csv_files:
            batch = _extract_batch_from_filename(csv_path)
            if batch_sizes is not None and batch not in batch_sizes:
                continue
            df = pd.read_csv(csv_path).sort_values("relative_time_s")
            time_s = df["relative_time_s"].to_numpy(dtype=float)
            power_kw = df["power_total_W"].to_numpy(dtype=float) / 1000.0

            if rolling_window > 1 and len(power_kw) >= rolling_window:
                kernel = np.ones(rolling_window) / rolling_window
                smoothed = np.convolve(power_kw, kernel, mode="same")
                half = rolling_window // 2
                smoothed[:half] = power_kw[:half]
                smoothed[-half:] = power_kw[-half:]
                power_kw = smoothed

            batches_found.append(batch)
            times.append(time_s)
            powers_kw.append(power_kw)

        if not batches_found:
            ax.set_title(f"{model_label} (no matching traces)")
            continue

        cmap = plt.get_cmap("tab10")
        for i, (b, t, p) in enumerate(zip(batches_found, times, powers_kw, strict=True)):
            ax.plot(t, p, label=f"batch={b}", color=cmap(i))

        label_char = panel_labels[row] if row < len(panel_labels) else ""
        gpu_suffix = "GPUs" if num_gpus > 1 else "GPU"
        ax.set_title(
            f"({label_char}) {model_label}: Total-GPU Power ({num_gpus} {gpu_suffix})",
            fontsize=13,
        )
        ax.set_ylabel("Power (kW)", fontsize=11)
        if row == 0:
            ax.legend(
                fontsize=9,
                ncol=len(batches_found),
                loc="lower center",
                frameon=True,
                framealpha=0.9,
            )
        ax.grid(True, alpha=0.3)
        if xlim_s is not None:
            ax.set_xlim(xlim_s[0], xlim_s[1])
        else:
            ax.set_xlim(left=0)

    axes[-1, 0].set_xlabel("Time (seconds)", fontsize=11)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return fig


def plot_logistic_fits(
    summary_df: pd.DataFrame,
    fits_df: pd.DataFrame,
    model_labels: list[str],
    *,
    fit_exclude_batches: dict[str, set[int]] | None = None,
    figsize: tuple[float, float] = (6.45, 5.2),
    dpi: int = 300,
    marker_size: float = 16.0,
    line_width: float = 1.8,
    grid_alpha: float = 0.25,
    title_fontsize: int = 12,
    label_fontsize: int = 10,
    tick_fontsize: int = 10,
    legend_fontsize: int = 9,
    save_path: Path | str | None = None,
) -> Figure:
    """Plot 3x1 stacked logistic fits: power, latency, throughput (Fig. 3/9).

    Multiple models overlaid per panel. Scatter dots for measured data,
    smooth fitted curves from LogisticModel parameters.

    Args:
        summary_df: DataFrame with columns: ``model_label``, ``max_num_seqs``,
            ``avg_power_watts_median``, ``avg_itl_ms_median`` (milliseconds),
            ``throughput_toks_s_median``.
        fits_df: DataFrame with columns: ``model_label``, ``metric``,
            ``L``, ``x0``, ``k``, ``b0``. Latency fits are in seconds.
        model_labels: Models to overlay in each panel.
        fit_exclude_batches: Per-model batch sizes to exclude from scatter dots.
        figsize: Figure size.
        dpi: Figure DPI.
        save_path: If given, save the figure to this path.
    """
    # Metric specs: (metric_name, summary_col, scale_factor, ylabel, title)
    # scale_factor converts summary column units to match logistic fit units.
    # Latency: summary has ms, fits have seconds -> multiply by 0.001.
    metric_specs = [
        (
            "power",
            "avg_power_watts_median",
            1.0,
            "W",
            "(a) Average GPU power consumption vs batch size",
        ),
        (
            "latency",
            "avg_itl_ms_median",
            0.001,
            "s/token",
            "(b) Average inter-token latency vs batch size",
        ),
        (
            "throughput",
            "throughput_toks_s_median",
            1.0,
            "tokens/s",
            "(c) Average token throughput vs batch size",
        ),
    ]

    fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=dpi, sharex=True)
    if fit_exclude_batches is None:
        fit_exclude_batches = {}

    for ax_idx, (ax, (metric_name, y_col, scale, ylabel, title)) in enumerate(zip(axes, metric_specs, strict=True)):
        xmins: list[float] = []
        xmaxs: list[float] = []

        for label in model_labels:
            mdf = summary_df[summary_df["model_label"] == label]
            exclude = fit_exclude_batches.get(label, set())
            if exclude:
                mdf = mdf[~mdf["max_num_seqs"].isin(exclude)]
            if y_col not in mdf.columns or mdf.empty:
                continue
            x = np.log2(mdf["max_num_seqs"].astype(float).to_numpy())
            if len(x) > 0:
                xmins.append(float(np.min(x)))
                xmaxs.append(float(np.max(x)))

        if not xmins:
            ax.set_title(title, fontsize=title_fontsize, loc="center")
            ax.set_ylabel(ylabel, fontsize=label_fontsize)
            ax.grid(True, alpha=grid_alpha)
            continue

        xs = np.linspace(min(xmins), max(xmaxs), 400)

        for label in model_labels:
            mdf = summary_df[summary_df["model_label"] == label]
            exclude = fit_exclude_batches.get(label, set())
            if exclude:
                mdf = mdf[~mdf["max_num_seqs"].isin(exclude)]
            if y_col not in mdf.columns or mdf.empty:
                continue

            x = np.log2(mdf["max_num_seqs"].astype(float).to_numpy())
            y = mdf[y_col].astype(float).to_numpy() * scale

            fit_row = fits_df[(fits_df["model_label"] == label) & (fits_df["metric"] == metric_name)]
            if fit_row.empty:
                continue

            fit = LogisticModel.from_dict(fit_row.iloc[0])
            ys_fit = np.array([fit.eval_x(float(xi)) for xi in xs])

            (line,) = ax.plot(
                xs,
                ys_fit,
                lw=line_width,
                label=label,
                zorder=2,
            )
            ax.scatter(
                x,
                y,
                s=marker_size,
                color=line.get_color(),
                zorder=3,
            )

        ax.set_title(title, fontsize=title_fontsize, loc="center")
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.grid(True, alpha=grid_alpha)
        ax.tick_params(axis="both", labelsize=tick_fontsize)

        if ax_idx == 2:
            ax.legend(frameon=True, fontsize=legend_fontsize, loc="best")

    axes[-1].set_xlabel(
        r"$\log_2(\mathrm{batch\ size})$",
        fontsize=label_fontsize,
    )

    fig.tight_layout(pad=0.35, h_pad=0.6)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return fig


def _lognorm_pdf(x: np.ndarray, sigma: float, scale: float) -> np.ndarray:
    """Standard lognormal PDF: f(x; sigma, scale) for x > 0."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    mask = x > 0
    xx = x[mask]
    out[mask] = (1.0 / (xx * sigma * np.sqrt(2.0 * np.pi))) * np.exp(-(np.log(xx / scale) ** 2) / (2.0 * sigma * sigma))
    return out


def plot_itl_distributions(
    latency_fits_df: pd.DataFrame,
    itl_samples_df: pd.DataFrame,
    model_label: str,
    *,
    batch_sizes: list[int] | None = None,
    inset_batch: int | None = None,
    show_hist: bool = True,
    hist_bins: int = 120,
    hist_alpha: float = 0.12,
    x_lo_q: float = 0.5,
    x_hi_q: float = 99.5,
    grid_n: int = 1200,
    figsize: tuple[float, float] = (7.2, 3.2),
    dpi: int = 160,
    save_path: Path | str | None = None,
) -> Figure:
    """Plot ITL mixture distribution overlay for one model (Fig. 4/10).

    Shows the fitted mixture PDF for each batch size overlaid, with
    optional histograms and an inset showing steady/stall component
    decomposition for one batch size.

    Args:
        latency_fits_df: DataFrame with columns: ``model_label``,
            ``max_num_seqs``, ``itl_mix_pi_steady``, ``itl_mix_sigma_steady``,
            ``itl_mix_scale_steady``, ``itl_mix_sigma_stall``,
            ``itl_mix_scale_stall``.
        itl_samples_df: DataFrame with columns: ``model_label``,
            ``max_num_seqs``, ``itl_s``.
        model_label: Which model to plot.
        batch_sizes: If given, only show these batch sizes. Otherwise show all found.
        inset_batch: Batch size for the component decomposition inset.
            Defaults to the largest batch size.
        show_hist: Whether to show histograms behind PDF curves.
        hist_bins: Number of histogram bins.
        hist_alpha: Histogram transparency.
        x_lo_q: Lower percentile for x-axis range.
        x_hi_q: Upper percentile for x-axis range.
        grid_n: Number of grid points for PDF evaluation.
        figsize: Figure size.
        dpi: Figure DPI.
        save_path: If given, save the figure to this path.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fits = latency_fits_df[latency_fits_df["model_label"] == model_label]
    samples = itl_samples_df[itl_samples_df["model_label"] == model_label]

    if fits.empty:
        raise ValueError(f"No latency fits found for model {model_label!r}")

    batches = sorted(int(b) for b in fits["max_num_seqs"].unique())
    if batch_sizes is not None:
        allowed = set(batch_sizes)
        batches = [b for b in batches if b in allowed]
    if not batches:
        raise ValueError(f"No matching batch sizes for model {model_label!r}")

    all_x = samples[samples["max_num_seqs"].isin(batches)]["itl_s"].to_numpy(dtype=float)
    if len(all_x) == 0:
        raise ValueError(f"No ITL samples found for model {model_label!r}")

    lo = float(np.percentile(all_x, x_lo_q))
    hi = float(np.percentile(all_x, x_hi_q))
    grid = np.linspace(lo, hi, grid_n)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    cmap = plt.get_cmap("tab10") if len(batches) <= 10 else plt.get_cmap("tab20")
    colors = {b: cmap(i % cmap.N) for i, b in enumerate(batches)}

    for b in batches:
        row = fits[fits["max_num_seqs"] == b].iloc[0]
        loc = float(row["itl_mix_loc"])
        pi = float(row["itl_mix_pi_steady"])
        s1 = float(row["itl_mix_sigma_steady"])
        sc1 = float(row["itl_mix_scale_steady"])
        s2 = float(row["itl_mix_sigma_stall"])
        sc2 = float(row["itl_mix_scale_stall"])

        shifted = grid - loc
        pdf_mix = pi * _lognorm_pdf(shifted, s1, sc1) + (1 - pi) * _lognorm_pdf(shifted, s2, sc2)

        c = colors[b]

        if show_hist:
            bsamp = samples[samples["max_num_seqs"] == b]["itl_s"].to_numpy(dtype=float)
            if len(bsamp) > 0:
                ax.hist(
                    bsamp,
                    bins=hist_bins,
                    range=(lo, hi),
                    density=True,
                    alpha=hist_alpha,
                    color=c,
                )

        ax.plot(grid, pdf_mix, linewidth=2.2, color=c, label=f"batch={b}")

    ax.set_title(f"(a) {model_label}: ITL distribution vs batch size")
    ax.set_xlabel("Inter-token latency (seconds)")
    ax.set_ylabel("Density")
    ax.legend(ncol=4, fontsize=9, frameon=True)
    ax.set_xlim(lo, hi)

    if inset_batch is None:
        inset_batch = max(batches)

    if inset_batch in batches:
        row = fits[fits["max_num_seqs"] == inset_batch].iloc[0]
        loc = float(row["itl_mix_loc"])
        pi = float(row["itl_mix_pi_steady"])
        s1 = float(row["itl_mix_sigma_steady"])
        sc1 = float(row["itl_mix_scale_steady"])
        s2 = float(row["itl_mix_sigma_stall"])
        sc2 = float(row["itl_mix_scale_stall"])

        bsamp = samples[samples["max_num_seqs"] == inset_batch]["itl_s"].to_numpy(
            dtype=float,
        )
        lo_i = float(np.percentile(bsamp, 0.5)) if len(bsamp) > 0 else lo
        hi_i = float(np.percentile(bsamp, 99.5)) if len(bsamp) > 0 else hi
        grid_i = np.linspace(lo_i, hi_i, 600)

        shifted_i = grid_i - loc
        pdf_steady = pi * _lognorm_pdf(shifted_i, s1, sc1)
        pdf_stall = (1 - pi) * _lognorm_pdf(shifted_i, s2, sc2)
        pdf_mix_i = pdf_steady + pdf_stall

        axins = inset_axes(
            ax,
            width="38%",
            height="55%",
            loc="lower right",
            bbox_to_anchor=(-0.1, 0.1, 1, 1),
            bbox_transform=ax.transAxes,
        )

        if len(bsamp) > 0:
            axins.hist(
                bsamp,
                bins=60,
                range=(lo_i, hi_i),
                density=True,
                alpha=0.20,
                color=colors[inset_batch],
            )

        axins.plot(
            grid_i,
            pdf_mix_i,
            lw=2.0,
            color=colors[inset_batch],
            label="mixture",
        )
        axins.plot(grid_i, pdf_steady, lw=1.6, ls="--", color="0.25", label="steady")
        axins.plot(grid_i, pdf_stall, lw=1.6, ls=":", color="0.25", label="stall")

        axins.set_title(f"(b) batch={inset_batch} components", fontsize=9)
        axins.set_xlim(lo_i, hi_i)
        axins.tick_params(axis="both", labelsize=8)
        axins.grid(True, alpha=0.25)
        axins.legend(fontsize=8, frameon=True, loc="upper right")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return fig


def _extract_batch_from_filename(path: Path) -> int:
    """Extract max_num_seqs from a trace CSV filename."""
    m = re.search(r"max_num_seqs_(\d+)", path.name)
    return int(m.group(1)) if m else 0
