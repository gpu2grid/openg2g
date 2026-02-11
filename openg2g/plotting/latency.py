"""Latency plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_latency_samples(
    latency_by_model: dict[str, list[float]],
    dt_ctrl: float,
    *,
    save_path: Path | str,
    Lth_by_model: dict[str, float] | None = None,
    title: str = "Sampled Average ITL",
    figsize: tuple[float, float] = (12, 4),
) -> None:
    """Plot sampled inter-token latency per model over time.

    Args:
        latency_by_model: ``{model_label: [avg_itl_at_each_ctrl_step]}``.
        dt_ctrl: Controller timestep in seconds.
        Lth_by_model: Optional latency thresholds to draw as horizontal lines.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for label, lats in latency_by_model.items():
        if not lats:
            continue
        t_ctrl = np.arange(len(lats)) * dt_ctrl
        ax.plot(t_ctrl / 60, lats, marker="o", markersize=1.5, linewidth=0.5, label=label)

    if Lth_by_model:
        for _label, lth in Lth_by_model.items():
            ax.axhline(lth, linestyle=":", linewidth=0.6, alpha=0.6)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Avg ITL (s)")
    ax.set_title(title)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)
