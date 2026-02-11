"""Batch schedule plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_batch_schedule(
    batch_log_by_model: dict[str, list[int]],
    dt_ctrl: float,
    *,
    save_path: Path | str,
    title: str = "Batch Size Schedule",
    figsize: tuple[float, float] = (12, 4),
    log2_y: bool = True,
) -> None:
    """Plot batch size schedule over time for each model.

    Args:
        batch_log_by_model: ``{model_label: [batch_size_at_each_ctrl_step]}``.
        dt_ctrl: Controller timestep in seconds.
        log2_y: If True, y-axis shows log2(batch_size).
    """
    fig, ax = plt.subplots(figsize=figsize)
    for label, batches in batch_log_by_model.items():
        if not batches:
            continue
        t_ctrl = np.arange(len(batches)) * dt_ctrl
        y = np.log2(np.array(batches, dtype=float)) if log2_y else np.array(batches, dtype=float)
        ax.step(t_ctrl / 60, y, where="post", label=label, linewidth=0.8)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("log\u2082(batch size)" if log2_y else "Batch size")
    ax.set_title(title)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)
