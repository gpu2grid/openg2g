"""Power profile plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_power_3ph(
    time_s: np.ndarray,
    kW_A: np.ndarray,
    kW_B: np.ndarray,
    kW_C: np.ndarray,
    *,
    save_path: Path | str,
    title: str = "DC Power by Phase",
    unit: str = "kW",
    scale: float = 1.0,
    linewidth: float = 0.5,
    training_window: tuple[float, float] | None = None,
    ramp_window: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (12, 4),
) -> None:
    """Plot three-phase power profiles over time.

    Args:
        time_s: Time in seconds.
        kW_A: Phase A power in kW.
        kW_B: Phase B power in kW.
        kW_C: Phase C power in kW.
        save_path: Where to save the figure.
        scale: Multiply kW values by this (e.g., 1e-3 for MW).
        training_window: Optional (t_start, t_end) in seconds for shaded overlay.
        ramp_window: Optional (t_start, t_end) in seconds for shaded overlay.
    """
    fig, ax = plt.subplots(figsize=figsize)
    t_min = time_s / 60.0
    ax.plot(t_min, kW_A * scale, label="Phase A", linewidth=linewidth)
    ax.plot(t_min, kW_B * scale, label="Phase B", linewidth=linewidth)
    ax.plot(t_min, kW_C * scale, label="Phase C", linewidth=linewidth)

    if training_window is not None:
        ax.axvspan(
            training_window[0] / 60,
            training_window[1] / 60,
            alpha=0.15,
            color="cornflowerblue",
            label="Training window",
        )
    if ramp_window is not None:
        ax.axvspan(
            ramp_window[0] / 60,
            ramp_window[1] / 60,
            alpha=0.15,
            color="orange",
            label="Ramp window",
        )

    ax.set_xlabel("Time (min)")
    ax.set_ylabel(f"Power ({unit})")
    ax.set_title(title)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)


def plot_per_model_power(
    time_s: np.ndarray,
    power_by_model_mw: dict[str, np.ndarray],
    *,
    save_path: Path | str,
    title: str = "Per-Model Power",
    figsize: tuple[float, float] = (12, 4),
) -> None:
    """Plot per-model power over time."""
    fig, ax = plt.subplots(figsize=figsize)
    t_min = time_s / 60.0
    for label, pw in power_by_model_mw.items():
        ax.plot(t_min, pw, label=label, linewidth=0.8)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Power (MW)")
    ax.set_title(title)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)
