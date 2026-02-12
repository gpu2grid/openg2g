"""Voltage trajectory plotting."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openg2g.types import GridState


def plot_allbus_voltages_per_phase(
    grid_states: list[GridState],
    time_s: np.ndarray,
    *,
    save_dir: Path | str,
    v_min: float = 0.95,
    v_max: float = 1.05,
    filename_template: str = "all_bus_voltages_phase_{label}.png",
    title_template: str = "All-Bus Voltages — Phase {label}",
    figsize: tuple[float, float] = (12, 4),
    linewidth: float = 0.3,
    alpha: float = 0.7,
    exclude_buses: Sequence[str] = (),
) -> None:
    """Plot per-phase voltage trajectories for all buses.

    Creates one figure per phase (A, B, C), each showing all bus voltages
    for that phase over time.

    Args:
        grid_states: List of GridState objects from the simulation.
        time_s: Time array (seconds), same length as grid_states.
        save_dir: Directory to save the three figures.
        exclude_buses: Bus names to exclude from plotting (case-insensitive).
    """
    save_dir = Path(save_dir)
    exclude_lower = {b.lower() for b in exclude_buses}

    for phase_label, phase_attr in [("A", "a"), ("B", "b"), ("C", "c")]:
        fig, ax = plt.subplots(figsize=figsize)

        bus_traces: dict[str, list[float]] = {}
        for gs in grid_states:
            for bus in gs.voltages.buses():
                if bus.lower() in exclude_lower:
                    continue
                v = getattr(gs.voltages[bus], phase_attr)
                if np.isnan(v):
                    continue
                if bus not in bus_traces:
                    bus_traces[bus] = []
                bus_traces[bus].append(v)

        t_min = time_s / 60.0
        for bus, trace in bus_traces.items():
            if len(trace) == len(time_s):
                ax.plot(t_min, trace, linewidth=linewidth, alpha=alpha, label=bus)

        ax.axhline(v_min, color="red", linestyle="--", linewidth=0.8)
        ax.axhline(v_max, color="red", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Voltage (pu)")
        ax.set_title(title_template.format(label=phase_label))
        ax.set_xlim(0, None)
        ax.set_ylim(0.93, 1.11)
        fig.tight_layout()

        fname = filename_template.format(label=phase_label)
        fig.savefig(str(save_dir / fname), bbox_inches="tight")
        plt.close(fig)


def plot_dc_bus_voltage(
    time_s: np.ndarray,
    Va: np.ndarray,
    Vb: np.ndarray,
    Vc: np.ndarray,
    *,
    save_path: Path | str,
    v_min: float = 0.95,
    v_max: float = 1.05,
    title: str = "DC Bus Voltage",
    figsize: tuple[float, float] = (12, 4),
) -> None:
    """Plot voltage at the DC bus for all three phases."""
    fig, ax = plt.subplots(figsize=figsize)
    t_min = time_s / 60.0
    ax.plot(t_min, Va, label="Phase A", linewidth=0.8)
    ax.plot(t_min, Vb, label="Phase B", linewidth=0.8)
    ax.plot(t_min, Vc, label="Phase C", linewidth=0.8)
    ax.axhline(v_min, color="red", linestyle="--", linewidth=0.8, label=f"V_min={v_min}")
    ax.axhline(v_max, color="red", linestyle="--", linewidth=0.8, label=f"V_max={v_max}")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Voltage (pu)")
    ax.set_title(title)
    ax.set_xlim(0, None)
    ax.set_ylim(0.93, 1.11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)
