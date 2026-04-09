"""Plotting functions and data-loading helpers for openg2g simulation results.

Reproduces the figures from the G2G paper, reading data from the library's
``SimulationLog`` and ``LLMDatacenterState`` objects.

This module lives outside the ``openg2g`` library on purpose: the library
exports simulation state and metrics, while all matplotlib-dependent
visualization code stays here.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from openg2g.datacenter.base import LLMDatacenterState
from openg2g.datacenter.offline import OfflineDatacenterState
from openg2g.datacenter.online import OnlineDatacenterState
from openg2g.grid.base import GridState

Figure = matplotlib.figure.Figure

# Bus color map (IEEE 13-bus, tab20-based)
# Deterministic colors so all voltage plots use consistent bus coloring.

BUS_COLOR_MAP: dict[str, Any] = {
    "611": (0.1216, 0.4667, 0.7059, 1.0),
    "632": (0.6824, 0.7804, 0.9098, 1.0),
    "633": (1.0, 0.4980, 0.0549, 1.0),
    "634": (1.0, 0.7333, 0.4706, 1.0),
    "645": (0.1725, 0.6275, 0.1725, 1.0),
    "646": (0.5961, 0.8745, 0.5412, 1.0),
    "650": (0.8392, 0.1529, 0.1569, 1.0),
    "652": (1.0, 0.5961, 0.5882, 1.0),
    "670": (0.5804, 0.4039, 0.7412, 1.0),
    "671": (0.7725, 0.6902, 0.8353, 1.0),
    "675": (0.5490, 0.3373, 0.2941, 1.0),
    "680": (0.7686, 0.6118, 0.5804, 1.0),
    "684": (0.8902, 0.4667, 0.7608, 1.0),
    "692": (0.9686, 0.7137, 0.8235, 1.0),
    "rg60": "black",
    "sourcebus": (0.4980, 0.4980, 0.4980, 1.0),
}


# Per-model time-series extraction


@dataclass
class PerModelTimeSeries:
    """Per-model time series extracted from ``SimulationLog.dc_states``."""

    time_s: np.ndarray
    power_w: dict[str, np.ndarray] = field(default_factory=dict)
    itl_s: dict[str, np.ndarray] = field(default_factory=dict)
    batch_size: dict[str, np.ndarray] = field(default_factory=dict)
    active_replicas: dict[str, np.ndarray] = field(default_factory=dict)


def _extract_per_model_timeseries(
    dc_states: Sequence[LLMDatacenterState],
) -> PerModelTimeSeries:
    """Build per-model arrays from a list of ``LLMDatacenterState`` objects.

    ``power_w`` is populated only when states are ``OfflineDatacenterState``
    (which carries ``power_by_model_w``).
    """
    if not dc_states:
        raise ValueError("dc_states is empty.")

    time_s = np.array([s.time_s for s in dc_states])

    model_labels = sorted(dc_states[0].batch_size_by_model.keys())

    batch_size: dict[str, np.ndarray] = {}
    active_replicas: dict[str, np.ndarray] = {}
    itl_s: dict[str, np.ndarray] = {}
    power_w: dict[str, np.ndarray] = {}

    for label in model_labels:
        batch_size[label] = np.array([s.batch_size_by_model.get(label, 0) for s in dc_states])
        active_replicas[label] = np.array([s.active_replicas_by_model.get(label, 0) for s in dc_states])
        itl_s[label] = np.array([s.observed_itl_s_by_model.get(label, float("nan")) for s in dc_states])

    if dc_states and isinstance(dc_states[0], OfflineDatacenterState):
        offline_states = cast(list[OfflineDatacenterState], dc_states)
        for label in model_labels:
            power_w[label] = np.array([s.power_by_model_w.get(label, 0.0) for s in offline_states])

    return PerModelTimeSeries(
        time_s=time_s,
        power_w=power_w,
        itl_s=itl_s,
        batch_size=batch_size,
        active_replicas=active_replicas,
    )


# Helper: bus ordering


def _bus_sort_key(b: str) -> tuple[int, str]:
    """Sort buses numerically when possible: '650' < '671' < '692'."""
    m = re.match(r"^(\d+)", str(b).strip())
    if m:
        return (int(m.group(1)), str(b))
    return (999999, str(b))


# Paper Fig. 5: 2-panel, 3-phase power (MW) + per-model average ITL


def plot_power_and_itl_2panel(
    dc_states: Sequence[LLMDatacenterState],
    *,
    t_train_start_s: float = 1000.0,
    t_train_end_s: float = 2000.0,
    t_ramp_start_s: float = 2500.0,
    t_ramp_end_s: float = 3600.0,
    ramp_label: str = "Less active GPUs",
    show_regimes: bool = True,
    figsize: tuple[float, float] = (7.2, 3.6),
    dpi: int = 300,
    save_path: Path | str | None = None,
) -> Figure:
    """2-panel figure: (a) 3-phase power in MW, (b) per-model average ITL.

    Args:
        dc_states: Sequence of datacenter state snapshots.
        save_path: If given, save figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    per_model = _extract_per_model_timeseries(dc_states)
    time_s = per_model.time_s
    kW_A = np.array([s.power_w.a / 1e3 for s in dc_states])
    kW_B = np.array([s.power_w.b / 1e3 for s in dc_states])
    kW_C = np.array([s.power_w.c / 1e3 for s in dc_states])
    avg_itl_by_model = per_model.itl_s

    t_min = np.asarray(time_s) / 60.0
    t_itl_min = t_min

    t_train_start_min = t_train_start_s / 60.0
    t_train_end_min = t_train_end_s / 60.0
    t_ramp_start_min = t_ramp_start_s / 60.0
    t_ramp_end_min = t_ramp_end_s / 60.0

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=figsize,
        dpi=dpi,
        constrained_layout=True,
    )

    title_pad = 4
    label_fs = 10
    tick_fs = 10
    legend_fs = 9
    lw_main = 0.8
    lw_lat = 0.6

    phase_colors = {
        "A": "#4E79A7",
        "B": "#59A14F",
        "C": "#9C755F",
    }

    # (a) Three-phase power
    ax = axes[0]
    ax.plot(t_min, np.asarray(kW_A) / 1e3, lw=lw_main, color=phase_colors["A"], label="Phase A")
    ax.plot(t_min, np.asarray(kW_B) / 1e3, lw=lw_main, color=phase_colors["B"], label="Phase B")
    ax.plot(t_min, np.asarray(kW_C) / 1e3, lw=lw_main, color=phase_colors["C"], label="Phase C")

    if show_regimes:
        ax.axvspan(t_train_start_min, t_train_end_min, color="tab:blue", alpha=0.10, zorder=0)
        ax.axvline(t_train_start_min, ls="--", lw=1.0, color="tab:blue", alpha=0.9)
        ax.axvline(t_train_end_min, ls="--", lw=1.0, color="tab:blue", alpha=0.9)

    ax.set_ylabel("Power (MW)", fontsize=label_fs)
    ax.set_title("(a) Synthetic three-phase data center power demand", pad=title_pad)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=tick_fs, labelbottom=False, bottom=True)

    handles, labels = ax.get_legend_handles_labels()
    if show_regimes:
        handles.append(Patch(facecolor="tab:blue", alpha=0.10, edgecolor="none"))
        labels.append("Training window")
    ax.legend(handles, labels, fontsize=legend_fs, ncol=4, loc="best", framealpha=0.9)

    # (b) Per-model average ITL
    ax = axes[1]
    for model, lat in avg_itl_by_model.items():
        ax.plot(t_itl_min, np.asarray(lat), lw=lw_lat, label=model)

    if show_regimes:
        ax.axvspan(
            t_ramp_start_min,
            t_ramp_end_min,
            color="tab:orange",
            alpha=0.12,
            zorder=0,
            label=ramp_label,
        )
        ax.axvline(t_ramp_start_min, ls="--", lw=1.0, color="tab:orange", alpha=0.9)
        ax.axvline(t_ramp_end_min, ls="--", lw=1.0, color="tab:orange", alpha=0.9)

    ax.set_ylim(0.0, 0.11)
    ax.set_ylabel("Avg ITL (s)", fontsize=label_fs)
    ax.set_title("(b) Per-model average ITL", pad=title_pad)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=legend_fs, ncol=1, loc="upper left", framealpha=0.9)
    ax.set_xlabel("Time (minutes)", fontsize=label_fs)
    ax.tick_params(labelsize=tick_fs)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", metadata={"Creation Time": None})
        plt.close(fig)
    return fig


# Paper Fig. 6 / Fig. 7: Per-phase all-bus voltages with bus colormap


def plot_allbus_voltages_per_phase(
    grid_states: list[GridState],
    time_s: np.ndarray,
    *,
    save_dir: Path | str,
    v_min: float = 0.95,
    v_max: float = 1.05,
    bus_color_map: dict[str, Any] | None = None,
    figsize_main: tuple[float, float] = (7.2, 2.0),
    figsize_c: tuple[float, float] = (7.2, 2.0),
    dpi: int = 200,
    legend_mode: str = "best",
    drop_buses: tuple[str, ...] = ("sourcebus",),
    reg_bus: str = "rg60",
    reg_label: str = "Regulator bus",
    reg_color: str = "black",
    reg_lw: float = 2.4,
    reg_zorder: int = 5,
    title_template: str = "Voltage trajectories (Phase {label})",
    shared_legend_phase: str = "B",
    filename_template: str = "allbus_voltages_phase_{label}.png",
    show_taps: bool = False,
) -> None:
    """Per-phase all-bus voltage plots with bus-specific colors and shared legend.

    Produces one PNG per phase (A, B, C).  Legend is placed on the
    ``shared_legend_phase`` (default B).

    Args:
        grid_states: List of ``GridState`` from ``SimulationLog``.
        time_s: Time array (seconds) aligned with *grid_states*.
        save_dir: Directory to write PNG files into.
        bus_color_map: {bus_name: color}. Defaults to ``BUS_COLOR_MAP``.
    """
    if bus_color_map is None:
        bus_color_map = dict(BUS_COLOR_MAP)

    save_dir = Path(save_dir)
    t_min = np.asarray(time_s) / 60.0

    drop_set = {str(b).strip().lower() for b in drop_buses}

    # Auto-assign colors for buses not in the color map
    all_plot_buses = set()
    for snap in grid_states[: min(10, len(grid_states))]:
        for bus in snap.voltages.buses():
            if bus.lower() not in drop_set:
                all_plot_buses.add(bus)
    missing = [
        b
        for b in sorted(all_plot_buses, key=_bus_sort_key)
        if b not in bus_color_map and b.lower() not in bus_color_map
    ]
    if missing:
        auto_cmap = plt.colormaps.get_cmap("tab20")
        for i, b in enumerate(missing):
            bus_color_map[b] = auto_cmap(i / max(len(missing) - 1, 1))
    reg_bus_lc = str(reg_bus).strip().lower()

    PHASE_MAP = {"A": 1, "B": 2, "C": 3}
    shared_legend_phase_int = PHASE_MAP.get(shared_legend_phase.upper(), 2)

    # Build buses-with-phase from grid_states
    buses_with_phase: dict[str, set[str]] = {"A": set(), "B": set(), "C": set()}
    for snap in grid_states[: min(10, len(grid_states))]:
        for bus in snap.voltages.buses():
            b_lc = bus.lower()
            if b_lc in drop_set:
                continue
            v = snap.voltages[bus]
            if not np.isnan(v.a):
                buses_with_phase["A"].add(bus)
            if not np.isnan(v.b):
                buses_with_phase["B"].add(bus)
            if not np.isnan(v.c):
                buses_with_phase["C"].add(bus)

    # Build V arrays per phase
    V_by_phase: dict[str, tuple[list[str], np.ndarray]] = {}
    for phase_letter in ("A", "B", "C"):
        buses = sorted(buses_with_phase.get(phase_letter, set()), key=_bus_sort_key)
        if not buses:
            continue
        V = np.full((len(buses), len(t_min)), np.nan, dtype=float)
        for k, snap in enumerate(grid_states):
            for i, b in enumerate(buses):
                if b in snap.voltages:
                    v = snap.voltages[b]
                    V[i, k] = getattr(v, phase_letter.lower())
        V_by_phase[phase_letter] = (buses, V)

    y_lo = 0.93
    y_hi = 1.11

    AX_RECT = (0.10, 0.18, 0.88, 0.72)

    shared_handles = None
    shared_labels = None

    for phase_letter in ("A", "B", "C"):
        if phase_letter not in V_by_phase:
            continue
        buses, V = V_by_phase[phase_letter]
        ph_int = PHASE_MAP[phase_letter]

        figsize = figsize_c if phase_letter == "C" else figsize_main
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes(AX_RECT)

        handles_by_label: dict[str, Any] = {}

        # Sort buses: reg_bus last so it draws on top
        ordered = [b for b in buses if b.lower() != reg_bus_lc]
        ordered += [b for b in buses if b.lower() == reg_bus_lc]

        for b in ordered:
            b_lc = b.lower()
            idx = buses.index(b)

            if b_lc == reg_bus_lc:
                color = reg_color
                lw = reg_lw
                zorder = reg_zorder
                label = reg_label
            else:
                color = bus_color_map.get(b, bus_color_map.get(b_lc, "gray"))
                lw = 1.2
                zorder = 2
                label = b

            (line,) = ax.plot(
                t_min,
                V[idx],
                label=label,
                color=color,
                linewidth=lw,
                zorder=zorder,
            )
            if label not in handles_by_label:
                handles_by_label[label] = line

        ax.axhline(v_min, linestyle="--", linewidth=2.0, alpha=0.9)
        ax.axhline(v_max, linestyle="--", linewidth=2.0, alpha=0.9)

        # Overlay regulator tap positions as bold step lines on secondary y-axis
        if show_taps and grid_states and grid_states[0].tap_positions is not None:
            TAP_STEP = 0.00625
            ax2 = ax.twinx()
            tap_colors = plt.colormaps.get_cmap("Set1")
            phase_lc = phase_letter.lower()

            # Collect all regulator names that match this phase
            all_reg_names = list(grid_states[0].tap_positions.regulators.keys())
            phase_regs = [r for r in all_reg_names if r.lower().endswith(phase_lc)]
            # If no phase-specific regs (e.g. ieee13 with reg1/reg2/reg3), show all
            if not phase_regs:
                phase_regs = all_reg_names

            for r_idx, reg_name in enumerate(phase_regs):
                tap_pu = np.array(
                    [gs.tap_positions.regulators.get(reg_name, 1.0) if gs.tap_positions else 1.0 for gs in grid_states]
                )
                tap_int = np.round((tap_pu - 1.0) / TAP_STEP).astype(int)
                ax2.step(
                    t_min,
                    tap_int,
                    where="post",
                    color=tap_colors(r_idx % 10),
                    linewidth=2.5,
                    alpha=0.7,
                    label=reg_name,
                    zorder=8,
                )

            ax2.set_ylabel("Tap Position (steps)", fontsize=8)
            ax2.tick_params(labelsize=7)
            ax2.legend(
                loc="lower right",
                fontsize=6,
                ncol=max(1, len(phase_regs) // 2),
                framealpha=0.85,
                title="Taps",
                title_fontsize=6,
            )

        ax.set_ylim(y_lo, y_hi)

        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Voltage (pu)")
        ax.set_title(title_template.format(label=phase_letter))
        ax.grid(True, alpha=0.3)

        if shared_handles is None and ph_int == shared_legend_phase_int:
            ordered_labels: list[str] = []
            for b in buses:
                lab = reg_label if b.lower() == reg_bus_lc else b
                if lab in handles_by_label and lab not in ordered_labels:
                    ordered_labels.append(lab)
            shared_labels = ordered_labels
            shared_handles = [handles_by_label[lab] for lab in shared_labels]

        if ph_int == shared_legend_phase_int and shared_handles is not None and shared_labels is not None:
            if legend_mode == "best":
                ax.legend(
                    shared_handles,
                    shared_labels,
                    loc="upper center",
                    ncol=6,
                    fontsize=7,
                    frameon=True,
                    framealpha=0.95,
                )
            else:
                fig.legend(
                    shared_handles,
                    shared_labels,
                    ncol=7,
                    fontsize=7,
                    frameon=True,
                    framealpha=0.95,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.02),
                )

        fig.savefig(
            save_dir / filename_template.format(label=phase_letter),
            bbox_inches="tight",
            metadata={"Creation Time": None},
        )
        plt.close(fig)


# Zone-envelope voltage plot (min/max per zone per phase)


ZONE_COLORS_DEFAULT = {
    "z1_sw": "#2196F3",
    "z2_nw": "#4CAF50",
    "z3_se": "#FF9800",
    "z4_ne": "#E91E63",
}


def plot_zone_voltage_envelope(
    grid_states: list[GridState],
    time_s: np.ndarray,
    *,
    zones: dict[str, list[str]],
    save_dir: Path | str,
    v_min: float = 0.95,
    v_max: float = 1.05,
    zone_colors: dict[str, str] | None = None,
    drop_buses: tuple[str, ...] = ("sourcebus",),
    title_template: str = "Voltage Envelope (Phase {label})",
    filename_template: str = "zone_voltage_envelope_phase_{label}.png",
    figsize: tuple[float, float] = (14, 5),
    dpi: int = 150,
    fill_alpha: float = 0.25,
    y_limits: tuple[float, float] | None = None,
    show_taps: bool = False,
) -> None:
    """Per-phase voltage envelope plot showing min/max per zone.

    For each zone and phase, plots the instantaneous minimum and maximum
    voltage across all buses in the zone, with a shaded fill between them.

    Args:
        grid_states: List of ``GridState`` from ``SimulationLog``.
        time_s: Time array (seconds) aligned with *grid_states*.
        zones: Mapping of zone_id -> list of bus names.
        save_dir: Directory to write PNG files into.
        zone_colors: Optional {zone_id: color} mapping.
    """
    if zone_colors is None:
        zone_colors = ZONE_COLORS_DEFAULT

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    t_min = np.asarray(time_s) / 60.0
    drop_set = {str(b).strip().lower() for b in drop_buses}

    # Build zone bus sets (lowered)
    zone_bus_sets: dict[str, set[str]] = {}
    for zid, buses in zones.items():
        zone_bus_sets[zid] = {b.lower() for b in buses} - drop_set

    # Discover which buses have which phases
    all_buses_with_phase: dict[str, set[str]] = {"a": set(), "b": set(), "c": set()}
    for snap in grid_states[: min(10, len(grid_states))]:
        for bus in snap.voltages.buses():
            b_lc = bus.lower()
            if b_lc in drop_set:
                continue
            v = snap.voltages[bus]
            if not np.isnan(v.a):
                all_buses_with_phase["a"].add(b_lc)
            if not np.isnan(v.b):
                all_buses_with_phase["b"].add(b_lc)
            if not np.isnan(v.c):
                all_buses_with_phase["c"].add(b_lc)

    n_steps = len(grid_states)

    for phase_letter in ("A", "B", "C"):
        ph_lc = phase_letter.lower()
        phase_buses = all_buses_with_phase.get(ph_lc, set())

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for zid in zones:
            color = zone_colors.get(zid, "gray")
            zone_buses = zone_bus_sets[zid] & phase_buses
            if not zone_buses:
                continue

            # Collect voltage timeseries for zone buses
            zone_buses_list = sorted(zone_buses)
            V = np.full((len(zone_buses_list), n_steps), np.nan)
            for k, snap in enumerate(grid_states):
                for i, b in enumerate(zone_buses_list):
                    for bus_name in snap.voltages.buses():
                        if bus_name.lower() == b:
                            V[i, k] = getattr(snap.voltages[bus_name], ph_lc)
                            break

            v_max_arr = np.nanmax(V, axis=0)
            v_min_arr = np.nanmin(V, axis=0)

            ax.plot(t_min, v_max_arr, color=color, linewidth=1.5, label=f"{zid} max")
            ax.plot(t_min, v_min_arr, color=color, linewidth=1.5, linestyle="--", label=f"{zid} min")
            ax.fill_between(t_min, v_min_arr, v_max_arr, color=color, alpha=fill_alpha)

        ax.axhline(v_min, color="red", linestyle=":", linewidth=2.0, alpha=0.8, label="V limits")
        ax.axhline(v_max, color="red", linestyle=":", linewidth=2.0, alpha=0.8)

        # Overlay regulator tap positions on secondary y-axis
        if show_taps and grid_states and grid_states[0].tap_positions is not None:
            TAP_STEP = 0.00625
            ax2 = ax.twinx()
            tap_cmap = plt.colormaps.get_cmap("Set1")
            phase_lc = phase_letter.lower()

            all_reg_names = list(grid_states[0].tap_positions.regulators.keys())
            phase_regs = [r for r in all_reg_names if r.lower().endswith(phase_lc)]
            if not phase_regs:
                phase_regs = all_reg_names

            for r_idx, reg_name in enumerate(phase_regs):
                tap_pu = np.array(
                    [gs.tap_positions.regulators.get(reg_name, 1.0) if gs.tap_positions else 1.0 for gs in grid_states]
                )
                tap_int = np.round((tap_pu - 1.0) / TAP_STEP).astype(int)
                ax2.step(
                    t_min,
                    tap_int,
                    where="post",
                    color=tap_cmap(r_idx % 10),
                    linewidth=2.5,
                    alpha=0.7,
                    label=reg_name,
                    zorder=8,
                )

            ax2.set_ylabel("Tap Position (steps)", fontsize=10)
            ax2.tick_params(labelsize=9)
            ax2.legend(
                loc="lower right",
                fontsize=7,
                ncol=max(1, len(phase_regs) // 2),
                framealpha=0.85,
                title="Taps",
                title_fontsize=7,
            )

        if y_limits is not None:
            ax.set_ylim(*y_limits)
        else:
            all_vals = []
            for zid in zones:
                zone_buses = zone_bus_sets[zid] & phase_buses
                if zone_buses:
                    all_vals.extend([v_min, v_max])
            ax.set_ylim(min(0.93, v_min - 0.02), max(1.07, v_max + 0.02))

        ax.set_xlabel("Time (minutes)", fontsize=13)
        ax.set_ylabel("Voltage (pu)", fontsize=13)
        ax.set_title(title_template.format(label=phase_letter), fontsize=14)
        ax.legend(loc="best", fontsize=9, ncol=3, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)

        fig.tight_layout()
        fig.savefig(
            save_dir / filename_template.format(label=phase_letter),
            bbox_inches="tight",
            metadata={"Creation Time": None},
        )
        plt.close(fig)


# Paper Fig. 8: 4-panel, batch, power/replica, ITL, throughput (OFO)


def plot_model_timeseries_4panel(
    dc_states: Sequence[LLMDatacenterState],
    model_labels: list[str],
    *,
    regime_shading: bool = True,
    t_regime_edges_s: tuple[float, ...] | None = None,
    regime_colors: tuple[str, ...] | None = None,
    regime_labels: tuple[str, ...] | None = None,
    figsize: tuple[float, float] = (7.2, 6.2),
    dpi: int = 300,
    save_path: Path | str | None = None,
) -> Figure:
    """4-panel OFO time-series: batch, power/replica, ITL, throughput.

    Args:
        dc_states: Sequence of datacenter state snapshots.
        model_labels: Ordered model labels for consistent color assignment.
        regime_shading: If True, draw colored background spans.
        t_regime_edges_s: Edge times (seconds) for regime spans.
            Defaults to (0, 950, 2000, 2950, 3100, 3600).
        regime_colors: Colors for each span (len = len(edges) - 1).
            Defaults to ("tab:gray", "tab:red", "tab:gray", "tab:red", "tab:green").
        regime_labels: Legend labels for unique regime colors.
            Defaults to ("Throughput-driven", "Voltage-driven", "Latency-driven").

    Returns:
        The matplotlib Figure object.
    """
    per_model = _extract_per_model_timeseries(dc_states)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_cycle:
        cmap = plt.get_cmap("tab10")
        color_cycle = [cmap(i) for i in range(10)]
    model_to_color = {lab: color_cycle[i % len(color_cycle)] for i, lab in enumerate(model_labels)}

    title_pad = 4
    label_fs = 10
    tick_fs = 10
    legend_fs = 9
    lw_main = 1.3

    fig, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=figsize,
        dpi=dpi,
        constrained_layout=True,
    )

    # Regime shading setup
    if t_regime_edges_s is None:
        t_regime_edges_s = (0, 950, 2000, 2950, 3100, 3600)
    if regime_colors is None:
        regime_colors = ("tab:gray", "tab:red", "tab:gray", "tab:red", "tab:green")
    if regime_labels is None:
        regime_labels = ("Throughput-driven", "Voltage-driven", "Latency-driven")

    t_edges_min = np.array(t_regime_edges_s, dtype=float) / 60.0

    def _apply_overlays(ax: Axes) -> None:
        if not regime_shading:
            return
        for i in range(len(t_edges_min) - 1):
            ax.axvspan(
                t_edges_min[i],
                t_edges_min[i + 1],
                color=regime_colors[i],
                alpha=0.12,
                zorder=0,
            )

    legend_patches: list[Patch] = []
    if regime_shading:
        seen_colors: set[str] = set()
        label_idx = 0
        for c in regime_colors:
            if c not in seen_colors:
                seen_colors.add(c)
                lab = regime_labels[label_idx] if label_idx < len(regime_labels) else ""
                label_idx += 1
                legend_patches.append(
                    Patch(
                        facecolor=c,
                        edgecolor=c,
                        linewidth=1.2,
                        alpha=0.12,
                        label=lab,
                    )
                )

    # (a) log2(batch)
    ax = axes[0]
    t_dc_min = per_model.time_s / 60.0

    for lab in model_labels:
        b = per_model.batch_size.get(lab)
        if b is None or b.size == 0:
            continue
        ax.step(
            t_dc_min[: b.size],
            np.log2(b),
            where="post",
            lw=lw_main,
            color=model_to_color[lab],
            label=lab,
        )

    _apply_overlays(ax)
    ax.set_ylabel(r"$\log_2(\mathrm{batch})$", fontsize=label_fs)
    ax.set_title("(a) Per-model batch size setting (log scale)", pad=title_pad)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=tick_fs)
    ymin, ymax = ax.get_ylim()
    ax.set_yticks(np.arange(np.floor(ymin), np.ceil(ymax) + 1))

    # (b) Power per active replica (kW)
    ax = axes[1]
    for lab in model_labels:
        if lab not in per_model.power_w:
            continue
        p_tot = per_model.power_w[lab]
        wrep = np.maximum(per_model.active_replicas[lab].astype(float), 1.0)
        y = p_tot / wrep / 1e3  # W -> kW
        ax.plot(per_model.time_s / 60.0, y, lw=1, color=model_to_color[lab], label=lab)

    _apply_overlays(ax)
    ax.set_ylabel("Power (kW)", fontsize=label_fs)
    ax.set_title("(b) Per-replica power measurement", pad=title_pad)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=tick_fs)

    # (c) Per-model ITL (legend placed here)
    ax = axes[2]
    for lab in model_labels:
        if lab not in per_model.itl_s:
            continue
        y = per_model.itl_s[lab]
        ax.plot(per_model.time_s / 60.0, y, lw=0.8, color=model_to_color[lab], label=lab)

    _apply_overlays(ax)
    ax.set_ylabel("ITL (s)", fontsize=label_fs)
    ax.set_title("(c) Per-model average ITL", pad=title_pad)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=tick_fs)
    ax.set_ylim(0, 0.2)
    ax.legend(fontsize=legend_fs, ncol=3, loc="upper left", framealpha=0.9)

    # (d) Total throughput per model (tokens/s, log scale)
    ax = axes[3]
    for lab in model_labels:
        if lab not in per_model.itl_s or lab not in per_model.batch_size:
            continue
        itl = per_model.itl_s[lab]
        bs = per_model.batch_size[lab].astype(float)
        wrep = per_model.active_replicas[lab].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            throughput = np.where(itl > 0, bs * wrep / itl, np.nan)
        ax.plot(per_model.time_s / 60.0, throughput, lw=lw_main, color=model_to_color[lab], label=lab)

    ax.set_yscale("log")
    _apply_overlays(ax)
    ax.set_ylabel("Tokens/s", fontsize=label_fs)
    ax.set_title("(d) Per-model total token throughput (log scale)", pad=title_pad)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=tick_fs)
    ax.set_xlabel("Time (minutes)", fontsize=label_fs)

    if legend_patches:
        fig.legend(
            handles=legend_patches,
            loc="upper center",
            ncol=len(legend_patches),
            fontsize=10,
            frameon=False,
            bbox_to_anchor=(0.5, 1.05),
            handlelength=2.2,
            handleheight=1.0,
            columnspacing=1.6,
            borderaxespad=0.2,
        )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", metadata={"Creation Time": None})
        plt.close(fig)
    return fig


# Standalone plots


def plot_power_3ph(
    time_s: np.ndarray,
    kW_A: np.ndarray,
    kW_B: np.ndarray,
    kW_C: np.ndarray,
    *,
    save_path: Path | str | None = None,
    title: str = "DC Power by Phase",
    t_train_start_s: float = 1000.0,
    t_train_end_s: float = 2000.0,
    t_ramp_start_s: float = 2500.0,
    t_ramp_end_s: float = 3000.0,
) -> Figure:
    """3-phase power in MW with training/ramp overlays."""
    t = np.asarray(time_s) / 60.0
    fig, ax = plt.subplots(figsize=(11, 3.2), dpi=160)
    ax.plot(t, np.asarray(kW_A) / 1e3, lw=1.0, label="Phase A")
    ax.plot(t, np.asarray(kW_B) / 1e3, lw=1.0, label="Phase B")
    ax.plot(t, np.asarray(kW_C) / 1e3, lw=1.0, label="Phase C")

    ax.axvspan(
        t_train_start_s / 60.0,
        t_train_end_s / 60.0,
        alpha=0.15,
        label=f"Training overlay ({t_train_start_s:.0f}\u2013{t_train_end_s:.0f}s)",
    )
    ax.axvspan(
        t_ramp_start_s / 60.0,
        t_ramp_end_s / 60.0,
        alpha=0.12,
        label=f"Ramp to 20% ({t_ramp_start_s:.0f}\u2013{t_ramp_end_s:.0f}s)",
    )

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Power (MW)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", metadata={"Creation Time": None})
        plt.close(fig)
    return fig


def plot_voltage_dc_bus(
    time_s: np.ndarray,
    Va: np.ndarray,
    Vb: np.ndarray,
    Vc: np.ndarray,
    *,
    v_min: float = 0.95,
    v_max: float = 1.05,
    save_path: Path | str | None = None,
    title: str = "Voltage at DC bus (3-phase)",
) -> Figure:
    """DC-bus 3-phase voltage with limit lines."""
    t = np.asarray(time_s) / 60.0
    fig, ax = plt.subplots(figsize=(11, 3.2), dpi=160)
    ax.plot(t, np.asarray(Va), lw=1.0, label="Va @ DC bus")
    ax.plot(t, np.asarray(Vb), lw=1.0, alpha=0.8, label="Vb @ DC bus")
    ax.plot(t, np.asarray(Vc), lw=1.0, alpha=0.8, label="Vc @ DC bus")
    ax.axhline(v_min, ls="--", lw=1.6, label="v_min")
    ax.axhline(v_max, ls="--", lw=1.6, label="v_max")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Voltage (pu)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=5, fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", metadata={"Creation Time": None})
        plt.close(fig)
    return fig


def plot_batch_schedule(
    per_model: PerModelTimeSeries,
    *,
    save_path: Path | str | None = None,
    title: str = "Per-model batch schedule (closed-loop)",
) -> Figure:
    """log2 batch schedule, step plot with integer y-ticks."""
    fig, ax = plt.subplots(figsize=(11, 2.6), dpi=160)

    t_min = per_model.time_s / 60.0
    for label, b in per_model.batch_size.items():
        ax.step(t_min[: b.size], np.log2(b), where="post", lw=1.4, label=label)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"$\log_2(\mathrm{batch\ size})$")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    ymin, ymax = ax.get_ylim()
    ax.set_yticks(np.arange(math.floor(ymin), math.ceil(ymax) + 1))

    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", metadata={"Creation Time": None})
        plt.close(fig)
    return fig


def plot_latency_samples(
    per_model: PerModelTimeSeries,
    *,
    itl_deadlines: dict[str, float] | None = None,
    save_path: Path | str | None = None,
    title: str = "Sampled ITL measurements (applied batch)",
) -> Figure:
    """Sampled average ITL per DC timestep with optional deadline overlays."""
    fig, ax = plt.subplots(figsize=(11, 2.8), dpi=160)

    t_min = per_model.time_s / 60.0
    for label, y in per_model.itl_s.items():
        ax.plot(t_min, y, lw=1.2, marker="o", ms=2.5, label=f"ITL[{label}]")
        if itl_deadlines and label in itl_deadlines:
            ax.axhline(float(itl_deadlines[label]), ls="--", lw=1.0)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Sampled avg ITL (s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", metadata={"Creation Time": None})
        plt.close(fig)
    return fig


def plot_per_model_power(
    per_model: PerModelTimeSeries,
    *,
    save_path: Path | str | None = None,
    title: str = "Per-model measured power (whole simulation)",
) -> Figure:
    """Per-model total power in MW."""
    fig, ax = plt.subplots(figsize=(11, 3.2), dpi=160)

    t_min = per_model.time_s / 60.0
    for label, y_w in per_model.power_w.items():
        ax.plot(t_min, np.asarray(y_w) / 1e6, lw=1.1, label=label)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Power (MW)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", metadata={"Creation Time": None})
        plt.close(fig)
    return fig


# Online multi-panel: GPU power, batch size, ITL, KV cache
# (ported from g2g/analyze.py, adapted for SimulationLog)


@dataclass
class OnlinePerModelTimeSeries:
    """Per-model time series extracted from online datacenter states."""

    time_s: np.ndarray
    measured_power_w: dict[str, np.ndarray] = field(default_factory=dict)
    batch_size: dict[str, np.ndarray] = field(default_factory=dict)
    itl_s: dict[str, np.ndarray] = field(default_factory=dict)
    kv_cache_pct: dict[str, np.ndarray] = field(default_factory=dict)
    num_requests_running: dict[str, np.ndarray] = field(default_factory=dict)


def extract_online_per_model_timeseries(
    dc_states: Sequence[OnlineDatacenterState],
) -> OnlinePerModelTimeSeries:
    """Build per-model arrays from a list of `OnlineDatacenterState` objects."""
    if not dc_states:
        raise ValueError("dc_states is empty.")

    time_s = np.array([s.time_s for s in dc_states])
    model_labels = sorted(dc_states[0].batch_size_by_model.keys())

    measured_power_w: dict[str, np.ndarray] = {}
    batch_size: dict[str, np.ndarray] = {}
    itl_s: dict[str, np.ndarray] = {}
    kv_cache_pct: dict[str, np.ndarray] = {}
    num_requests_running: dict[str, np.ndarray] = {}

    for label in model_labels:
        measured_power_w[label] = np.array([s.measured_power_w_by_model.get(label, 0.0) for s in dc_states])
        batch_size[label] = np.array([s.batch_size_by_model.get(label, 0) for s in dc_states])
        itl_s[label] = np.array([s.observed_itl_s_by_model.get(label, float("nan")) for s in dc_states])
        kv_cache_pct[label] = np.array(
            [
                s.prometheus_metrics_by_model.get(label, {}).get("kv_cache_usage_perc", float("nan")) * 100.0
                for s in dc_states
            ]
        )
        num_requests_running[label] = np.array(
            [s.prometheus_metrics_by_model.get(label, {}).get("num_requests_running", float("nan")) for s in dc_states]
        )

    return OnlinePerModelTimeSeries(
        time_s=time_s,
        measured_power_w=measured_power_w,
        batch_size=batch_size,
        itl_s=itl_s,
        kv_cache_pct=kv_cache_pct,
        num_requests_running=num_requests_running,
    )


def plot_online_timeseries(
    per_model: OnlinePerModelTimeSeries,
    model_labels: list[str],
    *,
    schedule_times_s: list[float] | None = None,
    figsize: tuple[float, float] = (14, 12),
    dpi: int = 200,
    save_path: Path | str | None = None,
) -> Figure:
    """Multi-panel online time series: GPU power, batch size, ITL, KV cache.

    Args:
        per_model: Extracted per-model time series from online states.
        model_labels: Ordered model labels for consistent color assignment.
        schedule_times_s: Optional list of batch size change times (seconds)
            to draw as vertical lines on all panels.
        figsize: Figure size in inches.
        dpi: Figure resolution.
        save_path: If given, save figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_cycle:
        cmap = plt.get_cmap("tab10")
        color_cycle = [cmap(i) for i in range(10)]
    model_to_color = {lab: color_cycle[i % len(color_cycle)] for i, lab in enumerate(model_labels)}

    t_s = per_model.time_s
    label_fs = 10
    tick_fs = 9
    legend_fs = 8

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=figsize, dpi=dpi, constrained_layout=True)

    def _draw_schedule_lines(ax: Axes) -> None:
        if schedule_times_s:
            for t in schedule_times_s:
                ax.axvline(t, color="gray", alpha=0.3, linewidth=0.7, linestyle=":")

    # (a) GPU Power (measured, per model)
    ax = axes[0]
    for lab in model_labels:
        if lab in per_model.measured_power_w:
            ax.plot(t_s, per_model.measured_power_w[lab], lw=0.8, color=model_to_color[lab], label=lab)
    ax.set_ylabel("Measured GPU Power (W)", fontsize=label_fs)
    ax.set_title("(a) Per-model measured GPU power")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=legend_fs, ncol=4, loc="upper right")
    ax.tick_params(labelsize=tick_fs)
    _draw_schedule_lines(ax)

    # (b) Batch size (from prometheus num_requests_running or state)
    ax = axes[1]
    for lab in model_labels:
        if lab in per_model.num_requests_running:
            y = per_model.num_requests_running[lab]
            if not np.all(np.isnan(y)):
                ax.step(t_s, y, where="post", lw=1.2, color=model_to_color[lab], label=f"{lab} (running)")
        if lab in per_model.batch_size:
            ax.step(
                t_s,
                per_model.batch_size[lab],
                where="post",
                lw=0.8,
                color=model_to_color[lab],
                linestyle="--",
                alpha=0.6,
                label=f"{lab} (set)",
            )
    ax.set_ylabel("Requests / Batch Size", fontsize=label_fs)
    ax.set_title("(b) Batch size and requests running")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=legend_fs, ncol=4, loc="upper right")
    ax.tick_params(labelsize=tick_fs)
    _draw_schedule_lines(ax)

    # (c) ITL
    ax = axes[2]
    for lab in model_labels:
        if lab in per_model.itl_s:
            ax.plot(t_s, per_model.itl_s[lab] * 1e3, lw=0.8, color=model_to_color[lab], label=lab)
    ax.set_ylabel("ITL (ms)", fontsize=label_fs)
    ax.set_title("(c) Per-model average ITL")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=legend_fs, ncol=4, loc="upper right")
    ax.tick_params(labelsize=tick_fs)
    _draw_schedule_lines(ax)

    # (d) KV cache usage
    ax = axes[3]
    for lab in model_labels:
        if lab in per_model.kv_cache_pct:
            y = per_model.kv_cache_pct[lab]
            if not np.all(np.isnan(y)):
                ax.plot(t_s, y, lw=1.0, color=model_to_color[lab], label=lab)
    ax.set_ylabel("KV Cache Usage (%)", fontsize=label_fs)
    ax.set_title("(d) KV cache usage")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=legend_fs, ncol=4, loc="upper right")
    ax.tick_params(labelsize=tick_fs)
    ax.set_xlabel("Time (s)", fontsize=label_fs)
    _draw_schedule_lines(ax)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", metadata={"Creation Time": None})
        plt.close(fig)
    return fig
