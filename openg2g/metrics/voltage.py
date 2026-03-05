"""Voltage violation metrics for all-bus, all-phase analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openg2g.grid.base import GridState


@dataclass
class VoltageStats:
    """Summary voltage statistics over a simulation run.

    Attributes:
        worst_vmin: Lowest voltage observed across all buses and phases (pu).
        worst_vmax: Highest voltage observed across all buses and phases (pu).
        violation_time_s: Total time with at least one bus-phase violating
            voltage bounds (seconds).
        integral_violation_pu_s: Integrated voltage violation magnitude
            across all bus-phase pairs (pu * s).
    """

    worst_vmin: float
    worst_vmax: float
    violation_time_s: float
    integral_violation_pu_s: float


def compute_allbus_voltage_stats(
    grid_states: list[GridState],
    *,
    v_min: float = 0.95,
    v_max: float = 1.05,
    exclude_buses: tuple[str, ...] = ("rg60",),
) -> VoltageStats:
    """Compute voltage violation statistics across all buses and phases.

    For each snapshot the integral violation sums
    `max(v_min - v, 0) + max(v - v_max, 0)` over every non-excluded
    bus-phase pair, then integrates over time.  A snapshot counts as
    "violated" when this sum is positive.

    Args:
        grid_states: Sequence of [`GridState`][openg2g.grid.base.GridState]
            objects from a simulation run.
        v_min: Lower voltage bound (pu).
        v_max: Upper voltage bound (pu).
        exclude_buses: Bus names to exclude from statistics (case-insensitive).
    """
    if len(grid_states) < 2:
        raise ValueError(
            f"At least two grid states are required to compute voltage statistics (got {len(grid_states)})."
        )

    times = np.array([gs.time_s for gs in grid_states], dtype=float)
    dt = float(np.median(np.diff(times)))

    # Collect bus-phase columns from the first snapshot (all snapshots
    # share the same set of buses for a given OpenDSS circuit).
    exclude = {b.lower() for b in exclude_buses}
    bus_names = [b for b in grid_states[0].voltages.buses() if b.lower() not in exclude]

    # Build (T, N) voltage matrix where N = num_buses * 3.
    T = len(grid_states)
    N = len(bus_names) * 3
    V = np.empty((T, N), dtype=float)
    for t, gs in enumerate(grid_states):
        col = 0
        for bus in bus_names:
            tp = gs.voltages[bus]
            V[t, col] = tp.a
            V[t, col + 1] = tp.b
            V[t, col + 2] = tp.c
            col += 3

    valid = ~np.isnan(V)
    worst_vmin = float(np.min(np.where(valid, V, np.inf)))
    worst_vmax = float(np.max(np.where(valid, V, -np.inf)))

    # Per-timestep violation: sum over all bus-phase pairs
    viol = np.where(valid, np.maximum(v_min - V, 0.0) + np.maximum(V - v_max, 0.0), 0.0)
    viol_sum = np.sum(viol, axis=1)  # shape (T,)

    violation_steps = int(np.count_nonzero(viol_sum > 0.0))
    integral_violation = float(np.sum(viol_sum * dt))

    return VoltageStats(
        worst_vmin=float(worst_vmin),
        worst_vmax=float(worst_vmax),
        violation_time_s=float(violation_steps * dt),
        integral_violation_pu_s=float(integral_violation),
    )
