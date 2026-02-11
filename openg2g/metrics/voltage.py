"""Voltage violation metrics for all-bus, all-phase analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openg2g.types import GridState


@dataclass
class VoltageStats:
    """Summary voltage statistics over a simulation run."""

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
    ``max(v_min - v, 0) + max(v - v_max, 0)`` over every non-excluded
    bus-phase pair, then integrates over time.  A snapshot counts as
    "violated" when this sum is positive.

    Args:
        grid_states: Sequence of GridState objects from a simulation run.
        v_min: Lower voltage bound (pu).
        v_max: Upper voltage bound (pu).
        exclude_buses: Bus names to exclude from statistics
            (case-insensitive).

    Returns:
        VoltageStats with worst-case min/max voltages, violation time,
        and integral violation magnitude.
    """
    if not grid_states:
        return VoltageStats(
            worst_vmin=float("nan"),
            worst_vmax=float("nan"),
            violation_time_s=0.0,
            integral_violation_pu_s=0.0,
        )

    exclude = {b.lower() for b in exclude_buses}

    times = np.array([gs.time_s for gs in grid_states], dtype=float)
    if len(times) > 1:
        dt = float(np.median(np.diff(times)))
    else:
        dt = 1.0

    worst_vmin = float("inf")
    worst_vmax = float("-inf")
    violation_steps = 0
    integral_violation = 0.0

    for gs in grid_states:
        snap_min = float("inf")
        snap_max = float("-inf")
        viol_sum = 0.0
        for bus in gs.voltages.buses():
            if bus.lower() in exclude:
                continue
            tp = gs.voltages[bus]
            for v in (tp.a, tp.b, tp.c):
                if np.isnan(v):
                    continue
                if v < snap_min:
                    snap_min = v
                if v > snap_max:
                    snap_max = v
                viol_sum += max(v_min - v, 0.0) + max(v - v_max, 0.0)

        if snap_min < worst_vmin:
            worst_vmin = snap_min
        if snap_max > worst_vmax:
            worst_vmax = snap_max

        if viol_sum > 0.0:
            integral_violation += viol_sum * dt
            violation_steps += 1

    return VoltageStats(
        worst_vmin=float(worst_vmin),
        worst_vmax=float(worst_vmax),
        violation_time_s=float(violation_steps * dt),
        integral_violation_pu_s=float(integral_violation),
    )
